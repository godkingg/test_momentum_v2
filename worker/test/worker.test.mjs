// Chạy: node --test worker/test/      (Node ≥ 20, không cần cài thêm gì)
import test from "node:test";
import assert from "node:assert/strict";
import { handle, verifySignature, parseDateOption, takeCooldown, authorize } from "../worker.js";

const NOW = Date.UTC(2026, 9, 6, 4, 0, 0);                    // 06/10/2026 11:00 giờ VN
const SECRET_GH = "ghp_SECRET_TOKEN_VALUE";
const enc = new TextEncoder();
const toHex = (buf) => Buffer.from(buf).toString("hex");

const kp = await crypto.subtle.generateKey("Ed25519", true, ["sign", "verify"]);
const PUBLIC_KEY = toHex(await crypto.subtle.exportKey("raw", kp.publicKey));

async function signedRequest(obj, { ts = Math.floor(NOW / 1000), tamper = false, method = "POST" } = {}) {
  const body = typeof obj === "string" ? obj : JSON.stringify(obj);
  const sig = toHex(await crypto.subtle.sign("Ed25519", kp.privateKey, enc.encode(String(ts) + body)));
  return new Request("https://worker.example/", {
    method, body: method === "POST" ? (tamper ? body + " " : body) : undefined,
    headers: { "x-signature-ed25519": sig, "x-signature-timestamp": String(ts), "content-type": "application/json" },
  });
}

const command = (over = {}) => ({
  type: 2, id: "i1", application_id: "111111111111111111", token: "INTERACTION_TOKEN_xyz", guild_id: "222222222222222222",
  channel_id: "333333333333333333", member: { user: { id: "444444444444444444" }, roles: ["555555555555555555"] },
  data: { name: "report", options: [] }, ...over,
});

function makeEnv(over = {}) {
  return { DISCORD_PUBLIC_KEY: PUBLIC_KEY, GH_TOKEN: SECRET_GH, GH_REPO: "me/momentum", ...over };
}

function fakeKV() {
  const m = new Map();
  return { get: async (k) => m.get(k) ?? null, put: async (k, v) => void m.set(k, v), delete: async (k) => void m.delete(k), _m: m };
}

/** Chạy handler, đợi các tác vụ nền (waitUntil), trả về { res, calls }. githubStatus: mã trả về của GitHub dispatch. */
async function run(obj, env = makeEnv(), { githubStatus = 204, now = NOW, reqOpts, githubThrows = false } = {}) {
  const calls = [], tasks = [];
  const fetchFn = async (url, init) => {
    calls.push({ url, init, body: init?.body ? JSON.parse(init.body) : null });
    if (url.startsWith("https://api.github.com/")) {
      if (githubThrows) throw new Error("network down " + SECRET_GH);
      return { status: githubStatus, ok: githubStatus < 300 };
    }
    return { status: 200, ok: true };
  };
  const res = await handle(await signedRequest(obj, reqOpts), env, { waitUntil: (p) => tasks.push(p) }, fetchFn, now);
  await Promise.all(tasks);
  const text = await res.text();
  return { res, calls, body: text ? (() => { try { return JSON.parse(text); } catch { return text; } })() : null };
}

const patchedText = (calls) => calls.filter((c) => c.init?.method === "PATCH").map((c) => c.body.content).join("\n");

// ------------------------------------------------------------------ chữ ký
test("GET trả 200 (health check)", async () => {
  const res = await handle(new Request("https://w.example/"), makeEnv(), { waitUntil() {} });
  assert.equal(res.status, 200);
});

test("PING hợp lệ → PONG", async () => {
  const { res, body } = await run({ type: 1 });
  assert.equal(res.status, 200);
  assert.deepEqual(body, { type: 1 });
});

test("chữ ký sai / thiếu / quá hạn / khoá hỏng → 401, không throw", async () => {
  assert.equal((await run({ type: 1 }, makeEnv(), { reqOpts: { tamper: true } })).res.status, 401);
  assert.equal((await run({ type: 1 }, makeEnv(), { reqOpts: { ts: Math.floor(NOW / 1000) - 3600 } })).res.status, 401);
  assert.equal((await run({ type: 1 }, makeEnv({ DISCORD_PUBLIC_KEY: "zz" }))).res.status, 401);
  assert.equal((await run({ type: 1 }, makeEnv({ DISCORD_PUBLIC_KEY: undefined }))).res.status, 401);
  const noHeaders = await handle(new Request("https://w.example/", { method: "POST", body: "{}" }), makeEnv(), { waitUntil() {} }, fetch, NOW);
  assert.equal(noHeaders.status, 401);
  assert.equal(await verifySignature("{}", "", "", PUBLIC_KEY, NOW), false);
});

// ------------------------------------------------------------------ luồng chính
test("/report: trả 'deferred' ngay, rồi kích hoạt workflow đúng tham số và sửa tin nhắn gốc", async () => {
  const { res, calls, body } = await run(command());
  assert.deepEqual(body, { type: 5 });
  const gh = calls.find((c) => c.url.startsWith("https://api.github.com/"));
  assert.equal(gh.url, "https://api.github.com/repos/me/momentum/actions/workflows/report_on_demand.yml/dispatches");
  assert.equal(gh.init.headers.authorization, `Bearer ${SECRET_GH}`);
  assert.equal(gh.init.headers["x-github-api-version"], "2022-11-28");
  assert.ok(gh.init.headers["user-agent"]);                                    // GitHub bắt buộc có User-Agent
  assert.deepEqual(gh.body, { ref: "main", inputs: { channel_id: "333333333333333333", requester_id: "444444444444444444", asof: "" } });
  const patch = calls.find((c) => c.init?.method === "PATCH");
  assert.equal(patch.url, "https://discord.com/api/v10/webhooks/111111111111111111/INTERACTION_TOKEN_xyz/messages/@original");
  assert.match(patch.body.content, /<@444444444444444444>.*ngay bây giờ.*3–5 phút/);
  assert.deepEqual(patch.body.allowed_mentions, { parse: [], users: [] });
  assert.ok(!JSON.stringify(calls.map((c) => c.body)).includes(SECRET_GH) || gh.init.headers.authorization.includes(SECRET_GH));
});

test("/report date: chuẩn hoá DD/MM/YYYY → ISO và nói rõ ngày trong tin nhắn", async () => {
  const { calls } = await run(command({ data: { name: "report", options: [{ name: "date", type: 3, value: "02/10/2026" }] } }));
  assert.equal(calls.find((c) => c.url.startsWith("https://api.github.com/")).body.inputs.asof, "2026-10-02");
  assert.match(patchedText(calls), /02\/10\/2026/);
});

test("parseDateOption: hợp lệ / không tồn tại / tương lai / quá xa / rác (kể cả chuỗi chèn lệnh)", () => {
  assert.deepEqual(parseDateOption("2026-10-06", NOW), { iso: "2026-10-06" });
  assert.deepEqual(parseDateOption("6-10-2026", NOW), { iso: "2026-10-06" });
  assert.deepEqual(parseDateOption("", NOW), { iso: "" });
  assert.deepEqual(parseDateOption(undefined, NOW), { iso: "" });
  for (const bad of ["31/02/2026", "2026-13-01", "2026-10-07", "2015-01-01", "abc", "2026-10-02; rm -rf /", "$(whoami)", "2026-10-02 rm"]) {
    assert.ok(parseDateOption(bad, NOW).error, `phải từ chối: ${JSON.stringify(bad)}`);
  }
  // khoảng trắng thừa được bỏ; kết quả luôn được DỰNG LẠI từ số nên không thể mang ký tự lạ vào workflow
  assert.deepEqual(parseDateOption(" 2026-10-02\n", NOW), { iso: "2026-10-02" });
  // 06/10 17:30 UTC = 07/10 00:30 giờ VN → "hôm nay" theo giờ VN đã là 07/10
  assert.deepEqual(parseDateOption("2026-10-07", Date.UTC(2026, 9, 6, 17, 30)), { iso: "2026-10-07" });
});

test("ngày sai → trả lời riêng tư (ephemeral) và KHÔNG gọi GitHub", async () => {
  const { body, calls } = await run(command({ data: { name: "report", options: [{ name: "date", value: "31/02/2026" }] } }));
  assert.equal(body.type, 4);
  assert.equal(body.data.flags, 64);
  assert.equal(calls.length, 0);
});

// ------------------------------------------------------------------ phân quyền
test("authorize: DM, sai server, whitelist user/role", () => {
  assert.match(authorize({ ...command(), guild_id: undefined }, makeEnv()), /server/);
  assert.match(authorize(command(), makeEnv({ ALLOWED_GUILD_IDS: "999" })), /không được phép/);
  assert.equal(authorize(command(), makeEnv({ ALLOWED_GUILD_IDS: "222222222222222222, 999" })), null);
  assert.equal(authorize(command(), makeEnv({ ALLOWED_USER_IDS: "444444444444444444" })), null);
  assert.equal(authorize(command(), makeEnv({ ALLOWED_ROLE_IDS: "555555555555555555" })), null);
  assert.match(authorize(command(), makeEnv({ ALLOWED_USER_IDS: "1", ALLOWED_ROLE_IDS: "2" })), /chưa được cấp quyền/);
});

test("người không được phép → ephemeral, không gọi GitHub", async () => {
  const { body, calls } = await run(command(), makeEnv({ ALLOWED_GUILD_IDS: "999" }));
  assert.equal(body.type, 4); assert.equal(body.data.flags, 64); assert.equal(calls.length, 0);
});

test("lệnh lạ và Worker thiếu cấu hình → ephemeral", async () => {
  const unknown = await run(command({ data: { name: "other" } }));
  assert.equal(unknown.body.data.flags, 64);
  const cfg = await run(command(), makeEnv({ GH_TOKEN: undefined }));
  assert.match(cfg.body.data.content, /chưa được cấu hình/); assert.equal(cfg.calls.length, 0);
});

// ------------------------------------------------------------------ cooldown
test("cooldown: cùng người trong 180s bị chặn, sau đó được; người khác trong 45s toàn cục bị chặn", async () => {
  const kv = fakeKV();
  const env = makeEnv({ COOLDOWN: kv });
  const first = await run(command(), env);
  assert.ok(first.calls.some((c) => c.url.startsWith("https://api.github.com/")));

  const again = await run(command(), env, { now: NOW + 30_000, reqOpts: { ts: Math.floor((NOW + 30_000) / 1000) } });
  assert.ok(!again.calls.some((c) => c.url.startsWith("https://api.github.com/")));
  assert.match(patchedText(again.calls), /thử lại sau ~\d+ giây/);

  const other = command({ member: { user: { id: "666666666666666666" }, roles: [] } });
  const blockedGlobal = await run(other, env, { now: NOW + 30_000, reqOpts: { ts: Math.floor((NOW + 30_000) / 1000) } });
  assert.ok(!blockedGlobal.calls.some((c) => c.url.startsWith("https://api.github.com/")));

  const okOther = await run(other, env, { now: NOW + 60_000, reqOpts: { ts: Math.floor((NOW + 60_000) / 1000) } });     // qua 45s toàn cục
  assert.ok(okOther.calls.some((c) => c.url.startsWith("https://api.github.com/")));

  const okAgain = await run(command(), env, { now: NOW + 200_000, reqOpts: { ts: Math.floor((NOW + 200_000) / 1000) } });
  assert.ok(okAgain.calls.some((c) => c.url.startsWith("https://api.github.com/")));
});

test("takeCooldown: không có KV thì cho qua; KV TTL luôn ≥ 60s", async () => {
  assert.equal((await takeCooldown({}, "u", NOW)).ok, true);
  const puts = [];
  const kv = { get: async () => null, put: async (k, v, o) => puts.push(o.expirationTtl), delete: async () => {} };
  await takeCooldown({ COOLDOWN: kv, USER_COOLDOWN_SECONDS: "10", GLOBAL_COOLDOWN_SECONDS: "5" }, "u", NOW);
  assert.deepEqual(puts.sort(), [60, 60]);
});

// ------------------------------------------------------------------ lỗi GitHub
test("GitHub 404/403: báo lỗi kèm gợi ý, KHÔNG lộ token, và nhả cooldown", async () => {
  const kv = fakeKV();
  const { calls } = await run(command(), makeEnv({ COOLDOWN: kv }), { githubStatus: 404 });
  const msg = patchedText(calls);
  assert.match(msg, /❌.*404.*GH_TOKEN/);
  assert.ok(!msg.includes(SECRET_GH) && !msg.includes("INTERACTION_TOKEN") );
  assert.equal(kv._m.size, 0);                                                  // cooldown đã được nhả → người dùng thử lại được ngay
});

test("lỗi mạng khi gọi GitHub: không lộ chi tiết lỗi (có thể chứa token)", async () => {
  const { calls } = await run(command(), makeEnv(), { githubThrows: true });
  const msg = patchedText(calls);
  assert.match(msg, /❌.*lỗi mạng/);
  assert.ok(!msg.includes(SECRET_GH));
});
