// Cloudflare Worker — endpoint cho slash command /report của Discord (Interactions Endpoint URL).
//
//   Discord ──POST──► Worker (xác thực chữ ký Ed25519, trả lời "đang xử lý" trong <3s)
//                        └─ waitUntil: kiểm tra quyền + cooldown ─► GitHub API (workflow_dispatch) ─► Actions tạo báo cáo
//                                                                    └─ sửa tin nhắn gốc: "đã nhận yêu cầu…"
//   Báo cáo (PNG + HTML) do GitHub Actions đăng vào ĐÚNG KÊNH người dùng gõ lệnh (bằng bot token).
//
// Một file duy nhất, không phụ thuộc thư viện → dán thẳng vào Cloudflare Dashboard (Workers → Edit code) hoặc `wrangler deploy`.
//
// Biến cấu hình (Settings → Variables and Secrets):
//   DISCORD_PUBLIC_KEY   (Variable)  Public Key của Application (Developer Portal → General Information)
//   GH_TOKEN             (Secret)    Fine-grained PAT, CHỈ repo này, quyền "Actions: Read and write"
//   GH_REPO              (Variable)  "owner/repo"
//   GH_REF               (Variable, tuỳ chọn, mặc định "main")
//   GH_WORKFLOW          (Variable, tuỳ chọn, mặc định "report_on_demand.yml")
//   ALLOWED_GUILD_IDS    (tuỳ chọn)  danh sách ID server được phép, cách nhau dấu phẩy. KHUYẾN NGHỊ đặt.
//   ALLOWED_USER_IDS / ALLOWED_ROLE_IDS (tuỳ chọn)  nếu đặt, người gọi phải thuộc 1 trong 2 danh sách
//   USER_COOLDOWN_SECONDS (mặc định 180) · GLOBAL_COOLDOWN_SECONDS (mặc định 45)
//   COOLDOWN             (KV namespace binding, tuỳ chọn nhưng KHUYẾN NGHỊ) lưu mốc cooldown — không có KV thì không chống spam được.

const API = "https://discord.com/api/v10";
const MAX_SKEW_SECONDS = 300;           // chống replay: timestamp chữ ký lệch quá 5 phút bị từ chối
const EPHEMERAL = 64;
const enc = new TextEncoder();

// ----------------------------------------------------------------------------- tiện ích
export function hexToBytes(hex) {
  if (typeof hex !== "string" || hex.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(hex)) throw new Error("hex không hợp lệ");
  return Uint8Array.from(hex.match(/../g) ?? [], (h) => parseInt(h, 16));
}

const json = (obj, status = 200) => new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
const ephemeral = (content) => json({ type: 4, data: { content, flags: EPHEMERAL, allowed_mentions: { parse: [] } } });
const list = (s) => (s ?? "").split(",").map((x) => x.trim()).filter(Boolean);

// ----------------------------------------------------------------------------- xác thực chữ ký của Discord
export async function verifySignature(body, signatureHex, timestamp, publicKeyHex, nowMs = Date.now()) {
  try {
    if (!signatureHex || !timestamp || !publicKeyHex) return false;
    if (!/^\d+$/.test(timestamp) || Math.abs(nowMs / 1000 - Number(timestamp)) > MAX_SKEW_SECONDS) return false;
    const keyBytes = hexToBytes(publicKeyHex);
    const data = enc.encode(timestamp + body);
    const sig = hexToBytes(signatureHex);
    let key;
    try {
      key = await crypto.subtle.importKey("raw", keyBytes, { name: "Ed25519" }, false, ["verify"]);
      return await crypto.subtle.verify({ name: "Ed25519" }, key, sig, data);
    } catch {
      // Runtime cũ của Cloudflare dùng tên thuật toán riêng
      key = await crypto.subtle.importKey("raw", keyBytes, { name: "NODE-ED25519", namedCurve: "NODE-ED25519" }, false, ["verify"]);
      return await crypto.subtle.verify("NODE-ED25519", key, sig, data);
    }
  } catch {
    return false;
  }
}

// ----------------------------------------------------------------------------- tham số ngày
/** Chấp nhận YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY. Trả về { iso } hoặc { error }. Chỉ cho ngày đã qua (≤ hôm nay giờ VN), trong 5 năm. */
export function parseDateOption(raw, nowMs = Date.now()) {
  if (raw === undefined || raw === null || String(raw).trim() === "") return { iso: "" };
  const s = String(raw).trim();
  let y, m, d, hit;
  if ((hit = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})$/))) [y, m, d] = [hit[1], hit[2], hit[3]];
  else if ((hit = s.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$/))) [d, m, y] = [hit[1], hit[2], hit[3]];
  else return { error: "Ngày không hợp lệ. Dùng dạng `2026-10-02` hoặc `02/10/2026`." };
  const [Y, M, D] = [Number(y), Number(m), Number(d)];
  const dt = new Date(Date.UTC(Y, M - 1, D));
  if (dt.getUTCFullYear() !== Y || dt.getUTCMonth() !== M - 1 || dt.getUTCDate() !== D) return { error: `Ngày \`${s}\` không tồn tại.` };
  const vnToday = new Date(nowMs + 7 * 3600 * 1000);                       // giờ Việt Nam = UTC+7
  const todayUtc = Date.UTC(vnToday.getUTCFullYear(), vnToday.getUTCMonth(), vnToday.getUTCDate());
  if (dt.getTime() > todayUtc) return { error: "Chỉ hỗ trợ ngày đã qua hoặc hôm nay. Bỏ trống để lấy báo cáo ngay bây giờ." };
  if (todayUtc - dt.getTime() > 5 * 366 * 86400 * 1000) return { error: "Ngày quá xa (tối đa 5 năm)." };
  return { iso: `${String(Y).padStart(4, "0")}-${String(M).padStart(2, "0")}-${String(D).padStart(2, "0")}` };
}

// ----------------------------------------------------------------------------- phân quyền
/** Trả về chuỗi lỗi (hiện riêng cho người gọi) hoặc null nếu được phép. */
export function authorize(interaction, env) {
  if (!interaction.guild_id) return "Lệnh này chỉ dùng được trong server.";
  const guilds = list(env.ALLOWED_GUILD_IDS);
  if (guilds.length && !guilds.includes(interaction.guild_id)) return "Server này không được phép dùng lệnh.";
  const allowUsers = list(env.ALLOWED_USER_IDS);
  const allowRoles = list(env.ALLOWED_ROLE_IDS);
  if (allowUsers.length || allowRoles.length) {
    const uid = interaction.member?.user?.id;
    const roles = interaction.member?.roles ?? [];
    if (!(allowUsers.includes(uid) || roles.some((r) => allowRoles.includes(r)))) return "Bạn chưa được cấp quyền dùng lệnh này.";
  }
  return null;
}

// ----------------------------------------------------------------------------- cooldown (Cloudflare KV, best-effort)
/**
 * Kiểm tra + đặt cooldown. KV nhất quán sau cùng (có thể trễ ~1 phút giữa các điểm biên) nên đây là chống spam "tương đối",
 * không phải khoá chặt. Không có KV binding → bỏ qua (không giới hạn).
 * Trả về { ok: true, release() } hoặc { ok: false, waitSeconds }.
 */
export async function takeCooldown(env, userId, nowMs = Date.now()) {
  const kv = env.COOLDOWN;
  if (!kv) return { ok: true, release: async () => {} };
  const userTtl = Number(env.USER_COOLDOWN_SECONDS ?? 180);
  const globalTtl = Number(env.GLOBAL_COOLDOWN_SECONDS ?? 45);
  const keys = { user: `cd:user:${userId}`, global: "cd:global" };
  let wait = 0;
  for (const k of Object.values(keys)) {
    const until = Number(await kv.get(k));
    if (until > nowMs) wait = Math.max(wait, Math.ceil((until - nowMs) / 1000));
  }
  if (wait > 0) return { ok: false, waitSeconds: wait };
  const put = (k, ttl) => kv.put(k, String(nowMs + ttl * 1000), { expirationTtl: Math.max(60, Math.ceil(ttl)) });   // KV yêu cầu TTL ≥ 60s
  await Promise.all([put(keys.user, userTtl), put(keys.global, globalTtl)]);
  return { ok: true, release: async () => { await Promise.all(Object.values(keys).map((k) => kv.delete(k))); } };
}

// ----------------------------------------------------------------------------- gọi API ngoài
async function editOriginal(interaction, env, content, fetchFn = fetch) {
  const url = `${API}/webhooks/${interaction.application_id}/${interaction.token}/messages/@original`;
  const r = await fetchFn(url, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ content, allowed_mentions: { parse: [], users: [] } }),
  });
  return r.ok;
}

/** Kích hoạt workflow trên GitHub. Trả về { ok } hoặc { ok:false, status }. KHÔNG bao giờ đưa token vào thông báo lỗi. */
export async function dispatchWorkflow(env, inputs, fetchFn = fetch) {
  const repo = env.GH_REPO;
  const workflow = env.GH_WORKFLOW || "report_on_demand.yml";
  const ref = env.GH_REF || "main";
  const r = await fetchFn(`https://api.github.com/repos/${repo}/actions/workflows/${workflow}/dispatches`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.GH_TOKEN}`,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
      "user-agent": "momentum-vn30-report-worker",
      "content-type": "application/json",
    },
    body: JSON.stringify({ ref, inputs }),
  });
  return r.status === 204 ? { ok: true } : { ok: false, status: r.status };
}

// ----------------------------------------------------------------------------- xử lý lệnh /report (chạy nền sau khi đã trả lời Discord)
export async function processReport(interaction, env, fetchFn = fetch, nowMs = Date.now()) {
  const user = interaction.member?.user ?? interaction.user ?? {};
  const mention = user.id ? `<@${user.id}>` : "";
  const dateOpt = (interaction.data?.options ?? []).find((o) => o.name === "date")?.value;
  const parsed = parseDateOption(dateOpt, nowMs);                                    // (đã kiểm tra ở handler; kiểm tra lại cho chắc)
  if (parsed.error) return editOriginal(interaction, env, `${mention} ❌ ${parsed.error}`, fetchFn);

  const cd = await takeCooldown(env, user.id ?? "anon", nowMs);
  if (!cd.ok) {
    return editOriginal(interaction, env, `${mention} ⏳ Vừa có báo cáo được yêu cầu — thử lại sau ~${cd.waitSeconds} giây.`, fetchFn);
  }
  let res;
  try {
    res = await dispatchWorkflow(env, { channel_id: String(interaction.channel_id), requester_id: String(user.id ?? ""), asof: parsed.iso }, fetchFn);
  } catch {
    res = { ok: false, status: 0 };
  }
  if (!res.ok) {
    await cd.release();                                                              // lỗi hệ thống không được tính vào cooldown của người dùng
    const hint = [401, 403, 404].includes(res.status) ? " (kiểm tra GH_TOKEN / GH_REPO / GH_WORKFLOW của Worker)" : "";
    return editOriginal(interaction, env, `${mention} ❌ Không khởi động được báo cáo (GitHub trả về ${res.status || "lỗi mạng"})${hint}.`, fetchFn);
  }
  const what = parsed.iso ? `báo cáo cuối phiên **${parsed.iso.split("-").reverse().join("/")}**` : "báo cáo **ngay bây giờ**";
  return editOriginal(interaction, env, `${mention} ⏳ Đã nhận yêu cầu ${what} — sẽ gửi vào kênh này sau khoảng 3–5 phút.`, fetchFn);
}

// ----------------------------------------------------------------------------- handler chính
export async function handle(request, env, ctx, fetchFn = fetch, nowMs = Date.now()) {
  if (request.method === "GET") return new Response("Momentum VN30 /report endpoint — OK", { status: 200 });
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });

  const body = await request.text();
  const ok = await verifySignature(body, request.headers.get("x-signature-ed25519"), request.headers.get("x-signature-timestamp"),
                                   env.DISCORD_PUBLIC_KEY, nowMs);
  if (!ok) return new Response("invalid request signature", { status: 401 });

  let interaction;
  try { interaction = JSON.parse(body); } catch { return new Response("bad request", { status: 400 }); }

  if (interaction.type === 1) return json({ type: 1 });                              // PING (Discord xác minh endpoint)
  if (interaction.type !== 2 || interaction.data?.name !== "report") return ephemeral("Lệnh không được hỗ trợ.");

  const denied = authorize(interaction, env);
  if (denied) return ephemeral(`⛔ ${denied}`);
  const parsed = parseDateOption((interaction.data.options ?? []).find((o) => o.name === "date")?.value, nowMs);
  if (parsed.error) return ephemeral(`❌ ${parsed.error}`);
  if (!env.GH_TOKEN || !env.GH_REPO) return ephemeral("❌ Worker chưa được cấu hình GH_TOKEN / GH_REPO.");

  ctx.waitUntil(processReport(interaction, env, fetchFn, nowMs));                    // làm việc nặng ở nền; Discord chỉ chờ tối đa 3 giây
  return json({ type: 5 });                                                          // DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE
}

export default {
  fetch: (request, env, ctx) => handle(request, env, ctx),
};
