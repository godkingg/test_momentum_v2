# -*- coding: utf-8 -*-
"""discord_webhook.py — Gửi báo cáo lên Discord qua WEBHOOK (không cần bot chạy 24/7, không cần token bot).

Mỗi tin nhắn gồm: 1 embed tóm tắt + ảnh PNG (hiển thị ngay trong embed) + file HTML (tải về mở bằng trình duyệt;
Discord không xem trước HTML). Tự retry khi bị rate-limit (429) hoặc lỗi 5xx. URL webhook là bí mật: KHÔNG bao giờ được in ra log.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from src.report_render import SIDE_TEXT, pct

WEBHOOK_PREFIXES = ("https://discord.com/api/webhooks/", "https://discordapp.com/api/webhooks/",
                    "https://ptb.discord.com/api/webhooks/", "https://canary.discord.com/api/webhooks/")
COLORS = {"green": 0x26A69A, "red": 0xEF5350, "amber": 0xD29922}
FIELD_MAX = 1000          # giới hạn Discord 1024 ký tự / field


class WebhookError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def _check_url(url: str) -> None:
    if not url or not url.startswith(WEBHOOK_PREFIXES):
        raise ValueError("DISCORD_WEBHOOK_URL không hợp lệ (phải bắt đầu bằng https://discord.com/api/webhooks/...)")


def _clip(lines: list[str], limit: int = FIELD_MAX) -> str:
    out, total = [], 0
    for i, ln in enumerate(lines):
        if total + len(ln) + 1 > limit:
            out.append(f"… và {len(lines) - i} dòng nữa (xem ảnh/HTML)")
            break
        out.append(ln)
        total += len(ln) + 1
    return "\n".join(out) or "—"


def _order_line(o: dict) -> str:
    pnl = f" ({pct(o['pnl'])})" if o["pnl"] is not None else ""
    return f"`{o['symbol']}` @{o['price']:,.2f}{pnl} — {o['label']}"


def build_embed(report: dict, png_name: str) -> dict:
    """Embed tóm tắt: KPI ngày + lệnh (MUA/BÁN/GIẢM) + vị thế đang giữ + so với 14:00."""
    m, p = report["meta"], report["portfolio"]
    partial = " (tạm tính)" if m["is_partial"] else ""
    day, bench = p["day_net"], p["day_bench"]
    on_demand = m.get("trigger") == "on_demand"
    if on_demand:
        head = f"⏱ Lệnh tạm tính lúc {m['generated_at'][:5]}" if m["mode"] == "preview" else "📊 Báo cáo cuối phiên"
        head += " (theo yêu cầu)"
    else:
        head = "⏱ Lệnh dự kiến 14:00" if m["mode"] == "preview" else "📊 Báo cáo cuối phiên 15:00"
    dd, mm, yy = m["asof"][8:], m["asof"][5:7], m["asof"][:4]
    title = f"{head} — {dd}/{mm}/{yy}"
    desc = (f"**Phiên {dd}/{mm}{partial}: {pct(day)}** · 1/N: {pct(bench)} · chênh {pct(day - bench)}\n"
            f"Lũy kế mô phỏng: **{pct(p['cum_net'], 1)}** (1/N {pct(p['cum_bench'], 1)}) · "
            f"{p['n_positions']} vị thế, đầu tư {p['exposure']:.0%}")
    fields = []
    for side, icon in (("BUY", "🟢"), ("SELL", "🔴"), ("REDUCE", "🟠")):
        rows = [o for o in report["orders"] if o["side"] == side]
        if rows:
            fields.append({"name": f"{icon} {SIDE_TEXT[side]} ({len(rows)})", "value": _clip([_order_line(o) for o in rows]), "inline": False})
    if not report["orders"]:
        fields.append({"name": "Lệnh", "value": "Không có lệnh nào" + (" tính đến thời điểm này" if m["is_partial"] else " trong phiên"), "inline": False})
    held = sorted(report["positions"], key=lambda x: -(x["pnl"] or 0))
    if held:
        fields.append({"name": f"📌 Đang giữ ({len(held)})", "value": _clip(["  ".join(f"`{h['symbol']}` {pct(h['pnl'], 1)}" for h in held)]), "inline": False})
    d = report.get("vs_preview")
    if d is not None:
        txt = lambda os_: ", ".join(f"{SIDE_TEXT[o['side']]} {o['symbol']}" for o in os_) or "—"
        fields.append({"name": "So với lệnh dự kiến 14:00", "value": _clip([f"Mới: {txt(d['added'])}", f"Không còn: {txt(d['removed'])}", f"Giữ nguyên: {txt(d['kept'])}"]), "inline": False})
    if m["warnings"]:
        fields.append({"name": "⚠ Lưu ý", "value": _clip([f"• {w}" for w in m["warnings"]]), "inline": False})
    color = COLORS["amber"] if m["is_partial"] else (COLORS["green"] if day >= 0 else COLORS["red"])
    return {"title": title, "description": desc, "color": color, "fields": fields, "image": {"url": f"attachment://{png_name}"},
            "footer": {"text": "Danh mục mô hình (paper) — không phải khuyến nghị đầu tư"}}


def _post_with_retry(url: str, payload: dict, files: dict, *, timeout: int, max_retries: int, post=requests.post, sleep=time.sleep,
                     params: dict | None = None, headers: dict | None = None):
    last = ""
    for attempt in range(max_retries):
        kwargs = dict(data={"payload_json": json.dumps(payload, ensure_ascii=False)}, files=files or None, timeout=timeout)
        if params:
            kwargs["params"] = params
        if headers:
            kwargs["headers"] = headers
        try:
            resp = post(url, **kwargs)
        except requests.RequestException as e:          # lỗi mạng: không đưa URL/token vào thông báo
            last = f"{type(e).__name__}"
            sleep(2 ** attempt)
            continue
        code = resp.status_code
        if 200 <= code < 300:
            return resp
        if code == 429:
            try:
                wait = float(resp.json().get("retry_after", 2))
            except Exception:
                wait = 2.0
            last = "HTTP 429 (rate limit)"
            sleep(min(wait, 30))
            continue
        if code >= 500:
            last = f"HTTP {code}"
            sleep(2 ** attempt)
            continue
        raise WebhookError(f"Discord từ chối (HTTP {code}): {resp.text[:300]}", status=code)
    raise WebhookError(f"Gửi Discord thất bại sau {max_retries} lần thử ({last})")


def send_report(webhook_url: str, report: dict, png_path: str | Path, html_path: str | Path | None = None, *,
                username: str = "Momentum VN30", mention: str = "", timeout: int = 60, max_retries: int = 4, post=requests.post, sleep=time.sleep):
    """Gửi embed + PNG (+ HTML). Đọc file vào bộ nhớ một lần để retry an toàn."""
    _check_url(webhook_url)
    png_path = Path(png_path)
    files = {"files[0]": (png_path.name, png_path.read_bytes(), "image/png")}
    attachments = [{"id": 0, "filename": png_path.name}]
    if html_path:
        html_path = Path(html_path)
        files["files[1]"] = (html_path.name, html_path.read_bytes(), "text/html")
        attachments.append({"id": 1, "filename": html_path.name})
    payload = {"username": username, "content": mention, "embeds": [build_embed(report, png_path.name)], "attachments": attachments,
               "allowed_mentions": {"parse": ["users", "roles"] if mention else []}}
    return _post_with_retry(webhook_url, payload, files, timeout=timeout, max_retries=max_retries, post=post, sleep=sleep, params={"wait": "true"})


def send_text(webhook_url: str, content: str, *, username: str = "Momentum VN30", timeout: int = 30, max_retries: int = 3, post=requests.post, sleep=time.sleep):
    """Gửi 1 thông báo văn bản ngắn (nghỉ lễ, thiếu dữ liệu, lỗi...)."""
    _check_url(webhook_url)
    payload = {"username": username, "content": content[:1900], "allowed_mentions": {"parse": []}}
    return _post_with_retry(webhook_url, payload, {}, timeout=timeout, max_retries=max_retries, post=post, sleep=sleep, params={"wait": "true"})


# ----------------------------------------------------------------------------- gửi vào kênh bằng bot token (slash command /report)
API = "https://discord.com/api/v10"


def _check_ids(channel_id: str, user_id: str | None = None) -> None:
    if not re.fullmatch(r"\d{15,25}", str(channel_id)):
        raise ValueError("channel_id không hợp lệ")
    if user_id and not re.fullmatch(r"\d{15,25}", str(user_id)):
        raise ValueError("requester_id không hợp lệ")


def _bot_headers(bot_token: str) -> dict:
    if not bot_token:
        raise ValueError("Thiếu DISCORD_BOT_TOKEN")
    return {"Authorization": f"Bot {bot_token}", "User-Agent": "MomentumVN30Report (github-actions, 1.0)"}


def send_report_channel(bot_token: str, channel_id: str, report: dict, png_path: str | Path, html_path: str | Path | None = None, *,
                        requester_id: str | None = None, timeout: int = 60, max_retries: int = 4, post=requests.post, sleep=time.sleep):
    """Đăng embed + PNG (+ HTML) vào kênh `channel_id` bằng bot (REST). Ping người gọi lệnh nếu có `requester_id`."""
    _check_ids(channel_id, requester_id)
    png_path = Path(png_path)
    files = {"files[0]": (png_path.name, png_path.read_bytes(), "image/png")}
    attachments = [{"id": 0, "filename": png_path.name}]
    if html_path:
        html_path = Path(html_path)
        files["files[1]"] = (html_path.name, html_path.read_bytes(), "text/html")
        attachments.append({"id": 1, "filename": html_path.name})
    payload = {"content": f"<@{requester_id}> báo cáo bạn yêu cầu:" if requester_id else "", "embeds": [build_embed(report, png_path.name)],
               "attachments": attachments, "allowed_mentions": {"parse": [], "users": [requester_id] if requester_id else []}}
    return _post_with_retry(f"{API}/channels/{channel_id}/messages", payload, files, timeout=timeout, max_retries=max_retries,
                            post=post, sleep=sleep, headers=_bot_headers(bot_token))


def send_text_channel(bot_token: str, channel_id: str, content: str, *, requester_id: str | None = None, timeout: int = 30,
                      max_retries: int = 3, post=requests.post, sleep=time.sleep):
    _check_ids(channel_id, requester_id)
    text = (f"<@{requester_id}> " if requester_id else "") + content
    payload = {"content": text[:1900], "allowed_mentions": {"parse": [], "users": [requester_id] if requester_id else []}}
    return _post_with_retry(f"{API}/channels/{channel_id}/messages", payload, {}, timeout=timeout, max_retries=max_retries,
                            post=post, sleep=sleep, headers=_bot_headers(bot_token))
