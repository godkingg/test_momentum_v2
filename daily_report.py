# -*- coding: utf-8 -*-
"""daily_report.py — Báo cáo lệnh hằng ngày của danh mục mô hình, gửi Discord (PNG + HTML).

Hai lần mỗi ngày giao dịch (giờ Việt Nam):
  python daily_report.py --mode preview   # ≈14:00 — lệnh DỰ KIẾN (nến hôm nay chưa chốt, còn kịp đặt lệnh ATC)
  python daily_report.py --mode final     # ≈15:00 — BÁO CÁO CUỐI PHIÊN (lệnh chốt, KPI ngày, so với lệnh 14:00)

Thiết kế để chạy trên GitHub Actions (miễn phí, không cần VPS/máy local) — xem .github/workflows/daily_report.yml.
Theo yêu cầu (slash command /report → Cloudflare Worker → workflow report_on_demand.yml):
  python daily_report.py --mode auto --on-demand --channel-id <id> --requester-id <id> [--asof YYYY-MM-DD]
  Tự chọn loại báo cáo theo giờ: đang trong phiên (09:00–15:00) → LỆNH TẠM TÍNH; sau 15:00 / cuối tuần / nghỉ lễ → BÁO CÁO CUỐI PHIÊN
  của phiên gần nhất; có --asof → báo cáo cuối phiên của ngày đó. KHÔNG ghi state/, không chờ giờ, không bỏ qua cuối tuần.

Biến môi trường (bí mật, lưu trong GitHub Secrets): DISCORD_WEBHOOK_URL (báo cáo theo lịch + dự phòng),
DISCORD_BOT_TOKEN (chỉ báo cáo theo yêu cầu: đăng vào đúng kênh người dùng gõ lệnh).

Trạng thái được lưu trong thư mục state/ (commit vào repo):
  ic_weights.json  trọng số IC ĐÓNG BĂNG (tính lại mỗi ngày làm lịch sử mô phỏng đổi → lệnh hôm qua có thể biến mất)
  universe.json    universe đã xác nhận — thiếu/thừa mã so với lần trước → DỪNG thay vì phát lệnh sai
  ledger.json      vị thế mô hình cuối phiên gần nhất, dùng để kiểm tra lịch sử tính lại có khớp không
  preview.json     lệnh dự kiến 14:00, để so với lệnh chốt 15:00
  daily_log.csv    nhật ký KPI từng ngày
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.backtest import backtest_with_exits
from src.data_loader import VN30_UNIVERSE, filter_by_coverage, load_ohlcv
from src.discord_webhook import WebhookError, send_report, send_report_channel, send_text, send_text_channel
from src.pipeline import ENGINE_PARAMS, FACTOR_NAMES, build_strategy_inputs, fit_ic_weights
from src.report_render import render_html, render_png
from src.reporting import build_report, check_ledger, positions_asof, positions_signature
from src.backtest import TOTAL_COST_RATE

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
DEFAULT_WAIT = {"preview": "14:00", "final": "15:00"}
FIELDS = ("close", "volume")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("preview", "final", "auto"), required=True, help="auto chỉ dùng với --on-demand (tự chọn theo giờ)")
    p.add_argument("--wait-until", default=None, help="HH:MM giờ VN — nếu chạy sớm hơn sẽ chờ tới giờ này (mặc định 14:00/15:00; 'none' = không chờ)")
    p.add_argument("--max-wait-min", type=int, default=45, help="Chạy sớm hơn mức này thì KHÔNG chờ (hữu ích khi chạy tay để test)")
    p.add_argument("--universe", nargs="+", default=VN30_UNIVERSE)
    p.add_argument("--start", default="2022-01-01", help="CỐ ĐỊNH: nhịp rebalance 5 phiên phụ thuộc vị trí ngày trong chuỗi dữ liệu")
    p.add_argument("--asof", default=None, help="YYYY-MM-DD: chạy cho ngày này (test/backfill) thay vì hôm nay")
    p.add_argument("--out-dir", default="reports/daily")
    p.add_argument("--state-dir", default="state")
    p.add_argument("--no-discord", action="store_true", help="Chỉ sinh file, không gửi")
    p.add_argument("--refit-weights", action="store_true", help="Ước lượng lại trọng số IC trên toàn bộ dữ liệu hiện có (đổi lịch sử mô phỏng!)")
    p.add_argument("--accept-universe-change", action="store_true")
    p.add_argument("--no-notice", action="store_true", help="Không gửi thông báo khi không có phiên giao dịch / thiếu dữ liệu")
    p.add_argument("--webhook-env", default="DISCORD_WEBHOOK_URL")
    p.add_argument("--mention", default=os.environ.get("DISCORD_MENTION", ""), help="VD <@123456789> để ping bạn")
    p.add_argument("--fetch-retries", type=int, default=3)
    p.add_argument("--on-demand", action="store_true", help="Chạy theo yêu cầu (slash command): không chờ giờ, không ghi state/, không bỏ qua cuối tuần")
    p.add_argument("--channel-id", default="", help="Kênh Discord nhận báo cáo theo yêu cầu (đăng bằng bot token)")
    p.add_argument("--requester-id", default="", help="ID người gọi lệnh — được ping khi báo cáo xong")
    p.add_argument("--bot-token-env", default="DISCORD_BOT_TOKEN")
    a = p.parse_args(argv)
    if a.asof and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", a.asof):
        p.error("--asof phải có dạng YYYY-MM-DD")
    for name in ("channel_id", "requester_id"):
        if getattr(a, name) and not re.fullmatch(r"\d{15,25}", getattr(a, name)):
            p.error(f"--{name.replace('_', '-')} phải là chuỗi số (ID Discord)")
    if a.mode == "auto" and not a.on_demand:
        p.error("--mode auto chỉ dùng cùng --on-demand")
    return a


# ----------------------------------------------------------------------------- tiện ích
def now_vn() -> datetime:
    return datetime.now(VN_TZ)


def log(msg: str) -> None:
    print(f"[{now_vn():%H:%M:%S}] {msg}", flush=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def wait_until(hhmm: str, max_wait_min: int, now_fn=now_vn, sleep_fn=time.sleep) -> float:
    """Chờ tới hh:mm (giờ VN) nếu đang sớm hơn nhưng không quá `max_wait_min`. Trả về số giây đã chờ."""
    h, m = map(int, hhmm.split(":"))
    now = now_fn()
    delta = (now.replace(hour=h, minute=m, second=0, microsecond=0) - now).total_seconds()
    if delta <= 0:
        return 0.0
    if delta > max_wait_min * 60:
        log(f"Chạy sớm {delta / 60:.0f} phút (> {max_wait_min}) — KHÔNG chờ, chạy ngay (có thể là chạy tay để test).")
        return 0.0
    log(f"Chờ {delta / 60:.1f} phút tới {hhmm} giờ VN...")
    sleep_fn(delta)
    return delta


def fetch_universe(symbols, start, end, loader, retries: int, sleep_fn=time.sleep) -> dict:
    """Tải dữ liệu; mã thiếu được tải lại tối đa `retries` lần. Còn thiếu → RuntimeError (không phát lệnh trên universe thiếu)."""
    ohlcv = loader(symbols, start, end, fields=FIELDS)
    missing = [s for s in symbols if s not in ohlcv["close"].columns]
    for attempt in range(retries):
        if not missing:
            break
        log(f"Thiếu {len(missing)} mã ({', '.join(missing)}) — thử lại lần {attempt + 1}/{retries}")
        sleep_fn(30 * (attempt + 1))
        extra = loader(missing, start, end, fields=FIELDS)
        ohlcv = {f: pd.concat([ohlcv[f], extra[f]], axis=1).sort_index() for f in FIELDS}
        missing = [s for s in symbols if s not in ohlcv["close"].columns]
    if missing:
        raise RuntimeError(f"Không tải được dữ liệu của: {', '.join(missing)} — dừng để tránh phát lệnh trên universe thiếu mã.")
    return ohlcv


class Sender:
    """Chọn đường gửi: theo yêu cầu → bot đăng vào kênh người gọi (dự phòng: webhook); theo lịch → webhook."""

    def __init__(self, args, post=None):
        self.args, self.post = args, post
        self.webhook = os.environ.get(args.webhook_env, "")
        self.token = os.environ.get(args.bot_token_env, "")
        self.use_channel = bool(args.on_demand and args.channel_id and self.token)

    @property
    def enabled(self) -> bool:
        return not self.args.no_discord and (self.use_channel or bool(self.webhook))

    def _kw(self) -> dict:
        return {"post": self.post} if self.post else {}

    def text(self, content: str) -> None:
        if not self.enabled:
            return
        if self.use_channel:
            try:
                send_text_channel(self.token, self.args.channel_id, content, requester_id=self.args.requester_id or None, **self._kw())
                return
            except (WebhookError, ValueError) as e:
                log(f"Không gửi được vào kênh yêu cầu ({e}) — thử webhook mặc định.")
                if not self.webhook:
                    raise
        send_text(self.webhook, content, **self._kw())

    def report(self, report: dict, png: Path, html_path: Path) -> str:
        """Trả về nơi đã gửi: 'channel' | 'webhook' | 'webhook-fallback'."""
        if self.use_channel:
            try:
                send_report_channel(self.token, self.args.channel_id, report, png, html_path, requester_id=self.args.requester_id or None, **self._kw())
                return "channel"
            except (WebhookError, ValueError) as e:
                log(f"Không đăng được vào kênh {self.args.channel_id} ({e}) — bot thiếu quyền? Thử webhook mặc định.")
                if not self.webhook:
                    raise
                send_report(self.webhook, report, png, html_path, mention=f"<@{self.args.requester_id}>" if self.args.requester_id else "", **self._kw())
                return "webhook-fallback"
        send_report(self.webhook, report, png, html_path, mention=self.args.mention, **self._kw())
        return "webhook"


def notify(sender: Sender, text: str, args) -> None:
    """Thông báo ngắn (nghỉ lễ, thiếu dữ liệu...). Theo yêu cầu thì LUÔN trả lời người dùng; theo lịch có thể tắt bằng --no-notice."""
    log(text)
    if args.on_demand or not args.no_notice:
        sender.text(text)


# ----------------------------------------------------------------------------- luồng chính
def run(args, *, loader=load_ohlcv, now_fn=now_vn, sleep_fn=time.sleep, post=None) -> int:
    now = now_fn()
    on_demand = args.on_demand
    today = pd.Timestamp(now.date())
    asof_req = pd.Timestamp(args.asof) if args.asof else None
    target_day = asof_req if asof_req is not None else today
    sender = Sender(args, post)
    if not on_demand and asof_req is None and today.weekday() >= 5:
        log("Cuối tuần — không có phiên giao dịch, bỏ qua.")
        return 0

    target = args.wait_until or DEFAULT_WAIT.get(args.mode, "none")
    if not on_demand and asof_req is None and target.lower() != "none":
        wait_until(target, args.max_wait_min, now_fn, sleep_fn)
        now = now_fn()

    state = Path(args.state_dir)
    out = Path(args.out_dir)
    log(f"Chế độ {args.mode.upper()}{' (theo yêu cầu)' if on_demand else ''} — phiên {target_day.date()} — tải {len(args.universe)} mã...")
    ohlcv = fetch_universe(args.universe, args.start, str(target_day.date()), loader, args.fetch_retries, sleep_fn)
    price_raw = ohlcv["close"].dropna(how="all").loc[: target_day + pd.Timedelta(hours=23, minutes=59)]   # cuối ngày: chịu được index có giờ
    last_day = None if price_raw.empty else price_raw.index.max().normalize()
    if last_day is None and on_demand:
        raise RuntimeError("Nguồn dữ liệu không trả về nến nào.")

    extra_warnings: list[str] = []
    mode = args.mode
    if on_demand:
        if asof_req is not None:
            if last_day != asof_req:
                notify(sender, f"⚠️ Ngày {asof_req:%d/%m/%Y} không có phiên giao dịch (nến gần nhất trước đó: "
                               f"{'—' if last_day is None else f'{last_day:%d/%m/%Y}'}). Hãy chọn ngày giao dịch khác.", args)
                return 0
            mode = "final"
        else:
            in_session = today.weekday() < 5 and dtime(9, 0) <= now.time() < dtime(15, 0)
            mode = "preview" if (in_session and last_day == today) else "final"
            if last_day != today:
                extra_warnings.append(f"Hôm nay ({today:%d/%m}) không có nến/không có phiên → báo cáo cho phiên gần nhất {last_day:%d/%m/%Y}.")
    elif last_day is None or last_day < target_day:
        notify(sender, f"⚠️ Chưa có nến ngày {target_day:%d/%m/%Y} từ nguồn dữ liệu (nghỉ lễ hoặc dữ liệu chậm; nến gần nhất: "
                       f"{None if last_day is None else last_day.date()}). Bỏ qua báo cáo {args.mode}.", args)
        return 0

    price_df, dropped = filter_by_coverage(price_raw)
    volume_df = ohlcv["volume"].reindex(index=price_df.index, columns=price_df.columns)

    # ---- universe phải khớp lần trước
    uni_path = state / "universe.json"
    expected = read_json(uni_path)
    if expected and not args.accept_universe_change and set(expected["symbols"]) != set(price_df.columns):
        thieu, thua = sorted(set(expected["symbols"]) - set(price_df.columns)), sorted(set(price_df.columns) - set(expected["symbols"]))
        raise RuntimeError(f"Universe thay đổi (thiếu {thieu}, thừa {thua}) — dừng. Kiểm tra dữ liệu hoặc chạy lại với --accept-universe-change.")
    if not on_demand:
        write_json(uni_path, {"symbols": list(price_df.columns), "dropped": dropped})

    # ---- trọng số IC đóng băng
    w_path = state / "ic_weights.json"
    saved_w = read_json(w_path)
    if args.refit_weights or not saved_w or set(saved_w["weights"]) != set(FACTOR_NAMES):
        log("Ước lượng trọng số IC (đóng băng vào state/ic_weights.json)...")
        weights = fit_ic_weights(price_df, volume_df)
        if not on_demand:
            write_json(w_path, {"fitted_on": str(price_df.index[-1].date()), "weights": weights})
        elif not saved_w:
            extra_warnings.append("Chưa có trọng số IC đóng băng (state/ic_weights.json — cần ít nhất 1 lần chạy theo lịch): trọng số được ước lượng tạm, "
                                  "lệnh có thể khác báo cáo theo lịch.")
    else:
        weights = saved_w["weights"]
    inp = build_strategy_inputs(price_df, volume_df, ic_weights=weights)

    # ---- engine v2 trên toàn bộ lịch sử → lệnh của hôm nay
    port_df, trade_df = backtest_with_exits(price_df, inp["comp_score"], inp["vol_spike"], inp["mom_6"], cost_rate=TOTAL_COST_RATE, **ENGINE_PARAMS)

    # ---- kiểm tra sổ lệnh: danh mục tính lại cho phiên trước có khớp bản đã lưu?
    if asof_req is not None and len(price_df) < 320:
        notify(sender, f"⚠️ Ngày {asof_req:%d/%m/%Y} quá sớm: chưa đủ ~320 phiên lịch sử để tính tín hiệu.", args)
        return 0
    warnings: list[str] = list(extra_warnings)
    ledger = read_json(state / "ledger.json")
    if ledger and len(price_df) >= 2:
        prev = price_df.index[-2]
        warnings = [f"Sổ lệnh lệch lịch sử tính lại: {m}" for m in check_ledger(
            ledger, positions_asof(price_df, inp["comp_score"], inp["vol_spike"], inp["mom_6"], ENGINE_PARAMS, TOTAL_COST_RATE, prev), prev)]

    pv = read_json(state / "preview.json")
    preview_orders = pv["orders"] if (not on_demand and mode == "final" and pv and pv.get("asof") == str(target_day.date())) else None

    asof = price_df.index.max()
    report = build_report(
        price_df=price_df, comp_score=inp["comp_score"], vol_spike=inp["vol_spike"], mom_6=inp["mom_6"], port_df=port_df, trade_df=trade_df,
        asof=asof, mode=mode, params=ENGINE_PARAMS, cost_rate=TOTAL_COST_RATE, dropped=dropped, preview_orders=preview_orders, warnings=warnings,
        generated_at=f"{now:%H:%M %d/%m/%Y}", data_note="nguồn vnstock/KBS", trigger="on_demand" if on_demand else "schedule")

    # ---- xuất file
    stem = f"momentum_vn30_{asof:%Y-%m-%d}_{mode}" + ("_ondemand" if on_demand else "")
    out.mkdir(parents=True, exist_ok=True)
    png, html_path = out / f"{stem}.png", out / f"{stem}.html"
    render_png(report, png)
    html_path.write_text(render_html(report), encoding="utf-8")
    write_json(out / f"{stem}.json", report)
    log(f"Đã tạo {png.name} ({png.stat().st_size // 1024} KB) + {html_path.name}; {len(report['orders'])} lệnh, {len(report['positions'])} vị thế.")

    # ---- gửi Discord
    if args.no_discord:
        log("--no-discord: không gửi.")
    elif not sender.enabled:
        log(f"⚠️ Không có đích gửi (thiếu {args.webhook_env}" + (f"/{args.bot_token_env}" if on_demand else "") + ") — bỏ qua gửi Discord.")
    else:
        log(f"Đã gửi Discord ({sender.report(report, png, html_path)}).")

    # ---- lưu trạng thái (chỉ báo cáo theo lịch)
    if not on_demand:
        if mode == "preview":
            write_json(state / "preview.json", {"asof": report["meta"]["asof"], "orders": report["orders"], "generated_at": report["meta"]["generated_at"]})
        else:
            write_json(state / "ledger.json", {"asof": report["meta"]["asof"], "positions": positions_signature(port_df.attrs["open_positions"]),
                                               "generated_at": report["meta"]["generated_at"]})
            _append_daily_log(state / "daily_log.csv", report)
    return 0


def _append_daily_log(path: Path, report: dict) -> None:
    """Mỗi ngày 1 dòng (ghi đè nếu chạy lại cùng ngày)."""
    p, m = report["portfolio"], report["meta"]
    row = {"date": m["asof"], "day_net": p["day_net"], "day_bench": p["day_bench"], "cum_net": p["cum_net"], "cum_bench": p["cum_bench"],
           "n_positions": p["n_positions"], "n_orders": len(report["orders"]), "n_buy": sum(o["side"] == "BUY" for o in report["orders"])}
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["date"] != row["date"]]
    rows.append({k: str(v) for k, v in row.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        code = run(args)
    except Exception as e:                       # báo lỗi lên Discord (không lộ secret) rồi thoát mã ≠ 0 để GitHub Actions đánh dấu failed
        msg = f"❌ Báo cáo {args.mode} lỗi: {type(e).__name__}: {e}"[:1500]
        print(msg, file=sys.stderr, flush=True)
        try:
            Sender(args).text(msg)
        except Exception:
            pass
        raise
    sys.exit(code)


if __name__ == "__main__":
    main()
