# -*- coding: utf-8 -*-
"""reporting.py — Dựng dữ liệu báo cáo hằng ngày (lệnh, vị thế, KPI, watchlist) từ kết quả engine v2.

Báo cáo là của DANH MỤC MÔ HÌNH (paper): giá khớp giả định = close, tỷ trọng = size / số vị thế (equal-weight).
Hai chế độ:
  - "preview" (≈14:00): nến hôm nay CHƯA CHỐT (giá = giá gần nhất, khối lượng chưa đủ phiên) → lệnh DỰ KIẾN.
  - "final"   (≈15:00): nến hôm nay đã chốt → lệnh CHỐT + KPI ngày + so sánh với lệnh dự kiến lúc 14:00.
Mọi giá trị trả về là kiểu JSON-serializable (str/float/int/list/dict).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import backtest_with_exits, invested_returns
from src.risk import equal_weight_benchmark
from src.strategy import PositionManager

# action của engine -> (hướng, nhãn tiếng Việt)
ACTION_LABELS: dict[str, tuple[str, str]] = {
    "BUY": ("BUY", "Vào lệnh theo rank"),
    "HARD_STOP": ("SELL", "Cắt lỗ cứng"),
    "TRAILING_STOP": ("SELL", "Trailing stop"),
    "VOL_EXIT": ("SELL", "Vol spike — thoát"),
    "VOL_REDUCE": ("REDUCE", "Vol spike — giảm 50%"),
    "SIGNAL_EXIT": ("SELL", "Rớt khỏi top rank"),
    "MOM_FLIP": ("SELL", "Momentum đảo chiều"),
}
CHART_DAYS = 120


def _d(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _f(x, nd: int = 6):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def extract_orders(trade_df: pd.DataFrame, asof, ranks_today: pd.Series) -> list[dict]:
    """Các lệnh của MỘT ngày từ trade log (BUY + mọi loại SELL), sắp xếp: BÁN trước, MUA sau."""
    if trade_df is None or len(trade_df) == 0:
        return []
    rows = trade_df[pd.to_datetime(trade_df["date"]) == pd.Timestamp(asof)]
    out = []
    for r in rows.itertuples():
        side, label = ACTION_LABELS.get(r.action, ("SELL", r.action))
        is_buy = r.action == "BUY"
        out.append({
            "side": side, "code": r.action, "label": label, "symbol": r.symbol,
            "price": _f(r.entry_px if is_buy else r.exit_px, 2),
            "size_text": "100%" if is_buy else ("50% vị thế" if r.action == "VOL_REDUCE" else "toàn bộ"),
            "pnl": None if is_buy else _f(r.pnl, 4),
            "hold_days": None if is_buy else int(r.hold_days),
            "reason": str(r.reason).replace("✅ ", ""),
            "rank": _f(ranks_today.get(r.symbol), 3),
        })
    return sorted(out, key=lambda o: (o["side"] == "BUY", o["symbol"]))


def build_positions(price_df: pd.DataFrame, port_df: pd.DataFrame, ranks_today: pd.Series, asof, params: dict) -> list[dict]:
    """Vị thế còn mở sau close `asof`, kèm mức cắt lỗ / trailing tính ra GIÁ để đặt lệnh điều kiện."""
    pos = port_df.attrs.get("open_positions", {})
    if not pos:
        return []
    prices = price_df.loc[asof]
    ret_today = price_df.pct_change(fill_method=None).loc[asof]
    idx = price_df.index
    n_pos = len(pos)
    out = []
    for sym, p in sorted(pos.items()):
        last = float(prices[sym])
        entry, peak = float(p["entry_price"]), float(max(p["peak_price"], last))
        out.append({
            "symbol": sym, "entry_date": _d(p["entry_date"]),
            "held_sessions": int(idx.get_loc(asof) - idx.get_loc(pd.Timestamp(p["entry_date"]))),
            "entry_price": _f(entry, 2), "last_price": _f(last, 2),
            "pnl": _f(last / entry - 1, 4), "day_ret": _f(ret_today[sym], 4),
            "from_peak": _f(last / peak - 1, 4), "size": _f(p["size"], 3), "weight": _f(p["size"] / n_pos, 4),
            "stop_price": _f(entry * (1 - params["stop_loss_pct"]), 2),
            "trail_price": _f(peak * (1 - params["trail_pct"]), 2),
            "rank": _f(ranks_today.get(sym), 3),
        })
    return out


def build_watchlist(price_df, vol_spike, mom_6, ranks_today, asof, params, held: set, bought_today: set,
                    is_rebalance_day: bool, sessions_to_rebalance: int, top_k: int = 8) -> list[dict]:
    """Top-K composite rank hôm nay + trạng thái: đang giữ / đã mua / vì sao chưa mua."""
    pm = PositionManager(
        stop_loss_pct=params["stop_loss_pct"], trail_pct=params["trail_pct"], vol_exit_thr=params["vol_exit_thr"],
        vol_reduce_thr=params["vol_reduce_thr"], signal_exit_rank=params["signal_exit_rank"], top_entry_rank=params["top_entry_rank"],
    )
    out = []
    for sym in ranks_today.dropna().sort_values(ascending=False).head(top_k).index:
        rank, vs, m6, px = float(ranks_today[sym]), float(vol_spike.loc[asof, sym]), float(mom_6.loc[asof, sym]), float(price_df.loc[asof, sym])
        if sym in bought_today:
            status = "Đã mua hôm nay"
        elif sym in held:
            status = "Đang giữ"
        else:
            ok, why = pm.check_entry(sym, px, rank, vs, m6)
            if not ok:
                status = f"Chưa đủ ĐK: {why}"
            elif is_rebalance_day:
                status = "Đủ điều kiện — hết slot"
            else:
                status = f"Đủ điều kiện — chờ rebalance (sau {sessions_to_rebalance} phiên)"
        out.append({"symbol": sym, "rank": _f(rank, 3), "mom_6": _f(m6, 4), "vol_spike": _f(vs, 2), "price": _f(px, 2), "status": status})
    return out


def diff_orders(preview: list[dict], final: list[dict]) -> dict:
    """So sánh lệnh dự kiến (14:00) với lệnh chốt (15:00) theo khóa (hướng, mã)."""
    key = lambda o: (o["side"], o["symbol"])
    p, f = {key(o): o for o in preview}, {key(o): o for o in final}
    return {
        "kept": [f[k] for k in f if k in p],
        "added": [f[k] for k in f if k not in p],      # xuất hiện khi chốt phiên (không có lúc 14:00)
        "removed": [p[k] for k in p if k not in f],    # có lúc 14:00 nhưng KHÔNG còn khi chốt
    }


def positions_signature(positions: dict) -> dict:
    """Dạng gọn để lưu sổ lệnh: {symbol: {entry_date, entry_price, size}}."""
    return {s: {"entry_date": _d(p["entry_date"]), "entry_price": round(float(p["entry_price"]), 4), "size": round(float(p["size"]), 4)}
            for s, p in positions.items()}


def positions_asof(price_df, comp_score, vol_spike, mom_6, params: dict, cost_rate: float, date) -> dict:
    """Vị thế mô hình tại close `date` (chạy lại engine — nhân quả — trên dữ liệu cắt tới `date`)."""
    cut = lambda df: df.loc[:date]
    port, _ = backtest_with_exits(cut(price_df), cut(comp_score), cut(vol_spike), cut(mom_6), cost_rate=cost_rate, **params)
    return positions_signature(port.attrs["open_positions"])


def check_ledger(ledger: dict | None, recomputed: dict, date) -> list[str]:
    """So sổ lệnh đã lưu (ngày `date`) với danh mục tính lại từ lịch sử. Trả về danh sách cảnh báo (rỗng = khớp)."""
    if not ledger or ledger.get("asof") != _d(date):
        return []
    saved = ledger.get("positions", {})
    msgs = []
    missing, extra = sorted(set(saved) - set(recomputed)), sorted(set(recomputed) - set(saved))
    if missing:
        msgs.append(f"Sổ lệnh có nhưng tính lại KHÔNG có: {', '.join(missing)}")
    if extra:
        msgs.append(f"Tính lại có nhưng sổ lệnh KHÔNG có: {', '.join(extra)}")
    for sym in sorted(set(saved) & set(recomputed)):
        a, b = saved[sym], recomputed[sym]
        if a["entry_date"] != b["entry_date"] or abs(a["entry_price"] / b["entry_price"] - 1) > 0.005 or abs(a["size"] - b["size"]) > 1e-6:
            msgs.append(f"{sym}: sổ lệnh (mua {a['entry_date']} @{a['entry_price']}) ≠ tính lại (mua {b['entry_date']} @{b['entry_price']})")
    return msgs


def build_report(
    *, price_df: pd.DataFrame, comp_score: pd.DataFrame, vol_spike: pd.DataFrame, mom_6: pd.DataFrame,
    port_df: pd.DataFrame, trade_df: pd.DataFrame, asof, mode: str, params: dict, cost_rate: float,
    dropped: list[str] | tuple = (), preview_orders: list[dict] | None = None, warnings: list[str] | tuple = (),
    generated_at: str = "", data_note: str = "", trigger: str = "schedule",
) -> dict:
    """Dữ liệu báo cáo đầy đủ cho 1 ngày giao dịch `asof` (phải có trong price_df và port_df)."""
    assert mode in ("preview", "final")
    asof = pd.Timestamp(asof)
    pos_idx = price_df.index.get_loc(asof)
    freq = params["rebal_freq"]
    is_rebalance = pos_idx % freq == 0
    to_rebalance = freq if is_rebalance else freq - pos_idx % freq

    ranks_today = comp_score.rank(axis=1, pct=True).loc[asof]
    orders = extract_orders(trade_df, asof, ranks_today)
    positions = build_positions(price_df, port_df, ranks_today, asof, params)
    held = {p["symbol"] for p in positions}
    bought = {o["symbol"] for o in orders if o["side"] == "BUY"}

    # ---- KPI danh mục
    row = port_df.loc[asof]
    bench = equal_weight_benchmark(price_df)
    inv = invested_returns(port_df)
    nav = (1 + inv).cumprod()
    b = bench.reindex(inv.index).fillna(0.0)
    nav_b = (1 + b).cumprod()
    win = inv.index[-CHART_DAYS:]
    rebase = lambda s: (s.loc[win] / s.loc[win[0]] - 1)
    exposure = float(sum(p["weight"] for p in positions))

    portfolio = {
        "day_net": _f(row["port_return"], 5), "day_gross": _f(row["gross_return"], 5), "day_cost": _f(row["cost"], 5),
        "day_bench": _f(bench.loc[asof], 5), "ret_5d": _f((1 + inv.iloc[-5:]).prod() - 1, 4),
        "bench_5d": _f((1 + b.iloc[-5:]).prod() - 1, 4),
        "cum_net": _f(nav.iloc[-1] - 1, 4), "cum_bench": _f(nav_b.iloc[-1] - 1, 4),
        "drawdown": _f(nav.iloc[-1] / nav.cummax().iloc[-1] - 1, 4), "max_drawdown": _f((nav / nav.cummax() - 1).min(), 4),
        "since": _d(inv.index[0]), "n_positions": len(positions), "exposure": _f(exposure, 4),
        "chart": {"dates": [_d(d) for d in win], "strategy": [_f(v, 5) for v in rebase(nav)], "benchmark": [_f(v, 5) for v in rebase(nav_b)]},
    }

    flips = sorted({o["symbol"] for o in orders if o["side"] != "BUY"} & bought)
    flip_warn = [f"{', '.join(flips)}: có cả lệnh BÁN và MUA cùng phiên (stop rồi vào lại do rank cao) — thực tế có thể bỏ cả hai"] if flips else []
    return {
        "meta": {
            "mode": mode, "trigger": trigger, "asof": _d(asof), "generated_at": generated_at, "data_note": data_note, "engine": "v2 (độ trễ 1 ngày)",
            "is_partial": mode == "preview", "cost_rate": cost_rate, "params": dict(params),
            "universe_size": int(price_df.shape[1]), "dropped": list(dropped), "warnings": list(warnings) + flip_warn,
            "is_rebalance_day": bool(is_rebalance), "sessions_to_rebalance": int(to_rebalance),
        },
        "orders": orders, "positions": positions, "portfolio": portfolio,
        "watchlist": build_watchlist(price_df, vol_spike, mom_6, ranks_today, asof, params, held, bought, bool(is_rebalance), int(to_rebalance)),
        "vs_preview": diff_orders(preview_orders, orders) if (mode == "final" and preview_orders is not None) else None,
    }
