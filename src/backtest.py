# -*- coding: utf-8 -*-
"""backtest.py — Backtest engine v2 với entry/exit rules (PositionManager) — ĐỘ TRỄ 1 NGÀY, KHÔNG LOOK-AHEAD.

QUY ƯỚC THỜI GIAN (quan trọng):
  - Signal/rank tại close ngày t chỉ dùng thông tin đến hết ngày t.
  - Mọi lệnh (BUY / SELL) khớp tại close ngày t. Giả định giá khớp = close, phần trượt giá
    nằm trong cost_rate (0.15% phí + 0.08% slippage).
  - Vị thế NẮM GIỮ từ close t-1 mới ăn return của ngày t. Vị thế vừa mở ở close t chỉ bắt đầu
    ăn return từ ngày t+1; vị thế bị đóng ở close t VẪN ăn return của ngày t (kể cả cú giảm
    kích hoạt stop-loss).

Engine cũ (Ngày 19-22) áp return của ngày t cho vị thế sau khi đã vào/ra theo giá ngày t
→ thiên lệch dương (xem src/backtest_legacy.py và tests/test_engine_no_lookahead.py).

Cost model (Ngày 20): GIỮ NGUYÊN giả định đã dùng ở project momentum-ridge-vn30
(0.15% transaction + 0.08% slippage = 0.23% mỗi lần BUY hoặc SELL), áp trên NOTIONAL
giao dịch (chỉ phần THAY ĐỔI của portfolio), trừ vào return của ngày thực hiện lệnh.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import PositionManager

# Giữ NGUYÊN giả định cost từ project momentum-ridge-vn30 (Ngày 13)
TRANSACTION_COST = 0.0015  # 0.15% phí giao dịch
SLIPPAGE = 0.0008          # 0.08% ước lượng bid-ask spread
TOTAL_COST_RATE = TRANSACTION_COST + SLIPPAGE


def backtest_with_exits(
    price_df: pd.DataFrame,
    composite_score: pd.DataFrame,
    vol_spike_df: pd.DataFrame,
    mom_6_df: pd.DataFrame,
    stop_loss_pct: float = 0.08,
    trail_pct: float = 0.12,
    vol_exit_thr: float = 3.0,
    vol_reduce_thr: float = 2.0,
    top_entry_rank: float = 0.65,
    signal_exit_rank: float = 0.40,
    max_positions: int = 10,
    rebal_freq: int = 5,
    cost_rate: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest đầy đủ với entry + exit rules, engine v2 (độ trễ 1 ngày).

    Thứ tự xử lý mỗi ngày t:
      A. Vị thế đang giữ (chốt ở close t-1) ăn return ngày t        -> gross_return
      B. Kiểm tra EXIT bằng giá/rank/vol tại close t (khớp ở close t)  -> trade log + cost
      C. ENTRY mới mỗi `rebal_freq` ngày bằng rank tại close t (khớp ở close t,
         bắt đầu ăn return từ ngày t+1)                                -> trade log + cost

    cost_rate: phí áp trên NOTIONAL giao dịch mỗi lần BUY hoặc SELL (mặc định 0.0 = không cost).
    Trả về (port_df, trade_df):
      port_df: index=date, cột port_return (net), gross_return, cost,
               n_positions (cuối ngày), n_positions_start (đầu ngày — vị thế thực sự ăn return hôm nay)
      trade_df: log từng giao dịch (BUY và mọi loại SELL), gồm pnl/hold_days
    """
    pm = PositionManager(
        stop_loss_pct=stop_loss_pct, trail_pct=trail_pct,
        vol_exit_thr=vol_exit_thr, vol_reduce_thr=vol_reduce_thr,
        signal_exit_rank=signal_exit_rank, top_entry_rank=top_entry_rank,
    )

    dates = price_df.index
    ret_df = price_df.pct_change(fill_method=None)
    comp_rank_df = composite_score.rank(axis=1, pct=True)

    portfolio_log, trade_log = [], []

    for i, date in enumerate(dates):
        if date not in comp_rank_df.index:
            continue

        prices_today = price_df.loc[date]
        ranks_today = comp_rank_df.loc[date]
        vol_today = vol_spike_df.loc[date] if date in vol_spike_df.index else pd.Series(dtype=float)
        mom6_today = mom_6_df.loc[date] if date in mom_6_df.index else pd.Series(dtype=float)

        # --- A. Ăn return hôm nay bằng vị thế chốt ở close t-1 (TRƯỚC mọi giao dịch hôm nay) ---
        n_start = len(pm.positions)
        gross_ret = 0.0
        if n_start > 0 and i > 0:
            for sym, pos in pm.positions.items():
                if sym in ret_df.columns and not pd.isna(ret_df.loc[date, sym]):
                    gross_ret += ret_df.loc[date, sym] * pos["size"] / n_start

        day_cost = 0.0  # tổng phí phát sinh trong ngày (theo tỷ trọng notional)

        # --- B. EXIT tại close t ---
        for sym in list(pm.positions.keys()):
            if sym not in prices_today or pd.isna(prices_today[sym]):
                continue
            action, reason, size_sell = pm.check_exit(
                sym, prices_today[sym], ranks_today.get(sym, 0), vol_today.get(sym, 0), mom6_today.get(sym, 0),
            )
            if action not in ("HOLD", "NO_POSITION"):
                pos = pm.positions[sym]
                pnl = (prices_today[sym] / pos["entry_price"]) - 1
                trade_log.append({
                    "date": date, "symbol": sym, "action": action, "reason": reason,
                    "entry_px": pos["entry_price"], "exit_px": prices_today[sym], "pnl": pnl,
                    "hold_days": (date - pos["entry_date"]).days, "size_sold": size_sell,
                })
                # Cost = tỷ trọng bán (size_sell / n_start) × cost_rate. n_start = số vị thế ngay TRƯỚC lệnh bán.
                if cost_rate > 0 and n_start > 0:
                    day_cost += cost_rate * size_sell / n_start
                pm.close_position(sym, size_sell)

        # --- C. ENTRY tại close t (mỗi rebal_freq ngày); vị thế mới ăn return từ ngày t+1 ---
        n_new_entries = 0
        if i % rebal_freq == 0:
            slots = max_positions - len(pm.positions)
            if slots > 0:
                cands = ranks_today.dropna().sort_values(ascending=False)
                for sym, rank in cands.items():
                    if slots <= 0:
                        break
                    if sym in pm.positions or sym not in prices_today or pd.isna(prices_today[sym]):
                        continue
                    ok, reason = pm.check_entry(sym, prices_today[sym], rank, vol_today.get(sym, 0), mom6_today.get(sym, 0), date)
                    if ok:
                        pm.open_position(sym, prices_today[sym], date, reason=f"rank={rank:.2f}")
                        trade_log.append({
                            "date": date, "symbol": sym, "action": "BUY", "reason": reason,
                            "entry_px": prices_today[sym], "exit_px": np.nan, "pnl": np.nan,
                            "hold_days": 0, "size_sold": 1.0,
                        })
                        n_new_entries += 1
                        slots -= 1

        # Entry mới -> cost tính trên tỷ trọng notional SAU khi biết tổng vị thế cuối ngày
        n_pos = len(pm.positions)
        if cost_rate > 0 and n_new_entries > 0 and n_pos > 0:
            day_cost += cost_rate * n_new_entries / n_pos

        portfolio_log.append({
            "date": date, "port_return": gross_ret - day_cost, "gross_return": gross_ret,
            "cost": day_cost, "n_positions": n_pos, "n_positions_start": n_start,
        })

    port_df = pd.DataFrame(portfolio_log).set_index("date")
    # Vị thế còn mở SAU close ngày cuối (dùng cho báo cáo hằng ngày): {symbol: {entry_price, entry_date, peak_price, size, ...}}
    port_df.attrs["open_positions"] = {sym: dict(pos) for sym, pos in pm.positions.items()}
    trade_df = pd.DataFrame(trade_log)
    return port_df, trade_df


def invested_returns(port_df: pd.DataFrame, col: str = "port_return") -> pd.Series:
    """Chuỗi return TỪ ngày đầu tiên thực sự có vị thế (n_positions_start > 0) trở đi.

    Khác cách cũ `ret[ret != 0]` (lọc theo GIÁ TRỊ): ở đây chỉ cắt giai đoạn warm-up chưa có
    vị thế, các ngày return = 0 hợp lệ sau đó (ví dụ giữ toàn tiền mặt) được GIỮ LẠI, nên
    Sharpe/Vol không bị lệch do bỏ ngày.
    """
    started = port_df["n_positions_start"] > 0
    if not started.any():
        return port_df[col].iloc[0:0]
    return port_df.loc[started.idxmax():, col]


def port_metrics(ret_series: pd.Series, ann_factor: int = 252) -> dict[str, float]:
    """Metrics chuẩn: Total/Ann Return, Ann Vol, Sharpe, Max Drawdown."""
    cum = (1 + ret_series).cumprod()
    total = cum.iloc[-1] - 1
    ann_r = (1 + total) ** (ann_factor / len(ret_series)) - 1
    ann_v = ret_series.std() * np.sqrt(ann_factor)
    sharpe = ann_r / ann_v if ann_v > 0 else np.nan
    max_dd = (cum / cum.cummax() - 1).min()
    return {"Total Return": total, "Ann Return": ann_r, "Ann Vol": ann_v, "Sharpe": sharpe, "Max DD": max_dd}


def sharpe_se(ret_series: pd.Series, ann_factor: int = 252) -> float:
    """Sai số chuẩn XẤP XỈ của Sharpe hằng năm: √(ann·(1 + SR_d²/2) / T), với SR_d = Sharpe theo ngày.
    Dùng để biết chênh lệch giữa 2 Sharpe có lớn hơn nhiễu hay không (T ≈ 700 ngày → SE ≈ 0.6)."""
    sd = ret_series.std()
    if not sd > 0:
        return np.nan
    sr_d = ret_series.mean() / sd
    return float(np.sqrt(ann_factor * (1 + 0.5 * sr_d ** 2) / len(ret_series)))


def calmar_ratio(metrics: dict[str, float]) -> float:
    """Calmar = Annualized Return / |Max Drawdown|."""
    if metrics["Max DD"] == 0 or np.isnan(metrics["Max DD"]):
        return float("nan")
    return metrics["Ann Return"] / abs(metrics["Max DD"])


def trade_statistics(trade_df: pd.DataFrame) -> dict:
    """Thống kê giao dịch: win rate, avg win/loss, avg hold days, exit breakdown."""
    if len(trade_df) == 0:
        return {}
    sells = trade_df[trade_df["action"] != "BUY"].copy()
    buys = trade_df[trade_df["action"] == "BUY"].copy()
    if len(sells) == 0:
        return {"n_buys": len(buys), "n_sells": 0}

    wins, losses = sells[sells["pnl"] > 0], sells[sells["pnl"] <= 0]
    return {
        "n_buys": len(buys), "n_sells": len(sells),
        "win_rate": len(wins) / len(sells),
        "avg_win_pnl": wins["pnl"].mean() if len(wins) else np.nan,
        "avg_loss_pnl": losses["pnl"].mean() if len(losses) else np.nan,
        "avg_hold_days": sells["hold_days"].mean(),
        "exit_breakdown": sells["action"].value_counts().to_dict(),
    }
