# -*- coding: utf-8 -*-
"""backtest_legacy.py — ENGINE CŨ (Ngày 19-22), CÓ LOOK-AHEAD CÙNG NGÀY. KHÔNG dùng cho kết quả nghiên cứu.

⚠️ Giữ lại CHỈ để (1) tái tạo con số cũ, (2) đo mức thiên lệch so với engine mới
(`src/backtest.py`), (3) làm đối chứng trong tests/test_engine_no_lookahead.py.

Lỗi: vị thế mở/đóng ở close ngày t vẫn ăn (hoặc né) return CỦA CHÍNH ngày t —
exit được quyết định bằng giá hôm nay rồi vị thế bị gỡ TRƯỚC khi tính return
hôm nay, nên cú giảm kích hoạt stop không bao giờ vào PnL; vị thế mới mở ở close t
vẫn hưởng return ngày t. Với tín hiệu NGẪU NHIÊN trên giá random-walk, engine này cho
Sharpe trung bình dương (+0.37 qua 40 seed) thay vì ≈ 0.

Docstring gốc:
backtest.py — Backtest engine với entry/exit rules (PositionManager).

Logic backtest_with_exits() giữ NGUYÊN từ day19_test.ipynb. Đây KHÔNG phải
backtest weight cố định theo rebalance đơn giản — mỗi mã có vòng đời
riêng (entry → hold → exit theo 6 rule), nên không thể dùng chung engine
weight-matrix của project momentum-ridge-vn30.

Cost model (Ngày 20): thêm transaction cost + slippage — GIỮ NGUYÊN giả
định đã dùng ở project momentum-ridge-vn30 (0.15% transaction + 0.08%
slippage = 0.23% tổng mỗi lần đổi vị thế), áp trên notional giao dịch tại
mỗi BUY và mỗi SELL (không phải trên toàn bộ portfolio — chỉ phần THAY ĐỔI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import PositionManager

# Giữ NGUYÊN giả định cost từ project momentum-ridge-vn30 (Ngày 13)
TRANSACTION_COST = 0.0015  # 0.15% round-trip
SLIPPAGE = 0.0008          # 0.08% ước lượng bid-ask spread
TOTAL_COST_RATE = TRANSACTION_COST + SLIPPAGE


def backtest_with_exits_legacy(
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
    """Backtest đầy đủ với entry + exit rules — giữ nguyên từ Ngày 19.

    cost_rate: phí áp trên NOTIONAL giao dịch mỗi lần BUY hoặc SELL (mặc định
    0.0 = tương thích ngược với Ngày 19, không cost). Dùng TOTAL_COST_RATE
    (0.23%) để tái tạo giả định cost giống project momentum-ridge-vn30.

    Trả về (port_df, trade_df):
      port_df: index=date, cột port_return (net cost nếu cost_rate>0),
               gross_return (trước cost), cost, n_positions
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

        n_pos_before_exit = len(pm.positions)
        day_cost = 0.0  # tổng phí phát sinh trong ngày (tính theo tỷ trọng notional)

        # Bước 1: kiểm tra EXIT cho vị thế đang giữ
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
                # Cost = size_sell (tỷ trọng bán) / n_positions TRƯỚC exit (notional weight
                # của vị thế đó trong portfolio) × cost_rate. n_pos_before_exit đảm bảo
                # cost tính trên đúng tỷ trọng portfolio ngay TRƯỚC lệnh bán.
                if cost_rate > 0 and n_pos_before_exit > 0:
                    day_cost += cost_rate * size_sell / n_pos_before_exit
                pm.close_position(sym, size_sell)

        # Bước 2: entry mới (mỗi rebal_freq ngày)
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
                        slots -= 1

        # Entry mới -> cost tính trên tỷ trọng notional SAU khi biết tổng vị thế
        # cuối ngày (n_pos hiện tại) — mỗi mã mới chiếm 1/n_pos của portfolio.
        n_pos = len(pm.positions)
        n_new_entries = sum(1 for t in trade_log if t["date"] == date and t["action"] == "BUY")
        if cost_rate > 0 and n_new_entries > 0 and n_pos > 0:
            day_cost += cost_rate * n_new_entries / n_pos

        # Bước 3: tính portfolio return hôm nay
        gross_ret = 0.0
        if n_pos > 0 and i > 0:
            for sym, pos in pm.positions.items():
                if sym in ret_df.columns and date in ret_df.index and not pd.isna(ret_df.loc[date, sym]):
                    gross_ret += ret_df.loc[date, sym] * pos["size"] / n_pos

        net_ret = gross_ret - day_cost
        portfolio_log.append({
            "date": date, "port_return": net_ret, "gross_return": gross_ret,
            "cost": day_cost, "n_positions": n_pos,
        })

    port_df = pd.DataFrame(portfolio_log).set_index("date")
    trade_df = pd.DataFrame(trade_log)
    return port_df, trade_df
