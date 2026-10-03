# -*- coding: utf-8 -*-
"""strategy.py — Composite signal + PositionManager (entry/exit rules).

Logic giữ NGUYÊN từ day19_test.ipynb (Cell "Xây dựng Exit Rules hoàn chỉnh").
Khác biệt so với chiến lược Long-only Q5 đơn giản (project momentum-ridge-vn30):
đây là chiến lược có QUẢN LÝ VỊ THẾ theo thời gian — 1 mã có thể được giữ
qua nhiều kỳ, và có 6 loại exit rule ưu tiên theo thứ tự cụ thể.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def composite_score(
    ranked_factors: dict[str, pd.DataFrame], ic_weights: dict[str, float],
) -> pd.DataFrame:
    """score_i = Σ (IC_k / Σ|IC|) × rank_k(factor_k_i) — composite IC-weighted.

    ranked_factors: {factor_name: cross-sectional rank DataFrame [date x symbol]}
    ic_weights: đã normalize (tổng |weight| = 1), ví dụ từ IC đo được ở Ngày 18-19.
    """
    factor_names = list(ranked_factors.keys())
    score = sum(ic_weights[name] * ranked_factors[name] for name in factor_names)
    return score


def normalize_ic_weights(ic_series: pd.Series) -> dict[str, float]:
    """Chuẩn hóa Mean IC từng factor thành weight (tổng |weight| = 1)."""
    total_abs = ic_series.abs().sum()
    return (ic_series / total_abs).to_dict()


class PositionManager:
    """Quản lý vị thế: khi nào MUA, khi nào BÁN. Logic giữ nguyên từ Ngày 19.

    Entry Rules:
      - composite_rank >= top_entry_rank (mặc định top 35%)
      - mom_6 > 0 (momentum dương)
      - vol_spike < vol_reduce_thr (không bất ổn)

    Exit Rules (ưu tiên từ cao → thấp):
      1. HARD STOP   : giá giảm >= stop_loss_pct từ giá mua
      2. TRAILING    : giá giảm >= trail_pct từ đỉnh gần nhất
      3. VOL EXIT    : vol_spike >= vol_exit_thr
      4. VOL REDUCE  : vol_spike >= vol_reduce_thr → bán 50%
      5. SIGNAL EXIT : composite_rank < signal_exit_rank (rớt khỏi top)
      6. MOM FLIP    : mom_6 < 0 (momentum đảo chiều)
    """

    def __init__(
        self,
        stop_loss_pct: float = 0.08,
        trail_pct: float = 0.12,
        vol_exit_thr: float = 3.0,
        vol_reduce_thr: float = 2.0,
        signal_exit_rank: float = 0.40,
        top_entry_rank: float = 0.65,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.trail_pct = trail_pct
        self.vol_exit_thr = vol_exit_thr
        self.vol_reduce_thr = vol_reduce_thr
        self.signal_exit_rank = signal_exit_rank
        self.top_entry_rank = top_entry_rank
        self.positions: dict[str, dict] = {}

    def check_exit(self, sym: str, current_price: float, composite_rank: float, vol_spike: float, mom_6: float):
        """Trả về (action, reason, size_to_sell)."""
        if sym not in self.positions:
            return "NO_POSITION", "", 0.0

        pos = self.positions[sym]
        entry_price, peak_price, current_size = pos["entry_price"], pos["peak_price"], pos["size"]

        if current_price > peak_price:
            self.positions[sym]["peak_price"] = current_price
            peak_price = current_price

        pnl_from_entry = (current_price / entry_price) - 1
        pnl_from_peak = (current_price / peak_price) - 1

        if pnl_from_entry <= -self.stop_loss_pct:
            return "HARD_STOP", f"Giảm {pnl_from_entry:.1%} từ giá mua (ngưỡng -{self.stop_loss_pct:.0%})", current_size
        if pnl_from_peak <= -self.trail_pct:
            return "TRAILING_STOP", f"Giảm {pnl_from_peak:.1%} từ đỉnh (ngưỡng -{self.trail_pct:.0%})", current_size
        if vol_spike >= self.vol_exit_thr:
            return "VOL_EXIT", f"Vol spike {vol_spike:.2f}x >= {self.vol_exit_thr}x", current_size
        if vol_spike >= self.vol_reduce_thr and current_size > 0.55:
            return "VOL_REDUCE", f"Vol spike {vol_spike:.2f}x >= {self.vol_reduce_thr}x", current_size * 0.5
        if composite_rank < self.signal_exit_rank:
            return "SIGNAL_EXIT", f"Composite rank {composite_rank:.2f} < {self.signal_exit_rank}", current_size
        if mom_6 < 0:
            return "MOM_FLIP", f"Mom-6 = {mom_6:.1%} < 0 (momentum đảo chiều)", current_size

        return "HOLD", "Tất cả điều kiện giữ OK", 0.0

    def check_entry(self, sym: str, price: float, composite_rank: float, vol_spike: float, mom_6: float, date=None):
        if sym in self.positions:
            return False, "Đã có vị thế"
        if composite_rank < self.top_entry_rank:
            return False, f"Rank {composite_rank:.2f} < {self.top_entry_rank}"
        if mom_6 <= 0:
            return False, f"Mom-6 {mom_6:.1%} <= 0"
        if vol_spike >= self.vol_reduce_thr:
            return False, f"Vol spike {vol_spike:.2f}x quá cao"
        return True, "✅ Entry OK"

    def open_position(self, sym: str, price: float, date, reason: str = ""):
        self.positions[sym] = {
            "entry_price": price, "entry_date": date, "peak_price": price,
            "size": 1.0, "entry_reason": reason,
        }

    def close_position(self, sym: str, size_to_sell: float | None = None):
        if sym not in self.positions:
            return
        pos = self.positions[sym]
        if size_to_sell is None or size_to_sell >= pos["size"]:
            del self.positions[sym]
        else:
            self.positions[sym]["size"] -= size_to_sell

    def get_portfolio_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.positions).T


def classify_signal(
    composite_rank: pd.Series, vol_spike: pd.Series, mom_6: pd.Series,
    top_pct: float = 0.35, bottom_pct: float = 0.35, vol_threshold: float = 2.0,
) -> pd.Series:
    """Signal cho từng mã tại 1 thời điểm: BUY / AVOID / REDUCE / EXIT / NEUTRAL.

    Dùng cho dashboard "hôm nay nên làm gì" khi chưa có vị thế cụ thể (entry-only view).
    Xem PositionManager.check_exit cho logic exit đầy đủ khi ĐÃ có vị thế.
    """
    signal = pd.Series("NEUTRAL", index=composite_rank.index)
    signal[composite_rank >= (1 - top_pct)] = "BUY"
    signal[composite_rank <= bottom_pct] = "AVOID"
    signal[vol_spike >= vol_threshold * 1.5] = "EXIT"
    signal[(vol_spike >= vol_threshold) & (vol_spike < vol_threshold * 1.5) & (signal == "BUY")] = "REDUCE"
    return signal
