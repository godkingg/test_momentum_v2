# -*- coding: utf-8 -*-
"""pipeline.py — Dựng signal của chiến lược (Composite IC-weighted) dùng chung cho launcher và báo cáo hằng ngày.

Trọng số IC có thể được TRUYỀN VÀO (đóng băng) thay vì tính lại trên toàn mẫu mỗi lần chạy: với chạy hằng ngày, tính lại
làm lịch sử mô phỏng thay đổi nhẹ mỗi ngày → lệnh hôm qua có thể "biến mất" khỏi lịch sử hôm nay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import assemble_dataset, cross_sectional_rank
from src.ic_analysis import single_factor_ic
from src.momentum_factors import momentum_12_1, momentum_6_1
from src.strategy import composite_score, normalize_ic_weights

N_FWD = 5
FACTOR_NAMES = ("Momentum_12_1", "Momentum_6_1", "Volume_Ratio", "TS_Momentum")
# Tham số PositionManager + engine (giữ nguyên như launcher.py); cost_rate truyền riêng
ENGINE_PARAMS = dict(
    stop_loss_pct=0.08, trail_pct=0.12, vol_exit_thr=3.0, vol_reduce_thr=2.0,
    top_entry_rank=0.65, signal_exit_rank=0.40, max_positions=10, rebal_freq=5,
)


def build_factors(price_df: pd.DataFrame, volume_df: pd.DataFrame):
    """4 factor chính — công thức giữ nguyên từ Ngày 18-19. Trả về (raw_factors, vol_20, vol_120)."""
    returns_df = price_df.pct_change(fill_method=None)
    vol_20 = returns_df.rolling(20).std()
    vol_120 = returns_df.rolling(120).std()

    mom_12 = momentum_12_1(price_df)
    target_vol = 0.15 / np.sqrt(252)
    ts_mom = np.sign(mom_12) * (target_vol / vol_20.replace(0, np.nan))

    return {
        "Momentum_12_1": mom_12,
        "Momentum_6_1": momentum_6_1(price_df),
        "Volume_Ratio": volume_df / volume_df.rolling(20).mean(),
        "TS_Momentum": ts_mom,
    }, vol_20, vol_120


def fit_ic_weights(price_df: pd.DataFrame, volume_df: pd.DataFrame) -> dict[str, float]:
    """Ước lượng trọng số IC (Purged K-Fold OOF) trên TOÀN BỘ dữ liệu hiện có."""
    raw, _, _ = build_factors(price_df, volume_df)
    dataset = assemble_dataset(raw, price_df, n_fwd=N_FWD, label_type="regression")
    if len(dataset) < 200:
        raise ValueError("Dataset quá nhỏ để đo IC đáng tin cậy — kiểm tra khoảng thời gian dữ liệu.")
    return normalize_ic_weights(single_factor_ic(dataset, list(raw), n_splits=5, embargo=5))


def build_strategy_inputs(price_df: pd.DataFrame, volume_df: pd.DataFrame, ic_weights: dict[str, float] | None = None) -> dict:
    """Trả về dict: comp_score, vol_spike, mom_6, raw_factors, ic_weights. ic_weights=None -> tự ước lượng."""
    raw, vol_20, vol_120 = build_factors(price_df, volume_df)
    if ic_weights is None:
        ic_weights = fit_ic_weights(price_df, volume_df)
    ranked = {name: cross_sectional_rank(f) for name, f in raw.items()}
    return {
        "comp_score": composite_score(ranked, ic_weights),
        "vol_spike": vol_20 / vol_120,
        "mom_6": raw["Momentum_6_1"],
        "raw_factors": raw,
        "ic_weights": ic_weights,
    }
