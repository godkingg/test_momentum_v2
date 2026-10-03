# -*- coding: utf-8 -*-
"""feature_engineering.py — Module xây dựng feature matrix cho ML alpha model.

Tách riêng thành module độc lập (không để hàm rải rác trong notebook) để
tái sử dụng cho các project sau này. Logic cross-sectional rank + build
dataset giữ nguyên từ Ngày 19 (day19_test.ipynb, Cell 5-6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cross_sectional_rank(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Rank percentile [0,1] mỗi ngày (cross-sectional) cho 1 factor.

    Rank thay vì dùng giá trị thô giúp các factor có scale khác nhau
    (momentum ~[-1,1], volume ratio ~[0, 5+]) so sánh được trực tiếp,
    và giảm ảnh hưởng của outlier cực trị.
    """
    return factor_df.rank(axis=1, pct=True)


def build_feature_matrix(raw_factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ghép nhiều factor (mỗi factor 1 DataFrame [date x symbol]) thành 1
    feature matrix long-format, mỗi factor được cross-sectional rank trước.

    Trả về DataFrame MultiIndex (date, symbol) x [feature_1, feature_2, ...].
    """
    ranked = {name: cross_sectional_rank(f) for name, f in raw_factors.items()}
    frames = []
    for name, r in ranked.items():
        r_clean = r.copy()
        r_clean.index = pd.to_datetime(r_clean.index).tz_localize(None)
        frames.append(r_clean.stack(future_stack=True).rename(name))

    feature_matrix = pd.concat(frames, axis=1)
    feature_matrix.index.names = ["date", "symbol"]
    return feature_matrix


def build_forward_return_label(
    price_df: pd.DataFrame, n_fwd: int = 5,
) -> pd.Series:
    """Label: forward return N ngày, dạng long-format (date, symbol) -> giá trị.

    N_FWD mặc định 5 (1 tuần giao dịch) — cân bằng giữa đủ tín hiệu và
    tránh label quá nhiễu (N_FWD=1 quá nhiễu ở VN do thanh khoản thấp).
    """
    fwd_ret = price_df.shift(-n_fwd) / price_df - 1
    fwd_ret.index = pd.to_datetime(fwd_ret.index).tz_localize(None)
    label = fwd_ret.stack(future_stack=True).rename("fwd_return")
    label.index.names = ["date", "symbol"]
    return label


def build_classification_label(
    price_df: pd.DataFrame, n_fwd: int = 5, top_pct: float = 0.3, bottom_pct: float = 0.3,
) -> pd.Series:
    """Label classification thay thế: 1 = top quintile/tercile forward return,
    0 = bottom, NaN = giữa (loại khỏi training) — dùng khi label regression
    quá noisy (phổ biến với universe nhỏ, thanh khoản thấp như VN30).

    Cross-sectional theo từng ngày: so sánh mã này với các mã khác CÙNG NGÀY,
    không so theo thời gian.
    """
    fwd_ret = price_df.shift(-n_fwd) / price_df - 1
    fwd_ret.index = pd.to_datetime(fwd_ret.index).tz_localize(None)

    def label_row(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        if len(valid) < 5:
            return pd.Series(np.nan, index=row.index)
        top_thresh = valid.quantile(1 - top_pct)
        bottom_thresh = valid.quantile(bottom_pct)
        out = pd.Series(np.nan, index=row.index)
        out[row >= top_thresh] = 1
        out[row <= bottom_thresh] = 0
        return out

    label_wide = fwd_ret.apply(label_row, axis=1)
    label = label_wide.stack(future_stack=True).rename("label_class")
    label.index.names = ["date", "symbol"]
    return label


def assemble_dataset(
    raw_factors: dict[str, pd.DataFrame], price_df: pd.DataFrame, n_fwd: int = 5,
    label_type: str = "regression", top_pct: float = 0.3, bottom_pct: float = 0.3,
) -> pd.DataFrame:
    """Pipeline đầy đủ: raw_factors -> feature matrix (ranked) + label -> dataset sạch.

    label_type: "regression" (fwd_return liên tục) hoặc "classification"
    (top/bottom quintile, 0/1, loại bỏ vùng giữa).
    """
    features = build_feature_matrix(raw_factors)

    if label_type == "classification":
        label = build_classification_label(price_df, n_fwd, top_pct, bottom_pct)
    else:
        label = build_forward_return_label(price_df, n_fwd)

    return features.join(label, how="inner").dropna()
