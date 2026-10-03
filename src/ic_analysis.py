# -*- coding: utf-8 -*-
"""ic_analysis.py — Module đo Information Coefficient (IC) dùng chung.

Tái sử dụng cho mọi factor mới (momentum, mean-reversion, volume, ML score...).
Logic Purged K-Fold giữ nguyên từ notebook Ngày 18-19 (day19_test.ipynb).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_sectional_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> pd.Series:
    """Spearman IC cross-sectional theo từng ngày. Trả về Series index=date."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "date": dates})
    ic_by_date = df.groupby("date").apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"])[0] if len(g) >= 5 else np.nan,
        include_groups=False,
    )
    return ic_by_date.dropna()


def purged_kfold_splits(n_samples: int, n_splits: int = 5, embargo: int = 5):
    """Time-ordered K-Fold với purging/embargo — tránh leakage giữa train/test
    liền kề theo thời gian. Giữ nguyên logic từ Ngày 13/18/19.
    """
    indices = np.arange(n_samples)
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        test_start, test_end = current, current + fold_size
        test_idx = indices[test_start:test_end]
        embargo_start = max(0, test_start - embargo)
        embargo_end = min(n_samples, test_end + embargo)
        train_idx = np.concatenate([indices[:embargo_start], indices[embargo_end:]])
        yield train_idx, test_idx
        current = test_end


def get_date_indexed_splits(df: pd.DataFrame, n_splits: int = 5, embargo: int = 5):
    """purged_kfold_splits nhưng chia theo NGÀY DUY NHẤT (không phải theo hàng),
    để mọi (date, symbol) cùng ngày luôn nằm cùng 1 phía train/test — tránh
    leakage xuyên mã trong cùng ngày.
    """
    unique_dates = sorted(df.index.get_level_values("date").unique())
    unique_dates_arr = np.array(unique_dates, dtype="datetime64[ns]")
    date_arr = df.index.get_level_values("date").values.astype("datetime64[ns]")

    for train_di, test_di in purged_kfold_splits(len(unique_dates), n_splits, embargo):
        train_dates = unique_dates_arr[train_di]
        test_dates = unique_dates_arr[test_di]
        train_mask = np.isin(date_arr, train_dates)
        test_mask = np.isin(date_arr, test_dates)
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def single_factor_ic(
    dataset: pd.DataFrame, feature_cols: list[str], label_col: str = "fwd_return",
    n_splits: int = 5, embargo: int = 5,
) -> pd.Series:
    """Mean IC (OOS, Purged K-Fold) cho từng factor đứng riêng lẻ.

    dataset cần MultiIndex (date, symbol) — dùng để get_date_indexed_splits
    chia đúng theo ngày duy nhất.
    """
    y = dataset[label_col]
    X = dataset[feature_cols]
    dates_arr = dataset.index.get_level_values("date").values.astype("datetime64[ns]")

    result = {}
    for feat in feature_cols:
        fold_ics = []
        for _, test_idx in get_date_indexed_splits(dataset, n_splits, embargo):
            if len(test_idx) == 0:
                continue
            ic_s = cross_sectional_ic(y.iloc[test_idx].values, X[feat].iloc[test_idx].values, dates_arr[test_idx])
            if len(ic_s) > 0:
                fold_ics.append(ic_s.mean())
        result[feat] = np.mean(fold_ics) if fold_ics else np.nan

    return pd.Series(result, name="Mean IC").sort_values(key=abs, ascending=False)


def ic_ir_stats(ic_series: pd.Series) -> dict[str, float]:
    """IC-IR (Information Ratio của IC) và các thống kê phụ trợ."""
    mean_ic, std_ic = ic_series.mean(), ic_series.std()
    return {
        "Mean IC": mean_ic,
        "Std IC": std_ic,
        "IC-IR": mean_ic / std_ic if std_ic else np.nan,
        "IC>0 (%)": (ic_series > 0).mean(),
        "N Obs": len(ic_series),
    }


def composite_score_ic(
    dataset: pd.DataFrame, feature_cols: list[str], ic_weights: dict[str, float],
    label_col: str = "fwd_return", n_splits: int = 5, embargo: int = 5,
) -> dict[str, float]:
    """Mean IC (OOS, Purged K-Fold) của COMPOSITE SCORE (IC-weighted linear
    combination), đo bằng đúng framework Purged K-Fold như Ridge/XGBoost để
    so sánh công bằng "táo với táo" — không chỉ so composite với single-factor.

    ⚠️ Lưu ý về tính công bằng của phép so sánh: composite_score ở đây dùng
    ic_weights CỐ ĐỊNH (thường tính từ TOÀN BỘ lịch sử, xem single_factor_ic())
    — không "refit" ic_weights riêng trên từng fold train như Ridge/XGBoost
    refit model. Điều này cho composite 1 LỢI THẾ CẤU TRÚC nhẹ (ít bị nhiễu
    do không có bước ước lượng tham số trên mỗi fold), cần nêu rõ khi diễn
    giải kết quả — composite thắng không đồng nghĩa "composite là phương
    pháp tốt hơn về bản chất", có thể một phần do ít bậc tự do hơn để overfit.

    Trả về dict giống ic_ir_stats(): Mean IC, Std IC, IC-IR, IC>0 (%), N Obs.
    """
    y = dataset[label_col]
    X = dataset[feature_cols]
    dates_arr = dataset.index.get_level_values("date").values.astype("datetime64[ns]")

    composite_pred = sum(ic_weights[feat] * X[feat] for feat in feature_cols)

    all_ic = []
    for _, test_idx in get_date_indexed_splits(dataset, n_splits, embargo):
        if len(test_idx) == 0:
            continue
        ic_s = cross_sectional_ic(y.iloc[test_idx].values, composite_pred.iloc[test_idx].values, dates_arr[test_idx])
        all_ic.append(ic_s)

    combined = pd.concat(all_ic).sort_index() if all_ic else pd.Series(dtype=float)
    return ic_ir_stats(combined)


def full_oos_ic_ir(
    dataset: pd.DataFrame, feature_col: str, label_col: str = "fwd_return",
    n_splits: int = 5, embargo: int = 5,
) -> dict[str, float]:
    """IC-IR đầy đủ trên TOÀN BỘ daily IC gộp từ mọi fold OOS (không chỉ mean
    của mean từng fold) — cho ước lượng std ổn định hơn single_factor_ic().
    """
    y = dataset[label_col]
    X = dataset[[feature_col]]
    dates_arr = dataset.index.get_level_values("date").values.astype("datetime64[ns]")

    all_ic = []
    for _, test_idx in get_date_indexed_splits(dataset, n_splits, embargo):
        if len(test_idx) == 0:
            continue
        ic_s = cross_sectional_ic(y.iloc[test_idx].values, X[feature_col].iloc[test_idx].values, dates_arr[test_idx])
        all_ic.append(ic_s)

    combined = pd.concat(all_ic).sort_index() if all_ic else pd.Series(dtype=float)
    return ic_ir_stats(combined)
