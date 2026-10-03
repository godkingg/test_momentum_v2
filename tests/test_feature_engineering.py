# -*- coding: utf-8 -*-
"""Kiểm tra feature_engineering.py: cross-sectional rank, dataset assembly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import (
    assemble_dataset, build_classification_label, build_feature_matrix,
    build_forward_return_label, cross_sectional_rank,
)


def _make_price_df(n: int = 200, n_assets: int = 10, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    cols = [f"SYM{i}" for i in range(n_assets)]
    log_rets = rng.normal(0.0003, 0.02, size=(n, n_assets))
    return pd.DataFrame(100 * np.exp(np.cumsum(log_rets, axis=0)), index=dates, columns=cols)


def test_cross_sectional_rank_is_percentile_0_to_1():
    price_df = _make_price_df()
    ranked = cross_sectional_rank(price_df)
    valid = ranked.dropna(how="all")
    assert (valid.min(axis=1) >= 0).all()
    assert (valid.max(axis=1) <= 1).all()


def test_cross_sectional_rank_is_per_row_not_per_column():
    """Rank phải tính THEO HÀNG (cross-sectional, cùng ngày) không phải theo cột (time-series)."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 2, 1]})
    ranked = cross_sectional_rank(df)
    # Ngày 0: A=1 < B=3 -> A rank thấp hơn B
    assert ranked.loc[0, "A"] < ranked.loc[0, "B"]
    # Ngày 2: A=3 > B=1 -> A rank cao hơn B
    assert ranked.loc[2, "A"] > ranked.loc[2, "B"]


def test_build_feature_matrix_multiindex_shape():
    price_df = _make_price_df()
    factors = {"f1": price_df.pct_change(5, fill_method=None), "f2": price_df.pct_change(20, fill_method=None)}
    fm = build_feature_matrix(factors)
    assert list(fm.index.names) == ["date", "symbol"]
    assert list(fm.columns) == ["f1", "f2"]


def test_assemble_dataset_regression_no_nan():
    price_df = _make_price_df()
    factors = {"f1": price_df.pct_change(5, fill_method=None)}
    dataset = assemble_dataset(factors, price_df, n_fwd=5, label_type="regression")
    assert dataset.isna().sum().sum() == 0
    assert "fwd_return" in dataset.columns


def test_assemble_dataset_classification_labels_are_0_or_1():
    price_df = _make_price_df()
    factors = {"f1": price_df.pct_change(5, fill_method=None)}
    dataset = assemble_dataset(factors, price_df, n_fwd=5, label_type="classification", top_pct=0.3, bottom_pct=0.3)
    assert set(dataset["label_class"].unique()).issubset({0.0, 1.0})


def test_classification_label_drops_middle_by_construction():
    """Middle 40% (giữa top 30% và bottom 30%) phải bị loại (NaN) trước dropna."""
    price_df = _make_price_df(n_assets=20)  # đủ mã để chia rõ 3 nhóm
    label = build_classification_label(price_df, n_fwd=5, top_pct=0.3, bottom_pct=0.3)
    non_na_pct = label.notna().mean()
    assert non_na_pct < 0.7  # phải loại ít nhất phần nào ở giữa
