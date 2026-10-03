# -*- coding: utf-8 -*-
"""Kiểm tra purged_kfold_splits() và get_date_indexed_splits() không rò rỉ dữ liệu."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ic_analysis import get_date_indexed_splits, purged_kfold_splits


def test_no_overlap_between_train_and_test():
    for train_idx, test_idx in purged_kfold_splits(n_samples=498, n_splits=5, embargo=5):
        assert len(set(train_idx) & set(test_idx)) == 0


def test_embargo_actually_removes_neighbors():
    n, embargo = 100, 5
    for train_idx, test_idx in purged_kfold_splits(n, n_splits=5, embargo=embargo):
        test_start, test_end = test_idx.min(), test_idx.max()
        near_boundary = set(range(max(0, test_start - embargo), test_start)) | \
                        set(range(test_end + 1, min(n, test_end + 1 + embargo)))
        assert near_boundary.isdisjoint(set(train_idx))


def _make_dataset(n_days: int = 100, n_assets: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"SYM{i}" for i in range(n_assets)]], names=["date", "symbol"])
    return pd.DataFrame({"feature": rng.normal(size=len(idx)), "fwd_return": rng.normal(size=len(idx))}, index=idx)


def test_date_indexed_splits_keep_same_day_together():
    """Mọi (date, symbol) của CÙNG 1 NGÀY phải nằm cùng phía train/test —
    không được xảy ra 1 ngày vừa ở train vừa ở test (leakage xuyên mã)."""
    dataset = _make_dataset()
    date_arr = dataset.index.get_level_values("date").values

    for train_idx, test_idx in get_date_indexed_splits(dataset, n_splits=5, embargo=3):
        train_dates = set(date_arr[train_idx])
        test_dates = set(date_arr[test_idx])
        assert train_dates.isdisjoint(test_dates)


def test_date_indexed_splits_cover_all_rows():
    dataset = _make_dataset()
    seen = set()
    for _, test_idx in get_date_indexed_splits(dataset, n_splits=5, embargo=3):
        seen |= set(test_idx)
    assert seen == set(range(len(dataset)))
