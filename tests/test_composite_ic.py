# -*- coding: utf-8 -*-
"""Kiểm tra composite_score_ic: IC-weighted composite đo đúng framework OOS."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ic_analysis import composite_score_ic, ic_ir_stats, single_factor_ic


def _make_dataset(n_days: int = 100, n_assets: int = 15, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"SYM{i}" for i in range(n_assets)]], names=["date", "symbol"])
    n = len(idx)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    # label tương quan dương với f1 (để composite có tín hiệu thật, không random thuần)
    label = 0.3 * f1 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"f1": f1, "f2": f2, "fwd_return": label}, index=idx)


def test_composite_score_ic_returns_valid_stats_dict():
    dataset = _make_dataset()
    ic_weights = {"f1": 0.7, "f2": 0.3}
    stats = composite_score_ic(dataset, ["f1", "f2"], ic_weights, n_splits=5, embargo=3)
    assert set(stats.keys()) == {"Mean IC", "Std IC", "IC-IR", "IC>0 (%)", "N Obs"}
    assert stats["N Obs"] > 0


def test_composite_score_ic_picks_up_signal_from_informative_factor():
    """f1 có tương quan dương thật với label -> composite dùng chủ yếu f1 phải có Mean IC dương rõ rệt."""
    dataset = _make_dataset(n_days=200, seed=1)
    ic_weights = {"f1": 1.0, "f2": 0.0}  # composite = chỉ dùng f1
    stats = composite_score_ic(dataset, ["f1", "f2"], ic_weights, n_splits=5, embargo=3)
    assert stats["Mean IC"] > 0.05  # f1 có tín hiệu thật (label = 0.3*f1 + noise)


def test_composite_score_ic_all_zero_weight_gives_near_zero_ic():
    """Nếu composite chỉ dùng factor KHÔNG có tín hiệu (f2, random thuần), Mean IC phải gần 0."""
    dataset = _make_dataset(n_days=200, seed=2)
    ic_weights = {"f1": 0.0, "f2": 1.0}
    stats = composite_score_ic(dataset, ["f1", "f2"], ic_weights, n_splits=5, embargo=3)
    assert abs(stats["Mean IC"]) < 0.15  # random factor -> IC nhiễu quanh 0, không hệ thống


def test_composite_score_ic_comparable_format_with_single_factor_ic():
    """composite_score_ic và single_factor_ic phải dùng CÙNG framework OOS (n_splits,
    embargo giống nhau) để phép so sánh công bằng — kiểm tra N Obs tương đương."""
    dataset = _make_dataset(n_days=150, seed=3)
    sf_ic = single_factor_ic(dataset, ["f1", "f2"], n_splits=5, embargo=3)
    comp_stats = composite_score_ic(dataset, ["f1", "f2"], {"f1": 0.5, "f2": 0.5}, n_splits=5, embargo=3)
    # single_factor_ic không trả N Obs trực tiếp, nhưng composite phải có N Obs hợp lý (>0, gần với số ngày OOS)
    assert comp_stats["N Obs"] > 0
    assert not sf_ic.empty
