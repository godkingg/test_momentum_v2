# -*- coding: utf-8 -*-
"""Kiểm tra momentum factors không dùng thông tin tương lai."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.momentum_factors import forward_return, momentum_6_1, momentum_12_1


def _make_price_df(n: int = 300, n_assets: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    cols = [f"SYM{i}" for i in range(n_assets)]
    log_rets = rng.normal(0.0003, 0.02, size=(n, n_assets))
    return pd.DataFrame(100 * np.exp(np.cumsum(log_rets, axis=0)), index=dates, columns=cols)


def test_momentum_at_t_unaffected_by_future_prices():
    """momentum_6_1(t) không đổi khi ta cắt bớt giá SAU ngày t."""
    price_df = _make_price_df()
    mom_full = momentum_6_1(price_df)

    cutoff = 200
    price_trunc = price_df.iloc[:cutoff]
    mom_trunc = momentum_6_1(price_trunc)

    stable_idx = price_trunc.index[130:]  # đủ warmup (126+21), tránh biên
    pd.testing.assert_frame_equal(mom_full.loc[stable_idx], mom_trunc.loc[stable_idx])


def test_momentum_12_1_at_t_unaffected_by_future_prices():
    price_df = _make_price_df()
    mom_full = momentum_12_1(price_df)

    cutoff = 280
    price_trunc = price_df.iloc[:cutoff]
    mom_trunc = momentum_12_1(price_trunc)

    stable_idx = price_trunc.index[260:]
    pd.testing.assert_frame_equal(mom_full.loc[stable_idx], mom_trunc.loc[stable_idx])


def test_forward_return_uses_future_by_design_and_is_documented():
    """forward_return() CỐ Ý dùng future price (đây là LABEL, không phải feature) —
    test này xác nhận hành vi đúng như thiết kế, không phải look-ahead bug."""
    price_df = _make_price_df()
    fwd = forward_return(price_df, n_fwd=5)
    expected_last5_nan = fwd.iloc[-5:]
    assert expected_last5_nan.isna().all().all(), "5 ngày cuối phải NaN (không có giá tương lai để tính)"
