# -*- coding: utf-8 -*-
"""Kiểm tra cost model trong backtest_with_exits: cost_rate=0 tương thích ngược,
cost_rate>0 không bao giờ cải thiện Sharpe, cost chỉ phát sinh ngày có giao dịch."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import TOTAL_COST_RATE, backtest_with_exits, port_metrics
from src.feature_engineering import cross_sectional_rank
from src.momentum_factors import momentum_6_1, momentum_12_1
from src.strategy import composite_score


def _setup(seed: int = 5):
    rng = np.random.default_rng(seed)
    n_days, n_assets = 300, 15
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"SYM{i}" for i in range(n_assets)]
    log_rets = rng.normal(0.0004, 0.018, size=(n_days, n_assets))
    price_df = pd.DataFrame(100 * np.exp(np.cumsum(log_rets, axis=0)), index=dates, columns=cols)
    volume_df = pd.DataFrame(rng.integers(1e5, 5e6, size=(n_days, n_assets)), index=dates, columns=cols).astype(float)

    mom6, mom12 = momentum_6_1(price_df), momentum_12_1(price_df)
    vol_ratio = volume_df / volume_df.rolling(20).mean()
    returns_df = price_df.pct_change(fill_method=None)
    vol_20, vol_120 = returns_df.rolling(20).std(), returns_df.rolling(120).std()
    vol_spike = vol_20 / vol_120

    ranked = {"m6": cross_sectional_rank(mom6), "m12": cross_sectional_rank(mom12), "vr": cross_sectional_rank(vol_ratio)}
    comp = composite_score(ranked, {"m6": 0.5, "m12": 0.3, "vr": 0.2})
    return price_df, comp, vol_spike, mom6


def test_zero_cost_rate_is_backward_compatible():
    """cost_rate=0.0 (mặc định) phải cho kết quả GIỐNG HỆT bản Ngày 19 (không cost)."""
    price_df, comp, vol_spike, mom6 = _setup()
    port_df, _ = backtest_with_exits(price_df, comp, vol_spike, mom6, max_positions=6, rebal_freq=5, cost_rate=0.0)
    pd.testing.assert_series_equal(port_df["port_return"], port_df["gross_return"], check_names=False)
    assert (port_df["cost"] == 0).all()


def test_positive_cost_rate_never_improves_sharpe():
    price_df, comp, vol_spike, mom6 = _setup()
    port_gross, _ = backtest_with_exits(price_df, comp, vol_spike, mom6, max_positions=6, rebal_freq=5, cost_rate=0.0)
    port_net, _ = backtest_with_exits(price_df, comp, vol_spike, mom6, max_positions=6, rebal_freq=5, cost_rate=TOTAL_COST_RATE)

    ret_gross = port_gross["port_return"]
    ret_net = port_net["port_return"]
    assert (port_net["cost"] > 0).sum() > 0, "Test setup cần ít nhất vài giao dịch phát sinh cost"

    sharpe_gross = port_metrics(ret_gross[ret_gross != 0])["Sharpe"]
    sharpe_net = port_metrics(ret_net[ret_net != 0])["Sharpe"]
    assert sharpe_net <= sharpe_gross


def test_cost_is_zero_on_days_without_trades():
    price_df, comp, vol_spike, mom6 = _setup()
    port_df, trade_df = backtest_with_exits(price_df, comp, vol_spike, mom6, max_positions=6, rebal_freq=5, cost_rate=TOTAL_COST_RATE)

    trade_dates = set(trade_df["date"]) if len(trade_df) else set()
    no_trade_days = port_df[~port_df.index.isin(trade_dates)]
    assert (no_trade_days["cost"] == 0).all()


def test_net_return_equals_gross_minus_cost():
    price_df, comp, vol_spike, mom6 = _setup()
    port_df, _ = backtest_with_exits(price_df, comp, vol_spike, mom6, max_positions=6, rebal_freq=5, cost_rate=TOTAL_COST_RATE)
    np.testing.assert_allclose(
        port_df["port_return"].values, (port_df["gross_return"] - port_df["cost"]).values, atol=1e-12,
    )
