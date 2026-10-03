# -*- coding: utf-8 -*-
"""risk.py — Metrics phụ trợ: benchmark, so sánh chiến lược."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import calmar_ratio, port_metrics, sharpe_se


def equal_weight_benchmark(price_df: pd.DataFrame) -> pd.Series:
    """1/N (equal-weight) benchmark daily return."""
    returns_df = price_df.pct_change(fill_method=None)
    return returns_df.mean(axis=1).dropna()


def compare_strategies(strategies: dict[str, pd.Series]) -> pd.DataFrame:
    """So sánh metrics đầy đủ (Ann Return, Vol, Sharpe, Max DD, Calmar) giữa
    nhiều chiến lược/benchmark, trên cùng khoảng thời gian overlap."""
    common_idx = None
    for ret in strategies.values():
        common_idx = ret.index if common_idx is None else common_idx.intersection(ret.index)

    rows = {}
    for name, ret in strategies.items():
        r = ret.loc[common_idx].dropna()
        m = port_metrics(r)
        m["Calmar"] = calmar_ratio(m)
        m["Sharpe SE"] = sharpe_se(r)      # chênh Sharpe < ~2 SE ⇒ chưa phân biệt được với nhiễu
        rows[name] = m
    return pd.DataFrame(rows).T


def excess_return_tstat(strategy: pd.Series, benchmark: pd.Series) -> float:
    """t-stat của return vượt trội hằng ngày so với benchmark (cùng khoảng ngày): mean/std·√T.
    Kiểm định ghép cặp — mạnh hơn nhiều so với so 2 Sharpe riêng lẻ vì 2 chuỗi cùng chịu 1 thị trường."""
    idx = strategy.index.intersection(benchmark.index)
    d = (strategy.loc[idx] - benchmark.loc[idx]).dropna()
    return float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else float("nan")
