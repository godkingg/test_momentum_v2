# -*- coding: utf-8 -*-
"""Smoke test: chạy launcher.main() end-to-end với dữ liệu GIẢ LẬP (không cần mạng/vnstock) —
đảm bảo pipeline engine v2 + ghi JSON/embed_data.js cho dashboard hoạt động."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import launcher


def _fake_ohlcv(symbols, start, end, fields=("close", "open", "volume"), **_):
    rng = np.random.default_rng(0)
    n, m = 900, len(symbols)
    idx = pd.bdate_range("2022-01-03", periods=n)
    drift = np.zeros((n, m))
    for t in range(1, n):                                    # drift tự tương quan -> momentum có tín hiệu thật
        drift[t] = 0.98 * drift[t - 1] + rng.normal(0, 0.0004, m)
    price = pd.DataFrame(100 * np.cumprod(1 + drift + rng.normal(0, 0.015, (n, m)), 0), idx, symbols)
    vol = pd.DataFrame(rng.lognormal(13, 0.4, (n, m)), idx, symbols)
    return {"close": price, "open": price.shift(1).fillna(price), "volume": vol}


def test_launcher_end_to_end_with_engine_v2_and_legacy_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "load_ohlcv", _fake_ohlcv)
    monkeypatch.setattr(launcher, "DASHBOARD_DIR", tmp_path)
    monkeypatch.setattr(launcher, "DATA_DIR", tmp_path / "data")
    launcher.main(["--universe", *[f"S{i:02d}" for i in range(20)], "--no-open", "--compare-legacy"])

    bt = json.loads((tmp_path / "data" / "backtest.json").read_text(encoding="utf-8"))
    assert bt["engine"]["version"] == "v2"
    cmp_ = bt["engine"]["comparison_vs_legacy"]
    assert cmp_ and {"sharpe_v2", "sharpe_legacy", "legacy_minus_v2"} <= cmp_.keys()
    vs = bt["vs_benchmark"]
    assert "Momentum+ExitRules (net cost)" in vs and "1/N Equal-Weight" in vs
    assert vs["Momentum+ExitRules (net cost)"]["Sharpe"] is not None
    assert len(bt["cumulative"]["dates"]) == len(bt["cumulative"]["strategy_net"]) > 100
    assert (tmp_path / "embed_data.js").exists() and (tmp_path / "data" / "today.json").exists()
