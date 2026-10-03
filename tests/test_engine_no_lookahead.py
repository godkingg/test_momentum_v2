# -*- coding: utf-8 -*-
"""Kiểm tra engine v2 (src/backtest.py) KHÔNG look-ahead cùng ngày, và đối chứng với engine cũ.

Quy ước: lệnh khớp ở close t; vị thế mới mở ở close t chỉ ăn return từ t+1; vị thế bị đóng ở close t
VẪN ăn return của ngày t (kể cả cú giảm kích hoạt stop).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import backtest_with_exits, invested_returns, port_metrics
from src.backtest_legacy import backtest_with_exits_legacy


def _flat_frames(n_days: int = 30, symbols=("A", "B", "C")):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    price = pd.DataFrame(100.0, index=dates, columns=list(symbols))
    ones = pd.DataFrame(1.0, index=dates, columns=list(symbols))
    return dates, price, ones


def test_new_position_does_not_earn_return_of_its_entry_day():
    """X nhảy +10% đúng ngày nó vào top rank. Engine v2: ngày đó chưa có vị thế -> return = 0.
    Engine cũ: vị thế vừa mở vẫn ăn +10% (look-ahead)."""
    dates, price, ones = _flat_frames(symbols=("A", "B", "C", "X"))
    D = 10
    price.loc[dates[D]:, "X"] = 110.0
    score = pd.DataFrame(1.0, index=dates, columns=price.columns)    # hòa nhau -> rank 0.625 < top_entry_rank: chưa ai được vào
    score.loc[dates[D]:, "X"] = 10.0                                  # từ ngày D, X là top (rank 1.0) -> vào lệnh đúng ngày D

    kw = dict(max_positions=1, rebal_freq=1, top_entry_rank=0.9, signal_exit_rank=0.1, stop_loss_pct=0.9, trail_pct=0.9)
    port_new, trades = backtest_with_exits(price, score, ones, ones, **kw)
    port_old, _ = backtest_with_exits_legacy(price, score, ones, ones, **kw)

    entry = trades[(trades["action"] == "BUY") & (trades["symbol"] == "X")]
    assert len(entry) == 1 and entry["date"].iloc[0] == dates[D]
    assert port_new.loc[dates[D], "gross_return"] == 0.0
    assert port_new["gross_return"].abs().max() < 1e-12, "X đứng yên sau ngày D -> không còn return nào"
    assert abs(port_old.loc[dates[D], "gross_return"] - 0.10) < 1e-9, "Đối chứng: engine cũ ăn +10% ngay ngày mua"


def test_stop_loss_day_loss_is_realized_in_portfolio_return():
    """Giữ A, A rơi -20% ngày S và bị HARD_STOP. Engine v2: -20% VÀO PnL ngày S. Engine cũ: bị né (=0)."""
    dates, price, ones = _flat_frames()
    S = 12
    price.loc[dates[S]:, "A"] = 80.0
    score = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=dates)
    score.loc[dates[S]:, "A"] = 0.0          # A tụt rank từ ngày S -> sau stop KHÔNG tái vào lại cùng ngày (C lên top, đứng yên)
    kw = dict(max_positions=1, rebal_freq=1, top_entry_rank=0.9, signal_exit_rank=0.1, stop_loss_pct=0.08, trail_pct=0.5)

    port_new, trades = backtest_with_exits(price, score, ones, ones, **kw)
    port_old, _ = backtest_with_exits_legacy(price, score, ones, ones, **kw)

    stop = trades[trades["action"] == "HARD_STOP"]
    assert len(stop) == 1 and stop["date"].iloc[0] == dates[S]
    assert abs(port_new.loc[dates[S], "gross_return"] - (-0.20)) < 1e-12
    assert abs(port_old.loc[dates[S], "gross_return"]) < 1e-12, "Đối chứng: engine cũ né mất cú giảm kích hoạt stop"


def test_cost_is_charged_on_trade_day_not_on_return_day():
    dates, price, ones = _flat_frames()
    score = pd.DataFrame({"A": 3.0, "B": 2.0, "C": 1.0}, index=dates)
    port, trades = backtest_with_exits(price, score, ones, ones, max_positions=1, rebal_freq=1,
                                       top_entry_rank=0.9, cost_rate=0.0023)
    buy_day = trades[trades["action"] == "BUY"]["date"].iloc[0]
    assert abs(port.loc[buy_day, "cost"] - 0.0023) < 1e-12          # mua 1 vị thế = 100% notional × 0.23%
    assert port.loc[buy_day, "n_positions_start"] == 0 and port.loc[buy_day, "n_positions"] == 1


def test_invested_returns_cuts_warmup_but_keeps_zero_days():
    idx = pd.bdate_range("2024-01-01", periods=6)
    port = pd.DataFrame({"port_return": [0, 0, 0.01, 0.0, -0.02, 0.0], "n_positions_start": [0, 0, 1, 1, 1, 1]}, index=idx)
    out = invested_returns(port)
    assert list(out.index) == list(idx[2:]) and (out == 0).sum() == 2


def _random_world(seed: int, n: int = 400, m: int = 15):
    """Return SỐ HỌC kỳ vọng 0 (KHÔNG dùng exp(cumsum): drift số học +σ²/2 làm mọi long-only có Sharpe ≈ +0.3).
    vol_spike tính từ chính dữ liệu (gồm return hôm nay) như pipeline thật; tín hiệu hoàn toàn ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    cols = [f"S{i}" for i in range(m)]
    r = rng.standard_t(4, size=(n, m)) * 0.018 / np.sqrt(2)
    price = pd.DataFrame(100 * np.cumprod(1 + r, 0), idx, cols)
    score = pd.DataFrame(rng.normal(size=(n, m)), idx, cols)
    ret = price.pct_change(fill_method=None)
    vs = (ret.rolling(20).std() / ret.rolling(120).std()).bfill().fillna(1.0)
    m6 = (price.shift(21) / price.shift(126) - 1).fillna(0.1)
    return price, score, vs, m6


def test_random_signal_has_no_alpha_under_v2_but_legacy_is_biased_with_tight_stops():
    """Tín hiệu ngẫu nhiên + giá không có alpha => kỳ vọng Sharpe ≈ 0. Stop chặt kích hoạt thường xuyên nên
    thiên lệch của engine cũ lộ rõ (đo ≈ +1.9 trên 60 seed); engine v2 phải ≈ 0."""
    kw = dict(stop_loss_pct=0.03, trail_pct=0.05)
    new, old = [], []
    for seed in range(12):
        price, score, vs, m6 = _random_world(seed)
        pn, _ = backtest_with_exits(price, score, vs, m6, **kw)
        po, _ = backtest_with_exits_legacy(price, score, vs, m6, **kw)
        new.append(port_metrics(invested_returns(pn))["Sharpe"])
        old.append(port_metrics(po["port_return"].iloc[130:])["Sharpe"])
    new, old = np.array(new), np.array(old)
    assert abs(new.mean()) < 0.5, f"Engine v2 không được có Sharpe dương hệ thống trên tín hiệu ngẫu nhiên: {new.mean():.2f}"
    assert (old - new).mean() > 0.8 and (old > new).mean() >= 0.9, "Đối chứng: engine cũ phải thiên lệch dương rõ rệt"


def test_alignment_future_signal_wins_same_day_signal_does_not():
    """Signal = return NGÀY MAI (nhìn trước có chủ đích) phải cho Sharpe cực cao; signal = return HÔM NAY (đã biết ở
    close t, hợp lệ) thì không. Nếu engine lệch ngày, cả hai sẽ cùng cao."""
    price, _, vs, m6 = _random_world(1, n=400, m=15)
    ret = price.pct_change(fill_method=None)
    kw = dict(max_positions=5, rebal_freq=1, top_entry_rank=0.8, signal_exit_rank=0.5, stop_loss_pct=9, trail_pct=9)
    sh = {}
    for name, sig in (("future", ret.shift(-1)), ("same_day", ret)):
        p, _ = backtest_with_exits(price, sig.fillna(0.0), vs * 0 + 1, m6 * 0 + 1, **kw)
        sh[name] = port_metrics(invested_returns(p))["Sharpe"]
    assert sh["future"] > 10 and sh["same_day"] < sh["future"] / 10
