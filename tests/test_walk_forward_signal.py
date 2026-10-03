# -*- coding: utf-8 -*-
"""Kiểm tra walk_forward_signal(): không look-ahead, khác biệt với K-fold OOS."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import fit_ridge, walk_forward_signal


def _make_dataset(n_days: int = 200, n_assets: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, [f"SYM{i}" for i in range(n_assets)]], names=["date", "symbol"])
    n = len(idx)
    f1 = rng.normal(size=n)
    label = 0.3 * f1 + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"f1": f1, "fwd_return": label}, index=idx)


def test_walk_forward_signal_shape_matches_universe():
    dataset = _make_dataset()
    signal = walk_forward_signal(dataset, ["f1"], model_fn=fit_ridge, retrain_every=21, min_train_days=60, alpha=1.0)
    symbols = sorted(dataset.index.get_level_values("symbol").unique())
    assert list(signal.columns) == symbols


def test_walk_forward_signal_early_days_are_nan_before_min_train():
    """Trước min_train_days ngày đầu tiên, chưa đủ dữ liệu train -> phải là NaN,
    KHÔNG được có prediction (nếu có, nghĩa là đã dùng ít dữ liệu hơn mức cho phép)."""
    dataset = _make_dataset(n_days=150)
    signal = walk_forward_signal(dataset, ["f1"], model_fn=fit_ridge, retrain_every=21, min_train_days=60, alpha=1.0)
    early_dates = signal.index[:60]
    assert signal.loc[early_dates].isna().all().all()


def test_walk_forward_signal_only_uses_past_data_stable_under_truncation():
    """Signal tại ngày t phải KHÔNG đổi khi cắt bớt dữ liệu SAU ngày t đủ xa
    (retrain_every) — xác nhận walk-forward chỉ dùng quá khứ, không nhìn tương lai."""
    dataset = _make_dataset(n_days=200, seed=3)
    signal_full = walk_forward_signal(dataset, ["f1"], model_fn=fit_ridge, retrain_every=21, min_train_days=60, alpha=1.0)

    cutoff_date = dataset.index.get_level_values("date").unique()[150]
    dataset_trunc = dataset[dataset.index.get_level_values("date") <= cutoff_date]
    signal_trunc = walk_forward_signal(dataset_trunc, ["f1"], model_fn=fit_ridge, retrain_every=21, min_train_days=60, alpha=1.0)

    # Ngày đủ xa cutoff (tránh biên do retrain_every) phải giống hệt nhau
    stable_dates = signal_trunc.index[100:-25]
    pd.testing.assert_frame_equal(signal_full.loc[stable_dates], signal_trunc.loc[stable_dates])


def test_walk_forward_signal_differs_from_naive_full_sample_fit():
    """Walk-forward (chỉ dùng quá khứ) phải cho kết quả KHÁC với việc fit 1 lần
    trên toàn bộ dataset rồi predict lại chính nó (full-sample fit dùng cả tương
    lai) — nếu giống hệt, nghi ngờ walk-forward đang vô tình look-ahead."""
    dataset = _make_dataset(n_days=200, seed=7)
    signal_wf = walk_forward_signal(dataset, ["f1"], model_fn=fit_ridge, retrain_every=21, min_train_days=60, alpha=1.0)

    full_model = fit_ridge(dataset[["f1"]], dataset["fwd_return"], alpha=1.0)
    full_pred = full_model.predict(dataset[["f1"]])
    full_signal = pd.Series(full_pred, index=dataset.index).unstack("symbol")

    valid_dates = signal_wf.dropna(how="all").index
    common_dates = valid_dates.intersection(full_signal.index)
    assert not signal_wf.loc[common_dates].equals(full_signal.loc[common_dates])


def test_walk_forward_signal_purges_label_horizon():
    """Nhãn fwd_return của ngày d dùng giá đến d+n_fwd. Xáo trộn nhãn của mọi hàng từ ngày P trở đi thì
    signal tại các ngày <= P+n_fwd KHÔNG được đổi (nếu đổi: train đang chứa nhãn chồng lấn với ngày dự đoán).
    Các ngày đủ xa sau đó PHẢI đổi (chứng minh test có độ nhạy)."""
    # Lưới refit (min_train_days=60, retrain_every=21) có điểm i=126. P=124 để điểm đó rơi vào (P, P+n_fwd]:
    # bản cũ (train mọi ngày < 126) sẽ thấy nhãn đã xáo của ngày 124-125 -> signal đổi; bản đã purge thì không.
    n_fwd, P = 5, 124
    dataset = _make_dataset(n_days=200, seed=11)
    dates = dataset.index.get_level_values("date")
    unique_dates = dates.unique()
    kw = dict(model_fn=fit_ridge, retrain_every=21, min_train_days=60, n_fwd=n_fwd, alpha=1.0)

    base = walk_forward_signal(dataset, ["f1"], **kw)

    shuffled = dataset.copy()
    mask = dates >= unique_dates[P]
    shuffled.loc[mask, "fwd_return"] = np.random.default_rng(0).permutation(shuffled.loc[mask, "fwd_return"].to_numpy())
    perturbed = walk_forward_signal(shuffled, ["f1"], **kw)

    early = base.index[(base.index >= unique_dates[60]) & (base.index <= unique_dates[P + n_fwd])]
    late = base.index[base.index >= unique_dates[P + n_fwd + 21 + 1]]
    pd.testing.assert_frame_equal(base.loc[early], perturbed.loc[early])
    assert not base.loc[late].equals(perturbed.loc[late])
