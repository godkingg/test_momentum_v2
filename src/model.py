# -*- coding: utf-8 -*-
"""model.py — Baseline model (Linear/Ridge) → XGBoost → SHAP, đánh giá bằng IC.

Pipeline: đơn giản trước (dễ diagnose), phức tạp sau. Ridge Coefficients và
XGBoost feature_importances_ được so sánh với SHAP values để xác nhận
feature nào THỰC SỰ drive prediction (feature importance thông thường có
thể gây hiểu lầm khi các feature tương quan với nhau).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

from src.ic_analysis import cross_sectional_ic, get_date_indexed_splits


def fit_linear_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Linear Regression thuần — baseline đơn giản nhất, dễ diagnose overfitting
    (so Ridge coefficients để xem regularization thay đổi gì)."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fit_ridge(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """Ridge — baseline chính, ưu tiên vì đơn giản/dễ diagnose/ít overfit hơn
    XGBoost trên universe nhỏ (28 mã)."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def fit_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series,
    n_estimators: int = 200, max_depth: int = 3, learning_rate: float = 0.05,
):
    """XGBoost — capture non-linearity + interaction giữa factor, thử SAU
    khi đã có baseline Ridge để so sánh cải thiện có ĐÁNG kể hay không."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def purged_kfold_oos_predictions(
    dataset: pd.DataFrame, feature_cols: list[str], label_col: str = "fwd_return",
    model_fn=fit_ridge, n_splits: int = 5, embargo: int = 5, **model_kwargs,
) -> pd.DataFrame:
    """Chạy Purged K-Fold CV, thu thập OOS predictions (date, symbol, y_true, y_pred)
    cho 1 model_fn bất kỳ (fit_linear_baseline / fit_ridge / fit_xgboost).

    Trả về DataFrame long-format các prediction OOS — dùng để tính IC hoặc
    backtest, KHÔNG dùng in-sample fit để tránh look-ahead.
    """
    X_all, y_all = dataset[feature_cols], dataset[label_col]
    dates_arr = dataset.index.get_level_values("date").values.astype("datetime64[ns]")
    syms_arr = dataset.index.get_level_values("symbol").values

    records = []
    for train_idx, test_idx in get_date_indexed_splits(dataset, n_splits, embargo):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        model = model_fn(X_all.iloc[train_idx], y_all.iloc[train_idx], **model_kwargs)
        pred = model.predict(X_all.iloc[test_idx])
        for dt, sym, yt, yp in zip(dates_arr[test_idx], syms_arr[test_idx], y_all.iloc[test_idx].values, pred):
            records.append({"date": dt, "symbol": sym, "y_true": yt, "y_pred": yp})

    return pd.DataFrame(records)


def walk_forward_signal(
    dataset: pd.DataFrame, feature_cols: list[str], label_col: str = "fwd_return",
    model_fn=fit_ridge, retrain_every: int = 21, min_train_days: int = 120,
    n_fwd: int = 5, **model_kwargs,
) -> pd.DataFrame:
    """Sinh signal wide-format [date x symbol] bằng walk-forward THẬT — CHỈ train
    trên dữ liệu QUÁ KHỨ trước mỗi ngày dự đoán, KHÔNG dùng K-fold.

    ⚠️ KHÁC BIỆT QUAN TRỌNG với purged_kfold_oos_predictions(): K-fold OOS (dùng
    để đo IC ở Ngày 19) có thể dùng dữ liệu TƯƠNG LAI để train fold đầu tiên
    (train_idx bao gồm các fold SAU test fold đó) — hợp lệ để đo "IC trung bình"
    nhưng KHÔNG hợp lệ để tạo 1 signal duy nhất dùng backtest theo trình tự thời
    gian (sẽ vô tình cho model "nhìn thấy tương lai" ở giai đoạn đầu backtest).
    walk_forward_signal() sửa vấn đề này: refit mỗi `retrain_every` ngày, mỗi lần
    chỉ dùng dữ liệu CÓ TRƯỚC ngày dự đoán.

    PURGE NHÃN (n_fwd): nhãn fwd_return của ngày d dùng giá đến d+n_fwd. Vì vậy ở ngày dự đoán
    thứ i, model CHỈ được train bằng các ngày d <= unique_dates[i - n_fwd - 1]. Bản cũ dùng mọi
    ngày d < ngày dự đoán → nhãn của n_fwd ngày cuối chứa giá TƯƠNG LAI (nằm sau ngày dự đoán).
    Đặt n_fwd=0 chỉ để tái tạo hành vi cũ khi so sánh; mặc định 5 = N_FWD của project.

    Trả về DataFrame [date x symbol] — pred_score, cùng shape như comp_score
    để dùng trực tiếp trong backtest_with_exits().
    """
    unique_dates = sorted(dataset.index.get_level_values("date").unique())
    symbols = sorted(dataset.index.get_level_values("symbol").unique())
    signal = pd.DataFrame(np.nan, index=pd.DatetimeIndex(unique_dates), columns=symbols)

    model = None
    for i, current_date in enumerate(unique_dates):
        if i < min_train_days:
            continue  # chưa đủ dữ liệu quá khứ để train lần đầu

        if model is None or i % retrain_every == 0:
            if i - n_fwd - 1 < 0:
                continue
            last_train_date = unique_dates[i - n_fwd - 1]       # purge: nhãn kết thúc trước ngày dự đoán
            train_mask = dataset.index.get_level_values("date") <= last_train_date
            train_data = dataset[train_mask]
            if len(train_data) < 50:
                continue
            model = model_fn(train_data[feature_cols], train_data[label_col], **model_kwargs)

        today_mask = dataset.index.get_level_values("date") == current_date
        today_data = dataset[today_mask]
        if len(today_data) == 0:
            continue
        pred = model.predict(today_data[feature_cols])
        today_symbols = today_data.index.get_level_values("symbol")
        signal.loc[current_date, today_symbols] = pred

    return signal


def evaluate_oos_predictions(oos_df: pd.DataFrame) -> dict[str, float]:
    """Đánh giá OOS predictions bằng IC (KHÔNG chỉ RMSE) — IC đo khả năng
    XẾP HẠNG đúng thứ tự mã tốt/xấu, quan trọng hơn RMSE cho long-only
    quintile strategy (không cần dự đoán đúng GIÁ TRỊ return, chỉ cần đúng
    THỨ TỰ tương đối)."""
    from scipy.stats import spearmanr

    daily_ic = oos_df.groupby("date").apply(
        lambda g: spearmanr(g["y_true"], g["y_pred"])[0] if len(g) >= 5 else np.nan,
        include_groups=False,
    ).dropna()

    rmse = np.sqrt(mean_squared_error(oos_df["y_true"], oos_df["y_pred"]))
    return {
        "Mean IC": daily_ic.mean(),
        "Std IC": daily_ic.std(),
        "IC-IR": daily_ic.mean() / daily_ic.std() if daily_ic.std() else np.nan,
        "IC>0 (%)": (daily_ic > 0).mean(),
        "RMSE": rmse,
        "N days": len(daily_ic),
    }


def compute_shap_values(xgb_model, X: pd.DataFrame):
    """Chạy shap.TreeExplainer trên XGBoost đã fit. Trả về (explainer, shap_values).

    So sánh SHAP importance (mean |shap value|) với xgb_model.feature_importances_
    thông thường: feature importance chuẩn của XGBoost đo "số lần feature
    được dùng để split" hoặc "gain trung bình", có thể đánh giá SAI khi các
    feature tương quan cao (một feature "che" đóng góp của feature khác).
    SHAP phân bổ đóng góp công bằng hơn theo lý thuyết game (Shapley values).
    """
    import shap

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def shap_importance_ranking(shap_values: np.ndarray, feature_cols: list[str]) -> pd.Series:
    """Mean |SHAP value| mỗi feature — thước đo importance theo SHAP,
    để so sánh trực tiếp với feature_importances_ chuẩn của XGBoost."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    return pd.Series(mean_abs_shap, index=feature_cols, name="mean_abs_shap").sort_values(ascending=False)


def compare_importance_methods(
    xgb_model, feature_cols: list[str], shap_values: np.ndarray,
) -> pd.DataFrame:
    """Bảng so sánh: XGBoost feature_importances_ (gain-based) vs SHAP importance.
    Rank khác nhau đáng kể là dấu hiệu có multicollinearity giữa features."""
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_cols, name="xgb_importance")
    shap_imp = shap_importance_ranking(shap_values, feature_cols)

    df = pd.DataFrame({"xgb_importance": xgb_imp, "shap_importance": shap_imp})
    df["xgb_rank"] = df["xgb_importance"].rank(ascending=False).astype(int)
    df["shap_rank"] = df["shap_importance"].rank(ascending=False).astype(int)
    df["rank_diff"] = (df["xgb_rank"] - df["shap_rank"]).abs()
    return df.sort_values("shap_importance", ascending=False)
