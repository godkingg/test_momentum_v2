# -*- coding: utf-8 -*-
"""Momentum factors — Ngày 18.

Tính 2 biến thể momentum "bỏ tháng gần nhất" (tránh short-term reversal),
so sánh sự khác biệt, và phân tích quintile để kiểm tra pattern
"winner tiếp tục thắng, loser tiếp tục thua" trên VN30-universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(price_df: pd.DataFrame) -> pd.DataFrame:
    """Momentum 12 tháng, bỏ 1 tháng gần nhất: giá(t-21)/giá(t-252) - 1.

    Đây là công thức momentum kinh điển (Jegadeesh-Titman). Lookback dài
    (252 phiên ~ 12 tháng) bắt xu hướng trung-dài hạn, bỏ 21 phiên gần
    nhất (~1 tháng) để loại short-term reversal (giá thường đảo chiều
    ngắn hạn ngay sau khi tăng/giảm mạnh).
    """
    return price_df.shift(21) / price_df.shift(252) - 1


def momentum_6_1(price_df: pd.DataFrame) -> pd.DataFrame:
    """Momentum 6 tháng, bỏ 1 tháng gần nhất: giá(t-21)/giá(t-126) - 1.

    Lookback ngắn hơn (126 phiên ~ 6 tháng) — bắt xu hướng gần hơn,
    phản ứng nhanh hơn với thay đổi regime so với Mom_12_1, nhưng cũng
    nhiễu hơn (ít observation hơn để "làm mượt" xu hướng).
    """
    return price_df.shift(21) / price_df.shift(126) - 1


def compare_momentum_variants(price_df: pd.DataFrame) -> pd.DataFrame:
    """So sánh Mom_6_1 và Mom_12_1: correlation, rank correlation, và
    % ngày 2 factor "đồng thuận" về dấu (cùng dương hoặc cùng âm).

    Khác biệt về CÁCH TÍNH: cùng công thức tổng quát price(t-21)/price(t-N)-1,
    chỉ khác N (126 vs 252 phiên) → Mom_6_1 nhạy hơn với biến động gần đây,
    Mom_12_1 ổn định hơn nhưng phản ứng chậm hơn khi xu hướng đổi chiều.
    """
    mom6, mom12 = momentum_6_1(price_df), momentum_12_1(price_df)
    common_mask = mom6.notna() & mom12.notna()

    corrs, rank_corrs, agree_pct = [], [], []
    for date in price_df.index:
        m6, m12 = mom6.loc[date][common_mask.loc[date]], mom12.loc[date][common_mask.loc[date]]
        if len(m6) < 5:
            continue
        corrs.append(m6.corr(m12))
        rank_corrs.append(m6.corr(m12, method="spearman"))
        agree_pct.append((np.sign(m6) == np.sign(m12)).mean())

    return pd.DataFrame({
        "metric": ["Pearson corr (mean)", "Spearman corr (mean)", "% ngày đồng thuận dấu (mean)"],
        "value": [np.mean(corrs), np.mean(rank_corrs), np.mean(agree_pct)],
    }).set_index("metric")


def forward_return(price_df: pd.DataFrame, n_fwd: int = 5) -> pd.DataFrame:
    """Forward return N ngày, dùng làm label để đánh giá factor / quintile."""
    return price_df.shift(-n_fwd) / price_df - 1


def assign_quintile(factor_slice: pd.Series, n_quintiles: int = 5) -> pd.Series:
    """Xếp hạng 1 lát cắt cross-sectional (1 ngày) thành quintile. Q1=thấp nhất, Q5=cao nhất."""
    try:
        return pd.qcut(factor_slice, n_quintiles, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=factor_slice.index)


def quintile_assignments(factor_df: pd.DataFrame, n_quintiles: int = 5) -> pd.DataFrame:
    """Gán quintile cho mọi mã, mọi ngày (cross-sectional theo từng ngày)."""
    return factor_df.apply(lambda row: assign_quintile(row.dropna(), n_quintiles), axis=1) \
        .reindex(columns=factor_df.columns)


def quintile_forward_returns(
    factor_df: pd.DataFrame, fwd_ret_df: pd.DataFrame, n_quintiles: int = 5,
) -> pd.DataFrame:
    """Return trung bình forward theo từng quintile — bảng chuẩn để kiểm tra
    pattern "winner tiếp tục thắng" (Q5 return > Q1 return, đơn điệu tăng
    Q1→Q5) hay không.

    Trả về DataFrame index=quintile [1..5], cột: mean_fwd_return, n_obs.
    """
    quintile_df = quintile_assignments(factor_df, n_quintiles)
    stacked_q = quintile_df.stack(future_stack=True).rename("quintile")
    stacked_r = fwd_ret_df.stack(future_stack=True).rename("fwd_return")
    merged = pd.concat([stacked_q, stacked_r], axis=1).dropna()

    result = merged.groupby("quintile")["fwd_return"].agg(["mean", "std", "count"])
    result.columns = ["mean_fwd_return", "std_fwd_return", "n_obs"]
    result.index = result.index.astype(int)
    return result.sort_index()


def is_monotonic_winner_pattern(quintile_returns: pd.DataFrame) -> dict:
    """Kiểm tra pattern 'winner tiếp tục thắng, loser tiếp tục thua':
    return trung bình phải tăng ĐƠN ĐIỆU từ Q1 → Q5, và Q5 - Q1 (spread) > 0.

    Thị trường VN nhỏ, thanh khoản thấp — pattern có thể YẾU hơn hoặc
    KHÔNG đơn điệu so với thị trường Mỹ (nơi momentum là anomaly kinh điển).
    """
    means = quintile_returns["mean_fwd_return"]
    is_monotonic = means.is_monotonic_increasing
    q5_q1_spread = means.iloc[-1] - means.iloc[0]
    return {
        "monotonic_Q1_to_Q5": bool(is_monotonic),
        "Q5_minus_Q1_spread": float(q5_q1_spread),
        "spread_direction": "winner-thắng-tiếp (như kỳ vọng)" if q5_q1_spread > 0
                             else "KHÔNG như kỳ vọng (Q1 > Q5)",
    }
