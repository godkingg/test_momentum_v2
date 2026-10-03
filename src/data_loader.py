# -*- coding: utf-8 -*-
"""Load dữ liệu giá + volume cho universe VN30, có rate-limit handling.

Nguồn: vnstock (source="KBS"). Universe và logic lọc coverage tái sử dụng
từ notebook Ngày 19 (day19_test.ipynb) — giữ nguyên danh sách 30 mã gốc.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

# Universe gốc — 30 mã, TCX/VPL bị loại tự động do coverage<95% (đúng như Ngày 19)
VN30_UNIVERSE = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MCH", "MSN", "MWG", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TCX", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

RATE_LIMIT_BATCH = 18
RATE_LIMIT_SLEEP = 65
MIN_COVERAGE = 0.95


def load_ohlcv(
    symbols: list[str],
    start: str,
    end: str,
    fields: tuple[str, ...] = ("close", "open", "volume"),
) -> dict[str, pd.DataFrame]:
    """Tải giá + volume qua vnstock (source KBS). Trả về dict {field: DataFrame[date x symbol]}."""
    from vnstock import Quote

    raw: dict[str, pd.DataFrame] = {}
    for i, sym in enumerate(symbols, 1):
        if i > 1 and (i - 1) % RATE_LIMIT_BATCH == 0:
            print(f"⏳ Nghỉ {RATE_LIMIT_SLEEP}s để tránh rate limit...")
            time.sleep(RATE_LIMIT_SLEEP)
        print(f"[{i}/{len(symbols)}] Loading {sym}...")
        try:
            df = Quote(symbol=sym, source="KBS").history(start=start, end=end, interval="1D")
            df.columns = [c.lower() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            raw[sym] = df.set_index("time")[list(fields)]
        except Exception as e:
            print(f"⚠️ {sym}: {type(e).__name__} - {e}")
            continue

    print(f"\n✅ Đã tải: {len(raw)}/{len(symbols)} mã")
    return {
        field: pd.DataFrame({sym: d[field] for sym, d in raw.items()}).sort_index()
        for field in fields
    }


def filter_by_coverage(
    price_df: pd.DataFrame, min_coverage: float = MIN_COVERAGE
) -> tuple[pd.DataFrame, list[str]]:
    """Loại mã có coverage < ngưỡng (mới niêm yết / thiếu dữ liệu). Logic từ Ngày 19.

    Trả về (price_df đã lọc, danh sách mã bị loại).
    """
    n_total = len(price_df)
    coverage = price_df.notna().sum() / n_total
    dropped = coverage[coverage < min_coverage].index.tolist()
    kept = [s for s in price_df.columns if s not in dropped]
    return price_df[kept], dropped


def train_test_split_by_date(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chia in-sample / out-of-sample theo mốc thời gian."""
    return df[df.index <= cutoff], df[df.index > cutoff]
