#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launcher: tải dữ liệu THẬT (vnstock) → momentum + volume factors → composite
IC-weighted score → PositionManager (entry/exit) → backtest_with_exits (CÓ
transaction cost + slippage mặc định) → so sánh 1/N benchmark → ghi JSON cho
dashboard (hành động hôm nay + backtest) → tự mở trình duyệt.

Cách dùng:
    python launcher.py                      # mặc định: có cost (0.23%/lệnh), tải đến hôm nay
    python launcher.py --no-cost             # backtest KHÔNG cost (tái tạo baseline Ngày 19)
    python launcher.py --end 2026-08-24      # cập nhật đến ngày cụ thể
    python launcher.py --no-open             # chỉ tính toán, không tự mở trình duyệt

ENGINE v2 (độ trễ 1 ngày): signal chốt ở close t, vị thế ăn return từ t+1 (xem src/backtest.py).
    python launcher.py --compare-legacy     # chạy thêm engine CŨ (có look-ahead cùng ngày) để đo mức chênh

⚠️ Dashboard bootstrap (scripts_bootstrap_old_numbers.py) hiển thị số của ENGINE CŨ — chỉ để minh họa.
Chạy launcher.py để thay bằng số của engine v2 trên dữ liệu mới nhất.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
import webbrowser
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import TOTAL_COST_RATE, backtest_with_exits, calmar_ratio, invested_returns, port_metrics, sharpe_se, trade_statistics
from src.backtest_legacy import backtest_with_exits_legacy
from src.data_loader import VN30_UNIVERSE, filter_by_coverage, load_ohlcv, train_test_split_by_date
from src.feature_engineering import assemble_dataset, cross_sectional_rank
from src.ic_analysis import single_factor_ic
from src.momentum_factors import momentum_12_1, momentum_6_1
from src.risk import compare_strategies, equal_weight_benchmark, excess_return_tstat
from src.pipeline import build_factors  # noqa: F401  (giữ launcher.build_factors cho notebook 06)
from src.strategy import PositionManager, classify_signal, composite_score, normalize_ic_weights

DASHBOARD_DIR = ROOT / "dashboard"
DATA_DIR = DASHBOARD_DIR / "data"

N_FWD = 5
IS_CUTOFF = "2025-12-31"
VOL_SPIKE_THRESHOLD = 2.0
TOP_PCT, BOTTOM_PCT = 0.35, 0.35


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--universe", nargs="+", default=VN30_UNIVERSE)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--no-cost", action="store_true", help="Backtest KHÔNG transaction cost (baseline Ngày 19)")
    p.add_argument("--compare-legacy", action="store_true",
                   help="Chạy thêm engine CŨ (look-ahead cùng ngày) trên cùng dữ liệu/tham số để đo mức chênh Sharpe")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"{'='*70}\n1. TẢI DỮ LIỆU — universe {len(args.universe)} mã\n{'='*70}")
    ohlcv = load_ohlcv(args.universe, args.start, args.end, fields=("close", "open", "volume"))
    price_df = ohlcv["close"].dropna(how="all")
    volume_df = ohlcv["volume"].reindex(columns=price_df.columns)
    if price_df.empty:
        sys.exit("❌ Không tải được dữ liệu — kiểm tra mạng hoặc kết nối vnstock.")

    print(f"\n{'='*70}\n2. LỌC UNIVERSE THEO COVERAGE (ngưỡng 95%)\n{'='*70}")
    price_df, dropped = filter_by_coverage(price_df)
    volume_df = volume_df[price_df.columns]
    print(f"Universe sau lọc: {len(price_df.columns)}/{len(args.universe)} mã"
          f"{f' (loại: {dropped})' if dropped else ''}")

    print(f"\n{'='*70}\n3. BUILD FACTORS (Ngày 18)\n{'='*70}")
    raw_factors, vol_20, vol_120 = build_factors(price_df, volume_df)
    vol_spike = vol_20 / vol_120
    feature_cols = list(raw_factors.keys())

    print(f"\n{'='*70}\n4. IC ANALYSIS (Purged K-Fold OOS) — Ngày 18\n{'='*70}")
    dataset = assemble_dataset(raw_factors, price_df, n_fwd=N_FWD, label_type="regression")
    if len(dataset) < 200:
        sys.exit("❌ Dataset quá nhỏ để đo IC đáng tin cậy — kiểm tra lại khoảng thời gian.")
    sf_ic = single_factor_ic(dataset, feature_cols, n_splits=5, embargo=5)
    print(sf_ic.round(4).to_string())
    ic_weights = normalize_ic_weights(sf_ic)

    cost_rate = 0.0 if args.no_cost else TOTAL_COST_RATE
    print(f"\n{'='*70}\n5. COMPOSITE SCORE (IC-Weighted) → BACKTEST VỚI EXIT RULES "
          f"(cost_rate={cost_rate:.2%})\n{'='*70}")
    ranked_factors = {name: cross_sectional_rank(f) for name, f in raw_factors.items()}
    comp_score = composite_score(ranked_factors, ic_weights)

    engine_kwargs = dict(
        stop_loss_pct=0.08, trail_pct=0.12, vol_exit_thr=3.0, vol_reduce_thr=2.0,
        top_entry_rank=0.65, signal_exit_rank=0.40, max_positions=10, rebal_freq=5, cost_rate=cost_rate,
    )
    port_df, trade_df = backtest_with_exits(price_df, comp_score, vol_spike, raw_factors["Momentum_6_1"], **engine_kwargs)
    port_ret = invested_returns(port_df)                       # cắt warm-up theo vị thế thật, KHÔNG lọc theo giá trị
    ew_ret = equal_weight_benchmark(price_df)

    strategies_dict = {"Momentum+ExitRules (net)": port_ret, "1/N Equal-Weight": ew_ret}
    if cost_rate > 0:
        strategies_dict["Momentum+ExitRules (gross, không cost)"] = invested_returns(port_df, "gross_return")
    strategies_cmp = compare_strategies(strategies_dict)
    print(strategies_cmp.round(4).to_string())
    t_ex = excess_return_tstat(port_ret, ew_ret)
    print(f"\nt-stat return vượt trội so với 1/N: {t_ex:+.2f}  (|t| < 2 ⇒ chưa khác 1/N có ý nghĩa thống kê)")
    if cost_rate > 0:
        print(f"Tổng cost đã trả: {port_df['cost'].sum():.2%}")

    engine_comparison = None
    if args.compare_legacy:
        port_old, _ = backtest_with_exits_legacy(price_df, comp_score, vol_spike, raw_factors["Momentum_6_1"], **engine_kwargs)
        r_old = port_old["port_return"].loc[port_ret.index.min():]
        s_new, s_old = port_metrics(port_ret)["Sharpe"], port_metrics(r_old)["Sharpe"]
        engine_comparison = {"sharpe_v2": round(float(s_new), 4), "sharpe_legacy": round(float(s_old), 4),
                             "legacy_minus_v2": round(float(s_old - s_new), 4), "sharpe_se_v2": round(float(sharpe_se(port_ret)), 4)}
        print(f"\n[Engine cũ vs v2] Sharpe legacy {s_old:.3f} | v2 {s_new:.3f} | chênh {s_old - s_new:+.3f} "
              f"(SE ≈ {sharpe_se(port_ret):.2f})")
    trade_stats = trade_statistics(trade_df)
    print(f"\nTrades: {trade_stats.get('n_buys', 0)} buy / {trade_stats.get('n_sells', 0)} sell | "
          f"Win rate: {trade_stats.get('win_rate', float('nan')):.1%}")

    print(f"\n{'='*70}\n6. DASHBOARD HÔM NAY — Signal + Action\n{'='*70}")
    latest = price_df.index[-1]
    snap = build_today_snapshot(price_df, volume_df, raw_factors, vol_spike, comp_score, ic_weights)
    print(f"BUY: {(snap['signal']=='BUY').sum()} | AVOID: {(snap['signal']=='AVOID').sum()} | "
          f"REDUCE: {(snap['signal']=='REDUCE').sum()} | EXIT: {(snap['signal']=='EXIT').sum()}")

    print(f"\n{'='*70}\n7. GHI JSON CHO DASHBOARD\n{'='*70}")
    write_dashboard_json(
        price_df=price_df, snap=snap, sf_ic=sf_ic, ic_weights=ic_weights,
        port_df=port_df, ew_ret=ew_ret, strategies_cmp=strategies_cmp,
        trade_df=trade_df, trade_stats=trade_stats, cost_rate=cost_rate,
        start=args.start, end=args.end, dropped_symbols=dropped, latest=latest,
        engine_comparison=engine_comparison, excess_t=t_ex,
    )
    print("Đã ghi dashboard/data/*.json")

    print(f"\n{'='*70}\n✅ HOÀN TẤT\n{'='*70}")
    index_path = (DASHBOARD_DIR / "index.html").resolve()
    if not args.no_open:
        print(f"🌐 Đang mở: {index_path}")
        webbrowser.open(f"file://{index_path}")
    else:
        print(f"Mở thủ công: {index_path}")


def build_today_snapshot(price_df, volume_df, raw_factors, vol_spike, comp_score, ic_weights) -> pd.DataFrame:
    """Snapshot ngày cuối: giá, momentum, vol_spike, composite rank, signal."""
    latest = price_df.index[-1]
    returns_df = price_df.pct_change(fill_method=None)

    snap = pd.DataFrame({
        "price": price_df.iloc[-1],
        "ret_1d": returns_df.iloc[-1],
        "mom_6": raw_factors["Momentum_6_1"].iloc[-1],
        "mom_12": raw_factors["Momentum_12_1"].iloc[-1],
        "vol_ratio": raw_factors["Volume_Ratio"].iloc[-1],
        "vol_spike": vol_spike.iloc[-1],
        "composite": comp_score.iloc[-1],
    }).dropna(subset=["mom_6", "mom_12"])

    snap["composite_rank"] = snap["composite"].rank(pct=True)
    snap["signal"] = classify_signal(
        snap["composite_rank"], snap["vol_spike"], snap["mom_6"],
        top_pct=TOP_PCT, bottom_pct=BOTTOM_PCT, vol_threshold=VOL_SPIKE_THRESHOLD,
    )
    return snap.sort_values("composite_rank", ascending=False)


def write_dashboard_json(
    *, price_df, snap, sf_ic, ic_weights, port_df, ew_ret, strategies_cmp,
    trade_df, trade_stats, cost_rate, start, end, dropped_symbols, latest,
    engine_comparison=None, excess_t=None,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    port_ret = invested_returns(port_df)

    # -- today.json: dashboard hành động hôm nay --
    holdings = [
        {
            "symbol": sym, "price": round(float(row["price"]), 2),
            "ret_1d": round(float(row["ret_1d"]), 4), "mom_6": round(float(row["mom_6"]), 4),
            "mom_12": round(float(row["mom_12"]), 4), "vol_spike": round(float(row["vol_spike"]), 3),
            "composite_rank": round(float(row["composite_rank"]), 3), "signal": row["signal"],
        }
        for sym, row in snap.iterrows()
    ]
    today_payload = {
        "meta": {
            "note": f"Nguồn: vnstock, khoảng {start} → {end}, tải lúc {pd.Timestamp.now():%Y-%m-%d %H:%M}.",
            "as_of": latest.strftime("%Y-%m-%d"),
            "universe_size": price_df.shape[1],
            "dropped_symbols": dropped_symbols,
        },
        "ic_weights": {k: round(float(v), 4) for k, v in ic_weights.items()},
        "holdings": holdings,
        "summary": {
            "n_buy": int((snap["signal"] == "BUY").sum()), "n_avoid": int((snap["signal"] == "AVOID").sum()),
            "n_reduce": int((snap["signal"] == "REDUCE").sum()), "n_exit": int((snap["signal"] == "EXIT").sum()),
            "n_neutral": int((snap["signal"] == "NEUTRAL").sum()),
        },
    }
    with open(DATA_DIR / "today.json", "w", encoding="utf-8") as f:
        json.dump(today_payload, f, indent=2, ensure_ascii=False)

    # -- backtest.json --
    def get_metrics(key: str) -> dict:
        return strategies_cmp.loc[key].to_dict() if key in strategies_cmp.index else {}

    strat_metrics = get_metrics("Momentum+ExitRules (net)")
    bench_metrics = get_metrics("1/N Equal-Weight")
    gross_metrics = get_metrics("Momentum+ExitRules (gross, không cost)")

    common_idx = port_ret.index.intersection(ew_ret.index)
    cum_strat_net = (1 + port_ret.loc[common_idx]).cumprod() - 1
    cum_bench = (1 + ew_ret.loc[common_idx]).cumprod() - 1
    cum_strat_gross = None
    if cost_rate > 0 and "gross_return" in port_df.columns:
        gross_ret = invested_returns(port_df, "gross_return").loc[common_idx]
        cum_strat_gross = (1 + gross_ret).cumprod() - 1

    recent_trades = trade_df.sort_values("date", ascending=False).head(30) if len(trade_df) else trade_df
    trades_out = []
    for _, r in recent_trades.iterrows():
        trades_out.append({
            "date": r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else None,
            "symbol": r["symbol"], "action": r["action"], "reason": str(r["reason"])[:80],
            "entry_px": round(float(r["entry_px"]), 2) if pd.notna(r["entry_px"]) else None,
            "exit_px": round(float(r["exit_px"]), 2) if pd.notna(r["exit_px"]) else None,
            "pnl": round(float(r["pnl"]), 4) if pd.notna(r["pnl"]) else None,
        })

    vs_benchmark = {
        "Momentum+ExitRules (net cost)": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in strat_metrics.items()},
        "1/N Equal-Weight": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in bench_metrics.items()},
    }
    if gross_metrics:
        vs_benchmark["Momentum+ExitRules (gross, không cost)"] = {
            k: (None if pd.isna(v) else round(float(v), 4)) for k, v in gross_metrics.items()
        }

    cumulative_payload = {
        "dates": [d.strftime("%Y-%m-%d") for d in cum_strat_net.index],
        "strategy_net": [round(float(x), 4) for x in cum_strat_net],
        "benchmark": [round(float(x), 4) for x in cum_bench],
    }
    if cum_strat_gross is not None:
        cumulative_payload["strategy_gross"] = [round(float(x), 4) for x in cum_strat_gross]

    backtest_payload = {
        "engine": {"version": "v2", "timing": "signal chốt close t → ăn return từ t+1; exit/entry khớp close t",
                   "comparison_vs_legacy": engine_comparison, "excess_tstat_vs_1N": None if excess_t is None else round(float(excess_t), 3)},
        "cost_info": {
            "cost_rate_applied": round(float(cost_rate), 4),
            "total_cost_paid": round(float(port_df["cost"].sum()), 4) if "cost" in port_df.columns else None,
            "note": "0.23%/lệnh (0.15% transaction + 0.08% slippage) — giữ nguyên giả định "
                    "từ project momentum-ridge-vn30 (Ngày 13)." if cost_rate > 0
                    else "Backtest KHÔNG có transaction cost (chạy với --no-cost).",
        },
        "vs_benchmark": vs_benchmark,
        "trade_stats": {
            "n_buys": trade_stats.get("n_buys", 0), "n_sells": trade_stats.get("n_sells", 0),
            "win_rate": round(float(trade_stats.get("win_rate", float("nan"))), 4) if trade_stats.get("win_rate") is not None and not pd.isna(trade_stats.get("win_rate", np.nan)) else None,
            "avg_win_pnl": round(float(trade_stats.get("avg_win_pnl", np.nan)), 4) if not pd.isna(trade_stats.get("avg_win_pnl", np.nan)) else None,
            "avg_loss_pnl": round(float(trade_stats.get("avg_loss_pnl", np.nan)), 4) if not pd.isna(trade_stats.get("avg_loss_pnl", np.nan)) else None,
            "avg_hold_days": round(float(trade_stats.get("avg_hold_days", np.nan)), 1) if not pd.isna(trade_stats.get("avg_hold_days", np.nan)) else None,
            "exit_breakdown": trade_stats.get("exit_breakdown", {}),
        },
        "recent_trades": trades_out,
        "cumulative": cumulative_payload,
        "ic_summary": {k: round(float(v), 4) for k, v in sf_ic.to_dict().items()},
    }
    with open(DATA_DIR / "backtest.json", "w", encoding="utf-8") as f:
        json.dump(backtest_payload, f, indent=2, ensure_ascii=False)

    regenerate_embed_js()


def regenerate_embed_js() -> None:
    def load(name: str):
        with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
            return json.load(f)

    payload = {"TODAY_DATA": load("today"), "BACKTEST_DATA": load("backtest")}
    with open(DASHBOARD_DIR / "embed_data.js", "w", encoding="utf-8") as f:
        for name, data in payload.items():
            f.write(f"const {name} = " + json.dumps(data, ensure_ascii=False) + ";\n")


if __name__ == "__main__":
    main()
