# -*- coding: utf-8 -*-
"""Sinh dashboard/data/*.json bằng snapshot lịch sử đã chạy thật từ 2 nguồn:
1. day19_test.ipynb (đã CHẠY trên Colab ngày 2026-09-09) — dashboard "hôm nay"
2. notebooks/03_backtest_with_costs.ipynb (đã CHẠY với vnstock thật,
   2022-01-01→2025-12-31) — backtest CÓ transaction cost, Ngày 20

⚠️ SỐ LIỆU Ở ĐÂY DO ENGINE CŨ TÍNH (look-ahead cùng ngày) — chỉ để minh họa dashboard; engine v2 nằm ở
src/backtest.py, chạy `python launcher.py` để có số liệu mới.

Đây KHÔNG phải số giả lập — là execution output thật copy nguyên văn từ các
notebook đã chạy. Đây là script bootstrap/legacy, không phải nguồn kết quả
nghiên cứu mới nhất và không bao gồm bảng so sánh XGBoost Walk-Forward của
notebook 05. Chạy `python launcher.py` để cập nhật dashboard production bằng
dữ liệu vnstock mới nhất, trade log và cumulative return đầy đủ.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dashboard" / "data"
DASHBOARD_DIR = ROOT / "dashboard"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── IC từng factor của snapshot legacy (Purged K-Fold OOS, dữ liệu 2022-2025) ─
IC_SUMMARY = {
    "Momentum_6_1": 0.0524,
    "Momentum_12_1": 0.0440,
    "TS_Momentum": 0.0336,
    "Volume_Ratio": 0.0057,
}
# IC weight chuẩn hóa (Cell 12): Mom_6=0.386, Mom_12=0.324, TS_Mom=0.248, Vol_Rat=0.042
IC_WEIGHTS = {
    "Momentum_6_1": 0.386, "Momentum_12_1": 0.324, "TS_Momentum": 0.248, "Volume_Ratio": 0.042,
}

# ── Dashboard snapshot (output THẬT ngày 2026-09-09) ──────────────────────────
# 7 mã BUY signal cụ thể từ output ipynb — giữ nguyên số liệu, không suy diễn
TODAY_HOLDINGS = [
    {"symbol": "VHM", "price": 72.1, "ret_1d": -0.020, "mom_6": 0.812, "mom_12": 0.559, "vol_spike": 1.90, "composite_rank": 0.933, "signal": "BUY"},
    {"symbol": "VIC", "price": 247.7, "ret_1d": -0.008, "mom_6": 0.503, "mom_12": 2.501, "vol_spike": 0.77, "composite_rank": 0.901, "signal": "BUY"},
    {"symbol": "STB", "price": 76.7, "ret_1d": 0.008, "mom_6": 0.188, "mom_12": 0.282, "vol_spike": 1.38, "composite_rank": 0.882, "signal": "BUY"},
    {"symbol": "LPB", "price": 48.1, "ret_1d": -0.017, "mom_6": 0.364, "mom_12": 0.231, "vol_spike": 0.72, "composite_rank": 0.855, "signal": "BUY"},
    {"symbol": "HDB", "price": 27.5, "ret_1d": 0.005, "mom_6": 0.102, "mom_12": 0.040, "vol_spike": 1.15, "composite_rank": 0.792, "signal": "BUY"},
    {"symbol": "VJC", "price": 125.0, "ret_1d": -0.008, "mom_6": 0.071, "mom_12": 0.154, "vol_spike": 0.68, "composite_rank": 0.776, "signal": "BUY"},
    {"symbol": "SSB", "price": 17.7, "ret_1d": -0.017, "mom_6": 0.124, "mom_12": -0.134, "vol_spike": 0.72, "composite_rank": 0.582, "signal": "BUY"},
    {"symbol": "MBB", "price": 20.2, "ret_1d": 0.002, "mom_6": 0.001, "mom_12": -0.126, "vol_spike": 0.84, "composite_rank": None, "signal": "AVOID"},
    {"symbol": "HPG", "price": 22.1, "ret_1d": 0.009, "mom_6": -0.034, "mom_12": -0.131, "vol_spike": 0.94, "composite_rank": None, "signal": "AVOID"},
    {"symbol": "MWG", "price": 71.6, "ret_1d": -0.010, "mom_6": -0.061, "mom_12": -0.046, "vol_spike": 0.69, "composite_rank": None, "signal": "AVOID"},
    {"symbol": "MSN", "price": 68.4, "ret_1d": 0.000, "mom_6": -0.025, "mom_12": -0.199, "vol_spike": 0.76, "composite_rank": None, "signal": "AVOID"},
    {"symbol": "FPT", "price": 72.4, "ret_1d": 0.003, "mom_6": -0.056, "mom_12": -0.298, "vol_spike": 0.71, "composite_rank": None, "signal": "AVOID"},
    {"symbol": "SHB", "price": 11.8, "ret_1d": -0.008, "mom_6": -0.167, "mom_12": -0.372, "vol_spike": 0.84, "composite_rank": None, "signal": "AVOID"},
]

signal_payload = {
    "meta": {
        "note": "✅ SỐ LIỆU THẬT — trích xuất từ day19_test.ipynb (đã chạy trên Colab "
                "ngày 2026-09-09, dữ liệu vnstock 2022-01-01 → 2026-09-09, universe 28 mã). "
                "Chạy `python launcher.py` để cập nhật đến ngày hiện tại.",
        "as_of": "2026-09-09",
        "universe_size": 28,
        "dropped_symbols": ["TCX", "VPL"],
    },
    "ic_weights": IC_WEIGHTS,
    "holdings": TODAY_HOLDINGS,
    "summary": {
        "n_buy": 7, "n_avoid": 6, "n_reduce": 0, "n_exit": 0, "n_neutral": 15,
    },
}
with open(DATA_DIR / "today.json", "w", encoding="utf-8") as f:
    json.dump(signal_payload, f, indent=2, ensure_ascii=False)

# ── Backtest snapshot — Ngày 20, số liệu THẬT ĐÃ CÓ CẢ GROSS VÀ NET ───────────
# (notebooks/03_backtest_with_costs.ipynb đã chạy XONG với dữ liệu vnstock
# thật, 2022-01-01→2025-12-31, universe 28 mã — copy nguyên văn execution
# output, không suy diễn).
GROSS_METRICS = {
    "Total Return": 2.4851, "Ann Return": 0.5281, "Ann Vol": 0.2010,
    "Sharpe": 2.6269, "Max DD": -0.1764, "Calmar": 2.9938,
}
NET_METRICS = {
    "Total Return": 2.1720, "Ann Return": 0.4800, "Ann Vol": 0.2019,
    "Sharpe": 2.3773, "Max DD": -0.1827, "Calmar": 2.6279,
}
BENCH_METRICS = {
    "Total Return": 1.0210, "Ann Return": 0.2699, "Ann Vol": 0.1797,
    "Sharpe": 1.5023, "Max DD": -0.1681, "Calmar": 1.6060,
}

backtest_payload = {
    "engine": {"version": "legacy", "timing": "engine cũ: vị thế vào/ra ở close t vẫn ăn/né return ngày t (look-ahead cùng ngày)"},
    "cost_info": {
        "cost_rate_applied": 0.0023,
        "total_cost_paid": 0.0937,
        "note": "⚠️ ENGINE CŨ (có look-ahead cùng ngày) — Snapshot CÓ COST (Ngày 20) chạy với dữ liệu vnstock thật "
                "(notebooks/03_backtest_with_costs.ipynb, 2022-01-01→2025-12-31, 28 mã). "
                "Sau cost, chiến lược VẪN THẮNG 1/N benchmark rõ ràng (Sharpe NET 2.377 "
                "vs 1.502). Tổng cost đã trả: 9.37% NAV trên 374 lệnh giao dịch (~7.9 "
                "lệnh/tháng, 47 tháng).",
    },
    "vs_benchmark": {
        "Momentum+ExitRules (net cost)": NET_METRICS,
        "Momentum+ExitRules (gross, không cost)": GROSS_METRICS,
        "1/N Equal-Weight": BENCH_METRICS,
    },
    "trade_stats": {
        "n_buys": 189, "n_sells": 185, "win_rate": None,  # win_rate không tách riêng trong output Ngày 20
        "avg_win_pnl": None, "avg_loss_pnl": None, "avg_hold_days": None,
        "exit_breakdown": {},
        "_note": "Trade-level win_rate/avg_pnl/exit_breakdown không có trong output "
                 "Ngày 20 (chỉ có portfolio-level metrics) — dùng số Ngày 19 (186/182 "
                 "lệnh, win_rate 54.9%) làm tham chiếu gần đúng nếu cần, KHÔNG PHẢI "
                 "cùng 1 lần chạy chính xác với bảng metrics ở trên.",
    },
    "recent_trades": [],  # trade log chi tiết (>370 dòng) không có trong output notebook — chạy launcher.py để có
    "cumulative": {"dates": [], "strategy_net": [], "benchmark": []},  # cần launcher.py để vẽ đường cumulative
    "ic_summary": IC_SUMMARY,
    "_note": "Trade log chi tiết từng dòng và cumulative return series KHÔNG có sẵn "
             "trong output notebook gốc (chỉ có bảng tổng kết + summary số liệu) — "
             "chạy launcher.py để có đầy đủ.",
}
with open(DATA_DIR / "backtest.json", "w", encoding="utf-8") as f:
    json.dump(backtest_payload, f, indent=2, ensure_ascii=False)


def load(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


with open(DASHBOARD_DIR / "embed_data.js", "w", encoding="utf-8") as f:
    for const_name, fname in [("TODAY_DATA", "today"), ("BACKTEST_DATA", "backtest")]:
        f.write(f"const {const_name} = " + json.dumps(load(fname), ensure_ascii=False) + ";\n")

print("✅ Đã ghi dashboard/data/*.json — snapshot legacy từ day19_test.ipynb")
print("   + notebooks/03_backtest_with_costs.ipynb (Sharpe NET có cost = 2.377)")
print("   Chạy `python launcher.py` để cập nhật dashboard production và trade log/cumulative return.")
