# CHANGELOG — v2 (engine backtest không look-ahead)

## Lỗi đã sửa
| # | Lỗi | Chỗ sửa | Chứng minh |
|---|---|---|---|
| 1 | Engine cũ: vị thế mở/đóng ở close `t` vẫn ăn/né return ngày `t` (cú giảm kích hoạt stop không vào PnL) | `src/backtest.py` viết lại; bản cũ giữ ở `src/backtest_legacy.py` chỉ để đối chứng | `tests/test_engine_no_lookahead.py` (kịch bản tất định + tín hiệu ngẫu nhiên + alignment) |
| 2 | `walk_forward_signal` train bằng nhãn 5 ngày chứa giá sau ngày dự đoán (thiếu purge) | `src/model.py`: tham số `n_fwd=5` | `tests/test_walk_forward_signal.py::test_walk_forward_signal_purges_label_horizon` (bắt được lỗi ở `n_fwd=0`) |
| 3 | Cắt warm-up bằng `ret[ret != 0]` (lọc theo giá trị, bỏ cả ngày return = 0 hợp lệ) | `invested_returns()`; notebook 03–05, `launcher.py` đã đổi | `tests/test_engine_no_lookahead.py::test_invested_returns_*` |

## Thêm mới
- `src/backtest.py`: `invested_returns`, `sharpe_se`; `port_df` có thêm `n_positions_start`.
- `src/risk.py`: cột `Sharpe SE` trong `compare_strategies`, hàm `excess_return_tstat`.
- `launcher.py`: cờ `--compare-legacy`; `backtest.json` có khối `engine` (version, so sánh cũ/v2, t-stat vs 1/N); dashboard `backtest.html` hiển thị banner phiên bản engine.
- `notebooks/06_engine_v2_rerun.ipynb`: chạy lại toàn bộ trên dữ liệu thật, in Δ (engine cũ − v2), lưu `reports/results_engine_v2.json`.
- `tests/test_engine_no_lookahead.py` (6 test), `tests/test_launcher_smoke.py` (end-to-end bằng dữ liệu giả lập).

## Thay đổi tài liệu / notebook
- `README.md`, `reports/research_note.md` (mục 11): số Sharpe cũ chuyển thành "KẾT QUẢ CŨ — không trích dẫn"; kết luận "ĐÃ CHỐT" bị hạ xuống "chưa xác nhận".
- Notebook 03–05: output cũ đã xóa (còn nguyên bản cũ ở `notebooks/legacy_v1_outputs/`), có banner, đã vá sang `invested_returns`.
- `scripts_bootstrap_old_numbers.py` + `dashboard/`: snapshot cũ được gắn nhãn **ENGINE CŨ**.

## Chưa làm (cần dữ liệu thật hoặc nằm ngoài phạm vi)
- **Chưa có số Sharpe engine v2 trên dữ liệu thật** → chạy notebook 06.
- Trọng số IC của Composite vẫn tính trên toàn mẫu (in-sample); tham số PositionManager chưa có out-of-sample; khớp lệnh tại close; mục 4.3 của research note (notebook 01) chưa kiểm tra quy ước thời gian.

## v2.1 — Báo cáo tự động hằng ngày lên Discord
- `daily_report.py`, `src/reporting.py`, `src/report_render.py`, `src/discord_webhook.py`, `src/pipeline.py` (signal dùng chung, trọng số IC đóng băng).
- `.github/workflows/daily_report.yml`: cron 13:50 → báo cáo 14:00, 14:50 → báo cáo 15:00 (giờ VN), tự chờ đúng giờ; `requirements-report.txt`; thư mục `state/`.
- `backtest_with_exits` gắn vị thế còn mở vào `port_df.attrs["open_positions"]` (không đổi chữ ký).
- `launcher.py` dùng `build_factors` từ `src/pipeline.py` (hành vi không đổi).
- Tests: `test_daily_report.py` (end-to-end, Discord giả), `test_reporting_render_webhook.py` — tổng 60 test.

## v2.2 — Slash command `/report` (xem báo cáo bất cứ lúc nào)
- `worker/worker.js`: Cloudflare Worker (1 file, không phụ thuộc) — xác thực Ed25519 + chống replay, phân quyền theo server/role/user, cooldown bằng KV, kích hoạt GitHub `workflow_dispatch`. 14 test Node (`worker/test/`).
- `.github/workflows/report_on_demand.yml` + `scripts/register_discord_commands.py`.
- `daily_report.py`: `--on-demand`, `--mode auto` (đang trong phiên → tạm tính; ngoài phiên/cuối tuần/lễ → cuối phiên gần nhất; `--asof` → ngày đã qua), `--channel-id`, `--requester-id`; không ghi `state/`; lỗi được báo về đúng kênh người gọi.
- `src/discord_webhook.py`: `send_report_channel` / `send_text_channel` (bot REST); `WebhookError.status`; tiêu đề báo cáo theo ngữ cảnh ("theo yêu cầu").
- Nhãn báo cáo đổi "Hôm nay" → "Phiên dd/mm" (đúng cả với báo cáo ngày cũ).
