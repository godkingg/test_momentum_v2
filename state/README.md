# state/ — trạng thái của báo cáo hằng ngày (workflow tự commit, đừng sửa tay)

| File | Vai trò |
|---|---|
| `ic_weights.json` | Trọng số IC **đóng băng**. Tính lại mỗi ngày sẽ làm lịch sử mô phỏng thay đổi → lệnh hôm qua có thể "biến mất". Chỉ tính lại khi chạy workflow với `refit_weights=true`. |
| `universe.json` | Universe đã xác nhận. Lần chạy sau thiếu/thừa mã → script DỪNG và báo lỗi thay vì phát lệnh sai. |
| `ledger.json` | Vị thế mô hình cuối phiên gần nhất; phiên sau dùng để kiểm tra lịch sử tính lại có khớp không. |
| `preview.json` | Lệnh dự kiến 14:00 — để báo cáo 15:00 so sánh. |
| `daily_log.csv` | Nhật ký KPI từng ngày (mỗi ngày 1 dòng). |
