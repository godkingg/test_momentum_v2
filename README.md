# Momentum Factors + ML Alpha + Exit Rules + Cost Model — VN30 Universe

> ## ⚠️ Bản v2 — đã thay engine backtest (đọc trước)
> Engine backtest cũ (Ngày 19–22) có **look-ahead cùng ngày**: vị thế mở/đóng ở close `t` vẫn ăn/né return của chính ngày `t`;
> walk-forward XGBoost cũ thiếu **purge nhãn**. Bản này thay bằng **engine v2** (`src/backtest.py`: signal chốt close `t`, vị thế ăn return từ `t+1`)
> và sửa `walk_forward_signal` (`n_fwd` purge).
> **Mọi Sharpe backtest cũ (2.71 / 2.68 / 2.64 / 2.38...) là số của engine cũ, đã chuyển xuống mục "Kết quả CŨ" và KHÔNG nên trích dẫn.**
> Số liệu engine v2 trên dữ liệu thật **chưa được tính** trong bản này (môi trường tạo bản v2 không truy cập được vnstock) —
> chạy `notebooks/06_engine_v2_rerun.ipynb` hoặc `python launcher.py --compare-legacy`, rồi điền vào mục "Kết quả engine v2".
> Chi tiết lỗi, bằng chứng và hạn chế còn lại: `reports/research_note.md` mục 11.

## Mục tiêu

**Ngày 18**: Kiểm tra pattern "winner tiếp tục thắng, loser tiếp tục thua"
(momentum classical) trên universe VN30-tương tự bằng 2 biến thể momentum
(6-1 và 12-1, bỏ tháng gần nhất tránh short-term reversal).

**Ngày 19**: So sánh 3 cách kết hợp nhiều factor thành 1 alpha score
(IC-Weighted / Ridge / XGBoost), dùng SHAP để xác nhận feature nào thực sự
drive prediction, đánh giá bằng IC (không chỉ RMSE).

**Ngày 20**: Test lại chiến lược Momentum + Exit Rules với **transaction
cost + slippage** — giữ nguyên giả định cost đã dùng ở project
`momentum-ridge-vn30` (0.15% + 0.08% = 0.23%/lệnh) — câu hỏi Ngày 19: Sharpe có còn
đứng vững sau chi phí giao dịch không? Kết luận cũ ("có") được tính bằng engine cũ nên
**cần xác nhận lại bằng engine v2** (xem mục Kết quả).

**Bản v2**: sửa look-ahead cùng ngày của engine backtest + purge nhãn của walk-forward,
thêm test chống look-ahead và notebook 06 để đo lại.

## Dataset

- Nguồn: `vnstock` (source KBS).
- Universe: 30 mã VN30-tương tự → lọc coverage (≥95%) còn **28 mã** (loại
  TCX, VPL).
- Khoảng thời gian: notebook đã chạy trên 2 giai đoạn khác nhau tại các thời
  điểm khác nhau — 2022-01-01→2025-12-31 (Ngày 20, backtest có cost) và
  2023-01-06→2025-12-24 (lần chạy lại gần nhất, Ngày 18-19). Khác biệt giai
  đoạn là BÌNH THƯỜNG (mỗi lần chạy `load_ohlcv()` lấy dữ liệu tính đến hiện
  tại), không phải lỗi — nhưng nghĩa là **IC factor đo được có thể thay đổi
  giữa các lần chạy**, cần verify định kỳ chứ không giả định mãi đúng.
- Forward return label: N_FWD = 5 ngày (1 tuần giao dịch).

## Phương pháp

```
Giá đóng cửa + volume (28 mã)
  → Momentum_6_1, Momentum_12_1, Volume_Ratio, TS_Momentum
  → Cross-sectional rank (feature_engineering.py)
  → IC Analysis, Purged K-Fold OOS (ic_analysis.py)
  → So sánh: Best single factor / IC-Weighted / Ridge / XGBoost (ic_analysis.py + model.py)
  → SHAP TreeExplainer trên XGBoost — so sánh với feature_importances_ chuẩn
  → Composite Score (IC-Weighted) → PositionManager (6 exit rule) → Backtest
  → Backtest engine v2 (độ trễ 1 ngày) CÓ transaction cost + slippage (0.23%/lệnh)
  → So sánh: Net vs Gross vs 1/N Equal-Weight, kèm Sharpe SE và t-stat return vượt trội
```

⚠️ **`launcher.py` (production, tạo dashboard BUY/SELL thật) dùng IC-Weighted
composite, KHÔNG dùng Ridge/XGBoost** — quyết định dựa trên kết quả đo được
rằng ML không cải thiện IC trên universe 28 mã (xem bảng dưới). ML (Ridge,
XGBoost, SHAP) vẫn có trong `src/model.py` và notebook 02 ở vai trò nghiên
cứu/so sánh, không phải signal production.

## Kết quả chính

### A. IC (không dùng engine backtest → không bị ảnh hưởng bởi lỗi look-ahead)

| Metric | Giá trị |
| --- | --- |
| Momentum_6_1 Mean IC (đứng riêng) | 0.0501 – 0.0525 (2 lần chạy độc lập) |
| IC-Weighted composite Mean IC | 0.0466 – 0.0572 (2 lần chạy độc lập) |
| Ridge Mean IC (all features) | 0.0134 – 0.0171 |
| XGBoost Mean IC (Purged K-Fold, Ngày 19) | 0.0279 – 0.0373 |

⚠️ Ngày 25 (`day25.ipynb`, dữ liệu tới 2026-09) cho thấy IC của Ridge/XGB/LGBM/RF trung bình ≈ 0 và đổi dấu theo fold (fold cuối âm ở cả 4 model) —
cần đối chiếu với bảng trên khi cập nhật dữ liệu (IC có thể không ổn định qua giai đoạn).

### B. Kết quả engine v2 (net cost 0.23%/lệnh, cùng cửa sổ cho mọi chiến lược)

⏳ **Điền sau khi chạy `notebooks/06_engine_v2_rerun.ipynb`** (số liệu nằm ở `reports/results_engine_v2.json`):

| Chiến lược | Sharpe ± 1.96·SE | Ann Return | Max DD | Win Rate | Rank IC | t(excess vs 1/N) |
| --- | --- | --- | --- | --- | --- | --- |
| IC-Weighted Composite | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Single Factor (Momentum_6_1) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| XGBoost walk-forward (purged) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| 1/N Equal-Weight | ⏳ | ⏳ | ⏳ | — | — | — |

Cách đọc: Sharpe SE ≈ 0.5–0.7 với ~3 năm dữ liệu ⇒ chênh lệch Sharpe giữa các chiến lược nhỏ hơn ~1.0 gần như chắc chắn là nhiễu;
dùng `t(excess vs 1/N)` (kiểm định ghép cặp) để so với benchmark. Notebook 06 cũng in cột **Δ (engine cũ − v2)** = lượng Sharpe bị thổi phồng bởi lỗi cũ trên dữ liệu thật của bạn.

### C. Bằng chứng đã xác minh trong bản v2 (không cần dữ liệu thật)

| Kiểm tra | Kết quả |
| --- | --- |
| Vị thế mở ở close `t` không ăn return ngày `t` (kịch bản tất định) | ✅ engine v2 = 0; engine cũ ăn +10% |
| Cú giảm kích hoạt stop vào PnL ngày `t` | ✅ engine v2 ghi −20%; engine cũ né = 0 |
| Signal = return ngày mai (nhìn trước) vs return hôm nay | ✅ Sharpe > 10 vs < 1/10 mức đó (engine không lệch ngày) |
| `walk_forward_signal` bất biến khi xáo nhãn tương lai | ✅ qua với `n_fwd=5`; **bắt được lỗi** với `n_fwd=0` |
| Tín hiệu ngẫu nhiên, stop mặc định (50 seed, giá không alpha, dữ liệu giả lập) | v2 −0.05 · cũ +0.13 (chênh +0.18 ± 0.08) |
| Tín hiệu ngẫu nhiên, stop chặt 3%/5% | v2 −0.09 · cũ +1.93 (chênh +2.02 ± 0.08) |
| `launcher.py` chạy end-to-end (dữ liệu giả lập) + JSON dashboard | ✅ `tests/test_launcher_smoke.py` |

→ Lỗi **có thật và đo được**, nhưng với tham số mặc định trên tín hiệu ngẫu nhiên nó chỉ giải thích ~0.1–0.2 Sharpe; mức ảnh hưởng thực tế lên chiến lược momentum
trên dữ liệu thị trường **chưa biết** cho tới khi chạy notebook 06.

### D. Kết quả CŨ — engine có look-ahead cùng ngày (⚠️ chỉ tham khảo, KHÔNG trích dẫn)

| Metric (engine cũ) | Giá trị cũ |
| --- | --- |
| Sharpe (có cost) — IC-Weighted Composite, notebook 05 | 2.7144 |
| Sharpe (có cost) — XGBoost Walk-Forward, notebook 05 (thiếu purge) | 2.6765 |
| Sharpe (có cost) — Single Factor (Momentum_6_1), notebook 05 | 2.6432 |
| Sharpe (có cost) — 1/N Equal-Weight, notebook 05 | 1.8023 |
| Notebook 04: Composite / Single / 1/N | 2.3773 / 2.1678 / 1.5023 |

Bản notebook cũ còn nguyên output nằm ở `notebooks/legacy_v1_outputs/`. Chênh lệch Composite–Single–XGBoost (≤ 0.07 Sharpe ở notebook 05) còn nhỏ hơn
Sharpe SE ≈ 0.6 nên **kết luận "Composite thắng" chưa bao giờ có ý nghĩa thống kê**, kể cả khi engine đúng.

⚠️ **Bài học phương pháp luận giữ nguyên**: IC cao hơn không tự động đồng nghĩa Sharpe cao hơn; quyết định chọn signal production phải dựa trên backtest
có cost + exit rules — nhưng backtest đó phải **không look-ahead** và so sánh phải kèm sai số chuẩn.

## Báo cáo tự động hằng ngày lên Discord (miễn phí · không VPS · không cần máy local)

Mỗi ngày giao dịch (thứ 2–6, giờ Việt Nam), GitHub Actions tự chạy và gửi vào kênh Discord của bạn **ảnh PNG + file HTML**:

| Giờ | Chế độ | Nội dung |
| --- | --- | --- |
| **14:00** | `preview` — lệnh dự kiến | Danh sách MUA/BÁN/GIẢM theo giá gần nhất (nến chưa chốt, còn kịp đặt lệnh ATC 14:30–14:45), vị thế đang giữ kèm **giá cắt lỗ / trailing** để đặt lệnh điều kiện, watchlist top rank |
| **15:00** | `final` — cuối phiên | Lệnh chốt, return ngày (net cost) so với 1/N, lũy kế, drawdown, và **so sánh với lệnh dự kiến 14:00** (mới / không còn / giữ nguyên) |

Cách hoạt động: Discord **webhook** (chỉ cần 1 URL, không cần bot online 24/7) + **GitHub Actions cron** (miễn phí; repo public không giới hạn phút, repo private ~2.000 phút/tháng, mỗi ngày dùng ≈ 25 phút).
PNG vẽ bằng matplotlib (không cần trình duyệt); HTML là file độc lập (không CDN/JS) — Discord không xem trước HTML nên nó là file đính kèm để tải về mở bằng trình duyệt.

### Cài đặt (≈10 phút)
1. **Tạo webhook**: Discord → Server Settings → Integrations → Webhooks → New Webhook → chọn kênh → *Copy Webhook URL*.
2. **Đẩy repo này lên GitHub** (thư mục `.github/workflows/` phải nằm trong repo).
3. **Lưu bí mật**: GitHub repo → Settings → Secrets and variables → Actions → *New repository secret* → tên `DISCORD_WEBHOOK_URL`, giá trị là URL vừa copy.
   (Tùy chọn: tab *Variables* → `DISCORD_MENTION` = `<@ID_Discord_của_bạn>` để được ping.)
4. **Chạy thử ngay** (đừng chờ tới 14:00): tab *Actions* → *Daily report (Discord)* → *Run workflow* → `mode=final`.
   Chạy vào cuối tuần/nghỉ lễ thì điền `asof` = ngày giao dịch gần nhất (vd `2026-10-02`) — chạy kiểu này **không commit state**.
5. Kiểm tra 3 điều ở lần chạy thử đầu: (a) bước tải dữ liệu **không bị chặn IP** của GitHub; (b) nến hôm nay **có xuất hiện lúc 14:00** (chạy `mode=preview` trong giờ giao dịch); (c) ảnh trong Discord hiển thị đúng.
   Từ đó lịch tự chạy; mỗi lần có thêm artifact (ảnh/HTML/JSON) tải về được trong tab *Actions* (giữ 14 ngày).

### Lịch chạy và độ trễ
| Cron (UTC) | Giờ VN | Việc |
| --- | --- | --- |
| `50 6 * * 1-5` | 13:50 | khởi động, **tự chờ tới đúng 14:00** rồi tải dữ liệu + gửi preview |
| `50 7 * * 1-5` | 14:50 | khởi động, **tự chờ tới đúng 15:00** rồi gửi báo cáo cuối phiên |

GitHub **không đảm bảo đúng giờ**: cron thường trễ 5–30 phút lúc cao điểm, hiếm khi bị bỏ qua hẳn. Lịch được đặt sớm 10 phút để bù; nếu GitHub trễ quá giờ hẹn thì báo cáo đến muộn tương ứng
(preview trễ quá ~14:25 là không còn ý nghĩa — hãy coi đó là tín hiệu tham khảo). Việc tải 30 mã mất ~2–3 phút nên ảnh thường đến 14:03–14:05 và 15:03–15:05.

### An toàn vận hành đã dựng sẵn
- **Trọng số IC đóng băng** (`state/ic_weights.json`): tính lại mỗi ngày làm lịch sử mô phỏng đổi nhẹ → lệnh hôm qua có thể biến mất. Chỉ tính lại khi chạy tay với `refit_weights=true`.
- **Chặn universe lệch**: thiếu/thừa mã so với lần trước (vd một mã tải lỗi) → script thử tải lại 3 lần rồi **dừng và báo lỗi lên Discord**, không phát lệnh trên dữ liệu thiếu.
- **Sổ lệnh** (`state/ledger.json`): mỗi phiên tính lại danh mục hôm qua và so với sổ đã lưu; lệch → cảnh báo ngay trong báo cáo (thường do nguồn dữ liệu sửa giá lịch sử, vd điều chỉnh cổ tức).
- **Cuối tuần** bỏ qua im lặng; **ngày lễ / chưa có nến** → gửi 1 dòng thông báo (tắt bằng `--no-notice`). Lỗi bất kỳ → báo lên Discord (không lộ URL webhook) và job báo đỏ để GitHub gửi mail.
- Commit `state/` mỗi ngày giao dịch cũng giữ repo luôn "có hoạt động" — GitHub tự tắt workflow theo lịch sau 60 ngày repo không có hoạt động.

### Giới hạn cần biết
- Đây là **danh mục mô hình (paper)**: giá khớp giả định = close, tỷ trọng equal-weight 1/số vị thế, chưa có T+2/thanh khoản. Báo cáo **không phải khuyến nghị đầu tư**, và mục *Kết quả engine v2* ở trên vẫn chưa có số thật — hãy chạy notebook 06 trước khi dựa vào tín hiệu.
- Lúc 14:00 **khối lượng hôm nay chưa đủ phiên** nên factor Volume_Ratio bị thấp hơn thực tế → thứ hạng có thể đổi lúc 15:00 (báo cáo cuối phiên nêu rõ lệnh nào thay đổi).
- Lịch sử mô phỏng bắt đầu từ `--start 2022-01-01` và **nhịp rebalance 5 phiên phụ thuộc vị trí ngày trong chuỗi dữ liệu** → đừng đổi `--start` giữa chừng (sẽ lệch toàn bộ lịch rebalance).
- Chiến lược có thể **bán rồi mua lại cùng mã trong cùng phiên** (thoát theo stop/rule nhưng rank vẫn cao): báo cáo tự cảnh báo; khi đặt lệnh thật có thể bỏ cả hai.
- Chưa kiểm chứng được trong môi trường tạo bản này: (1) GitHub có bị nguồn dữ liệu chặn IP không; (2) nến intraday lúc 14:00 có sẵn không → bước 5 ở trên.

Chạy tay trên máy: `python daily_report.py --mode final --no-discord --asof 2026-10-02` (sinh file vào `reports/daily/`).

## Lệnh slash `/report` — xem báo cáo bất cứ lúc nào (miễn phí · không VPS · không cần máy local)

```
Bạn gõ /report ──► Cloudflare Worker (free, luôn online) ──► GitHub Actions ──► bot đăng PNG + HTML vào ĐÚNG KÊNH bạn gõ lệnh
                   xác thực chữ ký, phân quyền, chống spam    tạo báo cáo (3–5 phút)
```

| Bạn gõ | Kết quả |
| --- | --- |
| `/report` lúc **đang trong phiên** (T2–T6, 09:00–15:00) | **Lệnh tạm tính** theo giá gần nhất (nến chưa chốt) |
| `/report` lúc **sau 15:00 / cuối tuần / nghỉ lễ** | **Báo cáo cuối phiên** của phiên giao dịch gần nhất (có ghi chú nếu hôm nay không có phiên) |
| `/report date:02/10/2026` (hoặc `2026-10-02`) | **Báo cáo cuối phiên** của ngày đó (phải là ngày có phiên, trong quá khứ) |

Vì sao cần Worker: Discord phải gửi lệnh tới một địa chỉ HTTPS **luôn sẵn sàng** (GitHub Actions chỉ chạy theo lịch/khi được gọi nên không làm được). Cloudflare Workers gói free (100.000 request/ngày) làm đúng việc đó — và không có máy nào của bạn phải bật.
Bot **không cần online** (chỉ dùng REST để đăng tin), nên không phải giữ tiến trình `discord.py` chạy như bot cũ.

### Cài đặt một lần (≈20 phút) — làm SAU khi báo cáo theo lịch đã chạy ít nhất 1 lần (để có `state/ic_weights.json`)
1. **Discord Developer Portal** → *New Application*. Ở *General Information* copy **Application ID** và **Public Key**. Ở tab *Bot* → *Reset Token* → copy **Bot Token** (không cần bật Privileged Intents).
2. **Mời bot vào server** bằng link (thay `APP_ID`): `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot%20applications.commands&permissions=52224`
   (quyền: xem kênh, gửi tin, nhúng link, đính kèm file). Bot phải có quyền ở **kênh bạn sẽ gõ `/report`**.
3. **GitHub**: thêm Secret `DISCORD_BOT_TOKEN`. Tạo **fine-grained PAT** (Settings → Developer settings → Fine-grained tokens): *Only select repositories* → repo này, quyền **Actions: Read and write**, đặt hạn dùng. Token này chỉ dùng ở bước 4.
4. **Cloudflare** (tài khoản free): *Workers & Pages → Create → Hello World → Edit code* → dán toàn bộ `worker/worker.js` → *Deploy*. Rồi *Settings → Variables and Secrets*:

   | Tên | Loại | Giá trị |
   | --- | --- | --- |
   | `DISCORD_PUBLIC_KEY` | Variable | Public Key ở bước 1 |
   | `GH_REPO` | Variable | `owner/repo` |
   | `GH_TOKEN` | **Secret** | PAT ở bước 3 |
   | `ALLOWED_GUILD_IDS` | Variable | ID server của bạn (**nên đặt**; bật *Developer Mode* → chuột phải server → Copy ID) |
   | `ALLOWED_ROLE_IDS` / `ALLOWED_USER_IDS` | Variable, tuỳ chọn | giới hạn người được dùng (cách nhau dấu phẩy) |
   | `COOLDOWN` | **KV binding** | tạo KV namespace rồi gắn vào Worker (**nên có**: chống spam, mặc định 3 phút/người, 45 giây toàn server) |

5. Developer Portal → *General Information* → **Interactions Endpoint URL** = địa chỉ Worker (`https://….workers.dev`) → *Save* (Discord tự gửi PING kiểm tra chữ ký; báo lỗi nghĩa là sai Public Key).
6. **Đăng ký lệnh** (chạy 1 lần ở bất kỳ đâu, kể cả Colab: `!pip -q install requests` rồi chạy lệnh dưới):
   `DISCORD_BOT_TOKEN=... python scripts/register_discord_commands.py --app-id APP_ID --guild-id SERVER_ID`
7. Vào kênh, gõ `/report`. Thấy "⏳ Đã nhận yêu cầu…" rồi 3–5 phút sau là báo cáo (bot ping bạn).

### Giới hạn và bảo mật cần biết
- **Không tức thời**: GitHub Actions khởi động + tải 30 mã mất **3–5 phút** (không thể nhanh hơn nếu không có server chạy liên tục). Nếu cần < 10 giây thì phải có máy luôn bật.
- **Chống lạm dụng**: ai gõ được `/report` đều tốn phút Actions và hạn mức vnstock. Hãy đặt `ALLOWED_GUILD_IDS`, cân nhắc `ALLOWED_ROLE_IDS`, và gắn KV. KV nhất quán sau cùng (trễ tới ~1 phút giữa các điểm biên) nên cooldown là chống spam *tương đối*, không phải khoá chặt.
- **Bí mật**: `GH_TOKEN` chỉ nằm ở Worker (Secret) và có quyền tối thiểu (chỉ Actions của 1 repo); `DISCORD_BOT_TOKEN` chỉ ở GitHub Secrets; **không token nào đi qua tham số workflow** (nên không hiện trong tab Actions, kể cả repo public). Worker xác thực chữ ký Ed25519 của Discord và từ chối request quá hạn > 5 phút (chống replay). Ngày nhập vào được kiểm tra ở Worker và ở script, workflow truyền qua `env:` (chống chèn lệnh).
- **PAT hết hạn** thì `/report` báo "GitHub trả về 401" — tạo PAT mới và cập nhật `GH_TOKEN`.
- **Báo cáo theo yêu cầu không ghi `state/`** (không đổi sổ lệnh/trọng số của báo cáo theo lịch). Nếu chưa có trọng số đóng băng, báo cáo sẽ ghi rõ cảnh báo.
- **Báo cáo ngày cũ là mô phỏng lại bằng dữ liệu hiện có** (trọng số IC đóng băng hôm nay), có thể khác với những gì báo cáo ngày đó đã gửi nếu nguồn dữ liệu điều chỉnh giá lịch sử.
- Bot không đăng được vào kênh (thiếu quyền) → báo cáo được gửi vào kênh của webhook mặc định thay thế.

## Hạn chế & Rủi ro

- **Kết luận "Composite tốt nhất" CHƯA được xác nhận** — dựa trên engine cũ và chênh lệch nhỏ hơn Sharpe SE.
  Giữ Composite trong `launcher.py` là lựa chọn tạm thời (đơn giản, ít turnover) chờ kết quả engine v2.
- **Trọng số IC của Composite tính trên toàn mẫu** rồi backtest chính mẫu đó (thiên lệch in-sample nhỏ); cách sửa đúng là
  tính trọng số expanding-window — chưa làm trong bản này.
- **Khớp lệnh tại close** ngày tín hiệu: không T+2, không giới hạn thanh khoản/market impact.
- **Benchmark 1/N** rebalance hằng ngày và không cost (hơi có lợi cho 1/N — thận trọng với chiến lược).
- **Engine v2 mới được kiểm chứng bằng test + dữ liệu giả lập**; kết quả trên dữ liệu thật phải chạy notebook 06.
- **Chưa có out-of-sample tách biệt** — tham số PositionManager
  (stop_loss=8%, trail=12%, top_entry_rank=0.65...) được set cố định, chưa
  test trên giai đoạn hoàn toàn chưa thấy để kiểm tra overfitting tham số.
- **Long-short TS-Momentum kém hiệu quả** do bull bias của thị trường VN —
  Short leg kéo hiệu suất xuống. Momentum long-short kinh điển (kiểu Mỹ)
  KHÔNG hoạt động tốt trực tiếp trên VN, cần chuyển sang long-only.
- **Ridge (K-fold) thua Momentum_6_1 đứng riêng** trên universe 28 mã (IC, không phụ thuộc engine).
  XGBoost walk-forward giao dịch nhiều hơn đáng kể (307 buy vs 189 của Composite ở lần chạy cũ); cần universe lớn hơn
  và market-impact model trước khi đánh giá lợi thế triển khai. So sánh Sharpe giữa 3 chiến lược phải chờ engine v2.
- **Đa cộng tuyến giữa Momentum_6_1 và Momentum_12_1** — Ridge coefficient
  của Momentum_12_1 gần như 0 dù IC riêng lẻ dương, dấu hiệu 2 factor tương
  quan cao (Pearson corr ~0.72, cùng công thức, chỉ khác lookback).
- **SHAP trong `notebooks/02_ml_alpha_prediction.ipynb` là phần MỞ RỘNG**
  theo yêu cầu — notebook gốc `day19_test.ipynb` không chứa SHAP. Đã chạy
  thành công end-to-end (bao gồm `shap.TreeExplainer`, summary plot,
  beeswarm plot) — xem output trong notebook đã upload gần nhất.
- **Cost model giả định đơn giản** — 0.23%/lệnh cố định, KHÔNG mô hình hóa
  market impact tăng theo kích thước lệnh, KHÔNG phân biệt cost mua vs bán
  (thuế bán 0.1% ở VN thường cao hơn mua), KHÔNG tính spread thay đổi theo
  thanh khoản từng mã. Đây là xấp xỉ hợp lý cho nghiên cứu ban đầu, chưa
  phải mô hình cost sát thực tế broker cụ thể.
- **Universe hẹp, VN30 được xác định tại ngày chạy** — 28 mã tự chọn, thiên về
  ngân hàng/blue-chip (tương lai có thể thay đổi).

## Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. (Tùy chọn) Dashboard mẫu với snapshot ENGINE CŨ (có banner cảnh báo) — đã sinh sẵn trong zip
python scripts_bootstrap_old_numbers.py

# 3. Chạy 5 notebook (Ngày 18 → 20) để tái tạo phân tích với dữ liệu vnstock hiện tại
jupyter notebook notebooks/01_momentum_factors.ipynb        # Momentum factors, IC, quintile
jupyter notebook notebooks/02_ml_alpha_prediction.ipynb     # Feature matrix, Ridge/XGBoost, SHAP
jupyter notebook notebooks/03_backtest_with_costs.ipynb     # Backtest CÓ transaction cost + slippage
jupyter notebook notebooks/04_single_factor_backtest.ipynb  # Single Factor vs Composite (có cost)
jupyter notebook notebooks/05_xgboost_walkforward_backtest.ipynb  # XGBoost walk-forward (purged) vs Single/Composite
jupyter notebook notebooks/06_engine_v2_rerun.ipynb          # ★ CHẠY CÁI NÀY: engine cũ vs v2, bảng kết quả thật + results_engine_v2.json

# 4. Chạy pipeline đầy đủ + backtest (mặc định CÓ cost) + dashboard
python launcher.py                      # có cost 0.23%/lệnh, tự mở dashboard sau khi xong
python launcher.py --no-cost            # backtest KHÔNG cost (tái tạo baseline Ngày 19)
python launcher.py --no-open            # chỉ tính toán, không tự mở trình duyệt
python launcher.py --compare-legacy     # chạy thêm engine CŨ trên cùng dữ liệu để đo mức Sharpe bị thổi phồng

# 5. Chạy test (look-ahead engine / purge walk-forward / purged K-fold / entry-exit rule / cost model / launcher smoke)
pytest tests/ -v
```

Sau khi chạy `launcher.py`, mở `dashboard/index.html` (hành động hôm nay:
BUY/AVOID/REDUCE/EXIT/NEUTRAL cho từng mã) và `dashboard/backtest.html`
(metrics đầy đủ NET vs GROSS vs 1/N, trade log, exit breakdown) trực tiếp
bằng trình duyệt — không cần server, dữ liệu được nhúng sẵn vào
`dashboard/embed_data.js`.

**Encoding**: toàn bộ file Python/Markdown trong project dùng UTF-8
(`# -*- coding: utf-8 -*-` ở đầu mỗi file `.py`), JSON ghi với
`ensure_ascii=False` để giữ nguyên tiếng Việt có dấu, tránh lỗi mojibake.

## Cấu trúc project

```
momentum-ml-alpha-vn30/
├── README.md
├── requirements.txt
├── daily_report.py                    # ★ báo cáo hằng ngày → Discord (PNG + HTML), chạy bằng GitHub Actions
├── requirements-report.txt            # thư viện tối thiểu cho báo cáo hằng ngày
├── .github/workflows/daily_report.yml # lịch 14:00 + 15:00 giờ VN
├── .github/workflows/report_on_demand.yml # chạy khi có lệnh /report (Worker kích hoạt)
├── worker/                            # Cloudflare Worker nhận slash command (worker.js + test Node)
├── scripts/register_discord_commands.py # đăng ký /report với Discord (1 lần)
├── state/                             # trọng số IC đóng băng, sổ lệnh, nhật ký (workflow tự commit)
├── launcher.py                        # orchestrator: data → factors → composite → backtest → dashboard JSON
├── scripts_bootstrap_old_numbers.py   # sinh dashboard bằng số THẬT từ day19_test.ipynb
├── data/                              # (rỗng — vnstock tải trực tiếp, không cache)
├── notebooks/
│   ├── 01_momentum_factors.ipynb      # Ngày 18: Mom_6_1 vs Mom_12_1, IC, quintile analysis
│   ├── 02_ml_alpha_prediction.ipynb   # Ngày 19: feature matrix, Ridge/XGBoost, SHAP, IC evaluation
│   ├── 03_backtest_with_costs.ipynb   # Ngày 20: backtest CÓ transaction cost + slippage (0.23%/lệnh)
│   ├── 04_single_factor_backtest.ipynb    # Ngày 21: Single Factor Momentum_6_1 vs Composite (có cost)
│   ├── 05_xgboost_walkforward_backtest.ipynb  # Ngày 22: XGBoost walk-forward (purged) vs Single/Composite/1N
│   ├── 06_engine_v2_rerun.ipynb       # ★ v2: chạy lại toàn bộ bằng engine mới + đo Δ so với engine cũ
│   └── legacy_v1_outputs/             # bản notebook 03-05 CŨ còn nguyên output (engine có look-ahead) — chỉ để đối chiếu
├── src/
│   ├── data_loader.py                 # vnstock, filter coverage
│   ├── momentum_factors.py            # Ngày 18: Mom_6_1, Mom_12_1, quintile analysis
│   ├── ic_analysis.py                 # Module IC/IC-IR dùng chung (Purged K-Fold) + composite_score_ic
│   ├── feature_engineering.py         # Ngày 19: cross-sectional rank, feature matrix, label
│   ├── model.py                       # Ngày 19: Linear/Ridge, XGBoost, SHAP + walk_forward_signal (Ngày 22; v2: purge n_fwd)
│   ├── strategy.py                    # Composite score, PositionManager (entry/exit rules)
│   ├── backtest.py                    # ★ ENGINE v2 (độ trễ 1 ngày): backtest_with_exits, invested_returns, port_metrics, sharpe_se
│   ├── backtest_legacy.py             # engine CŨ (look-ahead) — chỉ để đối chứng/đo chênh, KHÔNG dùng cho kết quả
│   └── risk.py                        # Benchmark, compare_strategies (+Sharpe SE), excess_return_tstat
├── reports/
│   └── research_note.md               # Observation/Hypothesis/Evidence/Conclusion, chuẩn Ngày 27
├── tests/
│   ├── test_no_lookahead.py
│   ├── test_purged_kfold.py
│   ├── test_feature_engineering.py
│   ├── test_position_manager.py
│   ├── test_cost_model.py             # Ngày 20: cost_rate=0 backward-compat, cost>0 không cải thiện Sharpe
│   ├── test_composite_ic.py           # Ngày 21: composite_score_ic() đo đúng framework OOS
│   ├── test_walk_forward_signal.py    # Ngày 22: walk_forward_signal() không look-ahead (+ test purge nhãn, v2)
│   ├── test_engine_no_lookahead.py    # ★ v2: engine không ăn return ngày vào lệnh, stop-loss vào PnL, đối chứng engine cũ
│   └── test_launcher_smoke.py         # ★ v2: launcher end-to-end bằng dữ liệu giả lập
└── dashboard/
    ├── index.html                     # Hành động hôm nay: BUY/AVOID/REDUCE/EXIT/NEUTRAL
    ├── backtest.html                  # Metrics NET vs GROSS vs 1/N, trade log, cost info (+ banner phiên bản engine)
    ├── vendor_chart.js                # Chart.js offline (không cần internet)
    ├── embed_data.js                  # dữ liệu nhúng sẵn (sinh bởi launcher.py hoặc bootstrap script)
    └── data/today.json, backtest.json
```
