# -_- coding: utf-8 -_-

# Research Note — Momentum Factors + ML Alpha + Exit Rules + Cost Model (VN30-universe)

> ## ⚠️ ĐÍNH CHÍNH ENGINE (v2) — đọc trước khi dùng bất kỳ con số Sharpe nào bên dưới
> Các mục **7, 8, 9** (và mọi Sharpe backtest trong các mục khác) được tính bằng **engine cũ có look-ahead cùng ngày**:
> vị thế mở/đóng ở close ngày `t` vẫn ăn/né return của chính ngày `t` (cú giảm kích hoạt stop không vào PnL).
> Walk-forward XGBoost cũ cũng train bằng nhãn 5 ngày chứa giá sau ngày dự đoán (thiếu purge).
> Cả hai đã được sửa (mục 11). **Các kết luận "ĐÃ CHỐT / ĐÃ XÁC NHẬN" ở mục 7-9 vì vậy bị hạ xuống "chưa xác nhận"
> cho tới khi chạy lại bằng `notebooks/06_engine_v2_rerun.ipynb`.** Mục 4-6 (IC, Purged K-Fold, SHAP) không dùng engine nên không bị ảnh hưởng.

## 1. Câu hỏi nghiên cứu

**Ngày 18**: Momentum 6-1 và 12-1 (bỏ tháng gần nhất) có tạo ra pattern
"winner tiếp tục thắng, loser tiếp tục thua" trên thị trường VN (28 mã
blue-chip/ngân hàng) hay không? Thị trường nhỏ, thanh khoản thấp — kết quả
có khác gì so với Mỹ (nơi momentum là anomaly kinh điển)?

**Ngày 19**: Kết hợp nhiều factor qua ML (Ridge, XGBoost) có cải thiện IC
so với factor tốt nhất đứng riêng lẻ không? SHAP có xác nhận feature
importance thông thường của XGBoost hay không?

## 2. Dữ liệu

- Universe: 30 mã → lọc coverage (≥95%) còn **28 mã** (loại TCX, VPL).
- Nguồn: `vnstock` (source KBS).
- Giai đoạn tham chiếu (day19_test.ipynb, đã chạy trên Colab): 2022-01-01 →
  2025-12-31 cho phần factor/IC/backtest; dashboard "hôm nay" chạy live
  ngày 2026-09-09.
- N_FWD = 5 ngày (1 tuần giao dịch) cho label forward return.

## 3. Phương pháp

```
Giá đóng cửa + volume (28 mã)
  ↓
Momentum_6_1, Momentum_12_1, Volume_Ratio, TS_Momentum (Ngày 18)
  ↓
Cross-sectional rank mỗi factor (percentile theo ngày) — feature_engineering.py
  ↓
IC Analysis (Purged K-Fold OOS) — ic_analysis.py (Ngày 18)
  ↓
So sánh: IC-Weighted composite / Ridge / XGBoost (Ngày 19)
  ↓
SHAP TreeExplainer trên XGBoost — xác nhận feature importance thật
  ↓
Composite Score (IC-Weighted) → PositionManager (6 exit rule) → Backtest
  ↓
So sánh với 1/N Equal-Weight benchmark
```

## 4. Kết quả chính — Ngày 18: Momentum Factors

### 4.1 IC từng factor (Purged K-Fold OOS, dữ liệu 2022-2025)

| Factor           | Mean IC    | Std IC | IC-IR |
| ---------------- | ---------- | ------ | ----- |
| **Momentum_6_1** | **0.0524** | 0.2745 | 0.191 |
| Momentum_12_1    | 0.0441     | 0.2942 | 0.149 |
| TS_Momentum      | 0.0335     | 0.2198 | 0.153 |
| Volume_Ratio     | 0.0057     | 0.2183 | 0.026 |

**Quan sát**: Momentum_6_1 có IC cao nhất trong 4 factor — nhất quán với lý
thuyết rằng lookback ngắn hơn (6 tháng) phản ứng nhanh hơn với thay đổi
regime so với 12 tháng, trên thị trường biến động như VN.

### 4.2 Pattern "winner tiếp tục thắng" — kết quả từ TS-Momentum long-short

Thử nghiệm ban đầu dùng **TS-Momentum long-short** (weight theo dấu momentum,
scale theo target volatility) **KHÔNG xác nhận rõ pattern momentum có lợi**:

| Portfolio                              | Ann Return | Ann Vol | Sharpe | Max DD  |
| -------------------------------------- | ---------- | ------- | ------ | ------- |
| TS-Momentum (vol-filtered, long-short) | 3.89%      | 8.01%   | 0.485  | -13.79% |
| 1/N Equal-Weight                       | 11.79%     | 20.91%  | 0.564  | -38.88% |

**Nguyên nhân (Observation → Hypothesis)**: Phân tách long/short contribution
cho thấy: Long-only ann return 18.30%, Short-only ann return 12.39%, EW
benchmark 11.79%. **Giả thiết**: thị trường VN giai đoạn 2022-2025 có
**bull bias** rõ rệt → chân SHORT của long-short strategy kéo hiệu suất
xuống thay vì đóng góp dương (như lý thuyết long-short kỳ vọng ở thị trường
trung lập). Đây là bằng chứng (Evidence) khớp với yêu cầu bài toán: "thị
trường VN nhỏ, thanh khoản thấp — kết quả có thể khác Mỹ."

**Conclusion**: Pattern momentum classical (Q5 thắng liên tục, Q1 thua liên
tục) có tồn tại về mặt IC (IC > 0 cho cả Mom_6_1 và Mom_12_1), nhưng
**không tự động chuyển thành lợi nhuận long-short tốt** trên VN do bull bias
— cần chuyển sang **long-only** để tránh short drag (xem mục 4.3).

### 4.3 Giải pháp: Composite Long-Only (top 35%)

Sau khi xác định vấn đề long-short, thử composite score (IC-weighted 4
factor) + long-only top 35% + rebalance tuần:

| Strategy                      | Total Ret   | Ann Ret    | Ann Vol | Sharpe    | Max DD  |
| ----------------------------- | ----------- | ---------- | ------- | --------- | ------- |
| EW Benchmark                  | 102.27%     | 26.95%     | 17.94%  | 1.502     | -16.81% |
| TS-Mom (original, long-short) | 16.26%      | 5.24%      | 9.26%   | 0.565     | -13.79% |
| Mom6 + EW (top 35%)           | 159.23%     | 38.08%     | 21.35%  | 1.784     | -21.27% |
| **Composite + EW (top 35%)**  | **173.09%** | **40.53%** | 20.86%  | **1.943** | -19.09% |
| Composite + TS (top 35%)      | 136.48%     | 33.85%     | 19.82%  | 1.708     | -19.17% |

→ **Composite + Equal-Weight (top 35%) đạt Sharpe cao nhất (1.9429), vượt
benchmark 1/N (1.5019)** trên giai đoạn in-sample này.

**Lưu ý về monthly rebalance** (thực tế hơn, ít giao dịch hơn): Sharpe giảm
xuống 0.898 (vs EW 1.502) — cho thấy phần lớn alpha của composite strategy
đến từ **rebalance tần suất cao** (tuần), không phải chỉ từ chất lượng
signal. Đây là điểm cần cân nhắc kỹ về turnover/transaction cost thực tế
(chưa được backtest có-cost trong notebook gốc).

## 5. Kết quả chính — Ngày 19: ML Alpha Prediction

### 5.1 So sánh 3 phương pháp kết hợp factor (Purged K-Fold OOS)

⚠️ **Cập nhật sau khi chạy lại với dữ liệu vnstock mới nhất** (2023-01-06 →
2025-12-24, 20571 quan sát) — số liệu dưới đây THAY THẾ bảng ở lần chạy
trước (giai đoạn 2022-2025 cũ). Khác biệt giữa 2 lần chạy là **bình thường**:
mỗi lần chạy `load_ohlcv()` lấy dữ liệu tính đến thời điểm hiện tại, nên
giai đoạn ước lượng khác nhau → IC đo được khác nhau. Đây không phải lỗi,
mà là bản chất của out-of-sample re-estimation.

| Method                                        | Mean IC                                                                                   | IC Std | IC>0% | IC-IR  | RMSE   |
| --------------------------------------------- | ----------------------------------------------------------------------------------------- | ------ | ----- | ------ | ------ |
| Best single factor (Momentum_6_1, đứng riêng) | 0.0501                                                                                    | —      | —     | —      | —      |
| **IC-Weighted composite**                     | _(chưa tính lại số cụ thể ở lần chạy 2023-2025 — xem Cell 6 của notebook 02 đã cập nhật)_ | —      | —     | —      | —      |
| Ridge (all features)                          | 0.0171                                                                                    | 0.2688 | 51.9% | 0.0637 | 0.0424 |
| XGBoost (all features)                        | 0.0336                                                                                    | 0.2119 | 57.2% | 0.1585 | 0.0426 |

**Kết quả xác nhận từ người dùng (chạy lại thực tế)**: Ridge và XGBoost đều
**THUA** Momentum_6_1 đứng riêng (0.0171 và 0.0336 đều < 0.0501) — nhất quán
với lần chạy trước. **Điểm khác biệt quan trọng với kết luận trước đây**:
người dùng xác nhận **chỉ Ridge/XGBoost thua single factor, KHÔNG có bằng
chứng cho thấy IC-Weighted composite cũng thua** — notebook 02 (bản gốc
trước khi cập nhật) **không tính lại IC-Weighted composite** để so sánh
trực tiếp trong cùng 1 bảng, nên kết luận "composite thắng" ở báo cáo cũ dựa
trên số liệu của MỘT lần chạy khác (giai đoạn 2022-2025), không phải cùng
lần chạy đang xét (2023-2025).

**Đã sửa**: `notebooks/02_ml_alpha_prediction.ipynb` mục 6 giờ tính
`composite_score_ic()` (hàm mới trong `src/ic_analysis.py`) ngay trong cùng
1 bảng so sánh với best-single-factor/Ridge/XGBoost — dùng ĐÚNG framework
Purged K-Fold OOS để so sánh công bằng. Chạy lại notebook 02 để có con số
IC-Weighted composite chính xác cho giai đoạn dữ liệu hiện tại, thay vì
suy luận/ngoại suy từ lần chạy trước.

⚠️ **Lưu ý về tính công bằng của phép so sánh** (đã thêm vào docstring
`composite_score_ic()`): IC-Weighted composite dùng `ic_weights` CỐ ĐỊNH
(tính 1 lần từ `single_factor_ic()` trên toàn dataset), không "refit" riêng
trên mỗi fold train như Ridge/XGBoost phải làm. Điều này cho composite 1 LỢI
THẾ CẤU TRÚC nhẹ (ít bậc tự do hơn để overfit trên universe nhỏ 28 mã) — nên
nếu composite thắng, đó một phần đến từ việc nó đơn giản hơn, không chỉ vì
nó "thông minh hơn" Ridge/XGBoost.

**Sharpe tốt nhất (backtest, Ngày 20)**: composite IC-Weighted (dùng trong
`launcher.py`) đạt Sharpe GROSS 2.627, NET (có cost) 2.377 — xem mục 7.

### 5.2 Ridge Coefficients vs IC Weights

Ridge coefficient cho Momentum_12_1 gần như bằng 0 (0.0018 ở lần chạy mới,
tương tự lần chạy trước) dù IC weight riêng lẻ là ~0.32 — dấu hiệu **đa
cộng tuyến** giữa Momentum_6_1 và Momentum_12_1 (2 factor có Pearson corr
trung bình 0.72 đo được ở Ngày 18, lần chạy 2023-2025). Ridge "chọn" dồn
phần lớn trọng số vào 1 trong 2 factor tương quan cao, khiến coefficient
của factor còn lại bị triệt tiêu dù bản thân nó có IC dương khi đứng riêng.

### 5.3 Khuyến nghị theo Occam's Razor (từ notebook gốc)

> Universe nhỏ (28 mã): Ridge thường ổn định hơn XGBoost. XGBoost cần
> ≥100 mã để tránh overfit. Nếu Ridge ≈ XGBoost về IC → dùng Ridge (đơn giản
> hơn). Chỉ nâng cấp lên XGBoost khi IC cải thiện > 20%.

Ở lần chạy mới: XGBoost Mean IC (0.0336) vượt Ridge (0.0171) khoảng ~96%,
vẫn vượt ngưỡng 20% — XGBoost có lý do để cân nhắc hơn Ridge theo tiêu chí
IC riêng. Nhưng **cả 2 đều thua Momentum_6_1 đứng riêng** (0.0501) — kết
luận thực dụng KHÔNG đổi: chưa có bằng chứng đủ mạnh để ưu tiên ML
(Ridge/XGBoost) trên universe 28 mã, dù XGBoost nhỉnh hơn Ridge.

## 6. Diễn giải theo Observation / Hypothesis / Evidence / Conclusion

- **Observation**: Ridge và XGBoost đều thua Momentum_6_1 đứng riêng trên CẢ
  2 lần chạy (giai đoạn 2022-2025 và 2023-2025) — kết quả ỔN ĐỊNH qua thời
  gian, không phải ngẫu nhiên 1 lần. IC-Weighted composite CẦN được tính lại
  trong cùng khung so sánh để xác nhận có tiếp tục thắng single factor hay
  không (xem mục 5.1) — đây là việc launcher.py ĐANG dùng, nên quan trọng
  phải verify định kỳ, không giả định mãi đúng.

- **Hypothesis A**: Bull bias của thị trường VN là nguyên nhân chính khiến
  long-short kém hiệu quả, KHÔNG PHẢI momentum factor tự nó yếu.

- **Hypothesis B**: Universe nhỏ (28 mã) hạn chế khả năng ML (Ridge/XGBoost)
  tận dụng interaction/non-linearity — cần universe lớn hơn (VN100+) để ML
  thực sự vượt trội linear combination đơn giản.

- **Evidence**: Ridge và XGBoost K-fold có Mean IC thấp hơn Momentum_6_1 ở
  các lần đo trước. Tuy nhiên, notebook 05 (engine cũ) cho thấy XGBoost Walk-Forward có
  Sharpe 2.6765, cao hơn Single Factor 2.6432 nhưng thấp hơn Composite
  2.7144; vì vậy không thể kết luận XGBoost luôn thua khi chỉ nhìn IC.

- **Conclusion**: Cả 2 hypothesis đều có bằng chứng ủng hộ và **không loại
  trừ lẫn nhau**. Composite vẫn là lựa chọn production vì đạt Sharpe cao nhất
  trong notebook 04 và 05, đồng thời turnover thấp hơn XGBoost. XGBoost có
  kết quả walk-forward tốt và vượt Single Factor về Sharpe, nhưng chi phí
  giao dịch cao hơn nhiều. Tất cả kết quả vẫn là nghiên cứu lịch sử/in-sample
  và cần được xác nhận bằng một giai đoạn out-of-sample độc lập.

## 7. Backtest với Exit Rules (PositionManager) — ⚠️ SỐ CỦA ENGINE CŨ

| Metric       | With Exits (GROSS, không cost) | EW Benchmark |
| ------------ | ------------------------------ | ------------ |
| Ann Return   | 36.67%                         | 11.79%       |
| Ann Vol      | 17.39%                         | 20.91%       |
| **Sharpe**   | **2.109**                      | 0.564        |
| Max DD       | -17.64%                        | -38.88%      |
| Total Return | 244.20%                        | 55.37%       |

**Trade statistics**: 186 lệnh mua / 182 lệnh bán, Win rate 54.9%, Avg win
+15.21% / Avg loss -5.66%, Avg hold 53 ngày.

**Exit breakdown**: MOM_FLIP (76) > TRAILING_STOP (41) > HARD_STOP (38) >
SIGNAL_EXIT (21) > VOL_REDUCE (6) — momentum đảo chiều là lý do exit phổ
biến nhất, xác nhận vai trò trung tâm của Momentum_6_1 trong composite score.

## 8. Ngày 20 — Cost Model (transaction cost + slippage) — ⚠️ SỐ CỦA ENGINE CŨ

**Đã hoàn thành VÀ đã chạy với dữ liệu vnstock thật** (2022-01-01 →
2025-12-31, universe 28 mã): `src/backtest.py::backtest_with_exits` nhận
tham số `cost_rate`, áp dụng phí trên notional giao dịch mỗi lần BUY/SELL.
Giữ NGUYÊN giả định cost đã dùng ở project `momentum-ridge-vn30`:

```
TRANSACTION_COST = 0.15%   # round-trip
SLIPPAGE         = 0.08%   # ước lượng bid-ask spread
TOTAL_COST_RATE  = 0.23%   # tổng mỗi lần đổi vị thế
```

### Kết quả THẬT (`notebooks/03_backtest_with_costs.ipynb`, đã chạy xong)

| Metric       | GROSS (không cost) | NET (có cost 0.23%/lệnh) | 1/N Benchmark |
| ------------ | ------------------ | ------------------------ | ------------- |
| Total Return | 248.51%            | 217.20%                  | 102.10%       |
| Ann Return   | 52.81%             | 48.00%                   | 26.99%        |
| Ann Vol      | 20.10%             | 20.19%                   | 17.97%        |
| **Sharpe**   | **2.627**          | **2.377**                | 1.502         |
| Max DD       | -17.64%            | -18.27%                  | -16.81%       |
| Calmar       | 2.994              | 2.628                    | 1.606         |

**374 lệnh giao dịch** (189 buy + 185 sell) trong 47 tháng (~7.9 lệnh/tháng).
**Tổng cost đã trả: 9.37% NAV** tích lũy toàn bộ giai đoạn backtest (~0.197%/
tháng). Sharpe giảm 0.250 điểm do cost (9.5% so với baseline GROSS).

**KẾT LUẬN CŨ (engine cũ — cần xác nhận lại bằng engine v2)**: Sau khi trừ transaction
cost + slippage, chiến lược Momentum+ExitRules **VẪN THẮNG** 1/N benchmark
rõ ràng (Sharpe NET 2.377 so với 1.502 của 1/N) — cost model đã lấp đúng
khoảng trống lớn nhất từng nêu ở các phiên bản trước của research note này.

**Lưu ý quan trọng**: đây là kết quả từ lần chạy dùng `comp_score` (IC-
Weighted composite), KHÔNG PHẢI Ridge hay XGBoost — khớp với thiết kế của
`launcher.py`. Composite score ở đây dùng `ic_weights` tính từ Mean IC của
4 factor: Momentum_6_1 (0.3870), Momentum_12_1 (0.3247), TS_Momentum
(0.2467), Volume_Ratio (0.0415) — đo trên chính giai đoạn 2022-2025 này.

## 9. Backtest trực tiếp Single Factor và XGBoost — ⚠️ SỐ CỦA ENGINE CŨ (kết luận cũ, CHƯA xác nhận lại)

### 9.1: Single Factor vs Composite (`notebooks/04_single_factor_backtest.ipynb`)

Đo IC riêng lẻ nhiều lần cho kết quả LẪN LỘN — có lần Momentum_6_1 đứng
riêng có IC cao hơn composite (0.0501 vs 0.0466), có lần composite cao hơn
(0.0572 vs 0.0525). Vì IC không nhất quán chỉ ra 1 phương án rõ ràng, cần
**backtest trực tiếp** (có cost, cùng entry/exit rules) để quyết định.

**Kết quả backtest có cost (lần chạy xác nhận cuối cùng)**:

| Method                       | Mean IC    | Sharpe (có cost) |
| ---------------------------- | ---------- | ---------------- |
| Single Factor (Momentum_6_1) | 0.0525     | 2.1678           |
| **IC-Weighted Composite**    | **0.0572** | **2.3773** 🏆    |
| 1/N Equal-Weight             | —          | 1.5023           |

**Composite cao hơn** ở backtest cũ (Sharpe 2.3773 vs 2.1678) — chênh 0.21 so với Sharpe SE ≈ 0.6 thì **không phân biệt được với nhiễu**, và đo bằng engine cũ.
**Quyết định giữ composite trong `launcher.py` là tạm thời**, chờ kết quả engine v2 (mục 11).

**Bài học phương pháp luận**: đây là minh chứng thực nghiệm cho nguyên tắc
"IC không phải toàn bộ câu chuyện" — Mean IC đo trung bình khả năng xếp
hạng trên TOÀN DẢI, còn Sharpe backtest phụ thuộc thêm vào cách signal
tương tác với entry/exit rules cụ thể của `PositionManager` (chỉ quan tâm
top/bottom 35%, không phải toàn bộ phân phối cross-sectional). Một signal
có IC nhỉnh hơn ở vùng giữa phân phối không nhất thiết tạo ra Sharpe cao
hơn nếu nó KÉM hơn đúng ở vùng cực trị (top/bottom 35%) mà chiến lược thực
sự giao dịch.

### 9.2: XGBoost Walk-Forward (`notebooks/05_xgboost_walkforward_backtest.ipynb`)

Notebook 05 đã chạy walk-forward trên dữ liệu thật. Signal được refit mỗi 21
phiên, chỉ dùng dữ liệu quá khứ và cần tối thiểu 252 phiên warm-up. Do đó,
không nên so sánh Total Return giữa các notebook nếu chưa kiểm tra cùng thời
gian hiệu lực.

| Strategy                     |    Mean IC | Total Return | Ann Return |    Ann Vol |     Sharpe |      Max DD | Calmar |
| ---------------------------- | ---------: | -----------: | ---------: | ---------: | ---------: | ----------: | -----: |
| XGBoost Walk-Forward         |     0.0391 |      145.52% |     59.17% |     22.11% |     2.6765 |     -14.23% | 4.1574 |
| Single Factor (Momentum_6_1) |     0.0525 |      132.88% |     54.87% |     20.76% |     2.6432 |     -15.30% | 3.5869 |
| **IC-Weighted Composite**    | **0.0572** |  **142.18%** | **58.04%** | **21.38%** | **2.7144** | **-18.27%** | 3.1776 |
| 1/N Equal-Weight             |          — |       73.20% |     32.87% |     18.24% |     1.8023 |     -16.75% | 1.9624 |

Các kết quả trên đều có transaction cost + slippage 0.23%/lệnh. Composite có
Sharpe cao nhất; XGBoost đứng thứ hai và vẫn vượt Single Factor ở Sharpe, dù
Mean IC thấp hơn cả hai. IC thấp hơn không hoàn toàn quyết định Sharpe.

XGBoost giao dịch thường xuyên hơn đáng kể:

| Strategy             | Buys | Sells | Win rate |   Avg hold | Total cost paid |
| -------------------- | ---: | ----: | -------: | ---------: | --------------: |
| XGBoost Walk-Forward |  307 |   303 |   56.11% | 16.40 ngày |          18.53% |
| Single Factor        |  209 |   203 |   57.14% | 51.34 ngày |          12.20% |
| Composite            |  189 |   185 |   55.14% | 52.35 ngày |           9.37% |

Exit của XGBoost chủ yếu là `SIGNAL_EXIT` (229), trong khi Composite chủ yếu
là `MOM_FLIP` (77) và `TRAILING_STOP` (42).

**Kết luận (engine cũ, chưa xác nhận lại)**: giữ tạm Composite trong `launcher.py`. Composite có
Sharpe cao nhất (nhưng chênh lệch giữa 3 chiến lược nhỏ hơn Sharpe SE) và ít turnover hơn; XGBoost là benchmark nghiên cứu đáng chú
ý, nhưng cần kiểm định out-of-sample và market impact trước khi triển khai.

## 10. Đề xuất bước tiếp theo

1. Theo dõi định kỳ: Composite thắng ở notebook 04 và tiếp tục có Sharpe cao
   nhất trong so sánh walk-forward của notebook 05.
2. Đánh giá XGBoost theo turnover và market impact, không chỉ theo Sharpe.
3. Thực hiện out-of-sample test riêng biệt vì tham số PositionManager vẫn
   được đánh giá trên cùng lịch sử nghiên cứu.
4. Mở rộng universe để kiểm tra Hypothesis B trên VN100 hoặc lớn hơn.
5. Verify lại SHAP và IC khi dữ liệu vnstock được cập nhật đáng kể.
6. Tinh chỉnh cost model theo phí mua/bán, thuế bán và market impact.

## 11. Engine v2 — sửa lỗi look-ahead và đo lại

### 11.1 Lỗi đã phát hiện
1. **Look-ahead cùng ngày trong `backtest_with_exits` (cũ).** Trong vòng lặp ngày `t`: exit được quyết định bằng giá `t` và vị thế bị gỡ *trước* khi
   tính return `t`; entry được mở bằng rank `t` rồi *vẫn ăn* return `t`. Hệ quả: cú giảm kích hoạt stop không bao giờ vào PnL; vị thế mới hưởng
   return của ngày chưa thể giao dịch. Lỗi được chứng minh bằng test tất định (`tests/test_engine_no_lookahead.py`), không cần dữ liệu thật.
2. **Thiếu purge nhãn trong `walk_forward_signal` (cũ).** Train bằng mọi ngày `< t`, nhưng nhãn `fwd_return` của 5 ngày cuối trước `t` dùng giá tới `t+4`.
   Test perturbation (xáo nhãn tương lai → signal quá khứ phải bất biến) bắt được lỗi ở `n_fwd=0` và qua ở `n_fwd=5`.

### 11.2 Quy ước mới (src/backtest.py, src/model.py)
- Signal/rank tại close `t` chỉ dùng thông tin đến hết `t`; lệnh khớp ở close `t`; **vị thế mới chỉ ăn return từ `t+1`**;
  vị thế bị đóng ở `t` **vẫn ăn return của `t`** (kể cả cú giảm kích hoạt stop). Cost 0.23% giữ nguyên, trừ vào ngày khớp lệnh.
- Walk-forward: train chỉ gồm ngày `≤ t − N_FWD − 1`.
- Cắt warm-up bằng `invested_returns()` (theo ngày có vị thế thật) thay cho `ret[ret != 0]` (lọc theo giá trị).
- Thêm `Sharpe SE` và `excess_return_tstat` để biết chênh lệch có vượt nhiễu hay không.

### 11.3 Mức độ ảnh hưởng — đo trên DỮ LIỆU GIẢ LẬP (không phải kết quả thị trường)
Tín hiệu ngẫu nhiên, giá không có alpha (return số học kỳ vọng 0), 50 seed × 700 ngày × 28 mã, cost = 0:

| Tham số exit | Sharpe v2 (≈ 0 là đúng) | Sharpe engine cũ | Chênh cũ − v2 |
| --- | --- | --- | --- |
| Mặc định (stop 8%, trail 12%) | −0.05 | +0.13 | **+0.18 ± 0.08** (~2 SE) |
| Stop chặt (3% / 5%) | −0.09 | +1.93 | **+2.02 ± 0.08** |

→ Độ lệch **phụ thuộc vào tần suất stop kích hoạt**. Với tham số mặc định trên tín hiệu ngẫu nhiên, lỗi chỉ giải thích ~0.1–0.2 Sharpe nên **không đủ để
giải thích riêng Sharpe 2.6–2.7 cũ**; nhưng với dữ liệu thật (xu hướng mạnh, stop kích hoạt nhiều hơn) mức chênh có thể lớn hơn — **chỉ chạy lại mới biết**.

### 11.4 Kết quả engine v2 trên dữ liệu thật
⏳ **CHƯA CHẠY** (môi trường tạo bản v2 không truy cập được vnstock). Chạy `notebooks/06_engine_v2_rerun.ipynb` (hoặc `python launcher.py --compare-legacy`);
notebook sinh `reports/results_engine_v2.json` + bảng so sánh engine cũ/v2 trên cùng dữ liệu. Điền kết quả vào README mục "Kết quả engine v2" và cập nhật lại mục 9-10.

### 11.5 Hạn chế còn lại (chưa sửa trong bản này)
- **Trọng số IC của Composite tính trên toàn mẫu** (`single_factor_ic` trên toàn dataset) rồi dùng backtest chính mẫu đó → có thiên lệch in-sample nhỏ.
  Cách sửa đúng: tính trọng số expanding-window (chỉ dùng IC tới `t − N_FWD − 1`).
- **Tham số PositionManager cố định**, chưa có giai đoạn out-of-sample tách biệt.
- **Khớp lệnh tại close** ngày tín hiệu (không T+2, không giới hạn thanh khoản/market impact); cost cố định 0.23%.
- **Benchmark 1/N** rebalance hằng ngày và không cost, trong khi chiến lược chịu cost → hơi có lợi cho 1/N (thận trọng), không phải lợi cho chiến lược.
- Mục 4.3 (Composite long-only top 35% rebalance tuần, từ notebook 01) dùng một backtest riêng — **chưa kiểm tra quy ước thời gian** của nó; coi là chưa xác nhận.
