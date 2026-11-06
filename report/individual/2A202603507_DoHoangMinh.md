# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603507
- **Họ và tên:** Đỗ Hoàng Minh
- **SEED:** 1881

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính
```md
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000    0.817      0.899         0.919   
​2. BoW + MLP (deep learning)         0.990    0.942      0.966         0.947   
3a. Encoder (MiniLM) + kNN           0.863    0.971      0.914         0.974   
3b. Encoder (MiniLM) + LogReg        0.952    0.962      0.957         1.000   
3c. Encoder (MiniLM) + MLP head      0.935    0.971      0.953         0.974   

                                 train (s)   tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.449       18105.355        0.041   
​2. BoW + MLP (deep learning)         9.212       17200.420        0.287   
3a. Encoder (MiniLM) + kNN           2.255         280.631       11.737   
3b. Encoder (MiniLM) + LogReg        2.635        1457.709        6.508   
3c. Encoder (MiniLM) + MLP head      4.491        1338.011        9.646   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.054           0.0  
​2. BoW + MLP (deep learning)          0.354           0.0  
3a. Encoder (MiniLM) + kNN           16.240           0.0  
3b. Encoder (MiniLM) + LogReg        10.693           0.0  
3c. Encoder (MiniLM) + MLP head      11.411           0.0  
- Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
- SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
- Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
- train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

```

## 3. Dự đoán và thực tế

| # | Thí nghiệm | Dự đoán ban đầu | Kết quả thực tế | Nhận xét |
|---|---|---|---|---|
| 1 | BoW + LogReg <br>*(Thí nghiệm 1.A)* | F1 ~ 0.87 <br>(Precision cao hơn Recall) | F1 = 0.899 <br>(Precision = 1.0, Recall = 0.817) | Mô hình kiểm soát tốt lỗi dự đoán nhầm (False Positive), có độ lệch với Kết quả tính tay. |
| 2 | BoW + MLP <br>*(Thí nghiệm 2.A)* | Val loss không giảm mãi; MLP vượt F1 LogReg | Val loss tối ưu ở epoch 6; F1 = 0.966 | Mạng nơ-ron học tốt các đặc trưng phức tạp nhưng có xu hướng quá khớp (overfit) rất nhanh từ các epoch đầu. |

## 4. Phân tích 50 case khó

### 4.1. Model tụt hiệu năng nhiều nhất và nhóm case gây lỗi

- **Model tụt nhiều nhất:** Mô hình **BoW + LogReg** tụt F1 nhiều nhất (từ 0.899 ở tập test xuống mức thấp nhất trong các mô hình dạng vector hóa cơ bản).
- **Nhóm case gây lỗi lớn nhất:** Nhóm tin nhắn chứa **từ ngữ lập dị/viết tắt/cố tình gõ sai chính tả** (như "fr33", "c4sh", "w1n") và các câu có **ngữ cảnh mập mờ/mỉa mai** (ngữ cảnh có từ mang nghĩa quảng cáo/quà tặng nhưng câu thực chất là trò chuyện thông thường).

---

### 4.2. Hai case cụ thể và cơ chế gây sai sót

| STT | Nội dung tin nhắn mẫu | Phân loại | Nguyên nhân & cơ chế khiến mô hình sai |
|---|---|---|---|
| 1 | *"Congrats! U have won a free ticket... Jk call me back."* | **Thực tế:** Ham (Nhầm thành Spam) | **Cơ chế:** Mô hình BoW đánh trọng số rất cao cho các từ độc lập như "Congrats", "won", "free ticket". Do không học được chuỗi văn bản và ngữ cảnh đảo ngược của từ "Jk" (just kidding) ở cuối câu, BoW + LogReg cộng dồn điểm số của các từ khóa quảng cáo và dự đoán nhầm thành Spam. |
| 2 | *"Urgent! Call 090xxx to get your c4sh back now."* | **Thực tế:** Spam (Nhầm thành Ham) | **Cơ chế:** Kẻ phát tin rác dùng kỹ thuật lách chữ ("c4sh" thay cho "cash"). Do từ điển của mô hình Bag-of-Words chỉ ghi nhận từ "cash", từ "c4sh" bị xem là từ chưa xuất hiện trong từ điển (Out-Of-Vocabulary / OOV) và bị gán trọng số 0, dẫn đến mô hình bỏ sót tin nhắn rác (loại F1 do Recall giảm). |

## 5. Thử thách cá nhân


## 6. Phương án cho sản phẩm thật
* **Phương án được chọn:** BoW + LogReg đáp ứng tốt nhất yêu cầu hệ thống thực tế.
* **Tốc độ xử lý:** Độ trễ p95 đạt 0.054 ms/tin, vượt xa yêu cầu (< 50 ms/tin) và xử lý mượt mà 18.105 tin/giây.
* **Chi phí vận hành:** 0.0 USD / 1.000 tin do chạy nhẹ nhàng trên CPU thông thường mà không cần GPU.
* **Chất lượng phân loại:** Đạt Precision tuyệt đối (1.000), loại bỏ hoàn toàn tình trạng gán nhầm tin nhắn quan trọng thành tin rác với F1-score đạt 0.899.
## 7. AI log và điều đã học

