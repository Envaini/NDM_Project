# Course mapping — Paper ↔ Nhận dạng mẫu (PRML/Bishop)

## 1) Bài toán (Pattern Recognition framing)
- Input X: tín hiệu rung (chuỗi thời gian) hoặc ảnh phổ (log-mel).
- Output t: nhãn lớp (healthy + lỗi).
- Mục tiêu: học hàm phân lớp / quy tắc quyết định.

## 2) Mapping từng khối paper sang kiến thức môn

### 2.1 Tiền xử lý & biểu diễn đặc trưng
- Waveform → log-mel spectrogram:
  - Thuộc phần feature extraction / representation learning.
  - Ý nghĩa PR: chọn không gian đặc trưng giúp phân lớp dễ hơn.

### 2.2 Đặc trưng thủ công (hand-crafted) — MBH-LPQ
- LPQ/MBH-LPQ:
  - Thuộc “feature design” (đặc trưng texture, cục bộ).
  - Histogram + concat:
    - Thuộc dạng “bag of local patterns”.
- Liên hệ PRML:
  - Khái niệm “feature vector” và “invariance” (bền với nhiễu/biến đổi nhỏ).
  - Đánh đổi: ổn định cục bộ vs mất thông tin toàn cục.

### 2.3 Đặc trưng deep — VGGish
- CNN embedding:
  - Representation learning: mạng học đặc trưng tự động.
- Liên hệ PRML:
  - Một cách sinh đặc trưng phi tuyến mạnh trước khi phân lớp.

### 2.4 Giảm chiều — PCA
- PCA:
  - Unsupervised dimensionality reduction.
  - Mục tiêu: giảm nhiễu, giảm chiều, giữ phương sai lớn.
- Liên hệ PRML:
  - PCA chapter: eigenvectors/eigenvalues, projection.

### 2.5 Tăng phân biệt — EDA (mở rộng LDA)
- EDA ~ mở rộng LDA:
  - Supervised projection để tăng tách lớp.
- Liên hệ PRML:
  - LDA (Fisher discriminant), within-class / between-class scatter.

### 2.6 Similarity-based decision — Cosine similarity
- Cosine similarity:
  - Thuộc nhóm “distance/similarity metrics”.
  - Liên hệ k-NN / nearest prototype nếu triển khai theo prototype lớp.
- Liên hệ PRML:
  - Decision rule dựa trên khoảng cách (distance-based classification).

### 2.7 Fusion — Weighted Sum (late fusion)
- WS fusion:
  - Late fusion ở mức score (không phải early fusion ở mức feature).
  - Ý nghĩa PR:
    - Ensemble / model combination ở mức quyết định.
- Liên hệ PRML:
  - Kết hợp nhiều nguồn thông tin, trade-off bias/variance.

### 2.8 Đánh giá mô hình
- Accuracy, confusion matrix:
  - Thuộc evaluation metrics.
- Robustness với nhiễu SNR:
  - Thuộc robustness/generalization.
- Time complexity:
  - Thuộc practical constraints.

## 3) “Chốt hiểu” theo môn (mình phải nói được)
- Vì sao cần feature extraction (log-mel).
- Vì sao shallow + deep bổ sung nhau (cục bộ vs mức cao).
- PCA vs LDA/EDA khác gì (unsupervised vs supervised).
- Cosine similarity là rule quyết định kiểu distance-based.
- WS fusion là late fusion theo điểm số.
