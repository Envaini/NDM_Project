# Phase 13 — Sổ tay toán + giải thích (NDM Project)

Tài liệu này tóm tắt các công thức/khái niệm toán dùng trong pipeline: log-mel, LPQ/MBH-LPQ, PCA+EDA, cosine matching, weighted-sum fusion, SNR, và các đại lượng đánh giá (loss/gradient ở mức khái niệm).

Quy ước:
- Vector cột ký hiệu đậm (ví dụ **x**), chuẩn Euclid: ||x||2.
- Cosine similarity: cos(a,b) = (a·b)/(||a|| ||b||).
- Khi nói “nguồn”: ưu tiên Paper (MBH-LPQ+VGGish fusion), PRML (Bishop), và doc chuẩn (librosa/torchaudio/scipy).

---

## 1) Loss và Gradient (khung tổng quát)

### 1.1 Loss (hàm mất mát)
**Định nghĩa:**
Loss L(θ) đo mức “sai” giữa dự đoán và nhãn, dùng để tối ưu tham số θ.

**Ví dụ hay gặp (classification):** Cross-entropy
\[
L = -\sum_{k=1}^{K} y_k \log(\hat{p}_k)
\]
với y là one-hot, \hat{p} là xác suất dự đoán.

**Ý nghĩa:**
Loss nhỏ hơn → mô hình phù hợp dữ liệu hơn.

**Nguồn:**
- Bishop (PRML) — phần logistic regression / neural networks.

**Câu nói miệng (vấn đáp):**
- “Loss là thước đo sai số; tối ưu là tìm θ để loss nhỏ nhất.”
- “Cross-entropy phạt mạnh những dự đoán tự tin nhưng sai.”

### 1.2 Gradient (đạo hàm theo tham số)
**Định nghĩa:**
Gradient ∇θ L là vector đạo hàm riêng của loss theo từng tham số θ.

**Cập nhật gradient descent:**
\[
\theta \leftarrow \theta - \eta \nabla_{\theta} L
\]
với η là learning rate.

**Ý nghĩa:**
Gradient chỉ hướng tăng nhanh nhất của loss → trừ gradient để đi xuống.

**Nguồn:**
- Bishop (PRML).

**Câu nói miệng:**
- “Gradient là ‘kim chỉ nam’ để biết chỉnh tham số theo hướng làm loss giảm.”

---

## 2) STFT và Mel / Log-mel

### 2.1 STFT (Short-Time Fourier Transform)
**Định nghĩa:**
Chia tín hiệu x[n] thành các cửa sổ ngắn rồi Fourier từng cửa sổ:
\[
X(m,\omega)=\sum_{n} x[n]\, w[n-m]\, e^{-j\omega n}
\]
m là chỉ số frame, w là cửa sổ (Hamming/Hann).

**Ý nghĩa:**
Biến tín hiệu 1D theo thời gian → biểu diễn thời gian–tần số.

**Nguồn:**
- Tài liệu DSP cơ bản / doc scipy.signal.stft (tham khảo).

**Câu nói miệng:**
- “STFT nhìn phổ theo từng lát thời gian, hợp cho tín hiệu không dừng.”

### 2.2 Mel filterbank và log-mel spectrogram
**Định nghĩa (mel-scale):**
Mel biến đổi tần số để gần cảm nhận của tai người:
\[
m = 2595 \log_{10}\left(1+\frac{f}{700}\right)
\]
Mel-spectrogram: áp các bộ lọc mel lên |STFT|^2 rồi lấy log.

**Log-mel:**
\[
\text{logmel} = 10\log_{10}(S_{\text{mel}} + \epsilon)
\]

**Ý nghĩa:**
- Nén động (log) + gom băng tần theo mel → ổn định hơn cho feature.

**Nguồn:**
- Librosa docs (melspectrogram, power_to_db) / Paper dùng log-mel.

**Câu nói miệng:**
- “Log-mel giúp đặc trưng bền hơn trước biến thiên biên độ và noise.”

---

## 3) LPQ và MBH-LPQ

### 3.1 LPQ (Local Phase Quantization)
**Định nghĩa (ý tưởng):**
Lấy pha cục bộ trong một lân cận quanh mỗi điểm ảnh bằng STFT/DFT cục bộ tại vài tần số mẫu u1..u4.
Tách phần thực/ảo rồi lượng tử theo dấu (>,0) tạo mã nhị phân.

**Mã 8-bit (khái niệm):**
- 4 điểm tần × (Re, Im) → 8 bit → mã 0..255.

**Ý nghĩa:**
LPQ giữ thông tin pha cục bộ → thường bền với mờ (blur) và biến thiên nhẹ.

**Nguồn:**
- Paper: phần mô tả LPQ + hình minh hoạ.

**Câu nói miệng:**
- “LPQ biến lân cận thành một mã 8-bit dựa trên dấu của Re/Im tại vài tần số.”

### 3.2 MBH-LPQ (Multi-Block Histogram LPQ)
**Định nghĩa:**
- Từ ảnh LPQ code (0..255), chia ảnh thành b khối (sub-blocks).
- Mỗi khối tạo histogram 256-bin, chuẩn hoá.
- Nối các histogram thành vector:
\[
\mathbf{h} = [H_1, H_2, \dots, H_b] \in \mathbb{R}^{256b}
\]

**Ý nghĩa:**
- Histogram theo block giữ cấu trúc thô theo không gian.
- Tăng mạnh so với LPQ “1 histogram toàn ảnh” vì giữ thông tin vị trí.

**Nguồn:**
- Paper: Algorithm MBH-LPQ + Table khảo sát b.

**Câu nói miệng:**
- “MBH-LPQ là LPQ + chia block + histogram từng block để giữ thêm thông tin không gian.”

---

## 4) PCA và EDA

### 4.1 PCA (Principal Component Analysis)
**Định nghĩa:**
PCA tìm trục chiếu tối đa phương sai dữ liệu.
Cho dữ liệu đã chuẩn hoá X, PCA tìm W sao cho:
\[
Z = XW
\]
và phương sai của Z là lớn nhất, W gồm các eigenvector ứng với eigenvalue lớn.

**Ý nghĩa:**
Giảm chiều, giảm nhiễu, giúp bước sau (EDA/LDA-like) ổn định hơn.

**Nguồn:**
- Bishop (PRML) — chương PCA.

**Câu nói miệng:**
- “PCA giữ hướng biến thiên lớn nhất để nén dữ liệu xuống ít chiều.”

### 4.2 EDA (Exponential Discriminant Analysis)
**Định nghĩa (ý tưởng):**
Tương tự LDA: dùng scatter trong-lớp Sw và giữa-lớp Sb.
EDA dùng biến đổi ma trận mũ để tăng tính phân biệt:
Giải bài toán trị riêng tổng quát dạng:
\[
\exp(S_b)\, v = \lambda\, \exp(S_w)\, v
\]

**Sw, Sb:**
- Sw: tổng phân tán trong lớp
- Sb: phân tán giữa các lớp

**Ý nghĩa:**
Tạo không gian chiếu phân biệt lớp tốt hơn, đặc biệt khi phân bố không “đẹp” tuyến tính.

**Nguồn:**
- Paper: phần PCA + EDA.

**Câu nói miệng:**
- “EDA giống LDA nhưng dùng expm(S) để nhấn mạnh cấu trúc phân biệt.”

---

## 5) Cosine similarity và Prototype matching

### 5.1 Cosine similarity
**Định nghĩa:**
\[
\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|}
\]

**Ý nghĩa:**
Đo “góc” giữa 2 vector → ít nhạy với scale biên độ.

**Nguồn:**
- Paper: cosine matching equation.

**Câu nói miệng:**
- “Cosine chỉ quan tâm hướng vector, không quan tâm độ lớn.”

### 5.2 Prototype classifier (trung bình theo lớp)
**Định nghĩa:**
Prototype lớp c:
\[
\mathbf{p}_c = \frac{1}{N_c}\sum_{i:y_i=c} \frac{\mathbf{x}_i}{\|\mathbf{x}_i\|}
\]
Dự đoán:
\[
\hat{y} = \arg\max_c \cos(\mathbf{x}, \mathbf{p}_c)
\]

**Ý nghĩa:**
Classifier đơn giản, nhanh, hợp khi feature đã “tách lớp” tốt.

**Nguồn:**
- Paper: matching dùng cosine.

**Câu nói miệng:**
- “Mỗi lớp đại diện bằng 1 vector trung bình; test chọn lớp có cosine lớn nhất.”

---

## 6) Weighted Sum (WS) fusion

### 6.1 WS ở mức điểm số (score-level fusion)
**Định nghĩa:**
Với điểm số s_i từ nhánh i, WS:
\[
S = \sum_{i=1}^{K} w_i s_i,\quad \sum_i w_i = 1
\]
Trong project: thường K=2 (deep VGGish và shallow MBH-LPQ):
\[
S = \alpha S_{\text{deep}} + (1-\alpha) S_{\text{shallow}}
\]

**Ý nghĩa:**
Trộn ưu điểm:
- deep: đặc trưng mức cao
- shallow: texture cục bộ
Khi domain đổi, hai nhánh có thể bù nhau.

**Nguồn:**
- ** Paper “deep–shallow features fusion” — weighted sum (Eq.20).

**Câu nói miệng:**
- “WS là trộn điểm số hai nhánh theo trọng số; chọn α tốt nhất bằng sweep.”

---

## 7) SNR (Signal-to-Noise Ratio)

### 7.1 Định nghĩa SNR (dB)
\[
\text{SNR(dB)} = 10\log_{10}\left(\frac{P_s}{P_n}\right)
\]
Ps: công suất tín hiệu, Pn: công suất nhiễu.

**Ý nghĩa:**
SNR cao → ít nhiễu; SNR thấp → nhiễu mạnh.

**Nguồn:**
- Paper: Paper “deep–shallow features fusion” — SNR (Eq.21).


**Câu nói miệng:**
- “SNR đo tỉ lệ năng lượng tín hiệu so với nhiễu theo dB.”

---

## 8) Ghi chú liên hệ trực tiếp với code của đồ án (Phase 10–12)
- log-mel: tạo ảnh (T × 64) rồi pad/crop về 96 frame → (96×64).
- LPQ: tạo mã 0..255 từ dấu (Re/Im) tại 4 tần số mẫu.
- MBH-LPQ: chia b block → histogram 256-bin từng block → vector 256b.
- PCA+EDA: giảm chiều rồi chiếu phân biệt lớp.
- Cosine + prototype: tính cosine similarity đến vector đại diện mỗi lớp.
- WS fusion: trộn score deep/shallow theo α.

(End)
