# Paper summary — Deep/Shallow feature fusion for bearing fault diagnosis (SciRep 2025)

## 1) Vấn đề & mục tiêu
- Bài toán: chẩn đoán lỗi vòng bi từ tín hiệu rung.
- Mục tiêu: phân lớp tình trạng máy (healthy + các loại lỗi) với độ chính xác cao, bền với nhiễu, thời gian chạy hợp lý.

## 2) Đóng góp chính (theo paper)
- Kết hợp 2 nhánh đặc trưng:
  - Deep: VGGish (embedding mức cao).
  - Shallow: MBH-LPQ (texture cục bộ từ log-mel).
- Giảm chiều + tăng phân biệt: PCA rồi EDA (mở rộng của LDA).
- So khớp bằng cosine similarity.
- Hợp nhất điểm bằng Weighted Sum (WS), thường tối ưu quanh tỉ lệ 0.8 (deep) và 0.2 (shallow).
- Có kiểm tra robustness với nhiễu Gaussian theo SNR và đo thời gian tính.

## 3) Pipeline (sơ đồ chữ) — BẮT BUỘC
Waveform
→ segmentation (cắt mẫu)
→ log-mel spectrogram
→ (nhánh A) VGGish → vector V_deep
→ (nhánh B) MBH-LPQ: LPQ(R) → chia b blocks → histogram 256-bin/block → concat → vector V_shallow
→ PCA
→ EDA
→ cosine similarity (ra score theo lớp)
→ Weighted Sum fusion: score = w_deep*score_deep + w_shallow*score_shallow
→ predicted label
→ evaluation: accuracy + confusion matrix + test nhiễu Gaussian theo SNR + time

## 4) Thành phần kỹ thuật (đủ để bạn “kể lại”)
### 4.1 Log-mel spectrogram
- Input: waveform 1D
- Output: ảnh 2D thời gian–tần số (log-mel)

### 4.2 Nhánh deep — VGGish
- Input: log-mel
- Output: embedding vector (paper dùng các tầng/điểm trích khác nhau như fc1_1/fc1_2/fc2 tùy thí nghiệm)

### 4.3 Nhánh shallow — MBH-LPQ
- LPQ: tính STFT cục bộ, lấy phần thực/ảo tại 4 điểm tần số, lượng tử dấu thành 8-bit code (0..255).
- MBH-LPQ: chia ảnh LPQ thành b blocks, mỗi block tạo histogram 256-bin, rồi nối lại thành vector b×256.
- Paper có quét R ∈ {3,5,7} và quét số block b (1→12) để chọn cấu hình tốt.

### 4.4 PCA rồi EDA
- PCA: giảm chiều, giữ thông tin chính.
- EDA: tăng phân biệt giữa lớp (dạng mở rộng LDA, dùng Sb/Sw).

### 4.5 Matching + Fusion
- Cosine similarity (công thức cosine similarity).
- Weighted Sum (WS) để gộp điểm hai nhánh.
- Tỉ lệ WS tối ưu thường quanh: deep 0.8, shallow 0.2 (theo paper).

### 4.6 Robustness với nhiễu (SNR)
- Paper thêm nhiễu Gaussian, đánh giá theo các mức SNR (ví dụ -6, -3, 0, 3, 6 dB).
- Kết luận xu hướng: fusion thường bền hơn từng nhánh riêng.

### 4.7 Time complexity
- Paper báo thời gian chạy (MBH-LPQ nhanh hơn VGGish; fusion chậm hơn VGGish một chút vì thêm nhánh shallow).

## 5) Dataset & protocol (theo paper)
- Paper đánh giá trên các bộ phổ biến như CWRU và PU (Paderborn có thể xuất hiện ở phần thảo luận).
- Thiết lập chia train/test dạng 80/20 và báo confusion matrix theo 10 lớp (tùy dataset).

Ghi chú triển khai đồ án của mình:
- Dự án của mình đang dùng CWRU (fs=48k, load=1HP, 10 classes) trong data/raw/CWRU.
- Mục tiêu reproduce trước: pipeline + baseline (log-mel → VGGish) và (log-mel → MBH-LPQ), sau đó fusion.

## 6) Kết quả & ablation (cái gì paper chứng minh)
- MBH-LPQ tốt hơn basic LPQ rõ rệt.
- VGGish mạnh nhất trong các deep baseline paper thử.
- Fusion WS (VGGish + MBH-LPQ) đạt tốt nhất hoặc gần tốt nhất.
- Quét b (số block) để tìm điểm cân bằng.
- Quét tỉ lệ WS để tìm deep/shallow ratio tối ưu.

## 7) Reproduce plan (Phase sau mình sẽ làm gì)
- Phase 3/4: dựng pipeline chạy được trên CWRU.
- Trích log-mel chuẩn.
- Làm nhánh deep (VGGish embedding).
- Làm nhánh shallow (MBH-LPQ) với R=7 và quét b.
- PCA → (tạm) LDA trước, rồi mới EDA nếu cần.
- Cosine similarity + WS fusion.
- Đánh giá: accuracy + confusion matrix + test noise SNR + thời gian.

## 8) Checklist “mình phải nói được” (tự kiểm)
- Nói được pipeline từ đầu đến cuối trong 60–90 giây.
- Nói rõ input/output của từng khối.
- Giải thích vì sao depth-deep + texture-shallow bổ sung nhau.
- Nói được WS là gộp điểm chứ không phải gộp vector.
