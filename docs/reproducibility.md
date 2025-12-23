# Reproducibility (Phase 1)

## Fixed seed
Seed chuẩn: 42
Mọi notebook/script đều phải set seed cho:
- Python random
- NumPy
- (nếu dùng) TensorFlow/PyTorch

## Config saving rule
Mỗi experiment tạo 1 thư mục:
results/exp_<YYYYMMDD_HHMM>_<short_name>/

Bắt buộc chứa:
- config.yaml (hoặc config.json)
- requirements.txt (pip freeze)
- notes.md (mô tả dữ liệu, seed, tham số, kết quả)

## Version logging
- Lưu pip freeze vào results/exp_.../requirements.txt
- Ghi chú version Python/OS nếu cần

## Dataset rule
Dataset để local tại:
data/raw/CWRU/...
Không commit dataset lên git.
