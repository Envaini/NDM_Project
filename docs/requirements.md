# NDM Project — Requirements (Phase 0)

## Project title
- VN: Phân loại trạng thái máy móc từ tín hiệu rung (CWRU, 48 kHz, 10 lớp)
- EN: Rolling Bearing Condition Classification from Vibration Signals (CWRU, 48 kHz, 10 classes)

## References
- Paper: Vibration signal analysis for rolling bearings faults diagnosis based on deep–shallow features fusion
- Textbook: Bishop — Pattern Recognition and Machine Learning (2006)

## Method overview
Log-Mel → MBH-LPQ + VGGish → PCA → EDA → cosine matching → weighted-sum fusion (mặc định 0.8 : 0.2).  
Chỉ số báo cáo: accuracy, macro-F1, confusion matrix. Toàn bộ artifacts lưu dưới `results/`.

## Dataset scope (frozen)

Nguồn: CWRU drive-end, fs = 48 kHz. Hai phần: tập lõi 10 lớp (train/val/test) và bộ bổ sung kiểm tra chéo-load.

### 1) Tập lõi 10 lớp (tải 1HP)
- Normal: `normal_baseline/Normal_1.mat`
- Fault 1HP: `B007_1.mat`, `B014_1.mat`, `B021_1.mat`,
  `IR007_1.mat`, `IR014_1.mat`, `IR021_1.mat`,
  `OR007@3_1.mat`, `OR021@3_1.mat`, `OR014@6_1.mat`

Nhãn cố định nằm trong `data/labels.csv`. Tách tập cố định 80/20 tại `data/splits/cwru_splits.json`.

### 2) Bộ bổ sung kiểm tra chéo-load / biến thể OR
- Normal (0/1/2/3): `Normal_0.mat`, `Normal_1.mat`, `Normal_2.mat`, `Normal_3.mat`
- OR biến thiên: `OR007@6_1.mat`, `OR007@12_1.mat`, `OR021@6_1.mat`, `OR021@12_1.mat`

Các file .mat lưu cục bộ (không commit):
- `data/raw/CWRU/normal_baseline/*.mat`
- `data/raw/CWRU/48k_drive_end_fault/*.mat`

## Split & protocol
- Chuẩn: dùng `data/splits/cwru_splits.json` (80/20, seed cố định) cho 10 lớp 1HP.
- Chéo-load (tùy chọn): train trên 1HP, test trên Normal_0/2/3 và/hoặc các biến thể OR; báo cáo riêng.

## Repository rules
- Không commit dữ liệu thô, kết quả, checkpoint: ignore `data/raw/**`, `results/**`, `*.mat`, `*.pt`, `*.pth`, `*.zip`.
- Phát hành qua GitHub Release 3 gói ZIP:
  - `NDM_Project_full.zip`  — mã nguồn + notebooks + configs
  - `CWRU_dataset.zip`      — các .mat
  - `results_artifacts.zip` — hình/bảng/model
- Cấu trúc repo:
  - `configs/`        – cấu hình thí nghiệm  
  - `data/`           – labels, splits (không chứa .mat)  
  - `docs/`           – tài liệu (requirements.md, v.v.)  
  - `experiments/`    – runner theo phase  
  - `notebooks/`      – notebook theo phase  
  - `original_paper/` – tài liệu tham khảo  
  - `results/`        – output sinh ra  
  - `src/`            – mã nguồn  
  - `Plan/`           – `Phase_Plan.pdf`

## Environment
- Python 3.10+; conda env: `ndm`
- Gói chính: numpy, scipy, pandas, scikit-learn, matplotlib, librosa, soundfile, torch/torchvision/torchaudio (CPU/GPU phù hợp)

## Reproducibility
- Seed mặc định: 42 cho NumPy/PyTorch/sklearn
- Mỗi run lưu: `results/run_{timestamp}/config.yaml`, confusion matrix, classification report

## Run
- Notebook: chạy trong thư mục `notebooks/` theo phase.
- Script ví dụ (fusion):
    python src/run_phase8_fusion.py --config configs/exp_fusion.yaml

## Deliverables
- Bảng điểm: accuracy, macro-F1, confusion matrix cho split chuẩn; kết quả chéo-load (nếu có).
- Artifacts: lưu dưới `results/`.
- Release: tag v0.x.y kèm 3 ZIP để tải nhanh.
