# NDM Project — Requirements (Phase 0)

## Project title
- VN: Phân loại trạng thái ổ lăn từ tín hiệu rung (CWRU, 48 kHz, 10 lớp)
- EN: Rolling Bearing Condition Classification from Vibration Signals (CWRU, 48 kHz, 10 classes)

## Goal
HIỂU → TÁI HIỆN (paper-core) → GIẢI THÍCH → MỞ RỘNG

Paper-core pipeline:
Log-Mel → (MBH-LPQ + VGGish) → PCA → EDA → cosine matching → WS sweep

## Dataset scope (frozen)
CWRU, drive-end, fs=48kHz, load=1HP, 10 classes.
- normal_baseline: Normal_1.mat
- 48k_drive_end_fault: B007_1.mat, B014_1.mat, B021_1.mat,
  IR007_1.mat, IR014_1.mat, IR021_1.mat,
  OR007@3_1.mat, OR007@6_1.mat, OR007@12_1.mat,
  OR014@6_1.mat,
  OR021@3_1.mat, OR021@6_1.mat, OR021@12_1.mat

## Folder rules
- data/raw/CWRU/... : chứa .mat (local)
- results/ : mọi output (figures/tables/models)
- notebooks/ : notebooks theo phase
