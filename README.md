# NDM Project — Bearing Condition Classification (CWRU)

Mục tiêu: HIỂU → TÁI HIỆN (paper-core) → GIẢI THÍCH → MỞ RỘNG

Paper-core pipeline:
Log-Mel → (MBH-LPQ + VGGish) → PCA → EDA → cosine matching → WS sweep

## Structure
configs/ data/ docs/ notebooks/ original_paper/ results/ src/ tests/

## Local environment
- Anaconda env: ndm
- Run: Jupyter Notebook tại thư mục NDM_Project

## Dataset (local, not committed)
data/raw/CWRU/normal_baseline/Normal_1.mat
data/raw/CWRU/48k_drive_end_fault/*.mat
