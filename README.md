# NDM Project — Bearing Condition Classification (CWRU)

Mục tiêu: HIỂU → TÁI HIỆN (paper-core) → GIẢI THÍCH → MỞ RỘNG

Paper-core pipeline: Log-Mel → (MBH-LPQ + VGGish) → PCA → EDA → cosine matching → WS sweep

## Structure
configs/ data/ docs/ experiments/ notebooks/ original_paper/ results/ src/ tests/

## Local environment
- Anaconda env: ndm
- Run: Jupyter Notebook tại thư mục NDM_Project

## Dataset (local, not committed)
data/raw/CWRU/normal_baseline/Normal_0.mat, Normal_1.mat, Normal_2.mat, Normal_3.mat  
data/raw/CWRU/48k_drive_end_fault/*.mat (bao gồm các load _0, _1, _2, _3 cho: B007, B014, B021, IR007, IR014, IR021; OR007@3, OR007@6, OR007@12, OR014@6, OR021@3, OR021@6, OR021@12)
