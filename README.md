# NDM Project â€” Bearing Condition Classification (CWRU)

Má»¥c tiÃªu: HIá»‚U â†’ TÃI HIá»†N (paper-core) â†’ GIáº¢I THÃCH â†’ Má»ž Rá»˜NG

Paper-core pipeline:
Log-Mel â†’ (MBH-LPQ + VGGish) â†’ PCA â†’ EDA â†’ cosine matching â†’ WS sweep

## Structure
configs/ data/ docs/ experiments/ notebooks/ original_paper/ results/ src/ tests/

## Local environment
- Anaconda env: ndm
- Run: Jupyter Notebook táº¡i thÆ° má»¥c NDM_Project

## Dataset (local, not committed)
data/raw/CWRU/normal_baseline/Normal_0.mat, Normal_1.mat, Normal_2.mat, Normal_3.mat
data/raw/CWRU/48k_drive_end_fault/*.mat  (bao gồm các load _0, _1, _2, _3 cho: B007, B014, B021, IR007, IR014, IR021, OR007@3, OR007@6, OR007@12, OR014@6, OR021@3, OR021@6, OR021@12)
