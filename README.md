# NDM Project — Bearing Condition Classification (CWRU)

[![Release](https://img.shields.io/github/v/release/Envaini/NDM_Project?display_name=tag)](https://github.com/Envaini/NDM_Project/releases)

## Download (Release)

- **All-in-one:** [NDM_Project_full.zip](https://github.com/Envaini/NDM_Project/releases/download/v0.1.0/NDM_Project_full.zip)
- **Dataset (CWRU 48k, loads 0/1/2/3):** [CWRU_dataset.zip](https://github.com/Envaini/NDM_Project/releases/download/v0.1.0/CWRU_dataset.zip)
- **Results & artifacts:** [results_artifacts.zip](https://github.com/Envaini/NDM_Project/releases/download/v0.1.0/results_artifacts.zip)

## Run
```bash
conda activate ndm
# notebook

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
