# NDM Project — Bearing Condition Classification (CWRU)

**Đồ án:** Chẩn đoán lỗi ổ lăn từ tín hiệu rung dựa trên kết hợp đặc trưng sâu và nông 

**Bài báo tham thiếu:** *Vibration signal analysis for rolling bearings faults diagnosis based on deep–shallow features fusion*

**Tài liệu môn học:** Bishop — *Pattern Recognition and Machine Learning* (2006)

**Link bài báo tham chiếu:** [Nature — s41598-025-93133-y](https://www.nature.com/articles/s41598-025-93133-y)  

**Link dataset:** [CWRU Bearing Data Center — Download](https://engineering.case.edu/bearingdatacenter/download-data-file)

[![Release](https://img.shields.io/github/v/release/Envaini/NDM_Project?display_name=tag)](https://github.com/Envaini/NDM_Project/releases)

## Lộ trình (Phase Plan)
Phase Plan là lộ trình triển khai đồ án (mục tiêu, nội dung, kết quả theo từng phase).
- [Đọc bản Phase Plan (PDF)](Plan/Phase_Plan.pdf)

## Report (báo cáo ngắn)

Report ngắn: trình bày Abstract, ý nghĩa ứng dụng và đóng góp của hướng tiếp cận (bài báo), trục xương sống (sơ đồ), nội dung đồ án, và tóm tắt slide 2–21.

- 📄 [Đọc Report (PDF)](report/Report.pdf)


## Download (Release)

- **All-in-one:** [NDM_Project_full.zip](https://github.com/Envaini/NDM_Project/releases/latest/download/NDM_Project_full.zip)
- **Dataset (CWRU 48k, loads 0/1/2/3):** [CWRU_dataset.zip](https://github.com/Envaini/NDM_Project/releases/latest/download/CWRU_dataset.zip)
- **Results & artifacts:** [results_artifacts.zip](https://github.com/Envaini/NDM_Project/releases/latest/download/results_artifacts.zip)

## Run
```bash
conda activate ndm
# notebook


Mục tiêu: HIỂU → TÁI HIỆN (paper-core) → GIẢI THÍCH → MỞ RỘNG

Paper-core pipeline: Log-Mel → (MBH-LPQ + VGGish) → PCA → EDA → cosine matching → WS sweep

## Structure
Plan/ configs/ data/ docs/ experiments/ notebooks/ original_paper/ papers/ report/ results/ src/ tests/

## Local environment
- Anaconda env: ndm
- Run: Jupyter Notebook tại thư mục NDM_Project

## Dataset (local, not committed)
data/raw/CWRU/normal_baseline/Normal_0.mat, Normal_1.mat, Normal_2.mat, Normal_3.mat  
data/raw/CWRU/48k_drive_end_fault/*.mat (bao gồm các load _0, _1, _2, _3 cho: B007, B014, B021, IR007, IR014, IR021; OR007@3, OR007@6, OR007@12, OR014@6, OR021@3, OR021@6, OR021@12)
