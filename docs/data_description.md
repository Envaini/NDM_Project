# Dataset Description — CWRU (fs=48k, load=1HP, Drive-End)

## 1) Dataset
- Dataset: Case Western Reserve University (CWRU) bearing dataset
- Sampling rate (paper-core): 48,000 Hz
- Load condition (paper-core): 1 HP
- Sensor channel used (paper-core): Drive-End (DE) vibration signal
- Task: 10-class classification (Healthy + 9 fault classes)

## 2) Folder layout (local)
- Project root: C:\Users\ADMIN\Desktop\NDM_Project
- Raw data:
  - data/raw/CWRU/normal_baseline/Normal_1.mat
  - data/raw/CWRU/48k_drive_end_fault/*.mat

## 3) Paper-core class taxonomy (10 classes)
Paper-core uses 10 classes:
- H (Healthy)
- IRF_007, IRF_014, IRF_021
- BF_007,  BF_014,  BF_021
- ORF_007, ORF_014, ORF_021

Notes:
- ORF files exist at multiple locations (@3, @6, @12). For paper-core we FIX location = @6.

## 4) File-to-label mapping
See: data/labels.csv

Selected 10 files:
- H:    normal_baseline/Normal_1.mat
- BF:   B007_1.mat, B014_1.mat, B021_1.mat
- IRF:  IR007_1.mat, IR014_1.mat, IR021_1.mat
- ORF@6: OR007@6_1.mat, OR014@6_1.mat, OR021@6_1.mat

## 5) Data sanity checks (to be executed)
- Each .mat must contain a 1-D DE signal vector (Drive-End).
- Sampling rate is 48kHz for the selected subset.
- Basic checks: signal length > 0, finite values, no NaN/Inf.
