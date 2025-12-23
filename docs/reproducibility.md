# Reproducibility

## Environment (Windows + Anaconda)
1) Create env
- conda env create -f configs/environment.yml
- conda activate ndm

2) Install python deps (pinned)
- pip install -r configs/requirements_pip.txt

3) Sanity check
- python -c "import numpy, scipy, pandas, sklearn, matplotlib, librosa; print('OK')"

## Seeds (mandatory)
- Python: random.seed(SEED)
- NumPy: np.random.seed(SEED)
- (Later) sklearn: set random_state=SEED where applicable

## Results logging (mandatory)
- Save configs (*.yaml/*.json) for every run into results/
- Save package versions:
  - pip freeze > results/<run_id>/requirements_pip.txt
