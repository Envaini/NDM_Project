# Baselines like the paper (checklist)

## Nhóm shallow (texture trên log-mel)
- LBP
- LDP
- basic LPQ: R ∈ {3, 5, 7}
- MBH-LPQ: cố định R=7, sweep b (số sub-block)

## Nhóm deep
- VGGish embedding 128-D (torchaudio)
  - Nếu không load được torchaudio trên máy → phải ghi rõ “fallback shallow-only”.

## Matching + Fusion
- Matching: cosine similarity với prototype theo lớp
- Fusion: Weighted Sum score: S = alpha * S_deep + (1-alpha) * S_shallow
- Sweep alpha để chọn tỉ lệ tốt nhất

## File kết quả liên quan
- results/run_all/<run_id>/paper_core_metrics.csv
- results/run_all/<run_id>/ws_sweep.csv
- results/run_all/<run_id>/ablation_summary.csv
- results/manifest.json (ghi seed, config, versions, artifacts)
