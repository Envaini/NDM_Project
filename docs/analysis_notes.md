# Analysis Notes (Phase 11/12/13)

## Mục tiêu đồ án
HIỂU → TÁI HIỆN (paper-core) → GIẢI THÍCH → Cải tiến → MỞ RỘNG

## Dataset đang dùng
- CWRU 48k drive-end, 10 lớp (Normal + IR/B/OR với các mức hư hỏng).
- Paper-core: ưu tiên theo cấu hình trong original_paper/protocol.yaml.
- Mở rộng: domain-shift train load A test load B (load 0/1/2/3).

## Những gì đã tái hiện từ paper (core)
- Nhóm shallow: LBP / LDP / basic LPQ (R=3,5,7) / MBH-LPQ (có sweep b).
- Nhóm deep: VGGish embedding (nếu torchaudio load được).
- Matching: cosine similarity với prototype theo lớp.
- Fusion: Weighted Sum ở mức score (alpha).

## Cải tiến / mở rộng đã làm
- Phase10: domain-shift (train load A → test load B), có matrix + summary.
- Phase11: phân tích gap diag vs off-diag (định lượng độ lệch miền).

## Lưu ý khoa học (rất quan trọng khi vấn đáp)
- Nếu train/test tách theo “đoạn cắt từ cùng bản ghi”, độ chính xác có thể bị “đẹp quá”.
- Vì vậy cần ghi rõ policy chia dữ liệu (anti-leak theo chunk/record hoặc theo file) và nêu hệ quả.

## Artefacts bắt buộc cần có ở root results/
- results/ablation_summary.csv
- results/ablation_plots.png
- tests/test_eda.py
- original_paper/baselines_like_paper.md
- original_paper/notes_diff_vs_paper.md
