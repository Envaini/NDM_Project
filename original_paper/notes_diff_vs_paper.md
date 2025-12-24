# Notes: Differences vs. the paper

## Dataset
- Paper dùng CWRU/PU/Paderborn (tuỳ bảng).
- Đồ án dùng CWRU 48k drive-end (và mở rộng load 0/1/2/3 cho domain-shift).

## Feature extraction
- Paper: MBH-LPQ chia sub-block theo lưới (2D).
- Nếu code chia block theo 1 chiều (chỉ theo trục dọc) thì “không đúng tinh thần paper” → cần ghi rõ hoặc sửa.

## Deep backbone
- Paper có thể báo YamNet/VGG16/VGGish tuỳ bảng.
- Đồ án ưu tiên VGGish (torchaudio) vì dễ tái hiện trên local.

## Split policy (anti-leak)
- Nếu split theo segment cắt từ cùng bản ghi: kết quả có thể “đẹp quá”.
- Giải pháp: split theo chunk/record trước (anti-leak), rồi mới segment trong mỗi chunk.

## Fusion
- Paper fusion ở mức score (WS).
- Đồ án giữ đúng WS score-level (alpha sweep).
