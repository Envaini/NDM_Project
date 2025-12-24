import os
import sys
import json
import random
from typing import Tuple, List

import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

# allow "import dataio/spectrogram/cnn_model" from src/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from dataio import load_splits_json, iter_segments_from_record
from spectrogram import make_spectrogram
from cnn_model import CNNBaseline


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str, title: str):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    results_dir = os.path.join(ROOT, "results")
    cfg_dir = os.path.join(ROOT, "configs")
    ensure_dir(results_dir)

    cnn_cfg = load_json(os.path.join(cfg_dir, "cnn.json"))
    spec_cfg = load_json(os.path.join(cfg_dir, "spectrogram.json"))

    seed = int(cnn_cfg.get("seed", 42))
    set_seed(seed)

    device_cfg = str(cnn_cfg.get("device", "auto")).lower()
    if device_cfg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    seg_len = int(cnn_cfg.get("seg_len", 4096))
    hop_len = int(cnn_cfg.get("hop_len", 4096))
    batch_size = int(cnn_cfg.get("batch_size", 64))
    epochs = int(cnn_cfg.get("epochs", 25))
    lr = float(cnn_cfg.get("lr", 1e-3))
    weight_decay = float(cnn_cfg.get("weight_decay", 1e-4))
    emb_dim = int(cnn_cfg.get("emb_dim", 128))

    print("ROOT   =", ROOT)
    print("device =", device)
    print("cnn_cfg =", cnn_cfg)
    print("spec_cfg =", spec_cfg)

    # load splits from Phase 4
    split_path = os.path.join(ROOT, "data", "splits", "cwru_splits.json")
    records_map, splits, meta = load_splits_json(ROOT, split_path)
    fs = int(meta["fs"])

    # anti-leak sanity
    tr, va, te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    print("split sizes:", {k: len(v) for k, v in splits.items()})
    print("intersections:", len(tr & va), len(tr & te), len(va & te))

    # ---- Build spectrogram dataset in memory (nhỏ, chạy CPU ổn) ----
    def build_xy(split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for rid in splits[split_name]:
            rec = records_map[rid]
            for seg in iter_segments_from_record(
                ROOT, rec, seg_len=seg_len, hop_len=hop_len, normalize=False
            ):
                S, _metaS = make_spectrogram(
                    seg,
                    fs=fs,
                    window_type=spec_cfg["window_type"],
                    win_length=int(spec_cfg["win_length"]),
                    hop_length=int(spec_cfg["hop_length"]),
                    n_fft=int(spec_cfg["n_fft"]),
                    use_mel=bool(spec_cfg["use_mel"]),
                    n_mels=int(spec_cfg["mel_bins"]),
                    fmin=float(spec_cfg["fmin"]),
                    fmax=spec_cfg["fmax"],
                    power=float(spec_cfg["power"]),
                    log_eps=float(spec_cfg["log_eps"]),
                    to_db=bool(spec_cfg["to_db"]),
                )
                S = np.asarray(S, dtype=np.float32)  # (F,T)
                X_list.append(S[None, ...])          # (1,F,T)
                y_list.append(rec["label"])

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=object)
        return X, y

    print("\nBuilding spectrogram tensors...")
    X_train, y_train_str = build_xy("train")
    X_val, y_val_str = build_xy("val")
    X_test, y_test_str = build_xy("test")

    print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

    # normalize by TRAIN stats (global)
    mean = float(X_train.mean())
    std = float(X_train.std() + 1e-6)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    print("train norm mean/std:", mean, std)

    # label encode
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)

    class_names = list(le.classes_)
    n_classes = len(class_names)
    print("Classes:", class_names)

    # torch datasets/loaders (ép dtype rõ ràng)
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CNNBaseline(n_classes=n_classes, emb_dim=emb_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = torch.nn.CrossEntropyLoss()

    def eval_loader(loader) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        model.eval()
        losses = []
        all_y, all_p = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device).float()
                yb = yb.to(device).long()
                logits = model(xb)
                loss = crit(logits, yb)
                losses.append(float(loss.item()))
                pred = torch.argmax(logits, dim=1)
                all_y.append(yb.cpu().numpy())
                all_p.append(pred.cpu().numpy())
        y_true = np.concatenate(all_y)
        y_pred = np.concatenate(all_p)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        return float(np.mean(losses)), float(acc), float(macro_f1), y_true, y_pred

    best = {"epoch": -1, "val_macro_f1": -1.0, "val_acc": -1.0}
    history = []

    best_path = os.path.join(results_dir, "cnn_baseline.pt")
    log_path = os.path.join(results_dir, "cnn_train_log.json")

    print("\nTraining...")
    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            train_losses.append(float(loss.item()))

        tr_loss = float(np.mean(train_losses))
        va_loss, va_acc, va_f1, _, _ = eval_loader(val_loader)

        row = {"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1}
        history.append(row)

        improved = (va_f1 > best["val_macro_f1"]) or (va_f1 == best["val_macro_f1"] and va_acc > best["val_acc"])
        if improved:
            best = {"epoch": ep, "val_macro_f1": va_f1, "val_acc": va_acc}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_names": class_names,
                    "cnn_cfg": cnn_cfg,
                    "spec_cfg": spec_cfg,
                    "train_norm": {"mean": mean, "std": std},
                    "best": best,
                },
                best_path,
            )

        print(
            f"Epoch {ep:02d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_acc={va_acc:.4f} | val_f1={va_f1:.4f}"
            + ("  <-- best" if improved else "")
        )

    save_json(log_path, {"best": best, "history": history})
    print("\nSaved:", best_path)
    print("Saved:", log_path)

    # test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    te_loss, te_acc, te_f1, y_true, y_pred = eval_loader(test_loader)
    print(f"\nTEST: loss={te_loss:.4f} acc={te_acc:.4f} macro_f1={te_f1:.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_path = os.path.join(results_dir, "cm_cnn.png")
    plot_confusion_matrix(cm, class_names, cm_path, title="CNN baseline confusion matrix (test)")
    print("Saved:", cm_path)

    # deep embeddings
    def extract_embeddings(X: np.ndarray, batch: int = 128) -> np.ndarray:
        embs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                xb = torch.tensor(X[i:i + batch], dtype=torch.float32).to(device)
                eb = model.forward_features(xb).cpu().numpy().astype(np.float32)
                embs.append(eb)
        return np.vstack(embs)

    Xtr_deep = extract_embeddings(X_train)
    Xva_deep = extract_embeddings(X_val)
    Xte_deep = extract_embeddings(X_test)

    np.save(os.path.join(results_dir, "X_train_deep.npy"), Xtr_deep)
    np.save(os.path.join(results_dir, "X_val_deep.npy"), Xva_deep)
    np.save(os.path.join(results_dir, "X_test_deep.npy"), Xte_deep)

    np.save(os.path.join(results_dir, "y_train.npy"), y_train.astype(np.int64))
    np.save(os.path.join(results_dir, "y_val.npy"), y_val.astype(np.int64))
    np.save(os.path.join(results_dir, "y_test.npy"), y_test.astype(np.int64))

    info = {
        "best": best,
        "test": {"loss": te_loss, "acc": te_acc, "macro_f1": te_f1},
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
            "X_train_deep": list(Xtr_deep.shape),
            "X_val_deep": list(Xva_deep.shape),
            "X_test_deep": list(Xte_deep.shape),
        },
        "class_names": class_names,
    }
    save_json(os.path.join(results_dir, "cnn_phase7_info.json"), info)
    print("Saved:", os.path.join(results_dir, "cnn_phase7_info.json"))

    print("\nPHASE 7 DONE.")


if __name__ == "__main__":
    main()
