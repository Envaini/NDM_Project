import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ----------------------------
# Labels
# ----------------------------
def read_labels_csv(root: str) -> pd.DataFrame:
    """
    Expect: root/data/labels.csv with columns: relpath,label
    """
    p = os.path.join(root, "data", "labels.csv")
    df = pd.read_csv(p)
    if "relpath" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must have columns: relpath,label")
    return df


# ----------------------------
# Load CWRU .mat -> DE signal
# ----------------------------
def pick_de_signal(mat: dict) -> Tuple[str, np.ndarray]:
    """
    Prefer keys like *_DE_time (CWRU common).
    Fallback: longest 1D vector.
    """
    keys = [k for k in mat.keys() if not k.startswith("__")]

    cand = [k for k in keys if ("DE" in k.upper() and "TIME" in k.upper())]
    if len(cand) > 0:
        k = cand[0]
        x = mat[k].squeeze()
        if x.ndim != 1:
            x = x.reshape(-1)
        return k, x.astype(np.float32)

    best_k, best_x, best_len = None, None, -1
    for k in keys:
        v = mat[k]
        if isinstance(v, np.ndarray):
            x = v.squeeze()
            if x.ndim == 1 and x.size > best_len:
                best_k, best_x, best_len = k, x, x.size

    if best_x is None:
        raise ValueError("Cannot find 1D signal in mat keys=" + str(keys))

    return best_k, best_x.astype(np.float32)


def load_de_signal(abs_mat_path: str) -> Tuple[str, np.ndarray]:
    mat = loadmat(abs_mat_path)
    key, x = pick_de_signal(mat)
    return key, x


# ----------------------------
# Anti-leak unit: "virtual record" (chunk) BEFORE segmentation
# ----------------------------
@dataclass(frozen=True)
class RecordUnit:
    record_id: str          # unique id
    relpath: str            # relative path from project root
    label: str
    start: int              # sample index
    end: int                # sample index (exclusive)


def make_virtual_records(
    root: str,
    labels_df: pd.DataFrame,
    fs: int = 48000,
    chunk_seconds: float = 1.0,
    drop_last: bool = True,
) -> List[RecordUnit]:
    """
    Create non-overlapping chunks per .mat file.
    Split is done on these chunks first (anti-leak), then segment inside each chunk.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    chunk_len = int(round(chunk_seconds * fs))
    if chunk_len <= 0:
        raise ValueError("chunk_len <= 0; check fs/chunk_seconds")

    records: List[RecordUnit] = []

    for _, row in labels_df.iterrows():
        rel = str(row["relpath"]).replace("\\", "/")
        lab = str(row["label"])
        abs_p = os.path.join(root, rel)

        _, x = load_de_signal(abs_p)
        n = x.size

        num_chunks = n // chunk_len if drop_last else int(np.ceil(n / chunk_len))
        for ci in range(num_chunks):
            s = ci * chunk_len
            e = min((ci + 1) * chunk_len, n)
            if drop_last and (e - s) < chunk_len:
                continue

            rid = f"{os.path.basename(rel)}::chunk{ci:03d}"
            records.append(RecordUnit(record_id=rid, relpath=rel, label=lab, start=int(s), end=int(e)))

    if len(records) == 0:
        raise RuntimeError("No records created. Check paths or chunk_seconds.")
    return records


def split_records_stratified(
    records: List[RecordUnit],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """
    Split by RecordUnit (chunk) => anti-leak.
    Return dict: {train:[record_id], val:[...], test:[...]}
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train+val+test must sum to 1.0")

    rng = np.random.default_rng(seed)

    by_label: Dict[str, List[str]] = {}
    for r in records:
        by_label.setdefault(r.label, []).append(r.record_id)

    train_ids, val_ids, test_ids = [], [], []
    for lab, ids in by_label.items():
        ids = ids.copy()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(np.floor(train_ratio * n))
        n_val = int(np.floor(val_ratio * n))
        n_test = n - n_train - n_val

        # nếu đủ nhiều chunk, ép có val/test tối thiểu 1
        if n >= 3:
            if n_val == 0:
                n_val = 1
                n_train = max(n_train - 1, 1)
            if n_test == 0:
                n_test = 1
                n_train = max(n_train - 1, 1)
            n_test = n - n_train - n_val

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    # anti-leak check
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise RuntimeError("Leakage: record_id overlap between splits!")

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def save_splits_json(
    root: str,
    out_relpath: str,
    records: List[RecordUnit],
    splits: Dict[str, List[str]],
    meta: Optional[Dict] = None
) -> str:
    out_path = os.path.join(root, out_relpath)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rec_map = {
        r.record_id: {"relpath": r.relpath, "label": r.label, "start": r.start, "end": r.end}
        for r in records
    }
    payload = {
        "meta": meta or {},
        "records": rec_map,
        "splits": splits
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return out_path


def load_splits_json(root: str, relpath: str) -> Tuple[Dict[str, dict], Dict[str, List[str]], Dict]:
    p = os.path.join(root, relpath)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["records"], payload["splits"], payload.get("meta", {})


# ----------------------------
# Segment + Normalize
# ----------------------------
def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    return (x - m) / (s + eps)


def iter_segments_from_record(
    root: str,
    rec: dict,
    seg_len: int,
    hop_len: int,
    normalize: bool = True
) -> Iterator[np.ndarray]:
    """
    rec: {"relpath","label","start","end"}
    """
    abs_p = os.path.join(root, rec["relpath"])
    _, x = load_de_signal(abs_p)
    x = x[int(rec["start"]):int(rec["end"])]

    n = x.size
    i = 0
    while i + seg_len <= n:
        seg = x[i:i + seg_len].copy()
        if normalize:
            seg = zscore(seg)
        yield seg
        i += hop_len


def summarize_splits(records_map: Dict[str, dict], splits: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Count per split per label
    """
    out: Dict[str, Dict[str, int]] = {}
    for split_name, ids in splits.items():
        c: Dict[str, int] = {}
        for rid in ids:
            lab = records_map[rid]["label"]
            c[lab] = c.get(lab, 0) + 1
        out[split_name] = dict(sorted(c.items(), key=lambda kv: kv[0]))
    return out
