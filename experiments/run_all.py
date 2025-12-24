
import os, re, json, math, hashlib, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as sps
import scipy.linalg as la


# --------------------------
# Utils
# --------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def l2norm(X: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())

def safe_relpath(p: Path, root: Path):
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p.resolve()).replace("\\", "/")

def get_git_commit(root: Path):
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.STDOUT)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def get_versions(pkgs):
    import importlib
    try:
        from importlib import metadata as importlib_metadata
    except Exception:
        import importlib_metadata

    vers = {}
    for p in pkgs:
        try:
            if p == "python":
                import sys
                vers[p] = sys.version.split()[0]
                continue
            vers[p] = importlib_metadata.version(p)
        except Exception:
            try:
                mod = importlib.import_module(p)
                vers[p] = getattr(mod, "__version__", None)
            except Exception:
                vers[p] = None
    return vers


# --------------------------
# CWRU parsing
# --------------------------
def parse_load_from_name(name: str):
    m = re.search(r"_(\d+)(?:\.mat)?$", name)
    return int(m.group(1)) if m else None

def parse_label_from_name(name: str):
    base = name.replace(".mat", "")
    if base.startswith("Normal"):
        return "H"
    if base.startswith("B007"): return "B007"
    if base.startswith("B014"): return "B014"
    if base.startswith("B021"): return "B021"
    if base.startswith("IR007"): return "IR007"
    if base.startswith("IR014"): return "IR014"
    if base.startswith("IR021"): return "IR021"
    # gom OR theo size, bỏ qua @3/@6/@12
    if base.startswith("OR007"): return "OR007"
    if base.startswith("OR014"): return "OR014"
    if base.startswith("OR021"): return "OR021"
    return None

def load_cwru_de_time(mat_path: Path):
    d = sio.loadmat(mat_path, squeeze_me=True)
    cand = None
    for k in d.keys():
        if k.endswith("_DE_time"):
            cand = k
            break
    if cand is None:
        raise KeyError(f"Không tìm thấy *_DE_time trong {mat_path.name}. Keys: {list(d.keys())[:20]}")
    x = d[cand].astype(np.float32)
    x = np.ravel(x)
    return x

def sample_segments(x: np.ndarray, seg_len: int, n_seg: int, rng: np.random.Generator):
    if len(x) <= seg_len:
        pad = seg_len - len(x) + 1
        x = np.pad(x, (0, pad), mode="wrap")
    max_start = len(x) - seg_len
    starts = rng.integers(0, max_start + 1, size=n_seg)
    segs = np.stack([x[s:s+seg_len] for s in starts], axis=0)
    return segs

def scan_metadata(cwru_root: Path):
    rows = []
    fault_dir = cwru_root / "48k_drive_end_fault"
    normal_dir = cwru_root / "normal_baseline"

    for p in list(fault_dir.glob("*.mat")) + list(normal_dir.glob("*.mat")):
        name = p.name
        y = parse_label_from_name(name)
        ld = parse_load_from_name(name)
        if y is None or ld is None:
            continue
        rows.append({"path": str(p), "name": name, "label": y, "load": ld})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Không scan được file *.mat theo format *_<load>.mat. Kiểm tra tên file và folder.")
    return df

def build_samples_single_load(df: pd.DataFrame, load: int, n_per_class: int, seg_len: int, seed: int):
    rng = np.random.default_rng(seed)
    sub = df[df["load"] == load].copy()
    labels = sorted(sub["label"].unique().tolist())

    X_wave = []
    y = []

    for lab in labels:
        files = sub[sub["label"] == lab]["path"].tolist()
        if len(files) == 0:
            continue
        need = n_per_class
        per_file = max(1, math.ceil(need / len(files)))
        got = 0
        for fp in files:
            x = load_cwru_de_time(Path(fp))
            segs = sample_segments(x, seg_len=seg_len, n_seg=per_file, rng=rng)
            take = min(segs.shape[0], need - got)
            X_wave.append(segs[:take])
            y += [lab] * take
            got += take
            if got >= need:
                break

    X_wave = np.concatenate(X_wave, axis=0).astype(np.float32)
    y = np.array(y)
    idx = rng.permutation(len(y))
    return X_wave[idx], y[idx]


# --------------------------
# Log-mel (96,64)
# --------------------------
def logmel_96x64(wave: np.ndarray, sr: int = 48000, n_fft: int = 1024, hop: int = 256, n_mels: int = 64, n_frames: int = 96):
    try:
        import librosa
    except Exception as e:
        raise ImportError("Thiếu librosa. Cài: pip install librosa") from e

    w = wave.astype(np.float32)
    w = w - w.mean()
    w = w / (np.max(np.abs(w)) + 1e-9)

    S = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)
    img = logS.T

    T = img.shape[0]
    if T >= n_frames:
        img = img[:n_frames, :]
    else:
        pad = n_frames - T
        img = np.pad(img, ((0, pad), (0, 0)), mode="edge")

    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn + 1e-9)
    return img.astype(np.float32)


# --------------------------
# LPQ + MBH-LPQ
# --------------------------
def lpq_codes(img: np.ndarray, R: int = 7):
    a = 1.0 / (2*R + 1)
    x = np.arange(-R, R+1)
    y = np.arange(-R, R+1)
    X, Y = np.meshgrid(x, y, indexing="xy")

    w1 = np.exp(-2j*np.pi*a*X)
    w2 = np.exp(-2j*np.pi*a*Y)

    f1 = w1
    f2 = w2
    f3 = w1*w2
    f4 = w1*np.conj(w2)

    def conv_complex(f):
        re = sps.convolve2d(img, np.real(f), mode="same", boundary="symm")
        im = sps.convolve2d(img, np.imag(f), mode="same", boundary="symm")
        return re, im

    re1, im1 = conv_complex(f1)
    re2, im2 = conv_complex(f2)
    re3, im3 = conv_complex(f3)
    re4, im4 = conv_complex(f4)

    bits = [
        (re1 > 0), (im1 > 0),
        (re2 > 0), (im2 > 0),
        (re3 > 0), (im3 > 0),
        (re4 > 0), (im4 > 0),
    ]
    code = np.zeros(img.shape, dtype=np.uint8)
    for i, b in enumerate(bits):
        code |= (b.astype(np.uint8) << i)
    return code

def best_grid(b: int):
    best = (1, b, 10**9)
    for gh in range(1, b+1):
        if b % gh == 0:
            gw = b // gh
            score = abs(gh - gw)
            if score < best[2]:
                best = (gh, gw, score)
    return best[0], best[1]

def mbh_lpq_feature(code_img: np.ndarray, b: int):
    H, W = code_img.shape
    gh, gw = best_grid(b)
    feats = []
    for i in range(gh):
        for j in range(gw):
            r0 = int(round(i * H / gh))
            r1 = int(round((i+1) * H / gh))
            c0 = int(round(j * W / gw))
            c1 = int(round((j+1) * W / gw))
            block = code_img[r0:r1, c0:c1]
            hist = np.bincount(block.ravel(), minlength=256).astype(np.float32)
            hist = hist / (hist.sum() + 1e-9)
            feats.append(hist)
    return np.concatenate(feats, axis=0).astype(np.float32)


# --------------------------
# PCA + EDA (stable) + fallback LDA
# --------------------------
def fit_pca_eda_stable(Xtr: np.ndarray, ytr: np.ndarray, pca_dim: int = 128, out_dim: int = None, reg: float = 1e-6):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xtr)

    d = Xs.shape[1]
    n = Xs.shape[0]
    p = min(pca_dim, d, max(2, n-1))
    pca = PCA(n_components=p, random_state=0)
    Xp = pca.fit_transform(Xs)

    classes = np.unique(ytr)
    mu = Xp.mean(axis=0, keepdims=True)

    Sw = np.zeros((p, p), dtype=np.float64)
    Sb = np.zeros((p, p), dtype=np.float64)

    for c in classes:
        Xc = Xp[ytr == c]
        muc = Xc.mean(axis=0, keepdims=True)
        Sw += (Xc - muc).T @ (Xc - muc)
        Sb += Xc.shape[0] * (muc - mu).T @ (muc - mu)

    Sw = Sw / max(1, (Xp.shape[0] - len(classes)))
    Sb = Sb / max(1, len(classes))

    # scale by trace to avoid expm overflow
    Sw = Sw / (np.trace(Sw) + 1e-9)
    Sb = Sb / (np.trace(Sb) + 1e-9)

    Sw += reg * np.eye(p)

    # try EDA (expm)
    try:
        A = la.expm(Sb)
        B = la.expm(Sw)
        A = (A + A.T) / 2.0
        B = (B + B.T) / 2.0
        if (not np.isfinite(A).all()) or (not np.isfinite(B).all()):
            raise ValueError("EDA expm produced inf/nan")
        w, V = la.eigh(A, B)
    except Exception:
        # fallback LDA: solve Sb v = λ Sw v
        w, V = la.eigh(Sb, Sw)

    idx = np.argsort(w)[::-1]
    V = V[:, idx]

    if out_dim is None:
        out_dim = min(len(classes) - 1, p)
        out_dim = max(2, out_dim)
    W = V[:, :out_dim].astype(np.float32)

    def transform(X: np.ndarray):
        Xs2 = scaler.transform(X)
        Xp2 = pca.transform(Xs2)
        Z = Xp2 @ W
        return Z.astype(np.float32)

    return transform


# --------------------------
# Cosine-prototype classifier
# --------------------------
def fit_prototypes(Xtr: np.ndarray, ytr: np.ndarray):
    Xn = l2norm(Xtr)
    classes = np.unique(ytr)
    protos = []
    for c in classes:
        pc = Xn[ytr == c].mean(axis=0, keepdims=True)
        pc = l2norm(pc)
        protos.append(pc)
    P = np.concatenate(protos, axis=0)
    return classes, P

def predict_with_protos(X: np.ndarray, classes: np.ndarray, P: np.ndarray):
    Xn = l2norm(X)
    scores = Xn @ P.T
    pred = classes[np.argmax(scores, axis=1)]
    return pred, scores


# --------------------------
# VGGish embed
# --------------------------
def try_init_vggish(sr: int = 48000):
    try:
        import torch
        import torchaudio
        from torchaudio.prototype.pipelines import VGGISH
        bundle = VGGISH
        model = bundle.get_model()
        model.eval()
        iproc = bundle.get_input_processor()
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        return True, model, iproc, resampler
    except Exception as e:
        print("[WARN] Không load được torchaudio VGGISH -> chạy shallow-only. Lỗi:", repr(e))
        return False, None, None, None

def vggish_embed_batch(Xwave, model, iproc, resampler, device="cpu"):
    """
    Xwave: (N, seg_len) numpy float32 (sr=48k)
    Trả: (N, 128) numpy float32
    Fix: nếu waveform quá ngắn => pad để iproc không trả về 0 examples.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    model.eval()
    model.to(device)

    # VGGish cần ~0.96s @16k => 15360 mẫu. Pad lên tối thiểu mức này.
    MIN_SAMPLES_16K = 15360

    feats = []
    with torch.no_grad():
        for w in Xwave:
            t = torch.from_numpy(w).float().to(device)

            # normalize nhẹ
            t = t - t.mean()
            t = t / (t.abs().max() + 1e-9)

            # resample 48k -> 16k
            t16 = resampler(t)

            # ép về 1D
            if t16.ndim != 1:
                t16 = t16.squeeze()
            t16 = t16.reshape(-1)

            # PAD nếu quá ngắn (nguyên nhân gây 0 examples)
            if t16.numel() < MIN_SAMPLES_16K:
                pad = MIN_SAMPLES_16K - t16.numel()
                t16 = F.pad(t16, (0, pad), mode="constant", value=0.0)

            # iproc: tùy version
            try:
                inp = iproc(t16)
            except TypeError:
                inp = iproc(t16, sample_rate=16000)

            # Nếu vẫn rỗng (hiếm), trả embedding 0 cho sample đó để pipeline không chết
            if inp is None or inp.numel() == 0 or (inp.ndim >= 1 and inp.shape[0] == 0):
                feats.append(np.zeros((128,), dtype=np.float32))
                continue

            inp = inp.to(device)

            out = model(inp)
            # out: (B,T,128) hoặc (B,128)
            if out.ndim == 3:
                v = out.mean(dim=1).squeeze(0)
            else:
                v = out.squeeze(0)

            feats.append(v.detach().cpu().numpy().astype(np.float32))

    return np.stack(feats, axis=0)

# --------------------------
# Paper-core run (Phase12 required artifacts)
# --------------------------
def extract_shallow_features(Xwave: np.ndarray, sr: int, lpq_R: int, b_list, audit_dir: Path):
    feats_b = {b: [] for b in b_list}
    audit_saved = False
    for w in Xwave:
        img = logmel_96x64(w, sr=sr)
        if not audit_saved:
            plt.figure()
            plt.imshow(img.T, aspect="auto", origin="lower")
            plt.title("Audit log-mel (96x64)")
            plt.tight_layout()
            plt.savefig(audit_dir / "audit_logmel.png", dpi=200)
            plt.close()
            audit_saved = True

        code = lpq_codes(img, R=lpq_R)
        for b in b_list:
            feats_b[b].append(mbh_lpq_feature(code, b=b))

    for b in b_list:
        feats_b[b] = np.stack(feats_b[b], axis=0).astype(np.float32)
    return feats_b

def save_confusion_matrix(y_true, y_pred, labels, out_png: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7,6))
    plt.imshow(cm, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def run_paper_core(df_meta: pd.DataFrame, cfg: dict, root: Path, run_dir: Path):
    seed = int(cfg["seed"])
    pc = cfg["paper_core"]
    load = int(pc["load"])
    n_per_class = int(pc["n_per_class"])
    seg_len = int(pc["seg_len"])
    sr = int(pc["sr"])
    test_size = float(pc["test_size"])
    val_size = float(pc["val_size"])

    lpq_R = int(cfg["lpq_R"])
    b_list = list(cfg["b_list"])
    alpha_list = list(cfg["alpha_list"])
    use_vggish = bool(cfg.get("use_vggish", True))

    seed_everything(seed)
    audit_dir = ensure_dir(run_dir / "audit")

    Xwave, y = build_samples_single_load(df_meta, load=load, n_per_class=n_per_class, seg_len=seg_len, seed=seed)

    # split train+val vs test
    X_trv, X_te, y_trv, y_te = train_test_split(Xwave, y, test_size=test_size, random_state=seed, stratify=y)
    # split train vs val
    X_tr, X_va, y_tr, y_va = train_test_split(X_trv, y_trv, test_size=val_size, random_state=seed, stratify=y_trv)

    # shallow
    print("[Phase12] paper-core: extract shallow ...")
    tr_sh = extract_shallow_features(X_tr, sr, lpq_R, b_list, audit_dir)
    va_sh = extract_shallow_features(X_va, sr, lpq_R, b_list, audit_dir)
    te_sh = extract_shallow_features(X_te, sr, lpq_R, b_list, audit_dir)

    # deep
    deep_ok = False
    Xtr_de = Xva_de = Xte_de = None
    model = iproc = resampler = None
    if use_vggish:
        deep_ok, model, iproc, resampler = try_init_vggish(sr=sr)
        if deep_ok:
            print("[Phase12] paper-core: extract deep (VGGish) ...")
            Xtr_de = vggish_embed_batch(X_tr, model, iproc, resampler)
            Xva_de = vggish_embed_batch(X_va, model, iproc, resampler)
            Xte_de = vggish_embed_batch(X_te, model, iproc, resampler)

    rows_metrics = []
    rows_ws = []
    rows_ablation = []

    # deep-only
    if deep_ok:
        de_tf = fit_pca_eda_stable(Xtr_de, y_tr, pca_dim=128)
        Ztr = de_tf(Xtr_de); Zva = de_tf(Xva_de); Zte = de_tf(Xte_de)
        cls, P = fit_prototypes(Ztr, y_tr)
        pv, Sv = predict_with_protos(Zva, cls, P)
        pt, St = predict_with_protos(Zte, cls, P)
        rows_metrics.append({"model":"VGGish", "best_b":None, "best_alpha":1.0,
                             "acc_val":accuracy_score(y_va, pv), "acc_test":accuracy_score(y_te, pt),
                             "f1_val":f1_score(y_va, pv, average="macro"), "f1_test":f1_score(y_te, pt, average="macro")})
    else:
        Sv = St = None
        cls = None

    # shallow sweep b + fusion sweep alpha
    best = {"best_model": None, "best_b": None, "best_alpha": None, "best_acc_val": -1.0}

    for b in b_list:
        sh_tf = fit_pca_eda_stable(tr_sh[b], y_tr, pca_dim=256)
        Ztr = sh_tf(tr_sh[b]); Zva = sh_tf(va_sh[b]); Zte = sh_tf(te_sh[b])
        cls2, P2 = fit_prototypes(Ztr, y_tr)
        pv_sh, Sv_sh = predict_with_protos(Zva, cls2, P2)
        pt_sh, St_sh = predict_with_protos(Zte, cls2, P2)

        accv = accuracy_score(y_va, pv_sh)
        acct = accuracy_score(y_te, pt_sh)

        rows_ablation.append({"b": int(b), "acc_val": float(accv), "acc_test": float(acct)})
        rows_metrics.append({"model":"MBH-LPQ", "best_b":int(b), "best_alpha":0.0,
                             "acc_val":accv, "acc_test":acct,
                             "f1_val":f1_score(y_va, pv_sh, average="macro"), "f1_test":f1_score(y_te, pt_sh, average="macro")})

        if accv > best["best_acc_val"]:
            best.update({"best_model":"MBH-LPQ", "best_b":int(b), "best_alpha":0.0, "best_acc_val":float(accv)})

        if deep_ok:
            # align class order
            all_cls = np.array(sorted(set(cls.tolist()) | set(cls2.tolist())))

            def align_scores(scores, cls_src):
                m = {c:i for i,c in enumerate(cls_src)}
                out = np.zeros((scores.shape[0], len(all_cls)), dtype=np.float32)
                for j,c in enumerate(all_cls):
                    out[:, j] = scores[:, m[c]] if c in m else -1e9
                return out

            Sv_de = align_scores(Sv, cls)
            St_de = align_scores(St, cls)
            Sv_sh_al = align_scores(Sv_sh, cls2)
            St_sh_al = align_scores(St_sh, cls2)

            for alpha in cfg["alpha_list"]:
                alpha = float(alpha)
                Sv_f = alpha*Sv_de + (1-alpha)*Sv_sh_al
                St_f = alpha*St_de + (1-alpha)*St_sh_al

                pv_f = all_cls[np.argmax(Sv_f, axis=1)]
                pt_f = all_cls[np.argmax(St_f, axis=1)]

                accv_f = accuracy_score(y_va, pv_f)
                acct_f = accuracy_score(y_te, pt_f)

                rows_ws.append({"b": int(b), "alpha": alpha, "acc_val": float(accv_f), "acc_test": float(acct_f)})

                if accv_f > best["best_acc_val"]:
                    best.update({"best_model":"Fusion", "best_b":int(b), "best_alpha":float(alpha), "best_acc_val":float(accv_f)})

    # --- Save required artifacts ---
    paper_core_metrics = pd.DataFrame(rows_metrics)
    paper_core_metrics.to_csv(run_dir / "paper_core_metrics.csv", index=False)

    ws_df = pd.DataFrame(rows_ws)
    ws_df.to_csv(run_dir / "ws_sweep.csv", index=False)

    ab_df = pd.DataFrame(rows_ablation).sort_values("acc_val", ascending=False)
    ab_df.to_csv(run_dir / "ablation_summary.csv", index=False)

    # confusion matrix for best
    labels = np.array(sorted(set(y_tr.tolist()) | set(y_te.tolist())))
    if best["best_model"] == "Fusion" and deep_ok:
        b = best["best_b"]; alpha = best["best_alpha"]

        # rebuild shallow scores
        sh_tf = fit_pca_eda_stable(tr_sh[b], y_tr, pca_dim=256)
        Ztr_sh = sh_tf(tr_sh[b]); Zte_sh = sh_tf(te_sh[b])
        cls2, P2 = fit_prototypes(Ztr_sh, y_tr)
        _, St_sh = predict_with_protos(Zte_sh, cls2, P2)

        # rebuild deep scores
        de_tf = fit_pca_eda_stable(Xtr_de, y_tr, pca_dim=128)
        Ztr_de = de_tf(Xtr_de); Zte_de = de_tf(Xte_de)
        cls, P = fit_prototypes(Ztr_de, y_tr)
        _, St_de = predict_with_protos(Zte_de, cls, P)

        all_cls = np.array(sorted(set(cls.tolist()) | set(cls2.tolist())))

        def align(scores, cls_src):
            m = {c:i for i,c in enumerate(cls_src)}
            out = np.zeros((scores.shape[0], len(all_cls)), dtype=np.float32)
            for j,c in enumerate(all_cls):
                out[:, j] = scores[:, m[c]] if c in m else -1e9
            return out

        St_f = alpha*align(St_de, cls) + (1-alpha)*align(St_sh, cls2)
        y_pred = all_cls[np.argmax(St_f, axis=1)]
        cm_path = run_dir / f"cm_fusion_b{b}_a{alpha:.1f}.png"
        save_confusion_matrix(y_te, y_pred, all_cls, cm_path, f"Paper-core CM (load={load}) Fusion b={b} a={alpha:.1f}")
    else:
        # best shallow
        b = best["best_b"]
        sh_tf = fit_pca_eda_stable(tr_sh[b], y_tr, pca_dim=256)
        Ztr_sh = sh_tf(tr_sh[b]); Zte_sh = sh_tf(te_sh[b])
        cls2, P2 = fit_prototypes(Ztr_sh, y_tr)
        y_pred, _ = predict_with_protos(Zte_sh, cls2, P2)
        cm_path = run_dir / f"cm_shallow_b{b}.png"
        save_confusion_matrix(y_te, y_pred, labels, cm_path, f"Paper-core CM (load={load}) MBH-LPQ b={b}")

    # metrics.csv (required): one-line summary + key numbers
    best_row = paper_core_metrics.sort_values("acc_val", ascending=False).head(1).iloc[0].to_dict()
    metrics = pd.DataFrame([{
        "paper_core_load": load,
        "best_model": best["best_model"],
        "best_b": best["best_b"],
        "best_alpha": best["best_alpha"],
        "best_acc_val": best["best_acc_val"],
        "best_acc_test": float(best_row.get("acc_test", np.nan)),
        "deep_ok": bool(deep_ok)
    }])
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    return {
        "deep_ok": deep_ok,
        "best": best,
        "artifacts": [
            run_dir / "paper_core_metrics.csv",
            run_dir / "ws_sweep.csv",
            run_dir / "ablation_summary.csv",
            run_dir / "metrics.csv",
            cm_path
        ]
    }


# --------------------------
# Phase11 analysis from Phase10 outputs (optional)
# --------------------------
def run_phase11_analysis(root: Path, phase10_dir: Path, out_dir: Path):
    out_dir = ensure_dir(out_dir)

    sum_csv = phase10_dir / "domain_shift_summary.csv"
    mat_csv = phase10_dir / "domain_shift_matrix.csv"
    if (not sum_csv.exists()) or (not mat_csv.exists()):
        print("[Phase12] Phase11 skip: không thấy Phase10 outputs:", str(sum_csv), str(mat_csv))
        return {"artifacts": []}

    sdf = pd.read_csv(sum_csv)
    piv = pd.read_csv(mat_csv, index_col=0)

    # heatmap
    plt.figure(figsize=(7,6))
    plt.imshow(piv.values, aspect="auto")
    plt.title("Domain shift accuracy (train load A -> test load B)")
    plt.xticks(range(piv.shape[1]), piv.columns.astype(str))
    plt.yticks(range(piv.shape[0]), piv.index.astype(str))
    plt.xlabel("test_load")
    plt.ylabel("train_load")
    plt.colorbar()
    plt.tight_layout()
    heat_png = out_dir / "heatmap_domain_shift.png"
    plt.savefig(heat_png, dpi=200)
    plt.close()

    # diag/off stats
    M = piv.values.astype(float)
    diag = np.diag(M)
    mask = ~np.eye(M.shape[0], dtype=bool)
    off = M[mask]

    stats = {
        "diag_mean": float(np.mean(diag)),
        "diag_std": float(np.std(diag)),
        "off_mean": float(np.mean(off)),
        "off_std": float(np.std(off)),
        "gap_diag_minus_off": float(np.mean(diag) - np.mean(off)),
        "min_diag": float(np.min(diag)),
        "max_diag": float(np.max(diag)),
        "min_off": float(np.min(off)),
        "max_off": float(np.max(off)),
    }
    stats_df = pd.DataFrame([stats])
    stats_csv = out_dir / "phase11_diag_off_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    return {"artifacts": [heat_png, stats_csv]}


# --------------------------
# Main run_all
# --------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config JSON")
    args = ap.parse_args()

    # ROOT = parent of experiments/
    root = Path(__file__).resolve().parents[1]

    cfg_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    # run id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_hash = sha256_bytes(cfg_path.read_bytes())[:12]
    run_id = f"{ts}_{cfg_hash}"

    out_root = root / cfg.get("out_root", "results/run_all")
    run_dir = ensure_dir(out_root / run_id)

    # save a copy config inside run folder
    (run_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # scan CWRU
    cwru_root = root / cfg["cwru_root"]
    df_meta = scan_metadata(cwru_root)

    artifacts = []
    print("[Phase12] RUN paper-core -> required artifacts ...")
    pc_res = run_paper_core(df_meta, cfg, root=root, run_dir=run_dir)
    artifacts += pc_res["artifacts"]

    # optional Phase11 analysis from Phase10 outputs
    if bool(cfg.get("run_phase11", True)):
        phase10_dir = root / cfg.get("phase10_dir", "results/phase10")
        phase11_out = root / cfg.get("phase11_out_dir", "results/phase11")
        print("[Phase12] RUN phase11 analysis (optional) ...")
        p11 = run_phase11_analysis(root=root, phase10_dir=phase10_dir, out_dir=phase11_out)
        artifacts += p11["artifacts"]

    # manifest
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "config_path": safe_relpath(cfg_path, root),
        "config_sha256": sha256_file(cfg_path),
        "run_id": run_id,
        "git_commit": get_git_commit(root),
        "versions": get_versions(["python","numpy","pandas","scikit-learn","scipy","matplotlib","torch","torchaudio","librosa"]),
        "artifacts": [safe_relpath(p, root) for p in artifacts if p is not None and Path(p).exists()]
    }

    # must be results/manifest.json (per requirement)
    results_dir = ensure_dir(root / "results")
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n[Phase12] DONE")
    print("Run folder:", str(run_dir.resolve()))
    print("Manifest:", str((results_dir / "manifest.json").resolve()))
    print("Required files present?")
    required = ["metrics.csv", "paper_core_metrics.csv", "ws_sweep.csv", "ablation_summary.csv"]
    for r in required:
        print(" -", r, "=>", (run_dir / r).exists())
    cm_files = list(run_dir.glob("cm_*.png"))
    print(" - cm_*.png =>", len(cm_files), "file(s)")
    if cm_files:
        print("   ", cm_files[0].name)

if __name__ == "__main__":
    main()
