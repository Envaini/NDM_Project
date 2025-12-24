import os, re, json, math
from pathlib import Path
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

# --------------------------
# Log-mel (paper-like): output (96, 64)
# --------------------------
def logmel_96x64(
    wave: np.ndarray,
    sr: int = 48000,
    n_fft: int = 1024,
    hop: int = 256,
    n_mels: int = 64,
    n_frames: int = 96
):
    import librosa
    w = wave.astype(np.float32)
    w = w - w.mean()
    w = w / (np.max(np.abs(w)) + 1e-9)

    S = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    logS = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)

    img = logS.T  # (T, 64)
    T = img.shape[0]
    if T >= n_frames:
        img = img[:n_frames, :]
    else:
        pad = n_frames - T
        img = np.pad(img, ((0, pad), (0, 0)), mode="edge")

    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn + 1e-9)
    return img.astype(np.float32)  # (96, 64)

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
    return np.concatenate(feats, axis=0)

# --------------------------
# PCA + (ổn định) "EDA" -> dùng generalized eigen trực tiếp (Sb, Sw)
# Tránh expm(...) vì overflow -> inf/NaN
# --------------------------
def fit_pca_eda(Xtr: np.ndarray, ytr: np.ndarray, pca_dim: int = 128, out_dim: int = None, reg: float = 1e-4):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xtr)

    d = Xs.shape[1]
    n = Xs.shape[0]
    p = min(pca_dim, d, max(2, n-1))
    pca = PCA(n_components=p, random_state=0)
    Xp = pca.fit_transform(Xs).astype(np.float64)

    classes = np.unique(ytr)
    mu = Xp.mean(axis=0, keepdims=True)

    Sw = np.zeros((p, p), dtype=np.float64)
    Sb = np.zeros((p, p), dtype=np.float64)

    for c in classes:
        Xc = Xp[ytr == c]
        muc = Xc.mean(axis=0, keepdims=True)
        Sw += (Xc - muc).T @ (Xc - muc)
        Sb += Xc.shape[0] * (muc - mu).T @ (muc - mu)

    # chuẩn hoá + regularize để Sw SPD
    Sw = (Sw + Sw.T) / 2.0
    Sb = (Sb + Sb.T) / 2.0
    Sw += reg * np.eye(p)

    # generalized eigen: Sb v = lambda Sw v
    w, V = la.eigh(Sb, Sw, check_finite=True)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]

    if out_dim is None:
        out_dim = min(len(classes) - 1, p)
        out_dim = max(2, out_dim)
    W = V[:, :out_dim].astype(np.float32)

    def transform(X: np.ndarray):
        Xs2 = scaler.transform(X)
        Xp2 = pca.transform(Xs2).astype(np.float32)
        Z = Xp2 @ W
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
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
# Main experiment
# --------------------------
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

def build_domain_samples(df: pd.DataFrame, load: int, n_per_class: int, seg_len: int, sr: int, seed: int):
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

def run_one_pair(
    df,
    train_load,
    test_load,
    outdir: Path,
    n_per_class=120,
    seg_len=48000,   # FIX: để VGGish ổn định (khoảng 1s ở 48kHz)
    sr=48000,
    lpq_R=7,
    b_list=(1,2,4,6,8,10,12),
    alpha_list=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    seed=42
):
    ensure_dir(outdir)
    seed_everything(seed)

    Xtr_wave, ytr = build_domain_samples(df, train_load, n_per_class, seg_len, sr, seed=seed)
    Xte_wave, yte = build_domain_samples(df, test_load,  n_per_class, seg_len, sr, seed=seed+123)

    Xtr_wave, Xval_wave, ytr, yval = train_test_split(
        Xtr_wave, ytr, test_size=0.2, random_state=seed, stratify=ytr
    )

    deep_ok = True
    try:
        import torch
        import torchaudio
        from torchaudio.prototype.pipelines import VGGISH
        bundle = VGGISH
        model = bundle.get_model()
        model.eval()
        iproc = bundle.get_input_processor()
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    except Exception as e:
        deep_ok = False
        model = None
        iproc = None
        resampler = None
        print("[WARN] Không load được torchaudio VGGISH -> bỏ deep branch. Lỗi:", repr(e))

    def vggish_embed_batch(Xwave: np.ndarray):
        if not deep_ok:
            return None
        import torch

        feats = []
        with torch.no_grad():
            for w in Xwave:
                t = torch.from_numpy(w).float().reshape(-1)
                t = t - t.mean()
                t = t / (t.abs().max() + 1e-9)

                # resampler thường thích (C, T)
                if t.ndim == 1:
                    t = t.unsqueeze(0)  # (1, T)
                t16 = resampler(t).squeeze(0).contiguous().reshape(-1)  # (T16,)

                # đảm bảo đủ dài để iproc không trả batch rỗng
                min_len16 = 16000  # ~1s
                if t16.numel() < min_len16:
                    pad = min_len16 - t16.numel()
                    t16 = torch.nn.functional.pad(t16, (0, pad), mode="constant", value=0.0)

                try:
                    inp = iproc(t16)
                except TypeError:
                    inp = iproc(t16, sample_rate=16000)

                # nếu batch rỗng -> fallback: pad thêm và chạy lại 1 lần
                if hasattr(inp, "shape") and inp.shape[0] == 0:
                    t16 = torch.nn.functional.pad(t16, (0, min_len16), mode="constant", value=0.0)
                    try:
                        inp = iproc(t16)
                    except TypeError:
                        inp = iproc(t16, sample_rate=16000)

                if hasattr(inp, "shape") and inp.shape[0] == 0:
                    # vẫn rỗng: trả embedding 0 để không crash
                    feats.append(np.zeros((128,), dtype=np.float32))
                    continue

                out = model(inp)
                if out.ndim == 3:
                    v = out.mean(dim=1).squeeze(0)
                else:
                    v = out.squeeze(0)
                feats.append(v.cpu().numpy().astype(np.float32))

        return np.stack(feats, axis=0)

    def shallow_feats_for_wave_batch(Xwave: np.ndarray):
        feats_b = {b: [] for b in b_list}
        audit_saved = False

        for w in Xwave:
            img = logmel_96x64(w, sr=sr)
            if not audit_saved:
                plt.figure()
                plt.imshow(img.T, aspect="auto", origin="lower")
                plt.title("Audit log-mel (paper-like)")
                plt.tight_layout()
                audit_dir = ensure_dir(outdir / "audit")
                plt.savefig(audit_dir / "logmel_example.png", dpi=200)
                plt.close()
                audit_saved = True

            code = lpq_codes(img, R=lpq_R)
            for b in b_list:
                feats_b[b].append(mbh_lpq_feature(code, b=b))

        for b in b_list:
            feats_b[b] = np.stack(feats_b[b], axis=0).astype(np.float32)
        return feats_b

    print(f"[Phase10] Extract shallow (MBH-LPQ) train/val/test ...")
    tr_sh = shallow_feats_for_wave_batch(Xtr_wave)
    va_sh = shallow_feats_for_wave_batch(Xval_wave)
    te_sh = shallow_feats_for_wave_batch(Xte_wave)

    Xtr_de = Xval_de = Xte_de = None
    if deep_ok:
        print(f"[Phase10] Extract deep (VGGish) train/val/test ...")
        Xtr_de = vggish_embed_batch(Xtr_wave)
        Xval_de = vggish_embed_batch(Xval_wave)
        Xte_de = vggish_embed_batch(Xte_wave)

        # nếu có NaN/Inf thì dập ngay
        Xtr_de = np.nan_to_num(Xtr_de, nan=0.0, posinf=0.0, neginf=0.0)
        Xval_de = np.nan_to_num(Xval_de, nan=0.0, posinf=0.0, neginf=0.0)
        Xte_de = np.nan_to_num(Xte_de, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    best = {"best_b": None, "best_alpha": None, "acc_val": -1}

    # deep-only
    if deep_ok:
        de_tf = fit_pca_eda(Xtr_de, ytr, pca_dim=128)
        Ztr_de = de_tf(Xtr_de); Zva_de = de_tf(Xval_de); Zte_de = de_tf(Xte_de)
        cls, P = fit_prototypes(Ztr_de, ytr)
        pv, sv = predict_with_protos(Zva_de, cls, P)
        pt, st = predict_with_protos(Zte_de, cls, P)
        rows.append({
            "model": "VGGish", "b": None, "alpha": 1.0,
            "acc_val": accuracy_score(yval, pv), "acc_test": accuracy_score(yte, pt),
            "f1_val": f1_score(yval, pv, average="macro"), "f1_test": f1_score(yte, pt, average="macro")
        })
    else:
        sv = st = None

    for b in b_list:
        sh_tf = fit_pca_eda(tr_sh[b], ytr, pca_dim=256)
        Ztr_sh = sh_tf(tr_sh[b]); Zva_sh = sh_tf(va_sh[b]); Zte_sh = sh_tf(te_sh[b])

        cls2, P2 = fit_prototypes(Ztr_sh, ytr)
        pv_sh, sv_sh = predict_with_protos(Zva_sh, cls2, P2)
        pt_sh, st_sh = predict_with_protos(Zte_sh, cls2, P2)

        rows.append({
            "model": "MBH-LPQ", "b": b, "alpha": 0.0,
            "acc_val": accuracy_score(yval, pv_sh), "acc_test": accuracy_score(yte, pt_sh),
            "f1_val": f1_score(yval, pv_sh, average="macro"), "f1_test": f1_score(yte, pt_sh, average="macro")
        })

        if deep_ok:
            for alpha in alpha_list:
                all_cls = np.array(sorted(set(cls.tolist()) | set(cls2.tolist())))

                def align_scores(scores, cls_src):
                    m = {c:i for i,c in enumerate(cls_src)}
                    out = np.full((scores.shape[0], len(all_cls)), -1e9, dtype=np.float32)
                    for j, c in enumerate(all_cls):
                        if c in m:
                            out[:, j] = scores[:, m[c]]
                    return out

                Sv_de = align_scores(sv, cls)
                Sv_sh = align_scores(sv_sh, cls2)
                St_de = align_scores(st, cls)
                St_sh = align_scores(st_sh, cls2)

                Sv = alpha*Sv_de + (1-alpha)*Sv_sh
                St = alpha*St_de + (1-alpha)*St_sh

                pv = all_cls[np.argmax(Sv, axis=1)]
                pt = all_cls[np.argmax(St, axis=1)]

                accv = accuracy_score(yval, pv)
                acct = accuracy_score(yte, pt)

                rows.append({
                    "model": "Fusion", "b": b, "alpha": float(alpha),
                    "acc_val": accv, "acc_test": acct,
                    "f1_val": f1_score(yval, pv, average="macro"),
                    "f1_test": f1_score(yte, pt, average="macro")
                })

                if accv > best["acc_val"]:
                    best.update({"best_b": b, "best_alpha": float(alpha), "acc_val": float(accv)})

    dfm = pd.DataFrame(rows)
    dfm.to_csv(outdir / "domain_shift_metrics.csv", index=False)
    (outdir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # confusion matrix
    if deep_ok and best["best_b"] is not None:
        b = best["best_b"]; alpha = best["best_alpha"]

        sh_tf = fit_pca_eda(tr_sh[b], ytr, pca_dim=256)
        Ztr_sh = sh_tf(tr_sh[b]); Zte_sh = sh_tf(te_sh[b])
        cls2, P2 = fit_prototypes(Ztr_sh, ytr)
        _, St_sh = predict_with_protos(Zte_sh, cls2, P2)

        de_tf = fit_pca_eda(Xtr_de, ytr, pca_dim=128)
        Ztr_de = de_tf(Xtr_de); Zte_de = de_tf(Xte_de)
        cls, P = fit_prototypes(Ztr_de, ytr)
        _, St_de = predict_with_protos(Zte_de, cls, P)

        all_cls = np.array(sorted(set(cls.tolist()) | set(cls2.tolist())))

        def align(scores, cls_src):
            m = {c:i for i,c in enumerate(cls_src)}
            out = np.full((scores.shape[0], len(all_cls)), -1e9, dtype=np.float32)
            for j, c in enumerate(all_cls):
                if c in m:
                    out[:, j] = scores[:, m[c]]
            return out

        St = alpha*align(St_de, cls) + (1-alpha)*align(St_sh, cls2)
        ypred = all_cls[np.argmax(St, axis=1)]
    else:
        b = dfm[dfm["model"]=="MBH-LPQ"].sort_values("acc_val", ascending=False).iloc[0]["b"]
        sh_tf = fit_pca_eda(tr_sh[int(b)], ytr, pca_dim=256)
        Ztr_sh = sh_tf(tr_sh[int(b)]); Zte_sh = sh_tf(te_sh[int(b)])
        cls2, P2 = fit_prototypes(Ztr_sh, ytr)
        ypred, _ = predict_with_protos(Zte_sh, cls2, P2)
        all_cls = np.array(sorted(set(ytr.tolist())))

    cm = confusion_matrix(yte, ypred, labels=all_cls)

    plt.figure(figsize=(7,6))
    plt.imshow(cm, aspect="auto")
    plt.title(f"Confusion Matrix (train_load={train_load}, test_load={test_load})")
    plt.xticks(range(len(all_cls)), all_cls, rotation=45, ha="right")
    plt.yticks(range(len(all_cls)), all_cls)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir / "cm_domain_shift.png", dpi=200)
    plt.close()

    fus = dfm[dfm["model"]=="Fusion"].copy()
    if not fus.empty:
        bs = fus.groupby("b")[["acc_val","acc_test"]].max().reset_index()
        bs.to_csv(outdir / "b_sweep.csv", index=False)
        plt.figure()
        plt.plot(bs["b"], bs["acc_val"], marker="o", label="val")
        plt.plot(bs["b"], bs["acc_test"], marker="o", label="test")
        plt.xlabel("b (sub-blocks)"); plt.ylabel("Accuracy"); plt.title("b_sweep")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "b_sweep_plot.png", dpi=200); plt.close()

        ws = fus.groupby("alpha")[["acc_val","acc_test"]].max().reset_index()
        ws.to_csv(outdir / "ws_sweep.csv", index=False)
        plt.figure()
        plt.plot(ws["alpha"], ws["acc_val"], marker="o", label="val")
        plt.plot(ws["alpha"], ws["acc_test"], marker="o", label="test")
        plt.xlabel("alpha (weight deep)"); plt.ylabel("Accuracy"); plt.title("ws_sweep")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "ws_sweep_plot.png", dpi=200); plt.close()

    best_row = dfm[dfm["model"]=="Fusion"].sort_values("acc_val", ascending=False).head(1)
    if best_row.empty:
        best_row = dfm.sort_values("acc_val", ascending=False).head(1)

    r = best_row.iloc[0].to_dict()
    r.update({"train_load": train_load, "test_load": test_load})
    return r

def main():
    seed = 42
    cwru_root = Path("data/raw/CWRU")
    out_root = ensure_dir(Path("results/phase10"))

    df = scan_metadata(cwru_root)
    loads = sorted(df["load"].unique().tolist())
    print("[Phase10] Loads found:", loads)
    print(df.groupby(["load","label"]).size().head(40))

    summary = []
    for tr in loads:
        for te in loads:
            od = out_root / f"train{tr}_test{te}"
            print(f"\n[Phase10] RUN train_load={tr} -> test_load={te}")
            row = run_one_pair(df, tr, te, od, seed=seed)
            summary.append(row)

    sdf = pd.DataFrame(summary)
    sdf.to_csv(out_root / "domain_shift_summary.csv", index=False)

    piv = sdf.pivot_table(index="train_load", columns="test_load", values="acc_test", aggfunc="max")
    piv.to_csv(out_root / "domain_shift_matrix.csv")

    print("\n[Phase10] DONE ->", str(out_root.resolve()))
    print("- domain_shift_summary.csv")
    print("- domain_shift_matrix.csv")
    print("- các folder trainA_testB/ chứa cm + sweep + audit")

if __name__ == "__main__":
    main()
