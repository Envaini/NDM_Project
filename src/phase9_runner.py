import os
import json
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import yaml
from scipy.io import loadmat
from scipy import linalg
from scipy.ndimage import convolve1d
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def find_1d_signal(mat: Dict) -> np.ndarray:
    cand = []
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
            a = np.squeeze(v)
            if a.ndim == 1 and a.size >= 1000:
                cand.append((a.size, k, a))
    if not cand:
        raise ValueError("Không tìm thấy tín hiệu 1D trong file .mat")
    cand.sort(key=lambda x: x[0], reverse=True)
    return cand[0][2].astype(np.float64)


def normalize_paper(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean()
    d = np.max(np.abs(x))
    if d < 1e-12:
        return x * 0.0
    return x / d


def segment_waveform(x: np.ndarray, seg_len: int, n_seg: int) -> np.ndarray:
    x = np.asarray(x).ravel()
    if x.size < seg_len:
        x = np.pad(x, (0, seg_len - x.size))
    max_start = max(0, x.size - seg_len)
    if max_start == 0:
        starts = np.zeros(n_seg, dtype=int)
    else:
        starts = np.linspace(0, max_start, num=n_seg, dtype=int)
    return np.stack([x[s:s + seg_len] for s in starts], axis=0)


def logmel_paper_like(x: np.ndarray, fs: int, n_fft: int, hop_length: int,
                     win_length: int, mel_bins: int, log_eps: float, target_frames: int) -> np.ndarray:
    import librosa

    S = librosa.feature.melspectrogram(
        y=x.astype(np.float32),
        sr=fs,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        center=False,
        power=2.0,
        n_mels=mel_bins,
        fmin=0.0,
        fmax=fs / 2,
    )
    S = np.log(S + log_eps).T  # (T, mel)

    T = S.shape[0]
    if T != target_frames:
        t_old = np.linspace(0.0, 1.0, num=T)
        t_new = np.linspace(0.0, 1.0, num=target_frames)
        out = np.empty((target_frames, S.shape[1]), dtype=np.float32)
        for m in range(S.shape[1]):
            out[:, m] = np.interp(t_new, t_old, S[:, m]).astype(np.float32)
    else:
        out = S.astype(np.float32)

    mn, mx = float(out.min()), float(out.max())
    if mx - mn > 1e-9:
        out = (out - mn) / (mx - mn)
    return out.astype(np.float32)  # (96,64)


def hist256(code_img: np.ndarray) -> np.ndarray:
    h = np.bincount(code_img.ravel().astype(np.int64), minlength=256).astype(np.float64)
    s = h.sum()
    if s > 0:
        h /= s
    return h


def lpq_code_image(img: np.ndarray, R: int) -> np.ndarray:
    img = img.astype(np.float64, copy=False)
    M = 2 * R + 1
    x = np.arange(-(M // 2), M // 2 + 1, dtype=np.float64)

    w0 = np.ones(M, dtype=np.complex128)
    w1 = np.exp(-2j * np.pi * x / M).astype(np.complex128)

    def filt_xy(wx, wy):
        tmp = convolve1d(img, wx, axis=1, mode="reflect")
        out = convolve1d(tmp, wy, axis=0, mode="reflect")
        return out

    F1 = filt_xy(w1, w0)
    F2 = filt_xy(w0, w1)
    F3 = filt_xy(w1, w1)
    F4 = filt_xy(w1, np.conj(w1))

    comps = [np.real(F1), np.imag(F1),
             np.real(F2), np.imag(F2),
             np.real(F3), np.imag(F3),
             np.real(F4), np.imag(F4)]
    bits = [(c >= 0).astype(np.uint8) for c in comps]

    code = np.zeros(img.shape, dtype=np.uint8)
    for i, b in enumerate(bits):
        code |= (b << i)
    return code


def basic_lpq_feature(img: np.ndarray, R: int) -> np.ndarray:
    return hist256(lpq_code_image(img, R=R))


def mbh_lpq_feature(img: np.ndarray, R: int, b: int) -> np.ndarray:
    code = lpq_code_image(img, R=R)
    blocks = np.array_split(code, b, axis=0)
    return np.concatenate([hist256(bl) for bl in blocks], axis=0)


def lbp_feature(img: np.ndarray, P: int, R: int, method: str) -> np.ndarray:
    from skimage.feature import local_binary_pattern
    lbp = local_binary_pattern(img, P=P, R=R, method=method).astype(np.uint8)
    return hist256(lbp)


def ldp_code_image(img: np.ndarray, k_top: int = 3) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    masks = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=np.float32),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=np.float32),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=np.float32),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=np.float32),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=np.float32),
    ]
    H, W = img.shape
    pad = np.pad(img, ((1, 1), (1, 1)), mode="reflect")
    resp = np.zeros((8, H, W), dtype=np.float32)
    for i, K in enumerate(masks):
        r = (
            K[0, 0] * pad[0:H, 0:W] + K[0, 1] * pad[0:H, 1:W + 1] + K[0, 2] * pad[0:H, 2:W + 2] +
            K[1, 0] * pad[1:H + 1, 0:W] + K[1, 1] * pad[1:H + 1, 1:W + 1] + K[1, 2] * pad[1:H + 1, 2:W + 2] +
            K[2, 0] * pad[2:H + 2, 0:W] + K[2, 1] * pad[2:H + 2, 1:W + 1] + K[2, 2] * pad[2:H + 2, 2:W + 2]
        )
        resp[i] = r

    topk = np.argpartition(resp, -k_top, axis=0)[-k_top:]
    code = np.zeros((H, W), dtype=np.uint8)
    for j in range(k_top):
        idx = topk[j]
        code |= (1 << idx).astype(np.uint8)
    return code


def ldp_feature(img: np.ndarray, k_top: int) -> np.ndarray:
    return hist256(ldp_code_image(img, k_top=k_top))


def vggish_feature_from_waveform(x: np.ndarray, fs: int, device: str) -> np.ndarray:
    import torch
    import torchaudio

    # cố thử pipeline VGGISH của torchaudio
    try:
        from torchaudio.prototype.pipelines import VGGISH
        input_sr = VGGISH.sample_rate
        proc = VGGISH.get_input_processor()
        model = VGGISH.get_model().to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(
            "VGGish không chạy được do torchaudio version. "
            "Cách fix nhanh: vào protocol.yaml -> deep.vggish.enabled=false (dùng VGG16). "
            f"Chi tiết lỗi: {e}"
        )

    wav = torch.from_numpy(x.astype(np.float32))
    wav = torchaudio.functional.resample(wav, fs, input_sr)

    min_len = int(input_sr * 1.0)
    if wav.numel() < min_len:
        wav = torch.nn.functional.pad(wav, (0, min_len - wav.numel()))

    with torch.no_grad():
        ex = proc(wav.to(device))
        emb = model(ex)
        if emb.ndim == 2:
            emb = emb.mean(dim=0)
        else:
            emb = emb.reshape(-1).mean()
    return emb.detach().cpu().numpy().astype(np.float32).ravel()


def vgg16_feature_batch(logmels: np.ndarray, layer: str, device: str, batch_size: int = 32) -> np.ndarray:
    import torch
    import torchvision
    import torchvision.transforms as T

    weights = torchvision.models.VGG16_Weights.DEFAULT
    model = torchvision.models.vgg16(weights=weights).to(device)
    model.eval()

    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    outs = []
    N = logmels.shape[0]
    for s in range(0, N, batch_size):
        x = torch.from_numpy(logmels[s:s + batch_size]).unsqueeze(1)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)
        x = norm(x)

        with torch.no_grad():
            f = model.features(x.to(device))
            f = model.avgpool(f)
            f = torch.flatten(f, 1)

            fc6 = model.classifier[2](model.classifier[1](model.classifier[0](f)))
            fc7 = model.classifier[5](model.classifier[4](model.classifier[3](fc6)))
            fc8 = model.classifier[6](fc7)

        if layer == "fc6":
            outs.append(fc6.cpu().numpy())
        elif layer == "fc7":
            outs.append(fc7.cpu().numpy())
        elif layer == "fc8":
            outs.append(fc8.cpu().numpy())
        else:
            raise ValueError("layer phải là fc6/fc7/fc8")

    return np.vstack(outs).astype(np.float32)


def fit_eda(X: np.ndarray, y: np.ndarray, reg_eps: float, n_components: int) -> np.ndarray:
    X = X.astype(np.float64)
    y = y.astype(np.int64)
    classes = np.unique(y)

    mu = X.mean(axis=0, keepdims=True)
    Sb = np.zeros((X.shape[1], X.shape[1]), dtype=np.float64)
    Sw = np.zeros_like(Sb)

    for c in classes:
        Xc = X[y == c]
        muc = Xc.mean(axis=0, keepdims=True)
        d = (muc - mu)
        Sb += Xc.shape[0] * (d.T @ d)
        X0 = Xc - muc
        Sw += (X0.T @ X0)

    Sw += reg_eps * np.eye(Sw.shape[0], dtype=np.float64)
    Sb = Sb / (np.trace(Sb) + 1e-12)
    Sw = Sw / (np.trace(Sw) + 1e-12)

    eSb = linalg.expm(Sb)
    eSw = linalg.expm(Sw)
    vals, vecs = linalg.eigh(eSb, eSw)
    idx = np.argsort(vals)[::-1]
    W = vecs[:, idx[:n_components]]
    return W.astype(np.float64)


def cosine_scores(Z: np.ndarray, cents: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float64)
    C = cents.astype(np.float64)
    Zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    Cn = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    return (Z / Zn) @ (C / Cn).T


def train_eval_branch(Xtr, ytr, Xva, yva, Xte, yte, pca_dim: int,
                      use_pca: bool, use_eda: bool, reg_eps: float, seed: int):
    classes = np.unique(ytr)

    if use_pca:
        pca = PCA(n_components=min(pca_dim, Xtr.shape[1]), random_state=seed)
        Xtr2 = pca.fit_transform(Xtr)
        Xva2 = pca.transform(Xva)
        Xte2 = pca.transform(Xte)
    else:
        Xtr2, Xva2, Xte2 = Xtr, Xva, Xte

    if use_eda:
        n_comp = min(len(classes) - 1, Xtr2.shape[1])
        W = fit_eda(Xtr2, ytr, reg_eps=reg_eps, n_components=n_comp)
        Ztr = Xtr2 @ W
        Zva = Xva2 @ W
        Zte = Xte2 @ W
    else:
        Ztr, Zva, Zte = Xtr2, Xva2, Xte2

    cents = np.vstack([Ztr[ytr == c].mean(axis=0) for c in classes])
    s_va = cosine_scores(Zva, cents)
    s_te = cosine_scores(Zte, cents)

    yhat_va = classes[np.argmax(s_va, axis=1)]
    yhat_te = classes[np.argmax(s_te, axis=1)]

    return {
        "scores_val": s_va,
        "scores_test": s_te,
        "yhat_val": yhat_va,
        "yhat_test": yhat_te,
        "acc_val": float(accuracy_score(yva, yhat_va)),
        "acc_test": float(accuracy_score(yte, yhat_te)),
        "f1_val": float(f1_score(yva, yhat_va, average="macro")),
        "f1_test": float(f1_score(yte, yhat_te, average="macro")),
    }


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str]


def build_cwru_splits(cfg: dict, project_root: str) -> SplitData:
    root = os.path.join(project_root, cfg["data"]["cwru_root"])
    fs = int(cfg["data"]["fs_hz"])
    seg_len = int(cfg["data"]["sample_len"])
    n_seg = int(cfg["data"]["samples_per_class"])

    n_tr = int(cfg["data"]["split"]["train_after_val_per_class"])
    n_val = int(cfg["data"]["split"]["val_per_class"])
    n_te = int(cfg["data"]["split"]["test_per_class"])

    classes_map = cfg["data"]["classes_10"]
    class_names = list(classes_map.keys())

    Xtr, ytr, Xva, yva, Xte, yte = [], [], [], [], [], []

    for ci, cname in enumerate(class_names):
        fpath = os.path.join(root, classes_map[cname])
        mat = loadmat(fpath)
        sig = find_1d_signal(mat)

        segs = segment_waveform(sig, seg_len=seg_len, n_seg=n_seg)
        segs = np.stack([normalize_paper(s) for s in segs], axis=0)

        Xtr.append(segs[:n_tr]);                 ytr.append(np.full(n_tr, ci, dtype=np.int64))
        Xva.append(segs[n_tr:n_tr+n_val]);       yva.append(np.full(n_val, ci, dtype=np.int64))
        Xte.append(segs[n_tr+n_val:n_tr+n_val+n_te]); yte.append(np.full(n_te, ci, dtype=np.int64))

    return SplitData(
        X_train=np.concatenate(Xtr, axis=0),
        y_train=np.concatenate(ytr, axis=0),
        X_val=np.concatenate(Xva, axis=0),
        y_val=np.concatenate(yva, axis=0),
        X_test=np.concatenate(Xte, axis=0),
        y_test=np.concatenate(yte, axis=0),
        class_names=class_names,
    )


def run_phase9(protocol_path: str = "original_paper/protocol.yaml") -> None:
    project_root = os.getcwd()
    with open(protocol_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    out_dir = cfg["output"]["results_dir"]
    cache_dir = cfg["output"]["cache_dir"]
    ensure_dir(out_dir)
    ensure_dir(cache_dir)
    ensure_dir(os.path.join(out_dir, "audit"))

    device = "cpu"
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass

    split = build_cwru_splits(cfg, project_root)
    fs = int(cfg["data"]["fs_hz"])

    sp = cfg["preprocess"]["spectrogram"]
    def npy(name): return os.path.join(cache_dir, name)

    if not os.path.isfile(npy("logmel_train.npy")):
        print("[Phase9] Extract log-mel (paper-like)...")
        logmel_train = np.stack([logmel_paper_like(x, fs, **sp) for x in tqdm(split.X_train)], axis=0)
        logmel_val   = np.stack([logmel_paper_like(x, fs, **sp) for x in tqdm(split.X_val)], axis=0)
        logmel_test  = np.stack([logmel_paper_like(x, fs, **sp) for x in tqdm(split.X_test)], axis=0)
        np.save(npy("logmel_train.npy"), logmel_train)
        np.save(npy("logmel_val.npy"), logmel_val)
        np.save(npy("logmel_test.npy"), logmel_test)
    else:
        logmel_train = np.load(npy("logmel_train.npy"))
        logmel_val   = np.load(npy("logmel_val.npy"))
        logmel_test  = np.load(npy("logmel_test.npy"))

    plt.figure(figsize=(5,3))
    plt.imshow(logmel_train[0].T, aspect="auto", origin="lower")
    plt.title("Audit log-mel (paper-like)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "audit", "logmel_example.png"), dpi=200)
    plt.close()

    use_pca = bool(cfg["reduce_and_classify"]["pca"]["enabled"])
    pca_dim = int(cfg["reduce_and_classify"]["pca"]["n_components"])
    use_eda = bool(cfg["reduce_and_classify"]["eda"]["enabled"])
    reg_eps = float(cfg["reduce_and_classify"]["eda"]["reg_eps"])

    rows = []
    sh = cfg["features"]["shallow"]

    if sh["lbp"]["enabled"]:
        P = int(sh["lbp"]["P"]); R = int(sh["lbp"]["R"]); method = str(sh["lbp"]["method"])
        print("[Phase9] LBP...")
        Xtr = np.stack([lbp_feature(im, P, R, method) for im in tqdm(logmel_train)], axis=0)
        Xva = np.stack([lbp_feature(im, P, R, method) for im in tqdm(logmel_val)], axis=0)
        Xte = np.stack([lbp_feature(im, P, R, method) for im in tqdm(logmel_test)], axis=0)
        res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
        rows.append({"model":"LBP", **{k:res[k] for k in ["acc_val","acc_test","f1_val","f1_test"]}})

    if sh["ldp"]["enabled"]:
        k_top = int(sh["ldp"]["k_top"])
        print("[Phase9] LDP...")
        Xtr = np.stack([ldp_feature(im, k_top) for im in tqdm(logmel_train)], axis=0)
        Xva = np.stack([ldp_feature(im, k_top) for im in tqdm(logmel_val)], axis=0)
        Xte = np.stack([ldp_feature(im, k_top) for im in tqdm(logmel_test)], axis=0)
        res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
        rows.append({"model":"LDP", **{k:res[k] for k in ["acc_val","acc_test","f1_val","f1_test"]}})

    if sh["lpq"]["enabled"]:
        for Rlpq in sh["lpq"]["R_grid"]:
            Rlpq = int(Rlpq)
            print(f"[Phase9] LPQ R={Rlpq}...")
            Xtr = np.stack([basic_lpq_feature(im, Rlpq) for im in tqdm(logmel_train)], axis=0)
            Xva = np.stack([basic_lpq_feature(im, Rlpq) for im in tqdm(logmel_val)], axis=0)
            Xte = np.stack([basic_lpq_feature(im, Rlpq) for im in tqdm(logmel_test)], axis=0)
            res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
            rows.append({"model":f"LPQ_R{Rlpq}", **{k:res[k] for k in ["acc_val","acc_test","f1_val","f1_test"]}})

    mbh = sh["mbh_lpq"]
    Rm = int(mbh["R"])
    b_grid = [int(x) for x in mbh["b_grid"]]

    best_b, best_val = None, -1.0
    best_mbh_val_scores, best_mbh_test_scores = None, None
    b_rows = []

    for b in b_grid:
        print(f"[Phase9] MBH-LPQ R={Rm}, b={b}...")
        Xtr = np.stack([mbh_lpq_feature(im, Rm, b) for im in tqdm(logmel_train)], axis=0)
        Xva = np.stack([mbh_lpq_feature(im, Rm, b) for im in tqdm(logmel_val)], axis=0)
        Xte = np.stack([mbh_lpq_feature(im, Rm, b) for im in tqdm(logmel_test)], axis=0)
        res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
        b_rows.append({"b":b, "acc_val":res["acc_val"], "acc_test":res["acc_test"]})
        if res["acc_val"] > best_val:
            best_val = res["acc_val"]
            best_b = b
            best_mbh_val_scores = res["scores_val"]
            best_mbh_test_scores = res["scores_test"]

    np.savetxt(os.path.join(out_dir, "b_sweep.csv"),
               np.array([[r["b"], r["acc_val"], r["acc_test"]] for r in b_rows]),
               delimiter=",", header="b,acc_val,acc_test", comments="")

    plt.figure()
    plt.plot([r["b"] for r in b_rows], [r["acc_val"] for r in b_rows], marker="o", label="val")
    plt.plot([r["b"] for r in b_rows], [r["acc_test"] for r in b_rows], marker="o", label="test")
    plt.xlabel("b (sub-blocks)"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "b_sweep_plot.png"), dpi=200)
    plt.close()

    deep = cfg["features"]["deep"]
    deep_scores_val = None
    deep_scores_test = None
    deep_name = None

    if deep["vggish"]["enabled"]:
        print("[Phase9] VGGish...")
        cache = os.path.join(cache_dir, "vggish_train.npy")
        if not os.path.isfile(cache):
            Xtr = np.stack([vggish_feature_from_waveform(x, fs, device) for x in tqdm(split.X_train)], axis=0)
            Xva = np.stack([vggish_feature_from_waveform(x, fs, device) for x in tqdm(split.X_val)], axis=0)
            Xte = np.stack([vggish_feature_from_waveform(x, fs, device) for x in tqdm(split.X_test)], axis=0)
            np.save(os.path.join(cache_dir, "vggish_train.npy"), Xtr)
            np.save(os.path.join(cache_dir, "vggish_val.npy"), Xva)
            np.save(os.path.join(cache_dir, "vggish_test.npy"), Xte)
        else:
            Xtr = np.load(os.path.join(cache_dir, "vggish_train.npy"))
            Xva = np.load(os.path.join(cache_dir, "vggish_val.npy"))
            Xte = np.load(os.path.join(cache_dir, "vggish_test.npy"))

        res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
        rows.append({"model":"VGGish", **{k:res[k] for k in ["acc_val","acc_test","f1_val","f1_test"]}})
        deep_scores_val, deep_scores_test = res["scores_val"], res["scores_test"]
        deep_name = "VGGish"

    if deep["vgg16"]["enabled"]:
        layer = str(deep["vgg16"]["layer"])
        print(f"[Phase9] VGG16_{layer}...")
        cache = os.path.join(cache_dir, f"vgg16_{layer}_train.npy")
        if not os.path.isfile(cache):
            Xtr = vgg16_feature_batch(logmel_train, layer=layer, device=device, batch_size=32)
            Xva = vgg16_feature_batch(logmel_val, layer=layer, device=device, batch_size=32)
            Xte = vgg16_feature_batch(logmel_test, layer=layer, device=device, batch_size=32)
            np.save(os.path.join(cache_dir, f"vgg16_{layer}_train.npy"), Xtr)
            np.save(os.path.join(cache_dir, f"vgg16_{layer}_val.npy"), Xva)
            np.save(os.path.join(cache_dir, f"vgg16_{layer}_test.npy"), Xte)
        else:
            Xtr = np.load(os.path.join(cache_dir, f"vgg16_{layer}_train.npy"))
            Xva = np.load(os.path.join(cache_dir, f"vgg16_{layer}_val.npy"))
            Xte = np.load(os.path.join(cache_dir, f"vgg16_{layer}_test.npy"))

        res = train_eval_branch(Xtr, split.y_train, Xva, split.y_val, Xte, split.y_test, pca_dim, use_pca, use_eda, reg_eps, seed)
        rows.append({"model":f"VGG16_{layer}", **{k:res[k] for k in ["acc_val","acc_test","f1_val","f1_test"]}})
        if deep_scores_val is None:
            deep_scores_val, deep_scores_test = res["scores_val"], res["scores_test"]
            deep_name = f"VGG16_{layer}"

    if deep_scores_val is None:
        raise RuntimeError("Không có deep branch nào chạy được (VGGish/VGG16).")

    alpha_grid = [float(a) for a in cfg["fusion"]["ws_alpha_grid"]]
    best_alpha, best_ws_val = None, -1.0
    best_pred_test = None
    classes = np.unique(split.y_train)

    ws_rows = []
    for a in alpha_grid:
        s_val = (1.0 - a) * best_mbh_val_scores + a * deep_scores_val
        s_te  = (1.0 - a) * best_mbh_test_scores + a * deep_scores_test
        yhat_val = classes[np.argmax(s_val, axis=1)]
        yhat_te  = classes[np.argmax(s_te, axis=1)]
        acc_val = float(accuracy_score(split.y_val, yhat_val))
        acc_te = float(accuracy_score(split.y_test, yhat_te))
        ws_rows.append({"alpha":a, "acc_val":acc_val, "acc_test":acc_te})
        if acc_val > best_ws_val:
            best_ws_val = acc_val
            best_alpha = a
            best_pred_test = yhat_te

    np.savetxt(os.path.join(out_dir, "ws_sweep.csv"),
               np.array([[r["alpha"], r["acc_val"], r["acc_test"]] for r in ws_rows]),
               delimiter=",", header="alpha,acc_val,acc_test", comments="")

    plt.figure()
    plt.plot([r["alpha"] for r in ws_rows], [r["acc_val"] for r in ws_rows], marker="o", label="val")
    plt.plot([r["alpha"] for r in ws_rows], [r["acc_test"] for r in ws_rows], marker="o", label="test")
    plt.xlabel("alpha (weight deep)"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ws_sweep_plot.png"), dpi=200)
    plt.close()

    cm = confusion_matrix(split.y_test, best_pred_test, labels=classes)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (alpha={best_alpha:.2f}, deep={deep_name}, b={best_b})")
    plt.colorbar()
    tick = np.arange(len(split.class_names))
    plt.xticks(tick, split.class_names, rotation=45, ha="right")
    plt.yticks(tick, split.class_names)
    plt.tight_layout()
    plt.xlabel("Pred"); plt.ylabel("True")
    plt.savefig(os.path.join(out_dir, "cm_paper_core.png"), dpi=200)
    plt.close()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "paper_core_metrics.csv"), index=False)

    with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump({"best_b": int(best_b), "best_alpha": float(best_alpha), "deep_branch": deep_name}, f, indent=2)

    tex = os.path.join(out_dir, "paper_core_table.tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\begin{tabular}{lcc}\\toprule\n")
        f.write("Model & Acc(val) & Acc(test)\\\\ \\midrule\n")
        for _, r in df.iterrows():
            av = r.get("acc_val", None); at = r.get("acc_test", None)
            avs = "-" if av is None or (isinstance(av, float) and math.isnan(av)) else f"{av*100:.2f}"
            ats = "-" if at is None or (isinstance(at, float) and math.isnan(at)) else f"{at*100:.2f}"
            f.write(f"{r['model']} & {avs} & {ats}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\n")
        f.write("\\caption{Phase 9 reproduction summary.}\\end{table}\n")

    print("[Phase9] DONE ->", out_dir)
