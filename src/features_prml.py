import numpy as np
from scipy import stats


def time_features(x: np.ndarray) -> np.ndarray:
    """
    x: 1D segment (raw)
    """
    x = x.astype(np.float64)
    eps = 1e-12

    mean = np.mean(x)
    std = np.std(x)
    rms = np.sqrt(np.mean(x**2) + eps)
    var = np.var(x)
    p2p = np.ptp(x)
    max_abs = np.max(np.abs(x)) + eps

    # shape descriptors
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, fisher=True, bias=False)  # excess kurtosis

    # common vibration factors
    crest = max_abs / rms
    mean_abs = np.mean(np.abs(x)) + eps
    shape = rms / mean_abs
    impulse = max_abs / mean_abs
    clearance = max_abs / ((np.mean(np.sqrt(np.abs(x))) + eps) ** 2)

    # zero-crossing rate (per sample)
    zcr = np.mean(np.diff(np.signbit(x)).astype(np.float64))

    return np.array([
        mean, std, var, rms, p2p, max_abs,
        skew, kurt,
        crest, shape, impulse, clearance,
        zcr
    ], dtype=np.float32)


def freq_features(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Frequency-domain features from power spectrum.
    """
    x = x.astype(np.float64)
    eps = 1e-12

    n = x.size
    # remove DC to make spectrum features more stable
    x0 = x - np.mean(x)

    # rFFT
    X = np.fft.rfft(x0)
    P = (np.abs(X) ** 2) / max(n, 1)  # power
    freqs = np.fft.rfftfreq(n, d=1.0/fs)

    P_sum = np.sum(P) + eps
    Pn = P / P_sum

    # spectral centroid, bandwidth
    centroid = np.sum(freqs * Pn)
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * Pn))

    # rolloff 85%
    cdf = np.cumsum(Pn)
    rolloff = freqs[np.searchsorted(cdf, 0.85)]

    # flatness (geom mean / arith mean)
    flatness = np.exp(np.mean(np.log(P + eps))) / (np.mean(P) + eps)

    # spectral entropy
    entropy = -np.sum(Pn * np.log(Pn + eps))

    # band energies: 5 equal bands in [0, fs/2]
    fmax = fs / 2.0
    edges = np.linspace(0.0, fmax, 6)
    band_e = []
    for i in range(5):
        m = (freqs >= edges[i]) & (freqs < edges[i+1])
        band_e.append(np.sum(Pn[m]))
    band_e = np.array(band_e, dtype=np.float32)

    return np.concatenate([
        np.array([centroid, bandwidth, rolloff, flatness, entropy], dtype=np.float32),
        band_e
    ]).astype(np.float32)


def extract_features(x: np.ndarray, fs: int) -> np.ndarray:
    return np.concatenate([time_features(x), freq_features(x, fs)]).astype(np.float32)
