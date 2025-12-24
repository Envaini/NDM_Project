import numpy as np

def _safe_import_scipy():
    try:
        from scipy.signal import stft, get_window
        return stft, get_window
    except Exception as e:
        raise ImportError(
            "Thiếu scipy. Cài scipy rồi chạy lại: pip install scipy"
        ) from e

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def mel_filterbank(
    fs: int,
    n_fft: int,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Trả về mel filterbank shape: (n_mels, n_freq_bins)
    n_freq_bins = n_fft//2 + 1 (rfft bins)
    """
    if fmax is None:
        fmax = fs / 2.0

    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, fs / 2.0, n_freqs)

    mmin = hz_to_mel(np.array([fmin], dtype=np.float64))[0]
    mmax = hz_to_mel(np.array([fmax], dtype=np.float64))[0]
    mpts = np.linspace(mmin, mmax, n_mels + 2)
    hzpts = mel_to_hz(mpts)

    # map hz points -> fft bin indices
    bins = np.floor((n_fft + 1) * hzpts / fs).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float64)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]
        if center == left:
            center = min(center + 1, n_freqs - 1)
        if right == center:
            right = min(right + 1, n_freqs - 1)

        # rising
        for k in range(left, center):
            fb[m - 1, k] = (k - left) / max(center - left, 1)
        # falling
        for k in range(center, right):
            fb[m - 1, k] = (right - k) / max(right - center, 1)

    return fb

def make_spectrogram(
    x: np.ndarray,
    fs: int,
    window_type: str = "hamming",
    win_length: int = 512,
    hop_length: int = 256,
    n_fft: int | None = None,
    center: bool = False,
    use_mel: bool = False,
    n_mels: int = 64,
    fmin: float = 0.0,
    fmax: float | None = None,
    power: float = 2.0,
    log_eps: float = 1e-8,
    to_db: bool = False,
):
    """
    Spectrogram chuẩn:
    - STFT -> |Z|^power -> (tuỳ chọn) Mel -> log (hoặc dB)
    Trả:
      S: (freq_bins, time_frames) hoặc (n_mels, time_frames)
      meta: dict (freqs, times, params)
    """
    stft, get_window = _safe_import_scipy()

    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        raise ValueError("x phải là 1D (tín hiệu 1 kênh).")

    if n_fft is None:
        n_fft = win_length

    if hop_length <= 0:
        raise ValueError("hop_length phải > 0")
    noverlap = win_length - hop_length
    if noverlap < 0:
        raise ValueError("hop_length không được lớn hơn win_length")

    # window
    win = get_window(window_type, win_length, fftbins=True)

    # STFT
    f, t, Z = stft(
        x,
        fs=fs,
        window=win,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=n_fft,
        detrend=False,
        return_onesided=True,
        boundary=None if not center else "zeros",
        padded=False,
    )

    mag = np.abs(Z)
    S = mag ** power  # power spectrogram

    if use_mel:
        fb = mel_filterbank(fs=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        S = fb @ S
        f_out = np.arange(n_mels)
    else:
        f_out = f

    # log scale
    S = np.maximum(S, 0.0)
    if to_db:
        # 10*log10(power) theo kiểu “dB”
        S = 10.0 * np.log10(S + log_eps)
    else:
        S = np.log(S + log_eps)

    meta = {
        "fs": fs,
        "window_type": window_type,
        "win_length": win_length,
        "hop_length": hop_length,
        "overlap": noverlap,
        "n_fft": n_fft,
        "use_mel": use_mel,
        "n_mels": n_mels if use_mel else None,
        "fmin": fmin if use_mel else None,
        "fmax": fmax if use_mel else None,
        "power": power,
        "log_eps": log_eps,
        "to_db": to_db,
        "freqs": f_out,
        "times": t,
    }
    return S.astype(np.float32), meta
