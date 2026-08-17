"""§4.2 ТЗ-01: спектр. Точная сетка §3.3 (n_fft=4096, hop=512).

Гармонические искажения (THD/чёт-нечет) сюда сознательно НЕ включены —
по ТЗ считаются "на устойчивых нотах (по F0)", а F0-трекинг это §4.6.
Дублировать pYIN здесь ради одной подметрики — плодить работу дважды;
посчитаны отдельным проходом поверх результатов §4.6 в `harmonics.py` /
`run_4_2_harmonics.py` (задача #24, закрыта) и вмёржены в 4_2_summary.parquet
по колонкам thd_median/thd_mean/even_odd_ratio_median."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import skew, kurtosis

N_FFT, HOP = 4096, 512
NAMED_BANDS = [
    ("sub", 20, 60), ("low", 60, 120), ("lowmid", 120, 250), ("mud", 250, 500),
    ("mid", 500, 2000), ("presence", 2000, 5000), ("sibilance", 5000, 8000), ("air", 8000, 16000),
]


def band_edges_log(f_lo, f_hi, bands_per_octave):
    n_oct = np.log2(f_hi / f_lo)
    n_bands = int(np.ceil(n_oct * bands_per_octave))
    edges = f_lo * 2.0 ** (np.arange(n_bands + 1) / bands_per_octave)
    return edges


def assign_bins_to_bands(freqs, edges):
    """Список массивов индексов бинов на полосу. Полосы без бинов — merge со
    следующей (иначе NNLS/LTAS клетка без данных)."""
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bands.append(idx)
    merged = []
    carry = np.array([], dtype=int)
    for idx in bands:
        carry = np.concatenate([carry, idx])
        if len(carry) > 0:
            merged.append(carry)
            carry = np.array([], dtype=int)
    if len(carry) > 0 and merged:
        merged[-1] = np.concatenate([merged[-1], carry])
    return [m for m in merged if len(m) > 0]


def compute_stft(x, sr):
    f, t, Z = stft(x, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    return f, t, np.abs(Z)


def detect_codec_cutoff(mono, sr, ref_band=(1000, 4000), floor_db_rel=50.0):
    """ТЗ-05 А3: lossy-кодеки (mp3 и т.п.) режут спектр на некоторой частоте
    — band_frac_air/sibilance после такого среза измеряют тишину, не звук.
    Ищем самую высокую частоту, где энергия ещё не упала на floor_db_rel
    дБ относительно опорной "нормальной" полосы (1-4кГц). Возвращает
    частоту среза в Гц, либо None, если среза не обнаружено (полоса живая
    до Найквиста)."""
    f, t, mag = compute_stft(mono, sr)
    spec_db = 20 * np.log10(np.mean(mag, axis=1) + 1e-12)
    ref_mask = (f >= ref_band[0]) & (f <= ref_band[1])
    if not ref_mask.any():
        return None
    ref_level = np.median(spec_db[ref_mask])
    active = spec_db > (ref_level - floor_db_rel)
    active_idx = np.where(active)[0]
    if len(active_idx) == 0:
        return None
    cutoff_idx = active_idx.max()
    # синквист сам по себе не "срез" — если активность доходит почти до
    # верха сетки частот, среза кодека нет, это просто конец STFT-сетки
    if f[cutoff_idx] >= 0.97 * f[-1]:
        return None
    return float(f[cutoff_idx])


def ltas(mag, f, bands_per_octave, f_lo=40, f_hi=16000):
    """Long-Term Average Spectrum по 1/n-октавным полосам."""
    edges = band_edges_log(f_lo, min(f_hi, f[-1]), bands_per_octave)
    band_bins = assign_bins_to_bands(f, edges)
    centers = np.array([f[idx].mean() for idx in band_bins])
    levels_db = np.array([20 * np.log10(np.mean(mag[idx, :]) + 1e-12) for idx in band_bins])
    return centers, levels_db


def spectral_slope(centers_hz, levels_db, f_lo=100, f_hi=10000):
    mask = (centers_hz >= f_lo) & (centers_hz <= f_hi)
    if mask.sum() < 3:
        return np.nan
    x = np.log2(centers_hz[mask])
    y = levels_db[mask]
    slope_per_octave, _ = np.polyfit(x, y, 1)
    return float(slope_per_octave)


def spectral_moments(mag, f):
    """Покадрово: centroid, spread, skewness, kurtosis — спектр как
    распределение вероятности по частоте."""
    p = mag / (mag.sum(axis=0, keepdims=True) + 1e-20)
    centroid = np.sum(f[:, None] * p, axis=0)
    spread = np.sqrt(np.sum(((f[:, None] - centroid[None, :]) ** 2) * p, axis=0))
    sk = np.zeros_like(centroid)
    ku = np.zeros_like(centroid)
    safe_spread = np.maximum(spread, 1e-9)
    sk = np.sum(((f[:, None] - centroid[None, :]) ** 3) * p, axis=0) / safe_spread ** 3
    ku = np.sum(((f[:, None] - centroid[None, :]) ** 4) * p, axis=0) / safe_spread ** 4 - 3
    return centroid, spread, sk, ku


def rolloff(mag, f, pct):
    cum = np.cumsum(mag, axis=0)
    total = cum[-1, :]
    thresh = pct * total
    idx = np.array([np.searchsorted(cum[:, j], thresh[j]) for j in range(mag.shape[1])])
    idx = np.clip(idx, 0, len(f) - 1)
    return f[idx]


def flatness(mag):
    """Wiener entropy: geometric mean / arithmetic mean спектра (0..1)."""
    log_mag = np.log(mag + 1e-12)
    geo_mean = np.exp(np.mean(log_mag, axis=0))
    arith_mean = np.mean(mag, axis=0) + 1e-12
    return geo_mean / arith_mean


def spectral_flux(mag):
    """Нормировано на RMS-магнитуду кадра — ТЗ-05 А1: без нормировки flux
    линейно масштабируется с гейном (не тембровая характеристика, а прокси
    абсолютной энергии), что ловится тестом на инвариантность к гейну."""
    d = np.diff(mag, axis=1, prepend=mag[:, :1])
    raw = np.sqrt(np.mean(np.maximum(d, 0) ** 2, axis=0))
    norm = np.sqrt(np.mean(mag ** 2, axis=0)) + 1e-12
    return raw / norm


def named_band_energy_fraction(mag, f):
    total = np.sum(mag ** 2, axis=0) + 1e-20
    out = {}
    for name, lo, hi in NAMED_BANDS:
        idx = np.where((f >= lo) & (f < hi))[0]
        out[name] = np.sum(mag[idx, :] ** 2, axis=0) / total if len(idx) else np.zeros(mag.shape[1])
    return out


def find_resonances(centers_hz, levels_db, smooth_span_oct=1.0, thresh_db=3.0):
    """Peak-picking по сглаженному LTAS: порог ОТНОСИТЕЛЬНЫЙ (над локальной
    сглаженной кривой), не абсолютный — узкий пик +3дБ важнее плоского
    подъёма на ту же величину."""
    log_f = np.log2(centers_hz)
    span_bins = max(1, int(round(smooth_span_oct / max(np.median(np.diff(log_f)), 1e-6))))
    kernel = np.ones(span_bins * 2 + 1) / (span_bins * 2 + 1)
    smoothed = np.convolve(levels_db, kernel, mode="same")
    residual = levels_db - smoothed

    peaks = []
    for i in range(1, len(residual) - 1):
        if residual[i] > thresh_db and residual[i] > residual[i-1] and residual[i] > residual[i+1]:
            half = residual[i] - 3.0
            left = i
            while left > 0 and residual[left] > half:
                left -= 1
            right = i
            while right < len(residual) - 1 and residual[right] > half:
                right += 1
            bw_oct = log_f[right] - log_f[left]
            q = (1.0 / bw_oct) if bw_oct > 1e-6 else np.nan
            peaks.append(dict(freq_hz=float(centers_hz[i]), amplitude_db=float(residual[i]), q_estimate=float(q)))
    return peaks


def analyze_file(path, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    f, t, mag = compute_stft(mono, sr)

    centers13, levels13 = ltas(mag, f, bands_per_octave=3)
    centers16, levels16 = ltas(mag, f, bands_per_octave=6)
    slope = spectral_slope(centers13, levels13)
    resonances = find_resonances(centers16, levels16)

    centroid, spread, sk, ku = spectral_moments(mag, f)
    roll85, roll95 = rolloff(mag, f, 0.85), rolloff(mag, f, 0.95)
    flat = flatness(mag)
    flux = spectral_flux(mag)
    bands = named_band_energy_fraction(mag, f)

    mfcc = None
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=20, n_fft=N_FFT, hop_length=HOP)
        mfcc_delta = librosa.feature.delta(mfcc)
    except Exception:
        mfcc_delta = None

    summary = dict(
        spectral_slope_db_per_oct=slope,
        n_resonances=len(resonances),
        top_resonance_freq_hz=(max(resonances, key=lambda r: r["amplitude_db"])["freq_hz"]
                                if resonances else np.nan),
        top_resonance_amp_db=(max(resonances, key=lambda r: r["amplitude_db"])["amplitude_db"]
                               if resonances else np.nan),
        centroid_hz_median=float(np.median(centroid)),
        spread_hz_median=float(np.median(spread)),
        skewness_median=float(np.median(sk)),
        kurtosis_median=float(np.median(ku)),
        rolloff85_hz_median=float(np.median(roll85)),
        rolloff95_hz_median=float(np.median(roll95)),
        flatness_median=float(np.median(flat)),
        flux_mean=float(np.mean(flux)),
        **{f"band_frac_{name}_median": float(np.median(bands[name])) for name, _, _ in NAMED_BANDS},
    )

    frames = dict(
        ltas_13=pd.DataFrame({"freq_hz": centers13, "level_db": levels13}),
        ltas_16=pd.DataFrame({"freq_hz": centers16, "level_db": levels16}),
        resonances=pd.DataFrame(resonances),
        moments=pd.DataFrame({"t_s": t, "centroid_hz": centroid, "spread_hz": spread,
                               "skewness": sk, "kurtosis": ku, "rolloff85_hz": roll85,
                               "rolloff95_hz": roll95, "flatness": flat, "flux": flux,
                               **{f"band_frac_{name}": bands[name] for name, _, _ in NAMED_BANDS}}),
        mfcc=pd.DataFrame(mfcc.T) if mfcc is not None else pd.DataFrame(),
        mfcc_delta=pd.DataFrame(mfcc_delta.T) if mfcc_delta is not None else pd.DataFrame(),
    )
    return summary, frames
