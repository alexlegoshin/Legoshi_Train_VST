"""§4.9 ТЗ-01: гармония, диссонанс и «приколы».

Sethares/Plomp-Levelt и Vassilakis — обе формулы по опубликованным работам
(Sethares 1993/2005, Vassilakis 2001), не переоткрываю, только реализую и
проверяю на школьных интервалах (унисон/квинта — низкий диссонанс,
малая секунда — высокий)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import stft

N_FFT, HOP = 4096, 512


def extract_partials(mag_frame, freqs, n_partials=12, min_rel_db=-40):
    """Топ-N пиков спектра кадра как парциалы (freq, amp)."""
    peak_idx = []
    for i in range(2, len(mag_frame) - 2):
        if (mag_frame[i] > mag_frame[i-1] and mag_frame[i] > mag_frame[i+1] and
                mag_frame[i] > mag_frame[i-2] and mag_frame[i] > mag_frame[i+2]):
            peak_idx.append(i)
    if not peak_idx:
        return np.array([]), np.array([])
    peak_idx = np.array(peak_idx)
    amps = mag_frame[peak_idx]
    peak_db = 20 * np.log10(amps + 1e-20)
    rel_db = peak_db - peak_db.max()
    keep = rel_db > min_rel_db
    peak_idx = peak_idx[keep]
    amps = amps[keep]
    if len(peak_idx) > n_partials:
        top = np.argsort(amps)[-n_partials:]
        peak_idx, amps = peak_idx[top], amps[top]
    return freqs[peak_idx], amps


def sethares_dissonance(freqs, amps):
    """Sethares (1993) — на основе кривой Plomp-Levelt."""
    if len(freqs) < 2:
        return 0.0
    b1, b2 = 3.5, 5.75
    s1, s2, dstar = 0.0207, 18.96, 0.24
    total = 0.0
    amps = amps / (amps.max() + 1e-20)
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            f1, f2 = min(freqs[i], freqs[j]), max(freqs[i], freqs[j])
            df = f2 - f1
            s = dstar / (s1 * f1 + s2)
            total += amps[i] * amps[j] * (np.exp(-b1 * s * df) - np.exp(-b2 * s * df))
    return float(total)


def vassilakis_roughness(freqs, amps):
    """Vassilakis (2001) — альтернативная модель, другой весовой член
    (амплитуды в степенях, не произведение напрямую)."""
    if len(freqs) < 2:
        return 0.0
    amps = amps / (amps.max() + 1e-20)
    total = 0.0
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            f1, f2 = min(freqs[i], freqs[j]), max(freqs[i], freqs[j])
            a1, a2 = amps[i], amps[j]
            fmin, amax = f1, max(a1, a2)
            amin = min(a1, a2)
            x = 0.24 / (0.0207 * fmin + 18.96)
            df = f2 - f1
            term = (amax * amin) ** 0.1 * 0.5 * (2 * amin / (amax + amin)) ** 3.11
            term *= (np.exp(-3.5 * x * df) - np.exp(-5.75 * x * df))
            total += term
    return float(total)


def dissonance_curve(mono, sr, n_partials=10):
    f, t, Z = stft(mono, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    mag = np.abs(Z)
    sethares = np.zeros(mag.shape[1])
    vassilakis = np.zeros(mag.shape[1])
    for k in range(mag.shape[1]):
        pf, pa = extract_partials(mag[:, k], f, n_partials=n_partials)
        sethares[k] = sethares_dissonance(pf, pa)
        vassilakis[k] = vassilakis_roughness(pf, pa)
    return t, sethares, vassilakis


def spectral_flux_series(mono, sr):
    f, t, Z = stft(mono, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    mag = np.abs(Z)
    d = np.diff(mag, axis=1, prepend=mag[:, :1])
    return t, np.sqrt(np.mean(np.maximum(d, 0) ** 2, axis=0))


def find_anomalies(t, series_dict, z_thresh=2.0, min_sep_s=1.0):
    """§4.9: кадры, где хотя бы одна из серий выбивается >2sigma от локальной
    нормы (скользящее окно 10с). "Приколы" — список с таймкодами."""
    anomalies = []
    win_frames = max(10, int(10.0 / (t[1] - t[0]) if len(t) > 1 else 100))
    for name, series in series_dict.items():
        s = np.asarray(series)
        z = np.full(len(s), np.nan)
        for i in range(len(s)):
            lo, hi = max(0, i - win_frames), min(len(s), i + win_frames)
            local = np.concatenate([s[lo:i], s[i+1:hi]])
            if len(local) < 10 or np.std(local) < 1e-12:
                continue
            z[i] = (s[i] - np.mean(local)) / np.std(local)
        idx = np.where(np.abs(z) > z_thresh)[0]
        last_t = -np.inf
        for i in idx:
            if t[i] - last_t >= min_sep_s:
                anomalies.append(dict(t_s=float(t[i]), metric=name, z_score=float(z[i]), value=float(s[i])))
                last_t = t[i]
    return pd.DataFrame(anomalies).sort_values("t_s") if anomalies else pd.DataFrame(
        columns=["t_s", "metric", "z_score", "value"])


def analyze_file(path, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    t, sethares, vassilakis = dissonance_curve(mono, sr)
    _, flux = spectral_flux_series(mono, sr)
    n = min(len(sethares), len(flux))
    t, sethares, vassilakis, flux = t[:n], sethares[:n], vassilakis[:n], flux[:n]

    corr_models = float(np.corrcoef(sethares, vassilakis)[0, 1]) if n > 2 else np.nan

    anomalies = find_anomalies(t, dict(dissonance=sethares, roughness=vassilakis, flux=flux))

    summary = dict(
        sethares_dissonance_median=float(np.median(sethares)),
        vassilakis_roughness_median=float(np.median(vassilakis)),
        sethares_vassilakis_correlation=corr_models,
        n_anomalies=len(anomalies),
    )
    frames = dict(
        curve=pd.DataFrame({"t_s": t, "sethares_dissonance": sethares,
                             "vassilakis_roughness": vassilakis, "spectral_flux": flux}),
        anomalies=anomalies,
    )
    return summary, frames
