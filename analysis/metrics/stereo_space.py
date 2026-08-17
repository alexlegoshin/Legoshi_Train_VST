"""§4.4 ТЗ-01: пространство и стерео. Только для настоящих стерео-миксов
(mix/reference/demo) — стемы дуал-моно, там нечего мерить, проверено
измерением ещё на шаге 1 (побайтовое сравнение L/R)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis.metrics.spectral import band_edges_log, assign_bins_to_bands


def _block_stats(L, R, sr, block_s=0.1):
    n = int(block_s * sr)
    nb = len(L) // n
    corr = np.zeros(nb)
    ms_ratio = np.zeros(nb)
    for i in range(nb):
        l, r = L[i*n:(i+1)*n], R[i*n:(i+1)*n]
        if np.std(l) > 1e-9 and np.std(r) > 1e-9:
            corr[i] = np.corrcoef(l, r)[0, 1]
        m, s = (l + r) / 2, (l - r) / 2
        ms_ratio[i] = np.sqrt(np.mean(s**2) + 1e-20) / np.sqrt(np.mean(m**2) + 1e-20)
    t = np.arange(nb) * block_s
    return t, corr, ms_ratio


def _band_analysis(L, R, sr, bands_per_octave=3, f_lo=40, f_hi=16000):
    from scipy.signal import stft
    n_fft = 8192
    f, t, ZL = stft(L, fs=sr, nperseg=n_fft, noverlap=n_fft - 2048, boundary=None)
    _, _, ZR = stft(R, fs=sr, nperseg=n_fft, noverlap=n_fft - 2048, boundary=None)
    edges = band_edges_log(f_lo, min(f_hi, f[-1]), bands_per_octave)
    band_bins = assign_bins_to_bands(f, edges)
    centers = np.array([f[idx].mean() for idx in band_bins])

    width_db, mono_loss_db, corr_by_band = [], [], []
    for idx in band_bins:
        zl, zr = ZL[idx, :], ZR[idx, :]
        e_l, e_r = np.sum(np.abs(zl) ** 2), np.sum(np.abs(zr) ** 2)
        zm, zs = (zl + zr) / 2, (zl - zr) / 2
        e_m, e_s = np.sum(np.abs(zm) ** 2), np.sum(np.abs(zs) ** 2)
        width_db.append(10 * np.log10((e_s + 1e-20) / (e_m + 1e-20)))
        expected_mono = (e_l + e_r) / 2  # §4.4: ожидание при полностью синфазных каналах
        mono_loss_db.append(10 * np.log10((e_m + 1e-20) / (expected_mono + 1e-20)))
        l_flat, r_flat = np.abs(zl).ravel(), np.abs(zr).ravel()
        corr_by_band.append(np.corrcoef(l_flat, r_flat)[0, 1] if len(l_flat) > 1 else np.nan)

    return pd.DataFrame(dict(freq_hz=centers, width_db=width_db,
                              mono_loss_db=mono_loss_db, correlation=corr_by_band))


def goniometer_stats(L, R):
    """Эксцентриситет облака M/S через собственные числа ковариации, не
    через circular mean угла.

    БАГ, который тут был: для моно-сигнала (L=R) угол мечется строго между
    0 градусов и 180 (в зависимости от знака сэмпла) — это ОДНА линия, но
    circular mean гасит противоположно направленные вектора в ноль, будто
    сигнал широкий и круговой. Дубль-моно и противофаза оба давали
    concentration~0 — ровно наоборот тому, что должно быть. Осевая мера
    (эллипс рассеяния, не направление вектора) не подвержена этой ошибке:
    прямая линия под любым углом — это всегда вырожденный эллипс с
    отношением осей ~0, круг/широкий стерео — ~1."""
    M, S = (L + R) / 2, (L - R) / 2
    cov = np.cov(np.stack([M, S]))
    eigvals = np.linalg.eigvalsh(cov)  # по возрастанию
    minor, major = eigvals[0], eigvals[1]
    axis_ratio = float(np.sqrt(max(minor, 0) / max(major, 1e-20)))  # 0=линия(моно), 1=круг(широко)
    # угол главной оси относительно M (0 град = моно-ось)
    _, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, 1]
    principal_angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
    return dict(goniometer_axis_ratio=axis_ratio, goniometer_principal_angle_deg=principal_angle_deg)


def analyze_file(path, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    if data.shape[1] < 2:
        return None, None  # моно-файл, блок неприменим
    L, R = data[:, 0], data[:, 1]

    overall_corr = float(np.corrcoef(L, R)[0, 1])
    e_l, e_r = np.mean(L ** 2), np.mean(R ** 2)
    balance_db = 10 * np.log10((e_l + 1e-20) / (e_r + 1e-20))
    M, S = (L + R) / 2, (L - R) / 2
    overall_ms_ratio = np.sqrt(np.mean(S**2) + 1e-20) / np.sqrt(np.mean(M**2) + 1e-20)

    band_df = _band_analysis(L, R, sr)
    t, corr_t, ms_t = _block_stats(L, R, sr)
    gonio = goniometer_stats(L, R)

    bad_mono_bands = band_df[band_df.mono_loss_db < -3.0]

    summary = dict(
        overall_correlation=overall_corr,
        balance_db=float(balance_db),
        overall_ms_ratio=float(overall_ms_ratio),
        n_bands_mono_loss_gt3db=len(bad_mono_bands),
        worst_mono_loss_db=float(band_df.mono_loss_db.min()) if len(band_df) else np.nan,
        worst_mono_loss_freq_hz=float(band_df.loc[band_df.mono_loss_db.idxmin(), "freq_hz"]) if len(band_df) else np.nan,
        **gonio,
    )
    frames = dict(
        by_band=band_df,
        blocks=pd.DataFrame({"t_s": t, "correlation": corr_t, "ms_ratio": ms_t}),
    )
    return summary, frames
