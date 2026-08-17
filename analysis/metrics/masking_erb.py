"""§4.8 ТЗ-01: маскирование. Готовых библиотек под многодорожечную
атрибуцию маскирования нет — почти всё руками.

Упрощения, которые стоит держать в голове:
- ERB-полосы через группировку STFT-бинов (не банк гамматон-фильтров) —
  как и в §4.2/§5, для устойчивости регрессии/анализа этого достаточно,
  полной остроты фильтра гамматон это не даёт.
- Spreading function — классическая формула Zwicker/Fastl для excitation
  pattern, в единицах ERB-rate вместо Барк (тот же функциональный вид,
  калибровка между шкалами не идентична — не заявляем лабораторную точность,
  только сравнение между дорожками/версиями).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import stft

N_FFT, HOP = 4096, 512


def hz_to_erb_rate(f):
    return 21.4 * np.log10(4.37 * f / 1000 + 1)


def erb_rate_to_hz(erb):
    return (10 ** (erb / 21.4) - 1) * 1000 / 4.37


def erb_band_edges(f_lo=50, f_hi=16000, n_bands=40):
    e_lo, e_hi = hz_to_erb_rate(f_lo), hz_to_erb_rate(f_hi)
    erb_edges = np.linspace(e_lo, e_hi, n_bands + 1)
    return erb_rate_to_hz(erb_edges)


def assign_bins_to_erb_bands(freqs, edges):
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        bands.append(idx)
    # смежные пустые полосы (ниже разрешения STFT на низких ERB) сливаем со следующей
    merged, carry = [], np.array([], dtype=int)
    for idx in bands:
        carry = np.concatenate([carry, idx])
        if len(carry):
            merged.append(carry)
            carry = np.array([], dtype=int)
    if len(carry) and merged:
        merged[-1] = np.concatenate([merged[-1], carry])
    return [m for m in merged if len(m) > 0]


def spreading_function_db(dz):
    """Zwicker/Fastl excitation pattern, dz в ERB-rate (аналог Барк).
    Асимметрична: крутой спад в сторону низких частот от маскера, пологий
    в сторону высоких (маскер "заливает" верх сильнее, чем низ)."""
    dz = dz + 0.474
    return 15.81 + 7.5 * dz - 17.5 * np.sqrt(1 + dz ** 2)


def build_spreading_matrix(band_centers_hz):
    erb = hz_to_erb_rate(band_centers_hz)
    dz = erb[:, None] - erb[None, :]  # [маскер, маскируемая]
    sf_db = spreading_function_db(dz)
    return 10 ** (sf_db / 10)  # линейная матрица весов [n_bands, n_bands]


def apply_temporal_masking(energy_bt, hop_s, forward_ms=200, backward_ms=20):
    """energy_bt: [n_bands, n_frames]. Прямое маскирование — экспоненциальный
    спад вперёд по времени; обратное — короткое и слабое, до маскера."""
    n_bands, n_frames = energy_bt.shape
    fwd_n = max(1, int(round(forward_ms / 1000 / hop_s)))
    bwd_n = max(1, int(round(backward_ms / 1000 / hop_s)))
    tau_fwd = fwd_n / 3.0  # постоянная спада, чтобы 200мс было "довольно распавшимся" хвостом
    kernel_fwd = np.exp(-np.arange(0, fwd_n) / tau_fwd)
    kernel_bwd = np.exp(-np.arange(0, bwd_n) / (bwd_n / 2.0)) * 0.3  # слабее и короче
    full_kernel = np.concatenate([kernel_bwd[::-1] * 0.3, [1.0], kernel_fwd[1:]])
    full_kernel /= full_kernel.max()

    out = np.zeros_like(energy_bt)
    for b in range(n_bands):
        out[b] = np.convolve(energy_bt[b], full_kernel, mode="same")
    return out


def erb_energy(mono, sr, band_edges):
    f, t, Z = stft(mono, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    mag2 = np.abs(Z) ** 2
    band_bins = assign_bins_to_erb_bands(f, band_edges)
    centers = np.array([f[idx].mean() for idx in band_bins])
    energy = np.array([mag2[idx, :].sum(axis=0) for idx in band_bins])
    hop_s = t[1] - t[0] if len(t) > 1 else HOP / sr
    return centers, t, hop_s, energy


def analyze_group(track_mono: dict, sr, n_bands=40, f_lo=50, f_hi=16000):
    """track_mono: {name: mono np.ndarray} — УЖЕ выровненные по времени
    (см. pipeline.deconv.run_5.load_aligned_stems), не сырые пути.

    ИСПРАВЛЕНО: раньше эта функция сама грузила файлы по путям и НЕ
    применяла выравнивание (§3.2/§5) — для §4.8 это реальная ошибка, а не
    мелочь: у вокальных стемов «основной трек» офсеты доходят до +1812/-1449
    сэмплов (~74мс между самыми разъехавшимися дорожками), и маскирование
    целиком завязано на то, что реально звучит ОДНОВРЕМЕННО. Без
    выравнивания "одновременность" была не настоящей. Возвращает audibility
    по дорожке и матрицу атрибуции (кто кого маскирует)."""
    edges = erb_band_edges(f_lo, f_hi, n_bands)
    energies, t_common, hop_s, centers = {}, None, None, None
    for name, mono in track_mono.items():
        c, t, hs, e = erb_energy(mono, sr, edges)
        centers, hop_s = c, hs
        if t_common is None or len(t) < len(t_common):
            t_common = t
        energies[name] = e

    n_frames = min(e.shape[1] for e in energies.values())
    energies = {k: v[:, :n_frames] for k, v in energies.items()}
    t_common = t_common[:n_frames]

    spread_matrix = build_spreading_matrix(centers)
    names = list(energies.keys())

    excitation = {}
    for name in names:
        exc = spread_matrix @ energies[name]  # [n_bands, n_frames], маскирующая энергия от этой дорожки
        excitation[name] = apply_temporal_masking(exc, hop_s)

    audibility = {}
    attribution_pairs = []
    for name in names:
        others = [n for n in names if n != name]
        if not others:
            audibility[name] = 1.0
            continue
        threshold = sum(excitation[o] for o in others)
        above = energies[name] > threshold
        audibility[name] = float(np.mean(above))

        # атрибуция: среди клеток, где name замаскирован, кто там доминирующий маскер
        masked_cells = ~above
        if masked_cells.sum() > 0:
            other_exc_stack = np.stack([excitation[o] for o in others])  # [n_other, n_bands, n_frames]
            dominant_idx = np.argmax(other_exc_stack, axis=0)  # [n_bands, n_frames]
            for oi, oname in enumerate(others):
                frac = float(np.mean((dominant_idx == oi) & masked_cells))
                if frac > 0:
                    attribution_pairs.append(dict(masked=name, masker=oname, fraction_of_masked_cells=frac))

    return dict(audibility=audibility, attribution=pd.DataFrame(attribution_pairs),
                band_centers=centers, t=t_common)
