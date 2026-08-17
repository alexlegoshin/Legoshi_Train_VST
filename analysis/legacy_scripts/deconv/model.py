"""Деконволюция микса по стемам, §5 ТЗ-01.

Модель:  M[f,t] ~= sum_i G_i[t] * H_i[f] * S_i[f,t] + R[f,t]

Реализация в два шага:
  1. Для каждой (полоса x блок) регрессией НЕОТРИЦАТЕЛЬНЫХ МНК по всем
     сырым (бин, кадр) внутри клетки находим a_i[band, block] — вклад
     энергии каждого стема. Это одна общая клетка на a_i = G_i*H_i,
     полосы группируются именно для того, чтобы внутри клетки было
     достаточно (бин, кадр)-точек для устойчивой регрессии, а не чтобы
     усреднить энергию в скаляр.
  2. Разложение a_i[band, block] на ранг-1: G_i[t] (огибающая) x H_i[f]
     (EQ-кривая) — через усечённое SVD (ближайшее ранг-1 приближение
     неотрицательной матрицы обычно неотрицательно на практике для
     физически осмысленных a_i; если знак съехал — берём abs и
     перенормируем).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.optimize import nnls
from scipy.signal import stft

from analysis.metrics.spectral import band_edges_log, assign_bins_to_bands  # общие утилиты, живут в ядре


@dataclass
class DeconvResult:
    stem_names: list
    G: dict       # name -> array[n_blocks] (усреднённая по частоте огибающая)
    H: dict       # name -> array[n_bands]   (усреднённая по времени EQ-кривая)
    band_centers: np.ndarray
    block_times: np.ndarray
    explained_fraction: float
    residual_energy: float
    mix_energy: float
    a_raw: dict    # name -> array[n_bands, n_blocks], сырые коэффициенты до ранг-1 факторизации


def deconvolve_channel(mix_ch, stems_ch: dict, sr, n_fft=4096, hop=512,
                        f_lo=60.0, f_hi=16000.0, bands_per_octave=6, block_s=0.5,
                        ridge=1e-6):
    names = list(stems_ch.keys())
    f, t, Mstft = stft(mix_ch, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    Sstft = {}
    for name in names:
        _, _, S = stft(stems_ch[name], fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
        Sstft[name] = S

    Mpow = np.abs(Mstft) ** 2
    Spow = {name: np.abs(S) ** 2 for name, S in Sstft.items()}

    edges = band_edges_log(f_lo, min(f_hi, f[-1]), bands_per_octave)
    band_bins = assign_bins_to_bands(f, edges)
    n_bands = len(band_bins)
    band_centers = np.array([f[idx].mean() for idx in band_bins])

    frame_dt = (t[1] - t[0]) if len(t) > 1 else hop / sr
    frames_per_block = max(1, int(round(block_s / frame_dt)))
    n_blocks = int(np.ceil(len(t) / frames_per_block))
    block_times = np.array([t[min(k * frames_per_block, len(t) - 1)] for k in range(n_blocks)])

    # Присутствие стема в полосе — суммарная энергия по всем кадрам этой
    # полосы. Полосы, где стем практически не звучит, дают вырожденный
    # (нулевой) столбец в регрессии; даже с ridge там может остаться числовой
    # шум, который потом рвёт ранг-1 разложение SVD (один выброс перевешивает
    # всю содержательную часть кривой). Такие пары (стем, полоса) исключаем
    # из регрессии заранее — это не гипотеза, а прямое знание, что стему
    # там нечем объяснять микс.
    presence = np.array([[Spow[name][bins].sum() if len(bins) else 0.0 for bins in band_bins]
                          for name in names])  # n_stems x n_bands
    presence_frac = presence / np.maximum(presence.max(axis=0, keepdims=True), 1e-30)
    # Относительного порога (доля от максимума В ЭТОЙ полосе) недостаточно:
    # в полосах, где у ВСЕХ стемов нет реального сигнала (частотный зазор
    # между их диапазонами), максимум сам по себе — это только пол
    # числового шума, и 1e-4 от шума всё ещё "проходит" порог. Добавляем
    # второе условие — относительно СОБСТВЕННОГО пика присутствия стема
    # по всем полосам: если тут у стема настолько меньше, чем в его лучшей
    # полосе, это шум, а не сигнал.
    own_floor = presence.max(axis=1, keepdims=True) * 1e-6
    active = (presence_frac > 1e-4) & (presence > own_floor)  # n_stems x n_bands

    a = {name: np.zeros((n_bands, n_blocks)) for name in names}
    explained_num, explained_den = 0.0, 0.0
    residual_energy_total = 0.0

    for bi, bins in enumerate(band_bins):
        active_idx = np.where(active[:, bi])[0]
        if len(active_idx) == 0:
            continue
        for k in range(n_blocks):
            frame_slice = slice(k * frames_per_block, min((k + 1) * frames_per_block, len(t)))
            y = Mpow[np.ix_(bins, range(frame_slice.start, frame_slice.stop))].ravel()
            X_full = np.stack([Spow[name][np.ix_(bins, range(frame_slice.start, frame_slice.stop))].ravel()
                                for name in names], axis=1)
            X = X_full[:, active_idx]  # только стемы, реально присутствующие в этой полосе
            if y.size == 0:
                continue
            # Ridge-регуляризация (§5.2 п.3, минимальная версия) поверх
            # отбора активных стемов: без неё даже среди активных столбцов
            # возможна плохая обусловленность на коротких блоках.
            col_scale = np.linalg.norm(X, axis=0).max() if X.size else 1.0
            lam = ridge * max(col_scale, 1e-12)
            X_aug = np.vstack([X, lam * np.eye(len(active_idx))])
            y_aug = np.concatenate([y, np.zeros(len(active_idx))])
            coefs_active, _ = nnls(X_aug, y_aug, maxiter=200)
            for j, i in enumerate(active_idx):
                a[names[i]][bi, k] = coefs_active[j]
            y_hat = X @ coefs_active
            explained_num += np.sum((y - y_hat) ** 2)
            explained_den += np.sum(y ** 2)
            residual_energy_total += max(0.0, np.sum(y) - np.sum(y_hat))

    explained_fraction = 1.0 - explained_num / max(explained_den, 1e-12)
    mix_energy = float(np.sum(Mpow))

    G, H, a_raw = {}, {}, {}
    for name in names:
        A = a[name]  # bands x blocks, non-negative by construction (NNLS)
        a_raw[name] = A
        if A.size == 0 or not np.any(A):
            G[name] = np.zeros(n_blocks)
            H[name] = np.zeros(n_bands)
            continue
        # rank-1 SVD approximation A ~= h (x) g
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        h = U[:, 0] * np.sqrt(S[0])
        g = Vt[0, :] * np.sqrt(S[0])
        # знак ранг-1 SVD произволен (h,g) или (-h,-g) — энергия неотрицательна,
        # поэтому фиксируем знак по большинству и берём abs
        if np.sum(h) < 0:
            h, g = -h, -g
        H[name] = np.abs(h)
        G[name] = np.abs(g)

    return DeconvResult(
        stem_names=names, G=G, H=H, band_centers=band_centers, block_times=block_times,
        explained_fraction=float(explained_fraction),
        residual_energy=float(residual_energy_total),
        mix_energy=mix_energy, a_raw=a_raw,
    )
