"""§4.2 довесок: гармонические искажения (THD, чёт/нечет) на устойчивых
нотах — задача #24. Считаем по нотам, которые уже сегментировал §4.6
(переиспользуем закэшированные `*.4_6_notes.parquet`, повторно pYIN не
гоняем — дорого и незачем, алгоритм там не менялся).

Область применения: файлы с "вокал" в пути (чистый монофонический
источник — THD там интерпретируем напрямую) И миксы/референс инженера сведения
(там F0-трекер в основном ловит ведущий вокал, но в спектре на частоте
k*f0 неизбежно есть энергия других инструментов — то же ограничение,
что и у тюн-артефактов в §4.6, честно перенесено сюда через флаг
`is_monophonic_source`). Демка исключена не нами: §4.6 её не обрабатывал
(её роль "demo", а не "mix"/"reference"), поэтому и THD для неё нет —
это унаследованный пробел, а не новый."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import soundfile as sf

N_HARMONICS = 8
MIN_NOTE_S = 0.15
SEARCH_FRAC = 0.03  # +-3% вокруг k*f0 — окно поиска локального пика в спектре ноты

EVEN_HARMONIC_IDX = [1, 3, 5, 7]  # k=2,4,6,8 (индекс = k-1)
ODD_HARMONIC_IDX = [2, 4, 6]      # k=3,5,7 (без основного тона k=1)


def is_vocal(path):
    p = path.lower()
    return "вокал" in p or "vocals" in p  # demucs-стемы «референс А» назван по-английски


def note_harmonic_amps(seg, sr, f0_hz, n_harmonics=N_HARMONICS):
    if len(seg) < int(0.05 * sr) or not np.isfinite(f0_hz) or f0_hz <= 0:
        return None
    win = np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(seg * win))
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    amps = np.full(n_harmonics, np.nan)
    for k in range(1, n_harmonics + 1):
        target = k * f0_hz
        if target >= freqs[-1]:
            break
        lo, hi = target * (1 - SEARCH_FRAC), target * (1 + SEARCH_FRAC)
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        if len(idx):
            amps[k - 1] = spec[idx].max()
    return amps


def analyze_notes(mono, sr, notes_df):
    """notes_df — датафрейм из кэша §4.6 (t_start,t_end,duration_s,f0_start,f0_end)."""
    per_note = []
    for _, note in notes_df.iterrows():
        if note.duration_s < MIN_NOTE_S:
            continue
        f0_note = float(np.nanmedian([note.f0_start, note.f0_end]))
        i0 = int(round(note.t_start * sr))
        i1 = int(round(note.t_end * sr))
        seg = mono[i0:i1]
        amps = note_harmonic_amps(seg, sr, f0_note)
        if amps is None or not np.isfinite(amps[0]) or amps[0] <= 0:
            continue

        harmonics_energy = np.nansum(amps[1:] ** 2)
        thd = float(np.sqrt(harmonics_energy) / amps[0])

        even_energy = np.nansum([amps[i] ** 2 for i in EVEN_HARMONIC_IDX if i < len(amps)])
        odd_energy = np.nansum([amps[i] ** 2 for i in ODD_HARMONIC_IDX if i < len(amps)])
        even_odd_ratio = float(even_energy / odd_energy) if odd_energy > 1e-20 else np.nan

        per_note.append(dict(t_start=float(note.t_start), duration_s=float(note.duration_s),
                              f0_hz=f0_note, thd=thd, even_odd_ratio=even_odd_ratio))

    per_note_df = pd.DataFrame(per_note)
    if len(per_note_df) == 0:
        summary = dict(n_notes_for_thd=0, thd_median=np.nan, thd_mean=np.nan,
                        even_odd_ratio_median=np.nan)
    else:
        summary = dict(
            n_notes_for_thd=int(len(per_note_df)),
            thd_median=float(per_note_df.thd.median()),
            thd_mean=float(per_note_df.thd.mean()),
            even_odd_ratio_median=float(per_note_df.even_odd_ratio.median()),
        )
    return summary, per_note_df


def analyze_file(path, notes_df, sr_expected=44100):
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)
    return analyze_notes(mono, sr, notes_df)
