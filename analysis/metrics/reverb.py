"""§4.5 ТЗ-01: реверберация и пространство.

Оценка «мокрости» через остаток деконволюции (§5, самая надёжная по ТЗ)
считается отдельно в `run_4_5_wetness.py` (задача #25, закрыта) и вмёржена
в 4_5_summary.parquet колонкой wetness_pct — только там, где есть §5
(миксы «основной трек» + контроль «контрольный трек»; для демки и «референс А»
без стемов — NaN, честно). Здесь — всё остальное: RT60/EDT по методу
Шрёдера на хвостах после изолированных onset'ов, DRR, C50/C80, спектр
хвоста, предзадержка."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def schroeder_edc(x):
    """Energy Decay Curve обратным интегрированием Шрёдера, в дБ
    относительно пика (0дБ в начале)."""
    energy = x.astype(np.float64) ** 2
    edc = np.cumsum(energy[::-1])[::-1]
    edc_db = 10 * np.log10(edc / (edc[0] + 1e-20) + 1e-20)
    return edc_db


def fit_decay(edc_db, sr, fit_range_db=(-5, -25), extrapolate_to_db=-60):
    """Линейная регрессия EDC в заданном диапазоне (T20 по умолчанию: -5..-25),
    экстраполяция до -60дБ. Возвращает время в секундах."""
    t = np.arange(len(edc_db)) / sr
    hi, lo = fit_range_db
    mask = (edc_db <= hi) & (edc_db >= lo)
    if mask.sum() < 5:
        return np.nan
    slope, intercept = np.polyfit(t[mask], edc_db[mask], 1)
    if slope >= 0:
        return np.nan
    return float(extrapolate_to_db / slope)


def find_isolated_onsets(x, sr, min_gap_s=0.3, tail_window_s=0.5):
    import librosa
    onsets = librosa.onset.onset_detect(y=x, sr=sr, units="samples", backtrack=True)
    isolated = []
    for i, onset in enumerate(onsets):
        next_onset = onsets[i+1] if i+1 < len(onsets) else len(x)
        gap_s = (next_onset - onset) / sr
        if gap_s >= min_gap_s:
            window = min(int(tail_window_s * sr), next_onset - onset, len(x) - onset)
            if window > int(0.05 * sr):
                isolated.append((onset, window))
    return isolated


def clarity_index(x_tail, sr, split_ms):
    n_split = int(split_ms / 1000 * sr)
    if n_split >= len(x_tail):
        return np.nan
    e_early = np.sum(x_tail[:n_split].astype(np.float64) ** 2)
    e_late = np.sum(x_tail[n_split:].astype(np.float64) ** 2)
    if e_late <= 0:
        return np.inf
    return float(10 * np.log10((e_early + 1e-20) / (e_late + 1e-20)))


def drr(x_tail, sr, direct_ms=50):
    n_d = int(direct_ms / 1000 * sr)
    if n_d >= len(x_tail):
        return np.nan
    e_direct = np.mean(x_tail[:n_d].astype(np.float64) ** 2)
    e_rest = np.mean(x_tail[n_d:].astype(np.float64) ** 2) if len(x_tail) > n_d else 1e-20
    return float(10 * np.log10((e_direct + 1e-20) / (e_rest + 1e-20)))


def tail_spectral_tilt(x_tail, sr, skip_ms=50):
    """Спектральный наклон ПОЗДНЕЙ части хвоста (после прямого звука) —
    что именно «мокрое», низ или верх."""
    n_skip = int(skip_ms / 1000 * sr)
    late = x_tail[n_skip:]
    if len(late) < sr // 20:
        return np.nan
    X = np.abs(np.fft.rfft(late))
    freqs = np.fft.rfftfreq(len(late), 1 / sr)
    mask = (freqs >= 100) & (freqs <= 8000) & (X > 0)
    if mask.sum() < 3:
        return np.nan
    slope, _ = np.polyfit(np.log2(freqs[mask]), 20 * np.log10(X[mask] + 1e-20), 1)
    return float(slope)


def predelay_estimate(x_tail, sr, direct_thresh_db=-20):
    """Время от онсета до момента, когда прямой звук стихает ниже
    direct_thresh_db относительно пика — грубая прокси-граница между
    прямым звуком и диффузным хвостом."""
    edc_db = schroeder_edc(x_tail)
    below = np.where(edc_db <= direct_thresh_db)[0]
    return float(below[0] / sr) if len(below) else np.nan


def analyze_file(path, sr_expected=44100, min_gap_s=0.3, tail_window_s=0.5):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    isolated = find_isolated_onsets(mono, sr, min_gap_s, tail_window_s)
    rows = []
    for onset, window in isolated:
        tail = mono[onset:onset+window]
        edc = schroeder_edc(tail)
        rt60 = fit_decay(edc, sr, fit_range_db=(-5, -25))
        edt = fit_decay(edc, sr, fit_range_db=(0, -10))
        rows.append(dict(
            onset_s=onset / sr, tail_len_s=window / sr,
            rt60_s=rt60, edt_s=edt,
            c50_db=clarity_index(tail, sr, 50), c80_db=clarity_index(tail, sr, 80),
            drr_db=drr(tail, sr), tail_spectral_tilt_db_per_oct=tail_spectral_tilt(tail, sr),
            predelay_s=predelay_estimate(tail, sr),
        ))
    df = pd.DataFrame(rows)

    def med(col):
        return float(df[col].median()) if len(df) and df[col].notna().any() else np.nan

    summary = dict(
        n_isolated_tails=len(df),
        rt60_s_median=med("rt60_s"), edt_s_median=med("edt_s"),
        c50_db_median=med("c50_db"), c80_db_median=med("c80_db"),
        drr_db_median=med("drr_db"),
        tail_spectral_tilt_median=med("tail_spectral_tilt_db_per_oct"),
        predelay_s_median=med("predelay_s"),
    )
    return summary, df
