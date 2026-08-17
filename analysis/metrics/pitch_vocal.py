"""§4.6 ТЗ-01: питч и вокал. Core: F0 (pYIN), нотная сегментация, интонация,
вибрато, реестр глиссандо/легато, признаки тюна.

Форманты (LPC), дыхания, сибилянты, отношение согласных/гласных считаются
отдельным проходом в `vocal_texture.py` / `run_4_6_texture.py` (задача #26,
закрыта) — переиспользует F0/voiced отсюда, чтобы не гонять pYIN дважды."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

HOP = 512  # точная сетка §3.3, 11.6мс при 44.1к — годится и для F0


def hz_to_cents(f0, ref=440.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200 * np.log2(f0 / ref)


def cents_to_nearest_semitone_deviation(cents):
    """Отклонение от равномерной темперации в центах, [-50, +50]."""
    semitone_cents = np.round(cents / 100) * 100
    return cents - semitone_cents


def extract_f0(x, sr, fmin=65.0, fmax=1000.0):
    import librosa
    f0, voiced_flag, voiced_prob = librosa.pyin(
        x, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=HOP, fill_na=np.nan)
    t = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=HOP)
    return t, f0, voiced_flag, voiced_prob


def segment_notes(t, f0, voiced, deriv_thresh_cents_per_100ms=50.0, min_note_s=0.05):
    """§4.6: переход = участок с монотонной производной F0 > 50 центов/100мс.

    БАГ, пойманный на синтетике: считать это как frame-to-frame производную,
    экстраполированную на 100мс — неправильно. Вибрато (типично 3-8Гц) даёт
    короткие, но резкие покадровые скачки, которые после экстраполяции легко
    превышают порог, и устойчивая нота с вибрато рассыпается на десяток
    ложных "переходов". Нужна РЕАЛЬНАЯ разница за окно ~100мс (не за один
    кадр), она у вибрато сама себя гасит на большей части цикла — колебание
    туда-обратно не даёт устойчивого монотонного тренда."""
    cents = hz_to_cents(f0)
    hop_s = t[1] - t[0] if len(t) > 1 else HOP / 44100
    win = max(1, int(round(0.1 / hop_s)))
    d_cents = np.full(len(t), np.nan)
    d_cents[win:] = cents[win:] - cents[:-win]  # изменение ЗА 100мс, не за кадр
    # первые win кадров: окно короче 100мс, но не оставлять их без производной
    # вовсе — иначе самое начало трека/фразы всегда классифицируется как нота
    for i in range(1, win):
        d_cents[i] = (cents[i] - cents[0]) * (0.1 / (i * hop_s))

    is_transition = np.zeros(len(t), dtype=bool)
    valid = voiced & np.isfinite(f0)
    for i in range(1, len(t)):
        if valid[i] and valid[i-1] and np.isfinite(d_cents[i]) and abs(d_cents[i]) > deriv_thresh_cents_per_100ms:
            if np.isfinite(d_cents[i-1]) and (np.sign(d_cents[i]) == np.sign(d_cents[i-1]) or not is_transition[i-1]):
                is_transition[i] = True
                if is_transition[i-1] is False and i > 0:
                    is_transition[i-1] = True  # включить точку начала перехода

    segments = []
    i = 0
    n = len(t)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        kind = "transition" if is_transition[i] else "note"
        while j < n and valid[j] and (is_transition[j] == is_transition[i]):
            j += 1
        dur = t[min(j, n-1)] - t[i]
        if dur >= min_note_s or kind == "transition":
            segments.append(dict(
                type=kind, t_start=float(t[i]), t_end=float(t[min(j-1, n-1)]),
                duration_s=float(dur), f0_start=float(f0[i]), f0_end=float(f0[min(j-1, n-1)]),
                interval_cents=float(cents[min(j-1, n-1)] - cents[i]),
            ))
        i = j
    return segments


def vibrato_analysis(t, f0, segment, hop_s):
    """Вибрато на устойчивой ноте: частота (Гц) и глубина (центы) через
    спектр остатка F0-центы минус медиана, на самой ноте."""
    mask = (t >= segment["t_start"]) & (t <= segment["t_end"])
    if mask.sum() < 8:
        return np.nan, np.nan
    cents = hz_to_cents(f0[mask])
    residual = cents - np.nanmedian(cents)
    residual = residual[np.isfinite(residual)]
    if len(residual) < 8:
        return np.nan, np.nan
    fft = np.abs(np.fft.rfft(residual * np.hanning(len(residual))))
    freqs = np.fft.rfftfreq(len(residual), hop_s)
    band = (freqs >= 3) & (freqs <= 10)  # типичный диапазон вибрато 3-10Гц
    if not np.any(band) or fft[band].max() < 1e-6:
        return np.nan, np.nan
    peak_idx = np.where(band)[0][np.argmax(fft[band])]
    rate_hz = float(freqs[peak_idx])
    depth_cents = float(np.std(residual) * np.sqrt(2))  # RMS->амплитуда синуса
    return rate_hz, depth_cents


def detect_tune_artifacts(t, f0, voiced, flat_var_cents=5.0, flat_min_s=0.2, jump_max_s=0.03, jump_min_cents=80):
    """§4.6: признаки тюна — аномально плоские участки (<5центов дольше 200мс)
    и резкие скачки короче 30мс."""
    cents = hz_to_cents(f0)
    hop_s = t[1] - t[0] if len(t) > 1 else HOP / 44100
    valid = voiced & np.isfinite(f0)

    flats = []
    i = 0
    n = len(t)
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i
        while j < n and valid[j] and abs(cents[j] - cents[i]) < flat_var_cents * 2:
            j += 1
        dur = t[min(j-1, n-1)] - t[i]
        window = cents[i:j]
        if dur >= flat_min_s and len(window) > 2 and np.std(window) < flat_var_cents:
            flats.append(dict(t_start=float(t[i]), t_end=float(t[min(j-1, n-1)]), duration_s=float(dur)))
        i = max(j, i + 1)

    jumps = []
    for i in range(1, n):
        if valid[i] and valid[i-1]:
            dc = abs(cents[i] - cents[i-1])
            if dc > jump_min_cents and hop_s <= jump_max_s:
                jumps.append(dict(t_s=float(t[i]), jump_cents=float(dc)))

    return flats, jumps


def analyze_file(path, sr_expected=44100, fmin=65.0, fmax=1000.0):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    t, f0, voiced, prob = extract_f0(mono, sr, fmin, fmax)
    hop_s = t[1] - t[0] if len(t) > 1 else HOP / sr
    segments = segment_notes(t, f0, voiced)

    notes = [s for s in segments if s["type"] == "note"]
    transitions = [s for s in segments if s["type"] == "transition"]

    for note in notes:
        cents_dev = cents_to_nearest_semitone_deviation(np.array([hz_to_cents((note["f0_start"] + note["f0_end"]) / 2)]))[0]
        note["intonation_deviation_cents"] = float(cents_dev)
        note["vibrato_rate_hz"], note["vibrato_depth_cents"] = vibrato_analysis(t, f0, note, hop_s)

    flats, jumps = detect_tune_artifacts(t, f0, voiced)

    voiced_frac = float(np.mean(voiced & np.isfinite(f0)))
    intonation_devs = [n["intonation_deviation_cents"] for n in notes if n["duration_s"] > 0.1]

    summary = dict(
        voiced_fraction=voiced_frac,
        n_notes=len(notes), n_transitions=len(transitions),
        median_note_duration_s=float(np.median([n["duration_s"] for n in notes])) if notes else np.nan,
        mean_abs_intonation_deviation_cents=float(np.mean(np.abs(intonation_devs))) if intonation_devs else np.nan,
        vibrato_rate_hz_median=float(np.nanmedian([n["vibrato_rate_hz"] for n in notes])) if notes else np.nan,
        vibrato_depth_cents_median=float(np.nanmedian([n["vibrato_depth_cents"] for n in notes])) if notes else np.nan,
        n_flat_tune_segments=len(flats), n_fast_jumps=len(jumps),
        f0_hz_median=float(np.nanmedian(f0[voiced])) if voiced.any() else np.nan,
    )
    frames = dict(
        f0=pd.DataFrame({"t_s": t, "f0_hz": f0, "voiced": voiced, "prob": prob}),
        notes=pd.DataFrame(notes),
        transitions=pd.DataFrame(transitions),
        flats=pd.DataFrame(flats),
        jumps=pd.DataFrame(jumps),
    )
    return summary, frames
