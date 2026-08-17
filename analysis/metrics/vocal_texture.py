"""§4.6 довесок: форманты (LPC), дыхания, сибилянты, отношение
согласные/гласные — задача #26. Только по вокальным дорожкам (is_vocal) —
на полном миксе LPC-форманты и детекторы дыхания/сибилянтов бессмысленны
(барабаны/тарелки дадут "сибилянты", инструментал даст "форманты" —
никакого отношения к вокальному тракту).

Голосовой/безголосый флаг переиспользуем из уже посчитанного §4.6 (кэш
*.4_6_f0.parquet), но НЕ по индексу кадра — своя STFT-сетка здесь
считается независимо (тот же hop, но librosa.pyin и scipy.stft не
гарантированно дают идентичные границы кадров), а по БЛИЖАЙШЕМУ времени
через merge_asof. Сопоставлять по индексу — тихая ловушка рассинхрона."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

HOP = 512
FRAME_LEN = 1024  # ~23мс при 44.1к — стандартное окно под LPC-форманты
LPC_ORDER = 12
FORMANT_RESAMPLE_SR = 10000  # см. lpc_formants — децимация перед LPC, не полная 44.1к
N_FORMANTS = 3
FORMANT_FMIN, FORMANT_FMAX = 90, 4000
FORMANT_MAX_BANDWIDTH_HZ = 400  # отбросить сильно задемпфированные корни (не резонанс, а шум LPC)

BREATH_BAND = (2000, 8000)
BREATH_BAND_FRAC_THRESH = 0.35
BREATH_FLATNESS_THRESH = 0.25
BREATH_MIN_RMS_DBFS = -45

SIBILANT_BAND = (5000, 10000)
SIBILANT_REL_THRESH_DB = 6.0  # над локальным медианным фоном полосы


def _frame_starts(n_samples, frame_len, hop):
    return np.arange(0, n_samples - frame_len + 1, hop)


def _voiced_lookup(t_query, f0_df):
    """merge_asof по времени — не по индексу кадра (см. докстринг модуля)."""
    q = pd.DataFrame({"t_s": t_query}).sort_values("t_s")
    ref = f0_df[["t_s", "voiced", "f0_hz"]].sort_values("t_s")
    merged = pd.merge_asof(q, ref, on="t_s", direction="nearest", tolerance=HOP / 44100 * 2)
    merged = merged.sort_index()
    voiced = merged["voiced"].fillna(False).to_numpy()
    return voiced


def lpc_formants(frame, sr, order=LPC_ORDER, n_formants=N_FORMANTS,
                  fmin=FORMANT_FMIN, fmax=FORMANT_FMAX, max_bw=FORMANT_MAX_BANDWIDTH_HZ,
                  resample_sr=FORMANT_RESAMPLE_SR):
    """LPC на децимированном кадре (44.1к -> 10к), а не на полной полосе.

    ПОЙМАНО НА СИНТЕТИКЕ: на полных 44.1кГц с order=16 LPC тратит бюджет
    полюсов на моделирование гармоник/шума до 22кГц, и вместо трёх чистых
    формант в районе 700/1200/2600Гц находится 1-2 смазанных, широкополосных
    корня (F2 полностью терялся, сливаясь с F1). Стандартный приём в
    формантных трекерах (Praat и т.п.) — сначала снизить частоту
    дискретизации примерно до 2×(верхняя нужная частота + запас), чтобы весь
    бюджет полюсов LPC работал в интересующей полосе. При resample_sr=10000
    (Найквист 5кГц, с запасом выше FORMANT_FMAX=4000) и order=12 те же три
    форманты восстанавливаются с точностью в единицы-десятки Гц."""
    import librosa
    from scipy.signal import resample
    if np.allclose(frame, 0) or len(frame) <= order:
        return [np.nan] * n_formants
    n_rs = max(int(round(len(frame) * resample_sr / sr)), order + 2)
    frame_rs = resample(frame, n_rs)
    pre = np.append(frame_rs[0], frame_rs[1:] - 0.97 * frame_rs[:-1])
    win = pre * np.hamming(len(pre))
    try:
        a = librosa.lpc(win, order=order)
    except Exception:
        return [np.nan] * n_formants
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 1e-6]
    if len(roots) == 0:
        return [np.nan] * n_formants
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * resample_sr / (2 * np.pi)
    r_mag = np.clip(np.abs(roots), 1e-6, 0.999999)
    bw = -(resample_sr / np.pi) * np.log(r_mag)
    mask = (freqs >= fmin) & (freqs <= fmax) & (bw < max_bw)
    freqs = np.sort(freqs[mask])
    out = list(freqs[:n_formants])
    while len(out) < n_formants:
        out.append(np.nan)
    return out


def band_energy_frac(mag_frame, freqs, lo, hi):
    total = np.sum(mag_frame ** 2) + 1e-20
    idx = np.where((freqs >= lo) & (freqs < hi))[0]
    return float(np.sum(mag_frame[idx] ** 2) / total) if len(idx) else 0.0


def frame_flatness(mag_frame, freqs=None, lo=None, hi=None):
    """Wiener entropy. Если задана полоса (lo,hi) — считать ТОЛЬКО в ней:
    посчитано по всему спектру, flatness ложно проваливается почти до нуля
    на полосовом сигнале — за пределами полосы амплитуда около нуля, это
    тащит geometric mean к нулю и топит реальную "шумность" внутри полосы
    (поймано на синтетике для детектора дыханий)."""
    if freqs is not None and lo is not None and hi is not None:
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        mag_frame = mag_frame[idx] if len(idx) else mag_frame
    log_mag = np.log(mag_frame + 1e-12)
    geo = np.exp(np.mean(log_mag))
    arith = np.mean(mag_frame) + 1e-12
    return float(geo / arith)


def detect_breaths(mono, sr, t, voiced, frame_len=FRAME_LEN, hop=HOP):
    """§4.6: дыхания — высокая энергия 2-8кГц, низкая тональность (прокси —
    spectral flatness), нет F0. По безголосым кадрам, сгруппировано в события
    (min_gap склеивает соседние безголосые кадры-кандидаты в одно дыхание)."""
    starts = _frame_starts(len(mono), frame_len, hop)
    is_breath = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        if voiced[i]:
            continue
        frame = mono[s:s + frame_len]
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-12)
        if rms_dbfs < BREATH_MIN_RMS_DBFS:
            continue
        spec = np.abs(np.fft.rfft(frame * np.hanning(frame_len)))
        freqs = np.fft.rfftfreq(frame_len, 1 / sr)
        band_frac = band_energy_frac(spec, freqs, *BREATH_BAND)
        flat = frame_flatness(spec, freqs, *BREATH_BAND)
        if band_frac > BREATH_BAND_FRAC_THRESH and flat > BREATH_FLATNESS_THRESH:
            is_breath[i] = True

    events = []
    i = 0
    n = len(starts)
    while i < n:
        if not is_breath[i]:
            i += 1
            continue
        j = i
        while j < n and is_breath[j]:
            j += 1
        events.append(dict(t_start=float(t[i]), t_end=float(t[min(j - 1, n - 1)]),
                            duration_s=float(t[min(j - 1, n - 1)] - t[i] + hop / sr)))
        i = j
    return events


def detect_sibilants(mono, sr, t, voiced, frame_len=FRAME_LEN, hop=HOP, med_win=21):
    """§4.6: сибилянты — транзиенты 5-10кГц без F0. Локальный пик энергии
    полосы над скользящим медианным фоном (та же логика, что peak-picking
    в spectral.find_resonances — относительный порог, не абсолютный)."""
    starts = _frame_starts(len(mono), frame_len, hop)
    band_db = np.full(len(starts), -np.inf)
    for i, s in enumerate(starts):
        if voiced[i]:
            continue
        frame = mono[s:s + frame_len]
        spec = np.abs(np.fft.rfft(frame * np.hanning(frame_len)))
        freqs = np.fft.rfftfreq(frame_len, 1 / sr)
        idx = np.where((freqs >= SIBILANT_BAND[0]) & (freqs < SIBILANT_BAND[1]))[0]
        e = np.sum(spec[idx] ** 2) if len(idx) else 0.0
        band_db[i] = 10 * np.log10(e + 1e-20)

    finite = np.isfinite(band_db)
    if finite.sum() < med_win:
        return []
    padded = np.where(finite, band_db, np.nanmedian(band_db[finite]))
    baseline = pd.Series(padded).rolling(med_win, center=True, min_periods=1).median().to_numpy()
    residual = band_db - baseline

    events = []
    for i in range(1, len(residual) - 1):
        if not finite[i]:
            continue
        if residual[i] > SIBILANT_REL_THRESH_DB and residual[i] > residual[i - 1] and residual[i] >= residual[i + 1]:
            events.append(dict(t_s=float(t[i]), level_db=float(band_db[i]), rel_db=float(residual[i])))
    return events


def analyze_file(path, f0_df, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    starts = _frame_starts(len(mono), FRAME_LEN, HOP)
    t = (starts + FRAME_LEN / 2) / sr
    voiced = _voiced_lookup(t, f0_df)

    formant_rows = []
    for i, s in enumerate(starts):
        if not voiced[i]:
            continue
        frame = mono[s:s + FRAME_LEN]
        f1, f2, f3 = lpc_formants(frame, sr)
        formant_rows.append(dict(t_s=float(t[i]), f1_hz=f1, f2_hz=f2, f3_hz=f3))
    formants_df = pd.DataFrame(formant_rows)

    breaths = detect_breaths(mono, sr, t, voiced)
    sibilants = detect_sibilants(mono, sr, t, voiced)

    frame_rms2 = np.array([np.mean(mono[s:s + FRAME_LEN] ** 2) for s in starts])
    voiced_energy_sum = float(np.sum(frame_rms2[voiced]))
    unvoiced_energy_sum = float(np.sum(frame_rms2[~voiced]))
    cons_vowel_ratio = float(unvoiced_energy_sum / voiced_energy_sum) if voiced_energy_sum > 1e-20 else np.nan

    summary = dict(
        formant_f1_hz_median=float(formants_df.f1_hz.median()) if len(formants_df) else np.nan,
        formant_f2_hz_median=float(formants_df.f2_hz.median()) if len(formants_df) else np.nan,
        formant_f3_hz_median=float(formants_df.f3_hz.median()) if len(formants_df) else np.nan,
        n_breath_events=len(breaths),
        breath_total_duration_s=float(sum(b["duration_s"] for b in breaths)),
        n_sibilant_events=len(sibilants),
        sibilant_rate_per_min=float(len(sibilants) / (len(mono) / sr / 60)) if len(mono) else np.nan,
        consonant_vowel_energy_ratio=cons_vowel_ratio,
    )
    frames = dict(formants=formants_df, breaths=pd.DataFrame(breaths), sibilants=pd.DataFrame(sibilants))
    return summary, frames
