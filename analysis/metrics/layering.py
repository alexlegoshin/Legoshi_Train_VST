"""§4.7 ТЗ-01: многослойность вокала. Ядро материала — переиспользует
F0-инфраструктуру §4.6. Пары дорожек сравниваются НА ОБЩЕЙ ВРЕМЕННОЙ ОСИ —
через сдвиги из alignment.parquet (шаг 2), иначе расхождение будет включать
фиксированный сдвиг экспорта, а не только реальную игру дублей."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def onset_time_divergence(x_a, x_b, sr, max_match_s=0.15):
    import librosa
    onsets_a = librosa.onset.onset_detect(y=x_a, sr=sr, units="time", backtrack=True)
    onsets_b = librosa.onset.onset_detect(y=x_b, sr=sr, units="time", backtrack=True)
    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return np.array([]), np.array([])
    dts = []
    matched_times = []
    for oa in onsets_a:
        idx = np.argmin(np.abs(onsets_b - oa))
        dt = onsets_b[idx] - oa
        if abs(dt) <= max_match_s:
            dts.append(dt * 1000)  # мс
            matched_times.append(oa)
    return np.array(dts), np.array(matched_times)


def pitch_divergence(t_a, f0_a, voiced_a, t_b, f0_b, voiced_b, offset_b_s=0.0):
    """t_b/f0_b сдвигаются на offset_b_s к системе координат A (offset_b_s —
    на сколько B ЗАПАЗДЫВАЕТ относительно референса; вычитаем, чтобы
    привести обе дорожки к единой оси)."""
    t_b_aligned = t_b - offset_b_s
    f0_b_interp = np.interp(t_a, t_b_aligned, f0_b, left=np.nan, right=np.nan)
    voiced_b_interp = np.interp(t_a, t_b_aligned, voiced_b.astype(float), left=0, right=0) > 0.5

    both_voiced = voiced_a & voiced_b_interp & np.isfinite(f0_a) & np.isfinite(f0_b_interp)
    if both_voiced.sum() == 0:
        return np.array([]), both_voiced, f0_b_interp
    cents_a = 1200 * np.log2(f0_a[both_voiced] / 440)
    cents_b = 1200 * np.log2(f0_b_interp[both_voiced] / 440)
    return cents_a - cents_b, both_voiced, f0_b_interp


def comb_filter_risk(dt_ms, d_cents, dt_thresh_ms=5.0, cents_thresh=5.0):
    """Риск гребёнки: доля моментов, где ОБА расхождения меньше порога."""
    if len(dt_ms) == 0 or len(d_cents) == 0:
        return 0.0
    frac_tight_time = float(np.mean(np.abs(dt_ms) < dt_thresh_ms))
    frac_tight_pitch = float(np.mean(np.abs(d_cents) < cents_thresh))
    # верхняя оценка одновременного риска (независимость не предполагаем,
    # берём минимум долей как консервативную границу совпадения)
    return dict(frac_tight_time=frac_tight_time, frac_tight_pitch=frac_tight_pitch,
                risk_upper_bound=min(frac_tight_time, frac_tight_pitch))


def analyze_pair(path_a, path_b, offset_a_s, offset_b_s, sr_expected=44100):
    import soundfile as sf
    from analysis.metrics.pitch_vocal import extract_f0

    xa, sr = sf.read(str(path_a), dtype="float64", always_2d=True)
    xa = xa.mean(axis=1)
    xb, sr2 = sf.read(str(path_b), dtype="float64", always_2d=True)
    xb = xb.mean(axis=1)
    assert sr == sr_expected and sr2 == sr_expected

    t_a, f0_a, voiced_a, _ = extract_f0(xa, sr)
    t_b, f0_b, voiced_b, _ = extract_f0(xb, sr)

    rel_offset_s = (offset_b_s - offset_a_s)  # относительный сдвиг B к A
    d_cents, both_voiced_mask, _ = pitch_divergence(t_a, f0_a, voiced_a, t_b, f0_b, voiced_b, rel_offset_s)

    dt_ms, matched_t = onset_time_divergence(xa, xb, sr)

    risk = comb_filter_risk(dt_ms, d_cents)
    simultaneity = float(np.mean(both_voiced_mask)) if len(both_voiced_mask) else 0.0

    # взаимная корреляция на общих (оба voiced) участках, по сырому сигналу
    corr = np.nan
    if both_voiced_mask.sum() > 10:
        idx_samples_a = (t_a[both_voiced_mask] * sr).astype(int)
        idx_samples_a = idx_samples_a[(idx_samples_a >= 0) & (idx_samples_a < len(xa))]
        idx_samples_b = np.clip((idx_samples_a + rel_offset_s * sr).astype(int), 0, len(xb) - 1)
        if len(idx_samples_a) > 10:
            corr = float(np.corrcoef(xa[idx_samples_a], xb[idx_samples_b])[0, 1])

    summary = dict(
        simultaneity_fraction=simultaneity,
        pitch_divergence_cents_median=float(np.median(np.abs(d_cents))) if len(d_cents) else np.nan,
        pitch_divergence_cents_std=float(np.std(d_cents)) if len(d_cents) else np.nan,
        time_divergence_ms_median=float(np.median(np.abs(dt_ms))) if len(dt_ms) else np.nan,
        time_divergence_ms_std=float(np.std(dt_ms)) if len(dt_ms) else np.nan,
        comb_risk_frac_tight_time=risk["frac_tight_time"] if isinstance(risk, dict) else np.nan,
        comb_risk_frac_tight_pitch=risk["frac_tight_pitch"] if isinstance(risk, dict) else np.nan,
        comb_risk_upper_bound=risk["risk_upper_bound"] if isinstance(risk, dict) else np.nan,
        mutual_correlation=corr,
    )
    return summary
