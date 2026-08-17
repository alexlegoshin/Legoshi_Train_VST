"""§4.1 ТЗ-01: громкость и динамика.

pyloudnorm даёт integrated_loudness "из коробки" (гейтинг по BS.1770-4), но
не отдаёт временной ряд momentary/short-term — переиспользуем его же
K-weighting фильтры (self._filters) и считаем ряд сами по той же формуле
стандарта (skользящее окно, без второго гейтинга — так принято для
временных рядов, гейтинг только для integrated)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyloudnorm as pyln
from scipy.stats import skew, kurtosis

from analysis.metrics.spectral import band_edges_log, assign_bins_to_bands

CH_GAIN = [1.0, 1.0, 1.0, 1.41, 1.41]


def _k_weight(x_mono_or_multi, meter: pyln.Meter):
    """Применяет ту же цепочку фильтров, что pyloudnorm использует внутри
    integrated_loudness, но возвращает отфильтрованный сигнал, не метрику."""
    data = x_mono_or_multi.copy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    for _, filter_stage in meter._filters.items():
        for ch in range(data.shape[1]):
            data[:, ch] = filter_stage.apply_filter(data[:, ch])
    return data


def _block_loudness_series(k_weighted, sr, block_s, hop_s):
    """LUFS-подобный ряд по формуле BS.1770 без гейтинга (momentary/short-term)."""
    block_n, hop_n = int(block_s * sr), int(hop_s * sr)
    n_ch = k_weighted.shape[1]
    n_blocks = max(0, (len(k_weighted) - block_n) // hop_n + 1)
    vals = np.full(n_blocks, -np.inf)
    times = np.arange(n_blocks) * hop_s + block_s / 2
    for j in range(n_blocks):
        seg = k_weighted[j*hop_n:j*hop_n+block_n]
        z = np.mean(seg ** 2, axis=0)
        s = sum(CH_GAIN[i] * z[i] for i in range(n_ch))
        if s > 0:
            vals[j] = -0.691 + 10 * np.log10(s)
    return times, vals


def true_peak_dbfs(x, sr, oversample=4):
    from scipy.signal import resample_poly
    xu = resample_poly(x, oversample, 1)
    return 20 * np.log10(max(np.max(np.abs(xu)), 1e-12))


def lra_ebu3342(short_term_vals):
    """LRA по упрощённому алгоритму EBU Tech 3342: отсечь ниже -70 LUFS,
    отсечь ниже (среднее оставшихся - 20дБ), взять разницу перцентилей 95-10."""
    v = short_term_vals[np.isfinite(short_term_vals)]
    v = v[v > -70]
    if len(v) == 0:
        return np.nan
    rel_thresh = np.mean(v) - 20
    v2 = v[v > rel_thresh]
    if len(v2) < 2:
        return np.nan
    return float(np.percentile(v2, 95) - np.percentile(v2, 10))


def dr_tt_style(x, sr, block_s=3.0):
    """DR в духе TT Dynamic Range Meter: блоки по 3с, RMS каждого, топ-20%
    по RMS усредняются, DR = 20*log10(пик / средний RMS топ-20%)."""
    block_n = int(block_s * sr)
    n_blocks = len(x) // block_n
    if n_blocks < 2:
        return np.nan
    rms = np.array([np.sqrt(np.mean(x[i*block_n:(i+1)*block_n] ** 2) + 1e-20) for i in range(n_blocks)])
    peak = np.max(np.abs(x))
    top_n = max(1, int(np.ceil(n_blocks * 0.2)))
    top_rms = np.sort(rms)[-top_n:].mean()
    return float(20 * np.log10(peak / max(top_rms, 1e-12)))


def crest_factor_by_band(x, sr, f_lo=40, f_hi=16000, bands_per_octave=3, frame_s=0.1):
    """Crest factor (peak/RMS, дБ) по 1/3-октавным полосам, покадрово
    (грубая сетка §3.3: hop~4410 при 44.1к = 100мс)."""
    n_fft = 8192
    hop = int(frame_s * sr)
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    mag = np.abs(Z)
    edges = band_edges_log(f_lo, min(f_hi, f[-1]), bands_per_octave)
    band_bins = assign_bins_to_bands(f, edges)
    band_centers = np.array([f[idx].mean() for idx in band_bins])

    cf = np.zeros((len(band_bins), len(t)))
    for bi, bins in enumerate(band_bins):
        band_mag = mag[bins, :]
        peak = band_mag.max(axis=0)
        rms = np.sqrt(np.mean(band_mag ** 2, axis=0) + 1e-20)
        cf[bi] = 20 * np.log10(np.maximum(peak, 1e-12) / np.maximum(rms, 1e-12))
    return band_centers, t, cf


def _smooth_envelope(x, sr, win_ms=1.0):
    """Огибающая по RMS в скользящем окне ~1мс — транзиент типа удара по
    барабану шумовой по своей природе, сырой |x| даёт дребезг на отдельных
    сэмплах вместо формы огибающей."""
    win = max(1, int(win_ms / 1000 * sr))
    power = x.astype(np.float64) ** 2
    kernel = np.ones(win) / win
    return np.sqrt(np.convolve(power, kernel, mode="same") + 1e-20)


def transient_events(x, sr, attack_search_ms=50, decay_search_ms=150):
    """На каждый onset: время атаки (10->90% пика), время спада (пик->50%),
    «панч» (пик первых 10мс относительно RMS первых 100мс). Атака/спад
    измеряются по сглаженной огибающей (~1мс RMS), не по сырому |x| —
    шумовой транзиент (удар) иначе даёт дребезг вместо формы огибающей."""
    import librosa
    onsets = librosa.onset.onset_detect(y=x, sr=sr, units="samples", backtrack=True)
    w10, w100 = int(0.010 * sr), int(0.100 * sr)
    search_att = int(attack_search_ms / 1000 * sr)
    search_dec = int(decay_search_ms / 1000 * sr)

    rows = []
    for onset in onsets:
        seg = x[onset:onset + search_att]
        if len(seg) < 10:
            continue
        env = _smooth_envelope(seg, sr)
        peak = env.max()
        if peak < 1e-6:
            continue
        above10 = np.where(env >= 0.1 * peak)[0]
        above90 = np.where(env >= 0.9 * peak)[0]
        if len(above10) == 0 or len(above90) == 0 or above90[0] <= above10[0]:
            continue
        attack_s = (above90[0] - above10[0]) / sr
        peak_idx_local = int(np.argmax(env))

        seg2_raw = x[onset + peak_idx_local: onset + peak_idx_local + search_dec]
        seg2 = _smooth_envelope(seg2_raw, sr)
        below50 = np.where(seg2 <= 0.5 * peak)[0]
        decay_s = float(below50[0] / sr) if len(below50) else np.nan

        if onset + w100 <= len(x):
            peak10 = np.max(np.abs(x[onset:onset + w10]))
            rms100 = np.sqrt(np.mean(x[onset:onset + w100] ** 2) + 1e-20)
            punch_db = float(20 * np.log10(max(peak10, 1e-12) / max(rms100, 1e-12)))
        else:
            punch_db = np.nan

        rows.append(dict(onset_s=onset / sr, attack_s=attack_s, decay_s=decay_s, punch_db=punch_db))
    return pd.DataFrame(rows)


def pumping_signature(x, sr, hop_ms=10, min_lag_ms=50, max_lag_ms=500, neighbor_ms=20):
    """§4.1: автокорреляция огибающей уровня, поиск периодического «дыхания»
    компрессора на лагах 50-500мс.

    БАГ, пойманный на реальном корпусе (синтетика с явной модуляцией его не
    ловила): автокорреляция гладкой огибающей монотонно убывает с лагом
    просто потому, что соседние по времени значения похожи — это свойство
    любого сигнала, не признак периодичности. argmax по диапазону тогда
    всегда берёт самый короткий лаг из разрешённых, независимо от того,
    есть ли реальная периодичность. На «основной трек» так и вышло:
    50.0мс (= нижняя граница диапазона) у всех версий без исключения,
    а кривая при печати оказалась гладкой без единого бугра.

    Правильный критерий — НАСТОЯЩИЙ локальный максимум (выше обоих соседей
    на +-neighbor_ms), а не глобальный максимум в разрешённом диапазоне.
    Если такого максимума нет — значит признаков периодического пампинга
    не обнаружено, и это надо честно вернуть, а не подставить границу."""
    hop = int(hop_ms / 1000 * sr)
    n = len(x) // hop
    env = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop] ** 2) + 1e-20) for i in range(n)])
    log_env = np.log(env + 1e-9)
    log_env -= log_env.mean()
    ac = np.correlate(log_env, log_env, mode="full")[len(log_env) - 1:]
    ac /= max(ac[0], 1e-12)
    lags_ms = np.arange(len(ac)) * hop_ms

    neighbor_n = max(1, int(round(neighbor_ms / hop_ms)))
    mask_idx = np.where((lags_ms >= min_lag_ms) & (lags_ms <= max_lag_ms))[0]
    mask_idx = mask_idx[(mask_idx - neighbor_n >= 0) & (mask_idx + neighbor_n < len(ac))]

    local_maxima = [i for i in mask_idx if ac[i] > ac[i - neighbor_n] and ac[i] > ac[i + neighbor_n]]
    if not local_maxima:
        return dict(pumping_score=0.0, pumping_period_ms=np.nan, pumping_detected=False)
    best = max(local_maxima, key=lambda i: ac[i])
    return dict(pumping_score=float(ac[best]), pumping_period_ms=float(lags_ms[best]), pumping_detected=True)


def analyze_file(path, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected, f"неожиданный sr: {sr}"
    mono = data.mean(axis=1)

    meter = pyln.Meter(sr)
    integrated = meter.integrated_loudness(data if data.shape[1] > 1 else mono)

    kw = _k_weight(data.copy(), meter)
    t_m, momentary = _block_loudness_series(kw, sr, block_s=0.4, hop_s=0.1)
    t_s, short_term = _block_loudness_series(kw, sr, block_s=3.0, hop_s=0.1)

    lra = lra_ebu3342(short_term)
    sample_peak = 20 * np.log10(max(np.max(np.abs(mono)), 1e-12))
    tp = true_peak_dbfs(mono, sr)
    plr = tp - integrated
    valid_st = short_term[np.isfinite(short_term)]
    psr = tp - short_term  # покадрово (может содержать -inf там, где short_term = -inf)

    crest_overall = 20 * np.log10(max(np.max(np.abs(mono)), 1e-12) /
                                    max(np.sqrt(np.mean(mono ** 2)), 1e-12))
    dr = dr_tt_style(mono, sr)

    band_centers, band_t, cf_bands = crest_factor_by_band(mono, sr)

    trans = transient_events(mono, sr)
    pump = pumping_signature(mono, sr)

    summary = dict(
        n_transients=len(trans),
        attack_ms_median=float(trans.attack_s.median() * 1000) if len(trans) else np.nan,
        decay_ms_median=float(trans.decay_s.median() * 1000) if len(trans) else np.nan,
        punch_db_median=float(trans.punch_db.median()) if len(trans) else np.nan,
        transient_density_per_s=float(len(trans) / (len(mono) / sr)),
        pumping_score=pump["pumping_score"], pumping_period_ms=pump["pumping_period_ms"],
        integrated_lufs=float(integrated),
        lra=lra,
        true_peak_dbfs=float(tp), sample_peak_dbfs=float(sample_peak),
        plr=float(plr),
        crest_factor_db=float(crest_overall),
        dr_tt=dr,
        short_term_p10=float(np.percentile(valid_st, 10)) if len(valid_st) else np.nan,
        short_term_p50=float(np.percentile(valid_st, 50)) if len(valid_st) else np.nan,
        short_term_p95=float(np.percentile(valid_st, 95)) if len(valid_st) else np.nan,
        short_term_var=float(np.var(valid_st)) if len(valid_st) else np.nan,
        short_term_skew=float(skew(valid_st)) if len(valid_st) > 2 else np.nan,
        short_term_kurtosis=float(kurtosis(valid_st)) if len(valid_st) > 2 else np.nan,
    )

    frames = dict(
        momentary=pd.DataFrame({"t_s": t_m, "lufs": momentary}),
        short_term=pd.DataFrame({"t_s": t_s, "lufs": short_term, "psr": psr}),
        crest_by_band=pd.DataFrame(
            cf_bands, index=[f"{c:.0f}Hz" for c in band_centers], columns=np.round(band_t, 3)),
        transients=trans,
    )
    return summary, frames
