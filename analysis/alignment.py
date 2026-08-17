"""GCC-PHAT alignment — общий алгоритм, не привязан к конкретному
материалу, вынесен в ядро (ТЗ-05 А4: используется оркестратором для
проверки синхронности дорожек трек-аута перед суммированием роли).

PHAT (phase transform) weighting flattens the cross-spectrum magnitude before
the inverse FFT, so the peak position depends only on phase (delay), not on
which signal is louder or brighter — robust to the timbral differences between
a raw stem and its processed appearance inside a mix.
"""
import numpy as np

# критерий уверенности, ТЗ-01 §3.2 / ТЗ-05 А4
CONFIDENCE_THRESHOLD = 1.3
Z_SCORE_THRESHOLD = 8.0


def gcc_phat(sig: np.ndarray, ref: np.ndarray, sr: int, max_shift_s: float | None = None):
    """Return (shift_samples, confidence, peak1, peak2, z_score) for sig relative to ref.

    Positive shift means `sig` lags `ref` (sig starts later).
    confidence = peak1 / peak2 of the two highest maxima separated by >1s.
    """
    n = sig.shape[-1] + ref.shape[-1]
    n_fft = 1 << (n - 1).bit_length()

    SIG = np.fft.rfft(sig, n=n_fft)
    REF = np.fft.rfft(ref, n=n_fft)
    R = SIG * np.conj(REF)
    R /= np.maximum(np.abs(R), 1e-12)  # PHAT: whiten magnitude, keep phase
    cc = np.fft.irfft(R, n=n_fft)
    cc = np.concatenate((cc[-(len(ref) - 1):], cc[:len(sig)]))  # center zero-lag
    lags = np.arange(-(len(ref) - 1), len(sig))

    if max_shift_s is not None:
        keep = np.abs(lags) <= int(max_shift_s * sr)
        cc, lags = cc[keep], lags[keep]

    order = np.argsort(cc)[::-1]
    peak1_idx = order[0]
    peak1 = cc[peak1_idx]
    min_sep = int(1.0 * sr)  # второй пик должен быть дальше 1с
    peak2 = 0.0
    for idx in order[1:]:
        if abs(lags[idx] - lags[peak1_idx]) > min_sep:
            peak2 = cc[idx]
            break

    shift = lags[peak1_idx]  # positive: `sig` lags `ref` by this many samples
    confidence = peak1 / max(peak2, 1e-9)
    # z-score пика относительно медианы корр.функции. Медиана и MAD, не
    # среднее/std — иначе сам пик (выброс) утягивает оценку разброса и
    # занижает свой же z-score.
    median = np.median(cc)
    mad = np.median(np.abs(cc - median)) * 1.4826 + 1e-12
    z_score = (peak1 - median) / mad
    return shift, confidence, peak1, peak2, z_score


def is_confident(confidence: float, z_score: float) -> bool:
    return confidence > CONFIDENCE_THRESHOLD and z_score > Z_SCORE_THRESHOLD


def activity_fraction(x, sr, hop_ms=10, rel_thresh_db=-30):
    """Доля кадров с активностью выше rel_thresh_db от пика — критерий
    для отбора «периодического/малоактивного» материала (типично ударные
    при активности <15% кадров)."""
    hop = int(sr * hop_ms / 1000)
    n = len(x) // hop
    if n == 0:
        return 0.0
    frame_rms = np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop] ** 2) + 1e-20) for i in range(n)])
    peak = frame_rms.max()
    if peak <= 0:
        return 0.0
    thresh = peak * 10 ** (rel_thresh_db / 20)
    return float(np.mean(frame_rms > thresh))


def bar_hypothesis_shift(period_s: float, max_bars: int, sig: np.ndarray, ref: np.ndarray, sr: int):
    """Периодический случай: перебрать сдвиги, кратные `period_s`
    (длительность такта), и вернуть тот, что даёт максимум простой
    кросс-корреляции огибающих. Используется, когда gcc_phat даёт
    низкую уверенность (несколько сопоставимых пиков — подпись периодики).

    Convention matches gcc_phat: positive shift means `sig` lags `ref`,
    i.e. sig[n] ~= ref[n - shift_samples].
    """
    def envelope(x, sr, hop_ms=10):
        hop = int(sr * hop_ms / 1000)
        n = len(x) // hop
        return np.array([np.sqrt(np.mean(x[i*hop:(i+1)*hop]**2) + 1e-12) for i in range(n)])

    def aligned_corr(a, b, shift_frames):
        if shift_frames >= 0:
            a_seg, b_seg = a[shift_frames:], b[:len(b) - shift_frames] if shift_frames else b
        else:
            a_seg, b_seg = a[:len(a) + shift_frames], b[-shift_frames:]
        m = min(len(a_seg), len(b_seg))
        if m < 10:
            return -np.inf
        return np.corrcoef(a_seg[:m], b_seg[:m])[0, 1]

    env_sig, env_ref = np.log(envelope(sig, sr)), np.log(envelope(ref, sr))
    hop_s = 0.01
    scores = {}
    for k in range(-max_bars, max_bars + 1):
        shift_s = k * period_s
        shift_frames = int(round(shift_s / hop_s))
        scores[shift_s] = aligned_corr(env_sig, env_ref, shift_frames)

    best_shift = max(scores, key=scores.get)
    best_score = scores[best_shift]
    # Математический факт: если материал строго периодичен с периодом ровно
    # `period_s` (лупующийся паттерн), то ЛЮБОЙ сдвиг, кратный этому
    # периоду, даёт тождественно тот же результат — различить их через
    # корреляцию огибающей в принципе нельзя, это не ограничение метода, а
    # свойство сигнала. Ловим это явно: несколько кандидатов в пределах 1%
    # от максимума — сдвиг неоднозначен.
    near_ties = [s for s, v in scores.items() if v >= best_score - 0.01 * abs(best_score)]
    ambiguous = len(near_ties) > 1
    return best_shift, best_score, ambiguous, sorted(near_ties)
