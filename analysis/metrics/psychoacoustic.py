"""§4.3 ТЗ-01: психоакустика через MoSQITo.

Два режима по стоимости, а не по важности:
- "quick" (стационарные метрики) — доли секунды на файл, гоним на весь корпус.
- "full" (покадровые time-varying) — ~1x реального времени НА КАЖДУЮ из трёх
  метрик (loudness/sharpness/roughness), то есть ~3x realtime на файл.
  На 84 файлах, включая 43 сырых тейка, это часы впустую — тейки не участвуют
  в сравнении версий. Покадровые считаем только на ключевых файлах (§CORE_ROLES).

Калибровка Па: физической калибровки уровня у этих записей нет и не будет.
Принята конвенция 0dBFS(peak) = 100dB SPL(peak) -> Pa = digital*2.0.
Абсолютные сон/акум цифры поэтому условны; сравнение МЕЖДУ версиями одной
песни при одной и той же конвенции — корректно, это и есть цель.

Fluctuation strength (vacil) в установленной версии MoSQITo (1.2.1) отсутствует
как класс — проверено ToolSearch по всему дереву пакета, нет даже черновика.
Не реализуем вручную сейчас: частичный прокси уже есть в §4.1 (pumping_signature,
автокорреляция огибающей на лагах 50-500мс) — уступает по психоакустической
строгости, но даёт хоть какой-то числовой сигнал про «пульсирует»."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

PA_CALIBRATION = 2.0  # 0dBFS peak -> 100dB SPL peak


def warmth_ratio(x, sr):
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=4096, noverlap=4096-512, boundary=None)
    mag2 = np.abs(Z) ** 2
    low = np.sum(mag2[(f >= 100) & (f < 500), :])
    high = np.sum(mag2[(f >= 1000) & (f < 4000), :])
    return float(low / max(high, 1e-20))


def harshness(x, sr):
    """Энергия 2-5кГц, взвешенная приближением A-weighting, нормированная на
    общую громкость (RMS)."""
    from scipy.signal import stft
    f, t, Z = stft(x, fs=sr, nperseg=4096, noverlap=4096-512, boundary=None)
    mag2 = np.abs(Z) ** 2
    band = (f >= 2000) & (f < 5000)
    # грубое A-weighting приближение: +пик чувствительности около 2-4кГц
    a_weight = 1.0 + 0.5 * np.exp(-((f - 3000) / 1500) ** 2)
    weighted = np.sum(mag2[band, :] * a_weight[band, None])
    total = np.sum(mag2) + 1e-20
    return float(weighted / total)


def quick_metrics(x_norm, sr):
    """Стационарные метрики — быстрые, для всего корпуса."""
    from mosqito.sq_metrics import loudness_zwst, sharpness_din_st, tnr_ecma_st
    pa = x_norm * PA_CALIBRATION
    loud = loudness_zwst(pa, sr)
    N = loud[0] if isinstance(loud, tuple) else loud
    S = sharpness_din_st(pa, sr, weighting="din")
    S = S[0] if isinstance(S, tuple) else S
    try:
        tnr_res = tnr_ecma_st(pa, sr, prominence=True)
        t_tnr = float(tnr_res[0]) if isinstance(tnr_res, tuple) else float(tnr_res)
    except Exception:
        t_tnr = np.nan

    return dict(
        loudness_sone_stationary=float(np.atleast_1d(N)[0]),
        sharpness_acum_stationary=float(np.atleast_1d(S)[0]),
        tonality_tnr_db=t_tnr,
        warmth_ratio=warmth_ratio(x_norm, sr),
        harshness=harshness(x_norm, sr),
    )


def full_timevarying(x_norm, sr):
    """Покадровые метрики — дорого, только для CORE_ROLES."""
    from mosqito.sq_metrics import loudness_zwtv, sharpness_din_tv, roughness_dw
    pa = x_norm * PA_CALIBRATION

    N, N_spec, bark_n, t_n = loudness_zwtv(pa, sr, field_type="free")
    S, t_s = sharpness_din_tv(pa, sr, weighting="din", skip=0.1)
    R, R_spec, bark_r, t_r = roughness_dw(pa, sr, overlap=0)

    summary = dict(
        loudness_sone_median=float(np.median(N)),
        loudness_sone_p95=float(np.percentile(N, 95)),
        sharpness_acum_median=float(np.median(S)),
        sharpness_acum_p95=float(np.percentile(S, 95)),
        roughness_asper_median=float(np.median(R)),
        roughness_asper_p95=float(np.percentile(R, 95)),
    )
    frames = dict(
        loudness=pd.DataFrame({"t_s": t_n, "sone": N}),
        sharpness=pd.DataFrame({"t_s": t_s, "acum": S}),
        roughness=pd.DataFrame({"t_s": np.ravel(t_r), "asper": np.ravel(R) if np.ndim(R) else np.full(len(np.ravel(t_r)), R)}),
    )
    return summary, frames
