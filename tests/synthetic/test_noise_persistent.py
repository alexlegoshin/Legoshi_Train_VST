"""Синтетика для §4.10 детектора наводок по тихим участкам (задача #29)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.metrics.noise import find_persistent_narrowband

SR = 44100


def make_hum_plus_busy_mix(hum_freq=108.0, hum_amp=0.01, dur_s=20.0, sr=SR,
                            burst_frac=0.85, seed=0, note_near_hum=True):
    """Постоянная узкополосная наводка (108Гц) через весь файл + громкие
    "музыкальные" блоки, часть из которых бьёт БАСОВОЙ НОТОЙ РЯДОМ С ТЕМИ ЖЕ
    108Гц (не общий широкополосный шум — реальный случай из проекта: §4.2
    нашёл настоящий музыкальный резонанс на той же частоте, что и наводка).
    Именно совпадение по частоте раздувает std бина 108Гц при подсчёте по
    всему файлу — общий широкополосный шум это не воспроизводит (его
    энергия размазана по тысячам бинов, отдельный бин почти не задет)."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    hum = hum_amp * np.sin(2 * np.pi * hum_freq * t)

    x = hum.copy()
    n_burst_samples = int(n * burst_frac)
    n_bursts = 12
    burst_len = n_burst_samples // n_bursts
    starts = rng.choice(np.arange(0, n - burst_len, burst_len), size=n_bursts, replace=False)
    for k, s in enumerate(starts):
        tb = np.arange(burst_len) / sr
        if note_near_hum and k % 2 == 0:
            # басовая нота прямо на частоте наводки — громкая, на порядок сильнее
            note = 0.4 * np.sin(2 * np.pi * hum_freq * tb) * np.hanning(burst_len)
        else:
            note = 0.3 * rng.standard_normal(burst_len)
        x[s:s + burst_len] += note
    return x


def test_quiet_frame_detector_finds_hum():
    x = make_hum_plus_busy_mix()
    candidates = find_persistent_narrowband(x, SR, f_lo=30, f_hi=1000)
    assert len(candidates) > 0, "детектор по тихим кадрам должен найти наводку 108Гц"
    freqs_found = [c["freq_hz"] for c in candidates]
    assert any(abs(f - 108.0) < 5 for f in freqs_found), \
        f"среди найденного нет 108Гц, получили {freqs_found}"


def test_no_false_positive_on_clean_busy_mix():
    """Без наводки, только громкие всплески — ложных "наводок" быть не должно."""
    x = make_hum_plus_busy_mix(hum_amp=0.0)
    candidates = find_persistent_narrowband(x, SR, f_lo=30, f_hi=1000)
    assert len(candidates) == 0, f"без реальной наводки нашли {len(candidates)} ложных кандидатов"


if __name__ == "__main__":
    test_quiet_frame_detector_finds_hum()
    test_no_false_positive_on_clean_busy_mix()
    print("Все синтетические тесты §4.10 детектора наводок прошли.")
