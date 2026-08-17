"""Синтетика для §4.2 THD/чёт-нечет (задача #24) — известный сигнал с
заданными амплитудами гармоник, до всякого доверия реальным нотам."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis.metrics.harmonics import analyze_notes, note_harmonic_amps

SR = 44100


def make_tone(f0, harmonic_amps, dur_s, sr=SR):
    """harmonic_amps: {k: amplitude} относительно амплитуды 1.0 у k=1."""
    t = np.arange(int(dur_s * sr)) / sr
    x = np.zeros_like(t)
    for k, a in harmonic_amps.items():
        x += a * np.sin(2 * np.pi * k * f0 * t)
    return x


def test_pure_tone_zero_thd():
    x = make_tone(220.0, {1: 1.0}, 1.0)
    amps = note_harmonic_amps(x, SR, 220.0)
    thd = np.sqrt(np.nansum(amps[1:] ** 2)) / amps[0]
    assert thd < 0.01, f"чистый тон без гармоник дал THD={thd:.4f}, ожидали ~0"


def test_known_thd_value():
    # k=2 амплитуда 0.3, k=3 амплитуда 0.1 относительно основного 1.0
    # THD = sqrt(0.3^2 + 0.1^2) / 1.0 = sqrt(0.09+0.01) = sqrt(0.10) = 0.3162
    x = make_tone(220.0, {1: 1.0, 2: 0.3, 3: 0.1}, 1.0)
    amps = note_harmonic_amps(x, SR, 220.0)
    thd = np.sqrt(np.nansum(amps[1:] ** 2)) / amps[0]
    expected = np.sqrt(0.3 ** 2 + 0.1 ** 2)
    assert abs(thd - expected) < 0.02, f"THD={thd:.4f}, ожидали {expected:.4f}"


def test_even_odd_ratio():
    # только чётные (k=2,4): чёт/нечет должно быть >> 1
    x_even = make_tone(220.0, {1: 1.0, 2: 0.4, 4: 0.2}, 1.0)
    amps_even = note_harmonic_amps(x_even, SR, 220.0)
    even_e = amps_even[1] ** 2 + amps_even[3] ** 2
    odd_e = amps_even[2] ** 2 + amps_even[4] ** 2 + amps_even[6] ** 2
    ratio_even = even_e / max(odd_e, 1e-20)

    # только нечётные (k=3,5): чёт/нечет должно быть << 1
    x_odd = make_tone(220.0, {1: 1.0, 3: 0.4, 5: 0.2}, 1.0)
    amps_odd = note_harmonic_amps(x_odd, SR, 220.0)
    even_e2 = amps_odd[1] ** 2 + amps_odd[3] ** 2
    odd_e2 = amps_odd[2] ** 2 + amps_odd[4] ** 2 + amps_odd[6] ** 2
    ratio_odd = even_e2 / max(odd_e2, 1e-20)

    assert ratio_even > 10, f"сигнал только с чётными гармониками дал чёт/нечет={ratio_even:.3f}"
    assert ratio_odd < 0.1, f"сигнал только с нечётными гармониками дал чёт/нечет={ratio_odd:.3f}"


def test_note_below_min_duration_skipped():
    # note короче MIN_NOTE_S (0.15с) должна быть отфильтрована analyze_notes
    x = make_tone(220.0, {1: 1.0, 2: 0.3}, 2.0)
    notes = pd.DataFrame([
        dict(t_start=0.0, t_end=0.05, duration_s=0.05, f0_start=220.0, f0_end=220.0),  # короткая
        dict(t_start=0.2, t_end=1.8, duration_s=1.6, f0_start=220.0, f0_end=220.0),     # нормальная
    ])
    summary, per_note = analyze_notes(x, SR, notes)
    assert summary["n_notes_for_thd"] == 1, f"ожидали 1 ноту после фильтра, получили {summary['n_notes_for_thd']}"


def test_full_pipeline_summary_matches_known_thd():
    x = make_tone(220.0, {1: 1.0, 2: 0.3, 3: 0.1}, 1.0)
    notes = pd.DataFrame([dict(t_start=0.0, t_end=1.0, duration_s=1.0, f0_start=220.0, f0_end=220.0)])
    summary, per_note = analyze_notes(x, SR, notes)
    expected = np.sqrt(0.3 ** 2 + 0.1 ** 2)
    assert abs(summary["thd_median"] - expected) < 0.02, summary


if __name__ == "__main__":
    test_pure_tone_zero_thd()
    test_known_thd_value()
    test_even_odd_ratio()
    test_note_below_min_duration_skipped()
    test_full_pipeline_summary_matches_known_thd()
    print("Все синтетические тесты §4.2 THD прошли.")
