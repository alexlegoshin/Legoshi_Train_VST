"""Синтетика для гейта по вокалу в скользящем окне (ТЗ-03). Вибрато в окне
агрегируется из уже провалидированных per-note значений (notes_df), не
пересчитывается заново по сырым кадрам — так что тестовые notes_df здесь
играют роль "уже посчитанного §4.6"."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pyloudnorm as pyln

from analysis.legacy_scripts.stitch.run_tz03_moving_window import analyze_file_windows, MAX_PLAUSIBLE_VIBRATO_CENTS

SR = 44100


def make_f0_df(t, voiced, f0_hz):
    return pd.DataFrame({"t_s": t, "voiced": voiced, "f0_hz": f0_hz, "prob": np.full(len(t), 0.5)})


def make_notes_df(centers, depths):
    return pd.DataFrame({
        "t_start": [c - 0.1 for c in centers], "t_end": [c + 0.1 for c in centers],
        "vibrato_depth_cents": depths,
    })


def test_gate_blocks_no_vocal_window():
    """Окно без вокала (pYIN voiced=False везде) должно дать NaN вибрато,
    даже если аудио громкое (не гасится энергетическим гейтом)."""
    dur = 8.0
    n = int(dur * SR)
    mono = 0.3 * np.random.default_rng(0).standard_normal(n)  # шум, не вокал
    t = np.arange(0, dur, 512 / SR)
    f0_df = make_f0_df(t, voiced=np.zeros(len(t), dtype=bool), f0_hz=np.full(len(t), np.nan))
    meter = pyln.Meter(SR)
    wdf = analyze_file_windows("x", mono, SR, f0_df, None, meter, notes_df=None)
    assert len(wdf) > 0, "энергетический гейт не должен был всё выкинуть"
    assert wdf.vibrato_depth_cents.isna().all(), "без вокала вибрато должно быть NaN везде"


def test_gate_passes_real_vocal_window():
    """Окно с подтверждённым вокалом (voiced=True и в миксе, и в стеме) и
    реальными нотами в notes_df должно агрегировать их медиану, не NaN."""
    dur = 8.0
    n = int(dur * SR)
    mono = 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
    t = np.arange(0, dur, 512 / SR)
    f0_df = make_f0_df(t, voiced=np.ones(len(t), dtype=bool), f0_hz=np.full(len(t), 220.0))
    stem_f0 = f0_df.copy()
    stem_f0["t_mix"] = stem_f0["t_s"]
    # ноты по всему треку (0-8с), у каждого 4с-окна должно найтись >=2
    notes_df = make_notes_df(centers=np.arange(0.5, 7.5, 0.7), depths=[35.0] * 10)

    meter = pyln.Meter(SR)
    wdf = analyze_file_windows("x", mono, SR, f0_df, stem_f0, meter, notes_df)
    assert wdf.vibrato_depth_cents.notna().any(), "с подтверждённым вокалом+нотами гейт не должен всё выкинуть"
    depths = wdf.vibrato_depth_cents.dropna()
    assert (depths == 35.0).all(), f"ожидали медиану 35.0 из notes_df, получили {depths.tolist()}"


def test_gate_blocks_when_stem_disagrees():
    """Даже если pYIN на "миксе" считает окно voiced (ложно), а стем в то же
    время НЕ voiced — гейт должен заблокировать (это и есть починка бага)."""
    dur = 8.0
    n = int(dur * SR)
    mono = 0.3 * np.random.default_rng(1).standard_normal(n)
    t = np.arange(0, dur, 512 / SR)
    f0_df = make_f0_df(t, voiced=np.ones(len(t), dtype=bool), f0_hz=np.full(len(t), 150.0))
    stem_f0 = make_f0_df(t, voiced=np.zeros(len(t), dtype=bool), f0_hz=np.full(len(t), np.nan))
    stem_f0["t_mix"] = stem_f0["t_s"]
    notes_df = make_notes_df(centers=[2.0, 3.0, 4.0], depths=[30.0, 32.0, 28.0])

    meter = pyln.Meter(SR)
    wdf = analyze_file_windows("x", mono, SR, f0_df, stem_f0, meter, notes_df)
    assert wdf.vibrato_depth_cents.isna().all(), \
        "стем говорит 'нет вокала' -> гейт должен заблокировать, даже если микс-трекер ошибся"


def test_implausible_vibrato_filtered_out():
    """Ноты с физически невозможной глубиной (>100 центов — классический
    сбой трекера на шуме/реверб-хвосте) должны быть отброшены при агрегации,
    не усреднены как настоящие."""
    dur = 8.0
    n = int(dur * SR)
    mono = 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(n) / SR)
    t = np.arange(0, dur, 512 / SR)
    f0_df = make_f0_df(t, voiced=np.ones(len(t), dtype=bool), f0_hz=np.full(len(t), 220.0))
    stem_f0 = f0_df.copy()
    stem_f0["t_mix"] = stem_f0["t_s"]
    # все "ноты" в этом окне — мусор трекера (470, 870 центов)
    notes_df = make_notes_df(centers=[2.0, 2.5, 3.0], depths=[470.0, 870.0, 520.0])

    meter = pyln.Meter(SR)
    wdf = analyze_file_windows("x", mono, SR, f0_df, stem_f0, meter, notes_df)
    assert wdf.vibrato_depth_cents.isna().all(), \
        f"мусорные ноты (>{MAX_PLAUSIBLE_VIBRATO_CENTS} центов) не должны давать число, " \
        f"получили {wdf.vibrato_depth_cents.dropna().tolist()}"


if __name__ == "__main__":
    test_gate_blocks_no_vocal_window()
    test_gate_passes_real_vocal_window()
    test_gate_blocks_when_stem_disagrees()
    test_implausible_vibrato_filtered_out()
    print("Все синтетические тесты гейта по вокалу прошли.")
