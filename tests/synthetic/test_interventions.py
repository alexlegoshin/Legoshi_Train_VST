"""Блок 6 (симулятор вмешательств): DSP-ходы должны реально двигать
метрику в заявленном направлении — иначе таблица «вмешательство → вектор
дельт» (roadmap.md, Блок 6) будет врать с самого начала."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.interventions import MOVES, compressor_soft, peaking_eq, saturation_soft, shelf_eq

SR = 44100


def _band_energy_db(x, sr, f_lo, f_hi):
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return 20 * np.log10(np.sqrt(np.mean(X[mask] ** 2)) + 1e-12)


def _pink_ish_noise(seconds, sr, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    white = rng.standard_normal(n)
    # грубый розовый шум через интегрирование в частотной области — просто
    # чтобы во ВСЕХ интересующих полосах (низ/presence/воздух) было что мерить
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    X[1:] /= np.sqrt(freqs[1:])
    return np.fft.irfft(X, n)


def test_bell_boost_raises_target_band_more_than_cut():
    x = _pink_ish_noise(4.0, SR)
    boosted = peaking_eq(x, SR, freq=300.0, gain_db=3.0, q=1.0)
    cut = peaking_eq(x, SR, freq=300.0, gain_db=-3.0, q=1.0)
    e_boost = _band_energy_db(boosted, SR, 250, 400)
    e_cut = _band_energy_db(cut, SR, 250, 400)
    assert e_boost > e_cut + 3.0, f"буст должен быть заметно громче среза в целевой полосе: {e_boost} vs {e_cut}"


def test_high_shelf_moves_only_the_high_band():
    x = _pink_ish_noise(4.0, SR)
    boosted = shelf_eq(x, SR, freq=10000.0, gain_db=2.0, kind="high")
    e_before_high = _band_energy_db(x, SR, 12000, 18000)
    e_after_high = _band_energy_db(boosted, SR, 12000, 18000)
    assert e_after_high > e_before_high + 1.0
    e_before_low = _band_energy_db(x, SR, 100, 300)
    e_after_low = _band_energy_db(boosted, SR, 100, 300)
    assert abs(e_after_low - e_before_low) < 0.5, "высокая полка не должна заметно трогать низ"


def test_low_shelf_moves_only_the_low_band():
    x = _pink_ish_noise(4.0, SR)
    cut = shelf_eq(x, SR, freq=80.0, gain_db=-2.0, kind="low")
    e_before_low = _band_energy_db(x, SR, 30, 70)
    e_after_low = _band_energy_db(cut, SR, 30, 70)
    assert e_after_low < e_before_low - 1.0


def test_compressor_reduces_sustained_burst_level():
    """На устойчиво громком участке (не единичный сэмпл-пик, а секция
    ~300мс, заведомо длиннее time_const_ms=30мс) компрессор должен заметно
    снизить уровень относительно исходного. Смотрим именно на УСТОЯВШУЮСЯ
    часть всплеска (пропускаем первые 100мс — там ещё сходится огибающая),
    не на crest factor всего файла — большая часть файла тихая и
    некомпрессируемая, разбавляет эффект до незаметности в агрегате."""
    n = int(4.0 * SR)
    t = np.arange(n) / SR
    x = 0.05 * np.sin(2 * np.pi * 440 * t)
    burst_start, burst_len = int(1.0 * SR), int(0.3 * SR)
    burst_end = burst_start + burst_len
    x[burst_start:burst_end] = 0.9 * np.sin(2 * np.pi * 440 * t[burst_start:burst_end])

    compressed = compressor_soft(x, SR)
    settle = burst_start + int(0.1 * SR)
    raw_level = np.sqrt(np.mean(x[settle:burst_end] ** 2))
    comp_level = np.sqrt(np.mean(compressed[settle:burst_end] ** 2))
    reduction_db = 20 * np.log10(raw_level / comp_level)
    assert reduction_db > 1.0, f"ожидали заметное снижение уровня всплеска, получили {reduction_db:.2f}дБ"


def test_saturation_adds_odd_harmonics():
    """Чистый тон 200Гц через мягкую сатурацию должен набрать энергию на
    3-й/5-й гармониках (нечётные — характерная черта tanh-сатурации)."""
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    x = 0.5 * np.sin(2 * np.pi * 200 * t)
    sat = saturation_soft(x, drive=3.0)
    e_3rd_before = _band_energy_db(x, SR, 580, 620)
    e_3rd_after = _band_energy_db(sat, SR, 580, 620)
    assert e_3rd_after > e_3rd_before + 10.0


def test_all_moves_run_on_mono_and_stereo_without_crashing():
    mono = _pink_ish_noise(1.0, SR)
    stereo = np.column_stack([mono, mono])
    for name, fn in MOVES.items():
        out_mono = fn(mono, SR)
        assert out_mono.shape == mono.shape, name
        assert np.all(np.isfinite(out_mono)), name
        out_stereo = fn(stereo, SR)
        assert out_stereo.shape == stereo.shape, name
        assert np.all(np.isfinite(out_stereo)), name


def test_exactly_ten_moves():
    """roadmap.md: ~8-10 канонических ходов, не больше без реальной нужды."""
    assert len(MOVES) == 10


if __name__ == "__main__":
    test_bell_boost_raises_target_band_more_than_cut()
    test_high_shelf_moves_only_the_high_band()
    test_low_shelf_moves_only_the_low_band()
    test_compressor_reduces_sustained_burst_level()
    test_saturation_adds_odd_harmonics()
    test_all_moves_run_on_mono_and_stereo_without_crashing()
    test_exactly_ten_moves()
    print("Все тесты DSP-ходов (Блок 6) прошли.")
