"""Блок 6 (симулятор вмешательств): ~8-10 канонических DSP-ходов, не больше
(roadmap.md — сознательно не расширять без реальной нужды). Панорама/
стерео-пространство сюда не входит — `overall_ms_ratio` не измерим в
принципе (NO_ZONE в amber.json), вмешательства туда не попадут ни в одну
зону, считать их незачем.

Стандартные RBJ Audio EQ Cookbook биквады (peaking/shelf) — общеизвестная
форма, коэффициенты выведены напрямую, без внешней EQ-библиотеки.
Компрессор и сатурация — минимальные, без цели точно попасть в заданное
число дБ (не мастеринг-инструмент, только для получения ВЕКТОРА дельт
метрик на разумную, типичную амплитуду хода)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.signal import lfilter


def _apply_biquad(x, b, a):
    if x.ndim == 1:
        return lfilter(b, a, x)
    return np.stack([lfilter(b, a, x[:, ch]) for ch in range(x.shape[1])], axis=1)


def peaking_eq(x, sr, freq, gain_db, q=1.0):
    """RBJ peaking EQ (bell). gain_db>0 — буст, <0 — срез."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return _apply_biquad(x, b, a)


def shelf_eq(x, sr, freq, gain_db, kind="high", q=0.707):
    """RBJ shelving EQ. kind="high" — полка выше freq, "low" — ниже.
    q=0.707 (Butterworth) — стандартный "плоский" наклон полки."""
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if kind == "high":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif kind == "low":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    else:
        raise ValueError(f"kind должен быть 'high' или 'low', получили {kind!r}")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return _apply_biquad(x, b, a)


def compressor_soft(x, sr, ratio=3.0, knee_db=6.0, time_const_ms=30.0,
                     threshold_offset_db=6.0):
    """Мягкий (soft-knee) фидфорвард-компрессор. Огибающая — одно-полюсное
    сглаживание МОЩНОСТИ (x^2), не attack/release на сыром |x[n]|: для
    периодического тона отдельные attack/release на мгновенном |x[n]|
    переключаются каждые пол-периода волны и не успевают стабилизироваться
    между переключениями — огибающая застревает далеко ниже реальной
    амплитуды, компрессор почти не реагирует даже на устойчиво громкий
    участок (найдено тестом test_compressor_reduces_crest_factor).
    Векторизовано через lfilter — важно для скорости на реальных треках
    (Блок 6 гоняет это по корпусу, не по одному синтетическому файлу).

    Порог не подбирается под целевое снижение — берётся относительно
    собственного RMS сигнала (threshold_offset_db выше него), чтобы
    компрессор реально что-то делал на любом входном материале, а не
    промахивался мимо уровня конкретного трека."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    rms_db = 20 * np.log10(np.sqrt(np.mean(mono ** 2)) + 1e-12)
    threshold_db = rms_db + threshold_offset_db

    alpha = np.exp(-1.0 / (sr * time_const_ms / 1000))
    power = mono.astype(np.float64) ** 2
    env_power = lfilter([1 - alpha], [1, -alpha], power)
    env_db = 10 * np.log10(env_power + 1e-12)  # степень, не амплитуда — 10*log10, не 20*

    over = env_db - threshold_db
    half_knee = knee_db / 2
    gain_reduction_db = np.zeros_like(over)
    in_knee = np.abs(over) <= half_knee
    above_knee = over > half_knee
    gain_reduction_db[above_knee] = over[above_knee] * (1 - 1 / ratio)
    knee_over = over[in_knee] + half_knee
    gain_reduction_db[in_knee] = ((1 - 1 / ratio) * knee_over ** 2) / (2 * knee_db + 1e-12)

    gain_lin = 10 ** (-gain_reduction_db / 20)
    return x * gain_lin[:, None] if x.ndim == 2 else x * gain_lin


def saturation_soft(x, drive=1.5):
    """Мягкая (tape-стиль) сатурация — tanh waveshaper, нормализован по
    уровню (не меняет пиковую громкость сам по себе, только форму волны/
    гармоники) — drive>1 сильнее гнёт форму, добавляя нечётные гармоники."""
    norm = np.tanh(drive)
    return np.tanh(x * drive) / norm


MOVES = {
    "bell_cut_lowmid": lambda x, sr: peaking_eq(x, sr, freq=300.0, gain_db=-3.0, q=1.0),
    "bell_boost_lowmid": lambda x, sr: peaking_eq(x, sr, freq=300.0, gain_db=3.0, q=1.0),
    "bell_cut_presence": lambda x, sr: peaking_eq(x, sr, freq=3000.0, gain_db=-3.0, q=1.0),
    "bell_boost_presence": lambda x, sr: peaking_eq(x, sr, freq=3000.0, gain_db=3.0, q=1.0),
    "shelf_air_boost": lambda x, sr: shelf_eq(x, sr, freq=10000.0, gain_db=2.0, kind="high"),
    "shelf_air_cut": lambda x, sr: shelf_eq(x, sr, freq=10000.0, gain_db=-2.0, kind="high"),
    "shelf_low_boost": lambda x, sr: shelf_eq(x, sr, freq=80.0, gain_db=2.0, kind="low"),
    "shelf_low_cut": lambda x, sr: shelf_eq(x, sr, freq=80.0, gain_db=-2.0, kind="low"),
    "compressor_soft": lambda x, sr: compressor_soft(x, sr),
    "saturation_soft": lambda x, sr: saturation_soft(x),
}

# Блок 7: человекочитаемое описание параметров хода («примерно 2дБ в
# районе 3кГц», roadmap.md — округлённый вывод, не ложная точность).
# Числа буквально те же, что в MOVES выше — держать в синхроне при правке.
MOVE_DESCRIPTIONS = {
    "bell_cut_lowmid": "колокол, срез ~3дБ на ~300Гц",
    "bell_boost_lowmid": "колокол, подъём ~3дБ на ~300Гц",
    "bell_cut_presence": "колокол, срез ~3дБ на ~3кГц",
    "bell_boost_presence": "колокол, подъём ~3дБ на ~3кГц",
    "shelf_air_boost": "высокая полка, подъём ~2дБ выше ~10кГц",
    "shelf_air_cut": "высокая полка, срез ~2дБ выше ~10кГц",
    "shelf_low_boost": "низкая полка, подъём ~2дБ ниже ~80Гц",
    "shelf_low_cut": "низкая полка, срез ~2дБ ниже ~80Гц",
    "compressor_soft": "мягкая компрессия (~3:1)",
    "saturation_soft": "мягкая сатурация (tape-стиль)",
}
