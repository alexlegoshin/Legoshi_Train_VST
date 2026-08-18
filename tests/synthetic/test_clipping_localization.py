"""Блок 2 (Этап 1, очистка/восстановление): detect_clipping локализует
отрезки клиппинга (таймкоды, доля трека), не только даёт bool — без
локализации "клиппинг есть" ничего не говорит о том, чинить один щелчок
или всю дорожку заново."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.engine import detect_clipping

SR = 44100


def test_no_clipping_on_clean_sine():
    n = 2 * SR
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    data = np.column_stack([x, x])
    info = detect_clipping(data, SR)
    assert info["clipped"] is False
    assert info["n_clipped_runs"] == 0
    assert info["clipped_fraction"] == 0.0
    assert info["clipped_regions_s"] == []


def test_localizes_two_known_regions():
    n = 5 * SR
    x = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    data = np.column_stack([x, x])
    r1_start, r1_len = int(1.0 * SR), 44
    r2_start, r2_len = int(3.5 * SR), 88
    data[r1_start:r1_start + r1_len, :] = 1.0
    data[r2_start:r2_start + r2_len, :] = -1.0

    info = detect_clipping(data, SR)
    assert info["clipped"] is True
    assert info["n_clipped_runs"] == 2
    (s1, e1), (s2, e2) = info["clipped_regions_s"]
    assert abs(s1 - 1.0) < 0.01 and abs((e1 - s1) - r1_len / SR) < 0.01
    assert abs(s2 - 3.5) < 0.01 and abs((e2 - s2) - r2_len / SR) < 0.01
    expected_fraction = (r1_len + r2_len) / n
    assert abs(info["clipped_fraction"] - expected_fraction) < 1e-6


def test_run_shorter_than_run_len_not_counted():
    """run_len=3 по умолчанию — 2 сэмпла подряд на полной шкале не считаются
    клиппингом (могли быть случайным пиком, не устойчивым плато)."""
    n = 2 * SR
    x = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    data = np.column_stack([x, x])
    data[1000:1002, :] = 1.0  # только 2 сэмпла подряд
    info = detect_clipping(data, SR, run_len=3)
    assert info["clipped"] is False


def test_clipping_on_single_channel_still_detected():
    """Клиппинг на ОДНОМ канале стерео-файла — тоже клиппинг, np.any по
    каналам в реализации, не np.all."""
    n = 2 * SR
    left = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    right = left.copy()
    right[1000:1010] = 1.0  # клиппинг только в правом канале
    data = np.column_stack([left, right])
    info = detect_clipping(data, SR)
    assert info["clipped"] is True


if __name__ == "__main__":
    test_no_clipping_on_clean_sine()
    test_localizes_two_known_regions()
    test_run_shorter_than_run_len_not_counted()
    test_clipping_on_single_channel_still_detected()
    print("Все тесты локализации клиппинга (Блок 2) прошли.")
