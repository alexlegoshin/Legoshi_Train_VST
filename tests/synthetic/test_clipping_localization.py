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
    r1, r2 = info["clipped_regions_s"]
    assert abs(r1["start_s"] - 1.0) < 0.01 and abs(r1["duration_s"] - r1_len / SR) < 0.01
    assert abs(r2["start_s"] - 3.5) < 0.01 and abs(r2["duration_s"] - r2_len / SR) < 0.01
    expected_fraction = (r1_len + r2_len) / n
    assert abs(info["clipped_fraction"] - expected_fraction) < 1e-6
    # оба отрезка короче CLICK_MAX_S (5мс) -> классифицируются как "click"
    assert r1["severity"] == "click" and r2["severity"] == "click"
    # клиппинг применён к обоим каналам одинаково -> оба в списке channels
    assert r1["channels"] == [0, 1]


def test_sustained_region_classified_correctly():
    """Отрезок длиннее CLICK_MAX_S (5мс) должен классифицироваться как
    "sustained" — разные режимы восстановления у любого реального
    инструмента (см. analysis/recommendations.py, category=declip_*)."""
    n = 2 * SR
    x = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    data = np.column_stack([x, x])
    long_len = int(0.02 * SR)  # 20мс, заведомо длиннее порога 5мс
    data[1000:1000 + long_len, :] = 1.0
    info = detect_clipping(data, SR)
    assert info["n_clipped_runs"] == 1
    assert info["clipped_regions_s"][0]["severity"] == "sustained"


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
    каналам в реализации, не np.all — и сообщается ИМЕННО этот канал, не
    оба (важно для рекомендации: чинить один канал — другая задача, чем
    оба сразу)."""
    n = 2 * SR
    left = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    right = left.copy()
    right[1000:1010] = 1.0  # клиппинг только в правом канале
    data = np.column_stack([left, right])
    info = detect_clipping(data, SR)
    assert info["clipped"] is True
    assert info["clipped_regions_s"][0]["channels"] == [1]


if __name__ == "__main__":
    test_no_clipping_on_clean_sine()
    test_localizes_two_known_regions()
    test_sustained_region_classified_correctly()
    test_run_shorter_than_run_len_not_counted()
    test_clipping_on_single_channel_still_detected()
    print("Все тесты локализации клиппинга (Блок 2) прошли.")
