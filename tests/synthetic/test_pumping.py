"""Регрессия на баг pumping_signature, пойманный на реальном корпусе:
argmax по диапазону лагов путает "монотонно убывающая гладкая огибающая"
с "периодическая огибающая" — обе дают высокое значение на самом коротком
разрешённом лаге. Нужен настоящий локальный максимум."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from analysis.metrics.loudness_dynamics import pumping_signature

SR = 44100


def test_monotonic_smooth_envelope_gives_no_false_pumping():
    """Огибающая, гладко и монотонно убывающая, без всякой периодики —
    ровно та ситуация, в которой на реальном «основной трек» старая версия
    молча выдавала границу диапазона (50мс) как будто нашла пампинг."""
    rng = np.random.default_rng(0)
    dur = 20.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 1.0 - 0.7 * t / dur  # монотонный линейный спад, без циклов
    x = rng.standard_normal(n) * env

    res = pumping_signature(x, SR)
    print(f"[monotonic] {res}")
    assert res["pumping_detected"] is False, "ложное срабатывание на монотонно убывающей огибающей"
    assert res["pumping_score"] == 0.0


def test_genuine_periodic_modulation_still_detected():
    """Не сломать то, что раньше работало — явная периодика 200мс всё ещё
    должна находиться."""
    rng = np.random.default_rng(1)
    dur = 20.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.abs(np.sin(np.pi * t / 0.2))  # период 200мс
    x = rng.standard_normal(n) * env

    res = pumping_signature(x, SR)
    print(f"[periodic] {res}")
    assert res["pumping_detected"] is True
    assert abs(res["pumping_period_ms"] - 200.0) < 15.0, f"период найден неверно: {res['pumping_period_ms']}"
    assert res["pumping_score"] > 0.5


if __name__ == "__main__":
    test_monotonic_smooth_envelope_gives_no_false_pumping()
    test_genuine_periodic_modulation_still_detected()
    print("ALL PUMPING REGRESSION TESTS PASSED")
