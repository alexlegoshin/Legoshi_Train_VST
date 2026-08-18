"""Блок 3 (измерение, «наложение дублей»): первый тест на layering.py —
код был написан для §4.7 (исследовательский корпус), но никогда не
покрывался тестами и не вызывался из живого пайплайна (roadmap.md, Блок 3).
Не рекомендация, только измерение расхождения по времени/питчу и верхняя
оценка риска гребёнки — фикс ждёт Блок 8 («с выбором»)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

from analysis.metrics.layering import analyze_pair

SR = 44100


def _make_vocal(seconds, sr, f0_base=250.0, cents_shift=0.0):
    """Вокалоподобный сигнал с вибрато и гармониками — не голая синусоида,
    та же конструкция, что и в test_block_v_regressions.py (pYIN на чистом
    тоне ведёт себя иначе, чем на реальном материале)."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0_curve = (f0_base * (2 ** (cents_shift / 1200))) + 15 * np.sin(2 * np.pi * 4 * t)
    phase = 2 * np.pi * np.cumsum(f0_curve) / sr
    return 0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)


def test_identical_dubs_show_low_divergence(tmp_path):
    x = _make_vocal(6.0, SR)
    path_a, path_b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(path_a), x, SR, subtype="FLOAT")
    sf.write(str(path_b), x, SR, subtype="FLOAT")

    summary = analyze_pair(path_a, path_b, offset_a_s=0.0, offset_b_s=0.0, sr_expected=SR)
    assert summary["simultaneity_fraction"] > 0.5, "оба дубля voiced почти всегда — совпадение по времени должно быть высоким"
    assert summary["pitch_divergence_cents_median"] < 3.0, summary
    assert summary["comb_risk_upper_bound"] > 0.5, "идентичные дубли — заведомо высокий риск гребёнки"


def test_pitch_shifted_dub_shows_divergence(tmp_path):
    """Дубль B спет на ~50 центов выше A, без сдвига по времени — должно
    отразиться в pitch_divergence, не во time_divergence."""
    a = _make_vocal(6.0, SR, cents_shift=0.0)
    b = _make_vocal(6.0, SR, cents_shift=50.0)
    path_a, path_b = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(path_a), a, SR, subtype="FLOAT")
    sf.write(str(path_b), b, SR, subtype="FLOAT")

    summary = analyze_pair(path_a, path_b, offset_a_s=0.0, offset_b_s=0.0, sr_expected=SR)
    assert 30.0 < summary["pitch_divergence_cents_median"] < 70.0, \
        f"ожидали ~50 центов расхождения, получили {summary['pitch_divergence_cents_median']}"


if __name__ == "__main__":
    test_identical_dubs_show_low_divergence(Path("/tmp"))
    test_pitch_shifted_dub_shows_divergence(Path("/tmp"))
    print("Все тесты layering.analyze_pair (Блок 3) прошли.")
