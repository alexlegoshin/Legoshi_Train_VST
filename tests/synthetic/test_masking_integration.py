"""Блок 4 (частотные конфликты, измерение): masking_erb.analyze_group уже
был написан (§4.8) и покрыт своими юнит-тестами раньше, но никогда не
вызывался из orchestrate.analyze_all_sources — этот тест проверяет именно
подключение (audibility в diagnostics по роли, attribution в _masking),
не сам алгоритм маскирования заново."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import soundfile as sf

import orchestrate

SR = 44100


def _tone(freq, seconds, amp=0.3, sr=SR):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_masking_wired_into_analyze_all_sources(tmp_path):
    dur = 6.0
    vocals = _tone(1000.0, dur, amp=0.3)
    bass = _tone(80.0, dur, amp=0.6)
    mix = vocals + bass
    to_stereo = lambda x: np.column_stack([x, x])

    mix_path = tmp_path / "mix.wav"
    vocals_path = tmp_path / "vocals.wav"
    bass_path = tmp_path / "bass.wav"
    sf.write(str(mix_path), to_stereo(mix), SR, subtype="FLOAT")
    sf.write(str(vocals_path), to_stereo(vocals), SR, subtype="FLOAT")
    sf.write(str(bass_path), to_stereo(bass), SR, subtype="FLOAT")

    _, diagnostics = orchestrate.analyze_all_sources(
        mix_path, {"vocals": vocals_path, "bass": bass_path},
        deep_psychoacoustics=False, is_ml_separated=True, track_name="masking_smoke")

    assert "audibility" in diagnostics["vocals"], diagnostics["vocals"]
    assert "audibility" in diagnostics["bass"], diagnostics["bass"]
    assert 0.0 <= diagnostics["vocals"]["audibility"] <= 1.0
    assert 0.0 <= diagnostics["bass"]["audibility"] <= 1.0
    # mix сама с собой не участвует в маскировании (сумма всего, не источник)
    assert "audibility" not in diagnostics["mix"]


def test_masking_skipped_with_single_source(tmp_path):
    """С одним источником (кроме mix) маскировать некого — не должно падать,
    просто нет audibility/_masking в выходе."""
    dur = 3.0
    vocals = _tone(1000.0, dur)
    to_stereo = lambda x: np.column_stack([x, x])

    mix_path = tmp_path / "mix.wav"
    vocals_path = tmp_path / "vocals.wav"
    sf.write(str(mix_path), to_stereo(vocals), SR, subtype="FLOAT")
    sf.write(str(vocals_path), to_stereo(vocals), SR, subtype="FLOAT")

    _, diagnostics = orchestrate.analyze_all_sources(
        mix_path, {"vocals": vocals_path},
        deep_psychoacoustics=False, is_ml_separated=True, track_name="masking_single")

    assert "audibility" not in diagnostics["vocals"]
    assert "_masking" not in diagnostics


def test_assert_consistent_sr_passes_when_all_equal():
    orchestrate._assert_consistent_sr({"vocals": 44100, "bass": 44100, "drums": 44100})  # не должно упасть


def test_assert_consistent_sr_raises_on_mismatch():
    """Регрессия (найдена код-ревью): раньше единый sr для analyze_group
    молча брался с последней обработанной роли без проверки, что
    остальные источники реально совпадают по частоте — латентный баг,
    сегодня не триггерится (оба режима входа приводят всё к
    engine.PIPELINE_SR раньше), но без явной проверки был бы виден только
    как систематически неверные audibility/атрибуция, без исключения."""
    with pytest.raises(ValueError, match="разных sr"):
        orchestrate._assert_consistent_sr({"vocals": 44100, "bass": 48000})


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_masking_wired_into_analyze_all_sources(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_masking_skipped_with_single_source(Path(tmp))
    test_assert_consistent_sr_passes_when_all_equal()
    test_assert_consistent_sr_raises_on_mismatch()
    print("Все тесты подключения маскирования (Блок 4) прошли.")
