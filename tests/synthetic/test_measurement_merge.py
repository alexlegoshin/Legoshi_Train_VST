"""orchestrate.analyze_all_sources: регрессия на баг код-ревью (найден на
реальном треке, не на синтетике) — track_avg-версия warmth_ratio (одно
число на весь трек, psychoacoustic.quick_metrics) молча затиралась
оконной Series того же имени из window_metrics, хотя legoshi_amber.json
объявляет зону warmth_ratio/mix с granularity="track_avg" и ждёт именно
скаляр. harshness такой проблемы не имеет — единственная зона (vocals)
объявлена granularity="window" и как раз должна получать оконную версию,
это НЕ регрессия, если harshness остаётся Series."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import soundfile as sf

import orchestrate

SR = 44100


def _tone(freq, seconds, amp=0.3, sr=SR):
    n = int(seconds * sr)
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def test_warmth_ratio_mix_stays_scalar_not_overwritten_by_window_series(tmp_path):
    dur = 10.0
    mix = _tone(220.0, dur, amp=0.3) + 0.05 * np.random.default_rng(0).standard_normal(int(dur * SR))
    to_stereo = lambda x: np.column_stack([x, x])

    mix_path = tmp_path / "mix.wav"
    sf.write(str(mix_path), to_stereo(mix), SR, subtype="FLOAT")

    measurements, _diag = orchestrate.analyze_all_sources(
        mix_path, {}, deep_psychoacoustics=False, is_ml_separated=True, track_name="warmth_regression")

    val = measurements.get(("warmth_ratio", "mix"))
    assert val is not None
    assert isinstance(val, (int, float, np.floating)), (
        f"warmth_ratio/mix должен остаться скаляром track_avg (legoshi_amber.json: "
        f"granularity=track_avg), получили {type(val)} — регрессия: оконная Series "
        f"снова затирает track_avg-число")


def test_harshness_vocals_still_gets_window_series_not_regressed_by_fix():
    """Убеждаемся, что фикс warmth_ratio не задел harshness — там оконная
    версия НУЖНА (единственная зона harshness/vocals объявлена
    granularity='window')."""
    dur = 6.0
    vocals = _tone(1000.0, dur, amp=0.3)
    to_stereo = lambda x: np.column_stack([x, x])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mix_path = tmp_path / "mix.wav"
        vocals_path = tmp_path / "vocals.wav"
        sf.write(str(mix_path), to_stereo(vocals), SR, subtype="FLOAT")
        sf.write(str(vocals_path), to_stereo(vocals), SR, subtype="FLOAT")

        measurements, _diag = orchestrate.analyze_all_sources(
            mix_path, {"vocals": vocals_path}, deep_psychoacoustics=False,
            is_ml_separated=True, track_name="harshness_no_regress")

        val = measurements.get(("harshness", "vocals"))
        assert val is not None
        assert isinstance(val, pd.Series), (
            f"harshness/vocals должен остаться оконной Series (granularity=window), "
            f"получили {type(val)} — фикс warmth_ratio не должен был это тронуть")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_warmth_ratio_mix_stays_scalar_not_overwritten_by_window_series(Path(tmp))
    test_harshness_vocals_still_gets_window_series_not_regressed_by_fix()
    print("Все тесты слияния измерений (warmth_ratio/harshness) прошли.")
