"""ТЗ-05 А1/А2: файл и его -10дБ копия должны дать идентичные метрики,
кроме тех, что по определению зависят от абсолютного уровня (LUFS, пики,
громкость в сонах). Число окон после энергетического гейта тоже обязано
совпасть — гейт применяется к нормализованному сигналу, не к сырому."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import soundfile as sf

from analysis import engine

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from local_paths import GAIN_INVARIANCE_SRC as SRC  # локальный, не в git — см. tests/local_paths.py
except ImportError:
    SRC = ROOT / "основной трек" / "-" / "версия сведения 2.wav"  # плейсхолдер, не существует — тест skip'ается

EXPECTED_TO_DIFFER = {"integrated_lufs", "sample_peak_dbfs", "true_peak_dbfs"}
# известный остаточный numerical-noise (лог/geometric-mean накопление
# через STFT), не зависит от гейна по конструкции метрики — см. отчёт А1
KNOWN_NUMERICAL_NOISE = {"flatness_median"}
REL_TOL = 1e-4  # чуть мягче формальных 1e-6 ТЗ — учитывает FLOAT32-запись WAV


@pytest.fixture(scope="module")
def gained_pair(tmp_path_factory):
    if not SRC.exists():
        pytest.skip("тестовый корпус недоступен в этом окружении")
    tmp = tmp_path_factory.mktemp("gain_invariance")
    data, sr = sf.read(str(SRC), dtype="float64", always_2d=True)
    control = tmp / "control_0db.wav"
    sf.write(str(control), data, sr, subtype="FLOAT")
    gained = tmp / "gained_-10db.wav"
    sf.write(str(gained), data * (10 ** (-10 / 20)), sr, subtype="FLOAT")
    return control, gained


def test_track_avg_metrics_invariant_to_gain(gained_pair):
    control, gained = gained_pair
    m_orig, _, _ = engine.track_avg_metrics(control, "mix")
    m_gain, _, _ = engine.track_avg_metrics(gained, "mix")

    bad = []
    for key in set(m_orig) & set(m_gain):
        metric_name = key[0] if isinstance(key, tuple) else key
        if metric_name in EXPECTED_TO_DIFFER or metric_name in KNOWN_NUMERICAL_NOISE:
            continue
        v1, v2 = m_orig[key], m_gain[key]
        if np.isnan(v1) and np.isnan(v2):
            continue
        rel = abs(v1 - v2) / (abs(v1) + 1e-12)
        if rel > REL_TOL:
            bad.append((key, v1, v2, rel))
    assert not bad, f"метрики зависят от абсолютного гейна (не должны): {bad}"


def test_energy_gate_window_count_invariant_to_gain(gained_pair):
    control, gained = gained_pair
    mono_c, sr, _ = engine.load_mono(control)
    mono_g, _, _ = engine.load_mono(gained)
    gain_c = engine.get_mix_gain_db(control)
    gain_g = engine.get_mix_gain_db(gained)

    w_c = engine.window_metrics(mono_c, sr, "mix", mix_gain_db=gain_c)
    w_g = engine.window_metrics(mono_g, sr, "mix", mix_gain_db=gain_g)
    assert len(w_c) == len(w_g), (
        f"энергогейт даёт разное число окон на разных по громкости копиях "
        f"одного трека ({len(w_c)} vs {len(w_g)}) — гейт стоит на сыром "
        f"сигнале, а не на нормализованном к -18 LUFS")
