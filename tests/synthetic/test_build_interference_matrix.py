"""Блок 6: проверка ПЛАМБИНГА build_interference_matrix.py на маленькой
синтетике — не гоняет настоящий корпус (это отдельный ручной запуск,
roadmap.md, Блок 6), только что baseline/after/дельта считаются
правильно и агрегация по медиане корректна."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

from analysis.build_interference_matrix import _measure, build_matrix, summarize
from analysis.interventions import MOVES
from analysis.verdict import load_preset

SR = 44100


def _make_mix(seconds=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
         + 0.05 * rng.standard_normal(n))
    return np.column_stack([x, x])


def test_summarize_takes_median_across_tracks():
    rows = [
        dict(move="bell_cut_lowmid", role="mix", track="a", metric="warmth_ratio", delta=-0.5),
        dict(move="bell_cut_lowmid", role="mix", track="b", metric="warmth_ratio", delta=-0.7),
        dict(move="bell_cut_lowmid", role="mix", track="a", metric="plr", delta=0.1),
    ]
    out = summarize(rows)
    assert out["bell_cut_lowmid"]["mix::warmth_ratio"]["median_delta"] == -0.6
    assert out["bell_cut_lowmid"]["mix::warmth_ratio"]["n"] == 2
    assert out["bell_cut_lowmid"]["mix::plr"]["n"] == 1


def test_build_matrix_produces_rows_for_zone_relevant_metrics(tmp_path):
    mix = _make_mix()
    path = tmp_path / "mix.wav"
    sf.write(str(path), mix, SR, subtype="FLOAT")

    zones = load_preset("legoshi_amber")
    zone_keys = {(z.metric, z.source) for z in zones}

    rows = build_matrix({"mix": [path]}, tmp_path, zone_keys)
    assert len(rows) > 0
    moves_seen = {r["move"] for r in rows}
    assert moves_seen == set(MOVES.keys()), f"ожидали строки по всем 10 ходам, получили {moves_seen}"
    for r in rows:
        assert r["role"] == "mix"
        assert (r["metric"], "mix") in zone_keys
        assert np.isfinite(r["delta"])


def test_build_matrix_skips_roles_without_zones(tmp_path):
    """Роли без зон в пресете (напр. если бы был передан несуществующий
    ключ) — не должны падать, просто не дают строк."""
    mix = _make_mix(seconds=1.0)
    path = tmp_path / "mix.wav"
    sf.write(str(path), mix, SR, subtype="FLOAT")
    rows = build_matrix({"mix": [path]}, tmp_path, zone_keys=set())
    assert rows == []


def _make_quiet_mix(seconds=8.0, seed=0):
    """~-60дБФС сырого RMS — заведомо ниже ENERGY_GATE_DBFS (-45), но
    выше него после нормализации к -18 LUFS (типичная громкость
    неразведённого стема, roadmap.md код-ревью)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = (0.001 * np.sin(2 * np.pi * 220 * t) + 0.0003 * np.sin(2 * np.pi * 3000 * t)
         + 0.0001 * rng.standard_normal(n))
    return np.column_stack([x, x])


def test_measure_normalizes_quiet_track_before_energy_gate(tmp_path):
    """БАГ (код-ревью, исправлен): _measure звал window_metrics без
    mix_gain_db (дефолт 0.0) — энергогейт резал окна СЫРОГО тихого
    сигнала целиком (wdf — пустой DataFrame без единой строки), все
    оконные метрики зоны пропадали молча. band_frac_lowmid — метрика,
    которая существует ТОЛЬКО в window_metrics (track_avg даёт
    одноимённую, но с суффиксом _median, см. spectral.analyze_file) —
    чистый маркер того, что окна вообще прошли гейт."""
    quiet = _make_quiet_mix()
    path = tmp_path / "quiet_mix.wav"
    sf.write(str(path), quiet, SR, subtype="FLOAT")

    out = _measure(path, "mix")
    assert ("band_frac_lowmid", "mix") in out, (
        "ни одной оконной метрики после _measure на тихом сыром треке — "
        "похоже, энергогейт снова режет ненормализованный сигнал (regression)")


if __name__ == "__main__":
    import tempfile
    test_summarize_takes_median_across_tracks()
    with tempfile.TemporaryDirectory() as tmp:
        test_build_matrix_produces_rows_for_zone_relevant_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_build_matrix_skips_roles_without_zones(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_measure_normalizes_quiet_track_before_energy_gate(Path(tmp))
    print("Все тесты пламбинга build_interference_matrix (Блок 6) прошли.")
