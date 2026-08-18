"""Блок 6: проверка ПЛАМБИНГА build_interference_matrix.py на маленькой
синтетике — не гоняет настоящий корпус (это отдельный ручной запуск,
roadmap.md, Блок 6), только что baseline/after/дельта считаются
правильно и агрегация по медиане корректна."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

from analysis.build_interference_matrix import build_matrix, summarize
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


if __name__ == "__main__":
    import tempfile
    test_summarize_takes_median_across_tracks()
    with tempfile.TemporaryDirectory() as tmp:
        test_build_matrix_produces_rows_for_zone_relevant_metrics(Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_build_matrix_skips_roles_without_zones(Path(tmp))
    print("Все тесты пламбинга build_interference_matrix (Блок 6) прошли.")
