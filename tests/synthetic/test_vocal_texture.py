"""Синтетика для §4.6 форманты/дыхания/сибилянты/согл.-гласн. (задача #26)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from analysis.metrics.vocal_texture import (
    lpc_formants, detect_breaths, detect_sibilants, analyze_file, _voiced_lookup, HOP, FRAME_LEN,
)

SR = 44100


def make_vowel(f0, formants_hz, dur_s, sr=SR, bw=80):
    """Импульсный источник (гармонический ряд) через ПАРАЛЛЕЛЬНЫЙ банк
    резонансных фильтров (сумма, не каскад) — так синтезируют форманты по
    классической модели: каждая формантная область — независимый резонанс
    голосового тракта, видимый в спектре одновременно с остальными.
    Каскад (последовательно) вместо этого перемножает АЧХ фильтров и почти
    полностью гасит дальние друг от друга резонансы — поймано именно на
    этом тесте, когда F1/F2 находились, а F3 систематически терялся."""
    n = int(dur_s * sr)
    period = int(round(sr / f0))
    source = np.zeros(n)
    source[::period] = 1.0
    x = np.zeros(n)
    for f in formants_hz:
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        b = [1 - r]
        a = [1, -2 * r * np.cos(theta), r ** 2]
        x += lfilter(b, a, source)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.5
    return x


def test_lpc_recovers_known_formants():
    x = make_vowel(120.0, [700, 1200, 2600], 0.5)
    frame = x[1000:1000 + FRAME_LEN]
    f1, f2, f3 = lpc_formants(frame, SR)
    assert abs(f1 - 700) < 50, f"F1={f1}, ожидали ~700"
    assert abs(f2 - 1200) < 80, f"F2={f2}, ожидали ~1200"
    assert abs(f3 - 2600) < 80, f"F3={f3}, ожидали ~2600"


def test_breath_detects_bandlimited_noise_not_pure_tone():
    rng = np.random.default_rng(0)
    dur = 1.0
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [2000, 8000], btype="bandpass", fs=SR, output="sos")
    breath_like = sosfilt(sos, noise)
    breath_like = breath_like / (np.max(np.abs(breath_like)) + 1e-9) * 0.3

    t_grid = np.arange(0, n - FRAME_LEN, HOP) / SR + (FRAME_LEN / 2) / SR
    voiced_all_false = np.zeros(len(t_grid), dtype=bool)

    events_breath = detect_breaths(breath_like, SR, t_grid, voiced_all_false)
    assert len(events_breath) > 0, "полосовой шум 2-8кГц должен дать хотя бы одно 'дыхание'"

    tone = 0.3 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
    events_tone = detect_breaths(tone, SR, t_grid, voiced_all_false)
    assert len(events_tone) == 0, f"чистый тон 1кГц не должен давать 'дыхание', получили {len(events_tone)}"

    silence = np.zeros(n)
    events_silence = detect_breaths(silence, SR, t_grid, voiced_all_false)
    assert len(events_silence) == 0, "тишина не должна давать 'дыхание'"


def test_sibilant_detects_high_freq_burst():
    rng = np.random.default_rng(1)
    dur = 2.0
    n = int(dur * SR)
    x = np.zeros(n)
    # фон — тихий широкополосный шум низкого уровня
    x += 0.01 * rng.standard_normal(n)
    # всплеск 5-10кГц в середине, ~80мс
    from scipy.signal import butter, sosfilt
    burst_n = int(0.08 * SR)
    burst = rng.standard_normal(burst_n)
    sos = butter(4, [5500, 9000], btype="bandpass", fs=SR, output="sos")
    burst = sosfilt(sos, burst)
    burst = burst / (np.max(np.abs(burst)) + 1e-9) * 0.4
    start = n // 2
    x[start:start + burst_n] += burst

    t_grid = np.arange(0, n - FRAME_LEN, HOP) / SR + (FRAME_LEN / 2) / SR
    voiced_all_false = np.zeros(len(t_grid), dtype=bool)
    events = detect_sibilants(x, SR, t_grid, voiced_all_false)
    assert len(events) > 0, "всплеск 5-10кГц должен дать хотя бы одно событие сибилянта"
    burst_t = start / SR
    assert any(abs(e["t_s"] - burst_t) < 0.15 for e in events), \
        f"событие должно попасть рядом с {burst_t:.2f}с, получили {[e['t_s'] for e in events]}"


def test_voiced_lookup_matches_by_time_not_index():
    # намеренно разные сетки времени (разный шаг) — merge_asof должен
    # всё равно сопоставить по ближайшему времени, а не съехать по индексу
    f0_df = pd.DataFrame({
        "t_s": np.arange(0, 2.0, 0.010),
        "voiced": np.arange(0, 2.0, 0.010) > 1.0,
        "f0_hz": 150.0,
    })
    t_query = np.arange(0.005, 1.995, 0.0116)  # другой шаг, сдвинутая фаза
    voiced = _voiced_lookup(t_query, f0_df)
    # для запросов после 1.0с должно быть voiced=True, до — False
    before = voiced[t_query < 0.95]
    after = voiced[t_query > 1.05]
    assert not before.any(), "до 1.0с в тестовых данных всё безголосое"
    assert after.all(), "после 1.0с в тестовых данных всё голосовое"


def test_consonant_vowel_ratio_via_analyze_file(tmp_path):
    import soundfile as sf
    dur = 2.0
    n = int(dur * SR)
    x = np.zeros(n)
    half = n // 2
    # первая половина — гласная (тон+гармоники), вторая — шумная "согласная" громче
    vowel = make_vowel(150.0, [700, 1200, 2500], dur / 2)
    x[:half] = vowel[:half]
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(n - half) * 0.6
    x[half:] = noise

    path = tmp_path / "synthetic_vowel_noise.wav"
    sf.write(str(path), np.column_stack([x, x]), SR)

    t_grid = np.arange(0, n - FRAME_LEN, HOP) / SR + (FRAME_LEN / 2) / SR
    voiced = t_grid < (half / SR)
    f0_df = pd.DataFrame({"t_s": t_grid, "voiced": voiced, "f0_hz": np.where(voiced, 150.0, np.nan)})

    summary, frames = analyze_file(path, f0_df)
    assert summary["consonant_vowel_energy_ratio"] > 1.0, \
        f"шумная половина громче — отношение согл/гласн должно быть >1, получили {summary['consonant_vowel_energy_ratio']}"


if __name__ == "__main__":
    test_lpc_recovers_known_formants()
    test_breath_detects_bandlimited_noise_not_pure_tone()
    test_sibilant_detects_high_freq_burst()
    test_voiced_lookup_matches_by_time_not_index()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_consonant_vowel_ratio_via_analyze_file(Path(d))
    print("Все синтетические тесты §4.6 форманты/дыхания/сибилянты прошли.")
