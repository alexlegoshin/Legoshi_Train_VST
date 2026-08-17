"""Синтетика для §4.9 chroma/тональность/аккорды/расстройка (задача #28)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.metrics.harmony_key import estimate_key, estimate_chord, vocal_bass_detuning, NOTE_NAMES

SR = 44100


def test_key_detection_recovers_root_and_mode():
    # C-мажорная гамма с акцентом на трезвучие C-E-G — должна вытянуть C major
    chroma = np.array([1.0, 0.1, 0.6, 0.1, 0.9, 0.5, 0.1, 0.95, 0.1, 0.4, 0.1, 0.5])
    root, mode, corr = estimate_key(chroma)
    assert root == "C" and mode == "major", f"получили {root} {mode} (corr={corr:.3f})"

    # тот же профиль, повёрнутый на 7 полутонов (G) — должен распознаться как G major
    chroma_g = np.roll(chroma, 7)
    root_g, mode_g, _ = estimate_key(chroma_g)
    assert root_g == "G" and mode_g == "major", f"получили {root_g} {mode_g}"


def test_key_detection_major_vs_minor():
    major_like = np.zeros(12); major_like[[0, 4, 7]] = 1.0  # C-E-G
    minor_like = np.zeros(12); minor_like[[0, 3, 7]] = 1.0  # C-Eb-G
    root_maj, mode_maj, _ = estimate_key(major_like)
    root_min, mode_min, _ = estimate_key(minor_like)
    assert root_maj == "C" and mode_maj == "major"
    assert root_min == "C" and mode_min == "minor"


def test_chord_triad_exact_match():
    c_major = np.zeros(12); c_major[[0, 4, 7]] = 1.0
    label, corr = estimate_chord(c_major)
    assert label == "C", f"получили {label}"
    assert corr > 0.99, f"точное совпадение с шаблоном должно дать corr~1, получили {corr}"

    c_minor = np.zeros(12); c_minor[[0, 3, 7]] = 1.0
    label_m, corr_m = estimate_chord(c_minor)
    assert label_m == "Cm", f"получили {label_m}"

    a_major = np.zeros(12); a_major[[9, 1, 4]] = 1.0  # A-C#-E
    label_a, _ = estimate_chord(a_major)
    assert label_a == "A", f"получили {label_a}"


def _harmonic_tone(f0, dur_s, sr=SR, amps=(1.0, 0.4, 0.2)):
    """Голая синусоида — патологический вход для pYIN (нет гармоник, не за
    что зацепиться): на ней даже точно синтезированные 2 октавы давали
    9-11 центов "расстройки" просто из-за шума трекера, не из-за бага
    расчёта. Реальный бас/вокал всегда богаче обертонами — воспроизводим
    это здесь, иначе тест мерит не то, что будет на реальных данных."""
    t = np.arange(int(dur_s * sr)) / sr
    x = np.zeros_like(t)
    for k, a in enumerate(amps, start=1):
        x += a * np.sin(2 * np.pi * k * f0 * t)
    return x * (0.4 / max(amps))


def test_vocal_bass_detuning_zero_when_exact_octaves():
    vocal = _harmonic_tone(220.0, 1.0)   # A3
    bass = _harmonic_tone(55.0, 1.0)     # A1, ровно на 2 октавы ниже
    summary, df = vocal_bass_detuning(vocal, bass, SR)
    assert summary["n_common_voiced_frames"] > 5
    assert summary["detuning_cents_median"] < 5, f"ожидали ~0 центов, получили {summary['detuning_cents_median']}"


def test_vocal_bass_detuning_detects_known_offset():
    vocal = _harmonic_tone(220.0, 1.0)
    # бас на 55*2^(30/1200) Гц — на 30 центов резче ровной октавы
    bass_f = 55.0 * (2 ** (30 / 1200))
    bass = _harmonic_tone(bass_f, 1.0)
    summary, df = vocal_bass_detuning(vocal, bass, SR)
    assert abs(summary["detuning_cents_median"] - 30) < 5, \
        f"ожидали ~30 центов, получили {summary['detuning_cents_median']}"
    assert summary["frac_gt_10cents"] > 0.9


def test_vocal_bass_detuning_ignores_perfect_fifth():
    """ТЗ-05 Блок В, баг 4: расстройка считалась по любым одновременно
    звучащим кадрам — терция/квинта (обычная гармония, не брак) засчитывались
    как расстройка (321 цент на реальном файле — треть, не порок). Фикс:
    сначала фильтр кадров-кандидатов на унисон/октаву (capture_cents=50),
    только по ним медиана. Чистая квинта (700 центов) далеко за пределами
    захвата — не должна попасть в оценку вовсе (n_common_voiced_frames~0)."""
    vocal = _harmonic_tone(220.0, 1.0)         # A3
    bass = _harmonic_tone(220.0 * (2 ** (-19 / 12)), 1.0)  # на квинту+октаву ниже (чистая квинта от ближайшей октавы)
    summary, df = vocal_bass_detuning(vocal, bass, SR)
    assert summary["n_common_voiced_frames"] > 5, "оба тона должны быть voiced — иначе тест ничего не проверяет"
    assert summary["n_candidate_unison_frames"] < 5, (
        f"квинта (~700 центов от ближайшей октавы) не должна пройти фильтр захвата "
        f"унисон/октава (capture_cents=50), получили n_candidate_unison_frames="
        f"{summary['n_candidate_unison_frames']}")
    assert np.isnan(summary["detuning_cents_median"]), (
        "при <5 кандидатов на унисон/октаву detuning_cents_median обязан быть NaN "
        "(не должен считаться по гармоническому интервалу вроде квинты)")


if __name__ == "__main__":
    test_key_detection_recovers_root_and_mode()
    test_key_detection_major_vs_minor()
    test_chord_triad_exact_match()
    test_vocal_bass_detuning_zero_when_exact_octaves()
    test_vocal_bass_detuning_detects_known_offset()
    test_vocal_bass_detuning_ignores_perfect_fifth()
    print("Все синтетические тесты §4.9 chroma/тональность/расстройка прошли.")
