"""ТЗ-05 Блок В: регрессионные тесты на баги, пойманные в исследовании,
для тех четырёх из семи, что ещё не были явно зафиксированы тестом
(остальные три уже покрыты: LPC-форманты — test_vocal_texture.py, детектор
наводок — test_noise_persistent.py, расстройка вокал-бас на унисоне —
test_harmony_key.py, туда же добавлена квинта ниже)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.signal import butter, sosfilt

from analysis.metrics import harmony_key, masking_erb, pitch_vocal
from analysis.metrics.vocal_texture import frame_flatness

SR = 44100


def test_frame_flatness_high_within_band_low_over_full_spectrum():
    """Баг 2: flatness как прокси тональности считалась по всему спектру и
    обнулялась на полосовом сигнале — за пределами полосы амплитуда около
    нуля, тащит geometric mean к нулю, топит реальную шумность внутри
    полосы. Полосовой шум 2-8кГц ВНУТРИ полосы почти белый (flatness~1),
    но на фоне всего 0-22кГц спектра выглядит почти чистым тоном (flatness~0)."""
    rng = np.random.default_rng(0)
    n = 1024
    sos = butter(4, [2000, 8000], btype="bandpass", fs=SR, output="sos")
    noise = sosfilt(sos, rng.standard_normal(n * 20))[-n:]  # прогрев фильтра
    spec = np.abs(np.fft.rfft(noise * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / SR)

    flat_full = frame_flatness(spec)
    flat_band = frame_flatness(spec, freqs, 2000, 8000)
    assert flat_full < 0.15, f"по всему спектру должно ложно проваливаться к 0, получили {flat_full}"
    assert flat_band > 0.6, f"внутри полосы шум должен быть близок к белому (flatness~1), получили {flat_band}"


def test_pyin_does_not_lock_to_fmin_on_long_file():
    """Баг 3: pYIN на длинном файле (~200с) залипал на нижней границе
    диапазона (66Гц вместо настоящих ~250Гц) при неудачном сочетании
    frame_length/resolution. pitch_vocal.extract_f0 использует
    frame_length=2048 (безопасная конфигурация, см. docstring
    harmony_key.vocal_bass_detuning про тот же баг на frame_length=4096) —
    тест фиксирует, что она остаётся такой."""
    dur = 200.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    # вокал с вибрато и гармониками, не голая синусоида — ближе к реальному входу
    f0_curve = 250 + 30 * np.sin(2 * np.pi * 4 * t) + 20 * np.sin(2 * np.pi * 0.1 * t)
    phase = 2 * np.pi * np.cumsum(f0_curve) / SR
    x = 0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)

    t0 = time.time()
    _, f0, _, _ = pitch_vocal.extract_f0(x, SR)
    elapsed = time.time() - t0
    med = np.nanmedian(f0)
    assert 200 <= med <= 300, (
        f"медиана F0={med:.1f}Гц вне ожидаемого 200-300Гц — похоже на залипание "
        f"на fmin=65Гц или другой сбой Viterbi на длинном файле (заняло {elapsed:.1f}с)")


def test_masking_requires_prealigned_input():
    """Баг 5: analyze_group раньше сама грузила файлы и не применяла
    выравнивание — маскирование считалось по НЕ настоящей одновременности.
    Два идентичных сигнала (узкополосные пачки шума в одной ERB-полосе) со
    сдвигом 74мс (реальный офсет с "основной трек", см. docstring
    analyze_group) должны маскировать друг друга ХУЖЕ, чем те же сигналы,
    выровненные (нулевой сдвиг) — совпадающие во времени пачки маскируют
    почти полностью, разъехавшиеся — нет."""
    dur = 6.0
    n = int(dur * SR)
    sos = butter(4, [900, 1300], btype="bandpass", fs=SR, output="sos")

    def burst_train(phase_s, burst_s=0.05, period_s=1.5, seed=1):
        rng = np.random.default_rng(seed)
        noise = sosfilt(sos, rng.standard_normal(n))
        noise = noise / (np.max(np.abs(noise)) + 1e-9)
        t = np.arange(n) / SR
        gate = (((t - phase_s) % period_s) < burst_s).astype(float)
        return noise * gate

    a = burst_train(phase_s=0.0)
    b_misaligned = burst_train(phase_s=0.074)  # тот же генератор — идентичный сигнал, сдвинутый по фазе
    b_aligned = burst_train(phase_s=0.0)        # выровнено — совпадает с a

    audibility_misaligned = masking_erb.analyze_group({"a": a, "b": b_misaligned}, SR)["audibility"]["a"]
    audibility_aligned = masking_erb.analyze_group({"a": a, "b": b_aligned}, SR)["audibility"]["a"]

    assert audibility_misaligned > audibility_aligned * 3, (
        f"рассинхрон 74мс должен давать заметно БОЛЬШЕ слышимости (меньше маскирования), чем "
        f"выровненные сигналы: misaligned={audibility_misaligned:.5f}, aligned={audibility_aligned:.5f}")
    assert audibility_aligned < 0.01, (
        f"полностью выровненные идентичные пачки в одной полосе должны маскировать почти "
        f"полностью (audibility~0), получили {audibility_aligned}")


def test_key_correlation_drops_under_distortion_like_chroma():
    """Баг 7: тональность по полному миксу врала из-за дисторшна —
    estimate_key не даёт отдельного флага "ненадёжно", вместо этого сама
    key_correlation и есть сигнал надёжности (Krumhansl-Schmuckler
    корреляция закономерно падает на "грязном", размазанном по всем 12
    тонам chroma — дисторшн-гитара даёт именно такой профиль, не чистые
    трезвучия). Тест фиксирует, что деградация действительно происходит —
    вызывающий код должен трактовать низкую corr как "ненадёжно"."""
    chroma_clean = np.zeros(12)
    chroma_clean[[0, 4, 7]] = 1.0  # чистое C-мажорное трезвучие
    _, _, corr_clean = harmony_key.estimate_key(chroma_clean)

    rng = np.random.default_rng(0)
    chroma_distorted = rng.uniform(0.3, 1.0, 12)  # энергия размазана по всем 12 тонам — типично для дисторшна
    _, _, corr_distorted = harmony_key.estimate_key(chroma_distorted)

    assert corr_clean > 0.75, f"чистое трезвучие должно давать высокую corr, получили {corr_clean}"
    assert corr_distorted < corr_clean - 0.15, (
        f"дисторшн-подобный (размазанный) chroma должен давать заметно более низкую corr: "
        f"clean={corr_clean:.3f}, distorted={corr_distorted:.3f}")


if __name__ == "__main__":
    test_frame_flatness_high_within_band_low_over_full_spectrum()
    test_pyin_does_not_lock_to_fmin_on_long_file()
    test_masking_requires_prealigned_input()
    test_key_correlation_drops_under_distortion_like_chroma()
    print("Все регрессионные тесты Блока В (ТЗ-05) прошли.")
