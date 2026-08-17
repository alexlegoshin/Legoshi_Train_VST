"""Регрессия на два бага refine_shift, пойманных на реальном корпусе, не на
синтетике — но синтетика должна ловить их впредь.
1) сырой dot product без нормировки уезжает туда, где громче;
2) окно сравнения от начала файла проваливается, если там тишина
   (дорожка вступает не с первой секунды — типично для баса/соло)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from analysis.legacy_scripts.align.run_alignment import refine_shift

SR = 44100


def test_refine_not_biased_by_loudness_envelope():
    """Истинный сдвиг с точным ответом внутри окна поиска, но с сигналом,
    чья громкость сильно растёт к концу трека — раньше raw dot product
    утаскивало ответ к громкому краю, а не к истинному совпадению."""
    rng = np.random.default_rng(0)
    dur_s = 20.0
    n = int(dur_s * SR)
    base = rng.standard_normal(n).astype(np.float32)
    ramp = np.linspace(0.1, 3.0, n).astype(np.float32)  # громкость растёт в 30 раз
    ref = base * ramp

    true_shift = 137  # сэмплов, намеренно внутри узкого окна поиска
    sig = np.zeros_like(ref)
    sig[true_shift:] = ref[:len(ref) - true_shift]

    best, score = refine_shift(sig, ref, SR, coarse_shift_samples=0, window_s=0.01)
    print(f"[loudness-ramp] true={true_shift} found={best} score={score:.3f}")
    assert best == true_shift, f"результат утащило громкостью: {best} != {true_shift}"


def test_refine_handles_silent_lead_in():
    """Дорожка (как бас в реальном корпусе) молчит первые несколько секунд —
    окно сравнения от нулевой позиции даст норму 0 и молча провалится."""
    rng = np.random.default_rng(1)
    dur_s = 20.0
    n = int(dur_s * SR)
    ref = np.zeros(n, dtype=np.float32)
    ref[int(5 * SR):] = rng.standard_normal(n - int(5 * SR)).astype(np.float32)  # вступает на 5-й секунде

    true_shift = -213
    sig = np.zeros_like(ref)
    if true_shift >= 0:
        sig[true_shift:] = ref[:len(ref) - true_shift]
    else:
        sig[:len(sig) + true_shift] = ref[-true_shift:]

    best, score = refine_shift(sig, ref, SR, coarse_shift_samples=0, window_s=0.01)
    print(f"[silent-lead-in] true={true_shift} found={best} score={score}")
    assert score is not None and np.isfinite(score) and score > -np.inf, \
        "уточнение молча провалилось (норма окна сравнения была 0)"
    assert best == true_shift, f"{best} != {true_shift}"


if __name__ == "__main__":
    test_refine_not_biased_by_loudness_envelope()
    test_refine_handles_silent_lead_in()
    print("ALL REFINE_SHIFT REGRESSION TESTS PASSED")
