"""Блок 4 (маскирование, ERB): регрессия на два бага код-ревью в
apply_temporal_masking:
1. Коэффициент 0.3 у обратного (backward) ядра временной маскировки
   применялся дважды подряд (0.3*0.3=0.09 вместо задуманных 0.3).
2. Куда серьёзнее: np.convolve(..., mode="same") центрирует свёртку по
   ГЕОМЕТРИЧЕСКОЙ середине асимметричного ядра (bwd_n обратных отсчётов +
   центр + fwd_n-1 прямых), а не по смысловому центру (индекс маскера) —
   при типичных forward_ms=200/backward_ms=20 весь temporal-masking
   отклик оказывался физически сдвинут на десятки кадров РАНЬШЕ реального
   маскера, а не после него, как задумано (forward-маскировка должна
   тянуться ПОСЛЕ маскера)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.metrics.masking_erb import apply_temporal_masking


def test_backward_kernel_peak_is_not_double_scaled():
    """hop_s=0.02 и backward_ms=20 (по умолчанию) дают bwd_n=1 — обратное
    ядро из ровно одного элемента, значение которого при построении
    должно остаться ровно 0.3 (после нормировки на пик=1.0 у центра),
    не 0.09."""
    hop_s = 0.02
    n_frames = 20
    impulse_frame = 10
    energy = np.zeros((1, n_frames))
    energy[0, impulse_frame] = 1.0

    out = apply_temporal_masking(energy, hop_s)
    before_impulse = out[0, impulse_frame - 1]
    assert abs(before_impulse - 0.3) < 0.02, (
        f"обратное ядро должно давать ~0.3 от пика непосредственно перед маскером, "
        f"получили {before_impulse:.3f} (0.09 означало бы старый баг двойного умножения на 0.3)")


def test_masker_peak_lands_on_its_own_frame_not_shifted_earlier():
    """Ключевая регрессия: отклик на импульс-маскер в кадре 10 обязан
    иметь свой пик РОВНО в кадре 10, не раньше. Старый баг (mode="same"
    на асимметричном ядре) сдвигал пик на много кадров назад — impulse на
    10 давал argmax на 6 при этих же параметрах."""
    hop_s = 0.02
    n_frames = 20
    impulse_frame = 10
    energy = np.zeros((1, n_frames))
    energy[0, impulse_frame] = 1.0

    out = apply_temporal_masking(energy, hop_s)
    assert np.argmax(out[0]) == impulse_frame, (
        f"пик отклика должен быть в кадре самого маскера ({impulse_frame}), "
        f"получили {np.argmax(out[0])} — отклик сдвинут во времени")
    assert out[0, impulse_frame] == 1.0
    # ничего до обратного окна (кадр 8 при bwd_n=1) — маскировка не может
    # тянуться в прошлое дальше собственного (короткого) обратного окна
    assert out[0, impulse_frame - 2] == 0.0


if __name__ == "__main__":
    test_backward_kernel_peak_is_not_double_scaled()
    test_masker_peak_lands_on_its_own_frame_not_shifted_earlier()
    print("Все тесты временной маскировки (Блок 4) прошли.")
