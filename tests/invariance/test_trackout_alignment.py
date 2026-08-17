"""ТЗ-05 А4: дорожка со сдвигом против главного микса должна быть
выровнена (сдвиг с точностью до нескольких мс), либо, если синхронизация
невозможна (некоррелированный сигнал), явно исключена, а не сложена
вслепую."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import soundfile as sf

import orchestrate

SR = 44100


def _make_signal(seconds=8.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    # широкополосный сигнал со структурой (не чистый шум) — ближе к
    # реальному аудио, GCC-PHAT должен находить сдвиг уверенно
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * 220 * t) + 0.5 * rng.standard_normal(n) * np.exp(-((t - 4) ** 2) / 2)
    return sig


def test_shifted_track_is_realigned(tmp_path):
    ref = _make_signal(seed=1)
    shift_samples = int(0.2 * SR)  # 200мс
    shifted = np.concatenate([np.zeros(shift_samples), ref])[:len(ref)]

    p = tmp_path / "vocal_double.wav"
    sf.write(str(p), shifted, SR, subtype="FLOAT")

    out_path, excluded = orchestrate.align_and_sum_tracks([p], ref, SR, tmp_path, "vocals")
    assert not excluded, f"выровненная дорожка ошибочно исключена: {excluded}"

    result, sr_out = sf.read(str(out_path))
    # после коррекции результат должен быть похож на ref, не на исходно
    # сдвинутую копию — проверяем корреляцию на общем участке
    m = min(len(result), len(ref))
    corr = np.corrcoef(result[:m], ref[:m])[0, 1]
    assert corr > 0.9, f"после выравнивания корреляция с референсом низкая: {corr:.3f}"


def test_uncorrelated_track_is_excluded(tmp_path):
    ref = _make_signal(seed=2)
    unrelated = _make_signal(seed=99)  # другой сигнал, синхронизировать нечего

    p = tmp_path / "bass.wav"
    sf.write(str(p), unrelated, SR, subtype="FLOAT")

    with pytest.raises(ValueError):
        # единственная дорожка роли, и она не синхронизируется -> список
        # выровненных пуст -> функция обязана упасть с понятной ошибкой,
        # не молча сложить мусор
        orchestrate.align_and_sum_tracks([p], ref, SR, tmp_path, "bass")
