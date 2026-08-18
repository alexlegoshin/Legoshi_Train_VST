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

    out_path, excluded, layering_pairs = orchestrate.align_and_sum_tracks([p], ref, SR, tmp_path, "vocals")
    assert layering_pairs == []  # один дубль в роли — сравнивать не с чем
    assert not excluded, f"выровненная дорожка ошибочно исключена: {excluded}"

    result, sr_out = sf.read(str(out_path))
    # после коррекции результат должен быть похож на ref, не на исходно
    # сдвинутую копию — проверяем корреляцию на общем участке
    m = min(len(result), len(ref))
    corr = np.corrcoef(result[:m], ref[:m])[0, 1]
    assert corr > 0.9, f"после выравнивания корреляция с референсом низкая: {corr:.3f}"


def _make_vocal_like(seconds, sr, seed=0):
    """Вокалоподобный сигнал (вибрато+гармоники, как в test_layering.py) +
    широкополосный всплеск — нужен и для уверенного GCC-PHAT (чистый тон
    даёт периодическую неоднозначность сдвига, задача #4), и для того,
    чтобы pYIN внутри layering.analyze_pair нашёл voiced-кадры."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0_curve = 250 + 15 * np.sin(2 * np.pi * 4 * t)
    phase = 2 * np.pi * np.cumsum(f0_curve) / SR
    voice = 0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    burst = 0.3 * rng.standard_normal(n) * np.exp(-((t - 4) ** 2) / 2)
    return voice + burst


def test_two_dubs_produce_layering_measurement(tmp_path):
    """Блок 3: два дубля одной роли — до суммирования должно появиться
    попарное измерение наложения (roadmap.md, «наложение дублей», код
    layering.py впервые реально подключён и вызван из пайплайна)."""
    ref = _make_vocal_like(8.0, SR, seed=3)
    shift_a, shift_b = int(0.05 * SR), int(0.09 * SR)
    dub_a = np.concatenate([np.zeros(shift_a), ref])[:len(ref)]
    dub_b = np.concatenate([np.zeros(shift_b), ref])[:len(ref)]

    p_a, p_b = tmp_path / "vocal_double_1.wav", tmp_path / "vocal_double_2.wav"
    sf.write(str(p_a), dub_a, SR, subtype="FLOAT")
    sf.write(str(p_b), dub_b, SR, subtype="FLOAT")

    out_path, excluded, layering_pairs = orchestrate.align_and_sum_tracks(
        [p_a, p_b], ref, SR, tmp_path, "vocals")
    assert not excluded, f"оба дубля структурные и коррелируют с ref, не должны быть исключены: {excluded}"
    assert len(layering_pairs) == 1  # 2 дубля -> ровно одна пара

    lp = layering_pairs[0]
    assert "error" not in lp, lp
    assert set(lp["pair"]) == {"vocal_double_1.wav", "vocal_double_2.wav"}
    # один и тот же исходник, только сдвинутый — питч не должен разъехаться
    assert lp["pitch_divergence_cents_median"] < 20.0, lp


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
