"""ТЗ-05 Б8: каждая (metric, source) в пресете обязана реально существовать
на выходе orchestrate.analyze_all_sources() — том же коде, что реально
собирает measurements.json. Иначе зона молча всегда NO_DATA, что и
случилось с formant_f3_hz (engine.formant_series() был написан, но
orchestrate.py его никогда не вызывал — зона была мертва с самого
создания пресета, пока этот тест не поймал несоответствие проверкой
через настоящий production-путь, а не переизобретёнными вызовами engine.py
напрямую). Тест намеренно "ломается" при добавлении новой зоны без
соответствующего кода в orchestrate.py — это его единственная задача, не
измерение качества метрик."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import soundfile as sf

import orchestrate
from analysis.verdict import load_preset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    # локальный, не в git — см. tests/local_paths.py
    from local_paths import KEY_COVERAGE_MIX as MIX_PATH, KEY_COVERAGE_STEMS as STEM_PATHS
except ImportError:
    # плейсхолдеры, не существуют — тест skip'ается
    MIX_PATH = ROOT / "референс А" / "+" / "1 референс А.mp3"
    STEM_PATHS = {
        "vocals": ROOT / "референс А" / "demucs_stems" / "vocals.wav",
        "bass": ROOT / "референс А" / "demucs_stems" / "bass.wav",
        "drums": ROOT / "референс А" / "demucs_stems" / "drums.wav",
        "other": ROOT / "референс А" / "demucs_stems" / "other.wav",
    }
SLICE_START_S = 75.0  # у "референс А" на 75-125с все 4 стема (в т.ч. тихие
                       # ударные) выше энергогейта ENERGY_GATE_DBFS=-45 —
                       # подобрано эмпирически, начало трека заведомо ниже
                       # гейта, окна отфильтруются в ноль и тест даст
                       # ложное "ключа нет"
SLICE_S = 50.0  # достаточно для нескольких окон 4с после гейта, быстро на MoSQITo


@pytest.fixture(scope="module")
def slices(tmp_path_factory):
    missing = [p for p in [MIX_PATH, *STEM_PATHS.values()] if not p.exists()]
    if missing:
        pytest.skip(f"тестовый корпус недоступен в этом окружении: {missing}")
    tmp = tmp_path_factory.mktemp("preset_key_coverage")
    out = {}
    # ВАЖНО: mix НЕ режется — mix_gain_db считается из целого mix
    # (integrated LUFS) и накладывается на стемы; обрезок в 50с даёт
    # другой LUFS, чем целый трек, и это меняет, какие окна стемов
    # проходят энергогейт (поймано эмпирически при отладке этого теста —
    # обрезанный mix давал -6.7дБ вместо -3.9дБ на целом файле, разницы
    # хватало, чтобы все окна drums ушли ниже гейта).
    out["mix"] = MIX_PATH
    for role, path in STEM_PATHS.items():
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        start = int(SLICE_START_S * sr)
        end = start + int(SLICE_S * sr)
        if end > len(data):
            start, end = 0, min(len(data), int(SLICE_S * sr))
        dst = tmp / f"{role}.wav"
        sf.write(str(dst), data[start:end], sr, subtype="FLOAT")
        out[role] = dst
    return out


@pytest.fixture(scope="module")
def measurement_keys(slices):
    """Настоящий production-путь: orchestrate.analyze_all_sources(), как
    его реально вызывает process_item() в режиме трек-аута (is_ml_separated
    выставлен так же, как для Demucs-стемов — этот корпус ими и является)."""
    stems = {r: p for r, p in slices.items() if r != "mix"}
    measurements, _ = orchestrate.analyze_all_sources(
        slices["mix"], stems, deep_psychoacoustics=True, is_ml_separated=True)
    return set(measurements.keys())  # {(metric, role), ...}


@pytest.mark.parametrize("preset_name", ["legoshi_amber"])
def test_every_zone_key_exists_in_orchestrator_output(preset_name, measurement_keys):
    zones = load_preset(preset_name)
    missing = [f"{z.source}/{z.metric} ({z.granularity})" for z in zones
               if (z.metric, z.source) not in measurement_keys]
    assert not missing, (
        "зоны пресета без соответствия в выходе analyze_all_sources() — "
        "зона всегда будет NO_DATA молча:\n" + "\n".join(missing))
