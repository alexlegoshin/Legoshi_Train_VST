"""ТЗ-05 А6: папка со смесью частот дискретизации/битности не должна
падать — все analyze_file() в analysis/metrics/*.py жёстко ждут 44100Гц
(assert), любой вход приводится к этому значению до анализа."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import soundfile as sf

import orchestrate
from analysis import engine


def _tone(seconds, sr, freq=220.0):
    t = np.arange(int(seconds * sr)) / sr
    return np.stack([np.sin(2 * np.pi * freq * t)] * 2, axis=1)


def test_ensure_sr_resamples_only_when_needed(tmp_path):
    p44 = tmp_path / "a.wav"
    sf.write(str(p44), _tone(1.0, 44100), 44100, subtype="PCM_16")
    same = engine.ensure_sr(p44, 44100, tmp_path)
    assert same == p44, "не должен пересоздавать файл, если sr уже совпадает"

    p48 = tmp_path / "b.wav"
    sf.write(str(p48), _tone(1.0, 48000), 48000, subtype="PCM_24")
    resampled = engine.ensure_sr(p48, 44100, tmp_path)
    assert resampled != p48
    info = sf.info(str(resampled))
    assert info.samplerate == 44100


def test_mixed_sr_bitdepth_trackout_does_not_crash(tmp_path):
    folder = tmp_path / "trackout"
    folder.mkdir()
    sf.write(str(folder / "main.wav"), _tone(6.0, 44100), 44100, subtype="PCM_16")
    sf.write(str(folder / "vocal.wav"), _tone(6.0, 48000), 48000, subtype="PCM_24")
    sf.write(str(folder / "bass.wav"), _tone(6.0, 44100), 44100, subtype="PCM_16")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    main_path, stems, excluded, input_formats = orchestrate.classify_trackout(folder, work_dir)

    assert sf.info(str(main_path)).samplerate == 44100
    for p in stems.values():
        assert sf.info(str(p)).samplerate == 44100, "источник после classify_trackout обязан быть на 44100"
    assert input_formats["vocal.wav"]["samplerate"] == 48000, "исходный sr должен быть сохранён для отчёта"
