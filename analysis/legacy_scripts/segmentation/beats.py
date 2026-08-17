"""Тактовая сетка, §3.4 п.3. Сознательно сохраняем РЕАЛЬНЫЕ тайминги найденных
битов, а не константную BPM-сетку, экстраполированную от нуля: на паузе
(2:13) и на подъёме перед вторым припевом (0:83) живое исполнение уходит от
номинального темпа на 60-150мс, и константная сетка накопила бы эту ошибку,
а не учла её."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]

NOMINAL_BPM = {"основной трек": 72.0, "контрольный трек": 73.0}
REF_MIX = {
    "основной трек": "основной трек/-/Итог сведения.wav",
    "контрольный трек": "контрольный трек/ТА/финальная/фин.mp3",
}


def build_beats(song, path):
    y, sr = sf.read(str(ROOT / path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=NOMINAL_BPM[song], tightness=100)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    local_bpm = np.concatenate([[np.nan], 60 / np.diff(beat_times)])

    df = pd.DataFrame(dict(
        song=song, beat_index=np.arange(len(beat_times)),
        time_s=beat_times, local_bpm=local_bpm,
        bar_index=np.arange(len(beat_times)) // 4,  # 4/4, как в ТЗ инженеру сведения
    ))
    global_bpm = float(np.atleast_1d(tempo)[0])
    print(f"{song}: {len(beat_times)} битов, глобальный tempo={global_bpm:.2f} "
          f"(номинал {NOMINAL_BPM[song]}), std локального bpm={np.nanstd(local_bpm):.2f}")
    return df


if __name__ == "__main__":
    frames = [build_beats(song, path) for song, path in REF_MIX.items()]
    out = pd.concat(frames, ignore_index=True)
    dest = ROOT / "_analysis" / "beats.parquet"
    out.to_parquet(dest, index=False)
    print(f"beats.parquet -> {dest}, {len(out)} строк")
