"""Шаг 11g: та же батарея, что на вокале (§11c), теперь на инструментальных
стемах Demucs — bass/drums/other, по каждой версии отдельно (включая
демку — в отличие от вокала, у демки инструментал полноценный, не тишина,
см. лог этого скрипта). Мотивация автора: он оценивал ОБЩИЙ микс, не
вокал отдельно — если модель раскладывается на вклад вокала и вклад
инструментала, вклад инструментала должен быть виден отдельно, и не
обязательно на тех же осях, где сработал вокал.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt

from analysis.metrics.psychoacoustic import warmth_ratio as warmth_ratio_fn, harshness as harshness_fn
from analysis.metrics.spectral import compute_stft, ltas, spectral_slope as spectral_slope_fn, \
    spectral_moments, named_band_energy_fraction

ROOT = Path(__file__).resolve().parents[4]
SEP_DIR = ROOT / "_analysis" / "separated" / "htdemucs_ft"
OUT_DIR = ROOT / "_analysis" / "diff"

WIN_S = 4.0
HOP_S = 1.0
ENERGY_GATE_DBFS = -45.0
LOWPASS_HZ = 10000.0
STEMS = ["bass", "drums", "other"]

STEM_DIRS = {
    "demo": "демка_аранж_основной_трек", "v2": "версия сведения 2", "v3": "версия сведения 3.",
    "v4": "версия сведения 4 ", "v5": "версия сведения 5", "v6": "версия сведения 6", "v7": "версия сведения 7",
    "референс Б": "референс Б - 27:4:2026, 18.58",
}
MIX_PATH_FOR_LUFS = {
    "demo": "основной трек/ТА/основной трек track out/демка_аранж_основной_трек.wav",
    "v2": "основной трек/-/версия сведения 2.wav", "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav", "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav", "v7": "основной трек/-/версия сведения 7.wav",
    "референс Б": "Песня в поддержку рака лёгких/+ но это моя грязная демка/референс Б - 27:4:2026, 18.58.wav",
}
TARGET_LUFS = -18.0
# референс А: свой Demucs-прогон из более раннего шага, другой путь/имена стемов
RADOST_STEMS = {"bass": ROOT / "референс А" / "demucs_stems" / "bass.wav",
                 "drums": ROOT / "референс А" / "demucs_stems" / "drums.wav",
                 "other": ROOT / "референс А" / "demucs_stems" / "other.wav"}
RADOST_MIX_PATH = "референс А/+/1 референс А.mp3"


def lowpass(x, sr, cutoff=LOWPASS_HZ, order=4):
    b, a = butter(order, cutoff / (sr / 2), btype="low")
    return filtfilt(b, a, x)


def analyze_windows(mono, sr):
    n = len(mono)
    win_n, hop_n = int(WIN_S * sr), int(HOP_S * sr)
    starts = np.arange(0, max(n - win_n, 0), hop_n)
    rows = []
    for s in starts:
        t0, t1 = s / sr, (s + win_n) / sr
        seg = mono[s:s + win_n]
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        if rms_dbfs < ENERGY_GATE_DBFS:
            continue
        f, t, mag = compute_stft(seg, sr)
        centers, levels = ltas(mag, f, bands_per_octave=3)
        keep = centers <= LOWPASS_HZ
        slope = spectral_slope_fn(centers[keep], levels[keep])
        centroid, spread, sk, ku = spectral_moments(mag, f)
        bands = named_band_energy_fraction(mag, f)
        warmth = warmth_ratio_fn(seg, sr)
        harsh = harshness_fn(seg, sr)
        rows.append(dict(
            t_start=t0, t_end=t1, rms_dbfs=rms_dbfs,
            band_frac_lowmid=float(np.median(bands["lowmid"])), band_frac_low=float(np.median(bands["low"])),
            band_frac_mud=float(np.median(bands["mud"])), band_frac_mid=float(np.median(bands["mid"])),
            band_frac_presence=float(np.median(bands["presence"])), skewness=float(np.median(sk)),
            spectral_slope=slope, warmth_ratio=warmth, harshness=harsh,
        ))
    return pd.DataFrame(rows)


def main():
    d1 = pd.read_parquet(ROOT / "_analysis" / "metrics" / "4_1_summary.parquet")
    all_rows = []

    for version, dirname in STEM_DIRS.items():
        mix_lufs_row = d1.loc[d1.path == MIX_PATH_FOR_LUFS[version], "integrated_lufs"]
        gain_db = TARGET_LUFS - float(mix_lufs_row.iloc[0]) if len(mix_lufs_row) else 0.0
        for stem in STEMS:
            path = SEP_DIR / dirname / f"{stem}.wav"
            if not path.exists():
                print(f"ПРОПУСК {version}/{stem}: нет {path}")
                continue
            data, sr = sf.read(str(path), dtype="float64", always_2d=True)
            mono = data.mean(axis=1) * (10 ** (gain_db / 20))
            mono = lowpass(mono, sr)
            wdf = analyze_windows(mono, sr)
            wdf["version"], wdf["stem"] = version, stem
            print(f"{version}/{stem}: {len(wdf)} окон")
            all_rows.append(wdf)

    mix_lufs = float(d1.loc[d1.path == RADOST_MIX_PATH, "integrated_lufs"].iloc[0])
    gain_db = TARGET_LUFS - mix_lufs
    for stem, path in RADOST_STEMS.items():
        if not path.exists():
            print(f"ПРОПУСК референс А/{stem}: нет {path}")
            continue
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        mono = data.mean(axis=1) * (10 ** (gain_db / 20))
        mono = lowpass(mono, sr)
        wdf = analyze_windows(mono, sr)
        wdf["version"], wdf["stem"] = "референс А", stem
        print(f"референс А/{stem}: {len(wdf)} окон")
        all_rows.append(wdf)

    table = pd.concat(all_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "instrumental_windows.parquet", index=False)
    table.to_csv(OUT_DIR / "instrumental_windows.csv", index=False)
    print(f"\nСохранено: {len(table)} окон -> {OUT_DIR / 'instrumental_windows.parquet'}")


if __name__ == "__main__":
    main()
