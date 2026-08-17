"""Шаг 11c: батарея метрик на ИЗОЛИРОВАННОМ Demucs-вокале, отдельно по
каждой версии инженера сведения + демке + референс А + референс Б (предложение Опуса,
проверка гипотезы автора «окраску давал вокал», а не громкость/яркость
всего микса — см. отчёт по Шагу 11).

Раньше (run_tz03_moving_window.py) для КП вокальные F0-метрики брались с
СЫРОГО трек-аута — одна и та же запись во всех версиях инженера сведения, поэтому
не могла объяснить разницу МЕЖДУ версиями. Демucs здесь разделяет каждую
версию ОТДЕЛЬНО — значит вокал уже несёт обработку конкретной версии
(компрессия, EQ, реверб, сатурация), и это именно то, что менял инженер сведения.

Оговорки Опуса, обе учтены:
  - хвост реверба на разделённом вокале не мерить: гейт по RMS-энергии
    окна (голос должен реально звучать, не затухающий хвост) — тот же
    принцип, что ENERGY_GATE_DBFS в run_tz03_moving_window.py.
  - выше 10кГц по разделённому вокалу не мерить: Demucs оставляет
    широкополосный артефактный шум на разделении, особенно на верхах —
    сигнал НЧ-фильтруется (Butterworth, 10кГц, zero-phase) перед любым
    спектральным расчётом, band_frac_air/sibilance не считаются вовсе
    (они целиком выше или на границе фильтра).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt

from analysis.metrics.psychoacoustic import warmth_ratio as warmth_ratio_fn, full_timevarying
from analysis.metrics.spectral import compute_stft, ltas, spectral_slope as spectral_slope_fn, \
    spectral_moments, named_band_energy_fraction

ROOT = Path(__file__).resolve().parents[4]
SEP_DIR = ROOT / "_analysis" / "separated" / "htdemucs_ft"
OUT_DIR = ROOT / "_analysis" / "diff"

WIN_S = 4.0
HOP_S = 1.0
ENERGY_GATE_DBFS = -45.0
LOWPASS_HZ = 10000.0

# демо+инженер сведения: собственный Demucs-прогон этого сеанса. референс А/референс Б: уже
# были разделены раньше (ТЗ-02 Задача 3), пути другие — переиспользуем.
VOCAL_PATHS = {
    "demo": SEP_DIR / "демка_аранж_основной_трек" / "vocals.wav",
    "v2": SEP_DIR / "версия сведения 2" / "vocals.wav",
    "v3": SEP_DIR / "версия сведения 3." / "vocals.wav",
    "v4": SEP_DIR / "версия сведения 4 " / "vocals.wav",
    "v5": SEP_DIR / "версия сведения 5" / "vocals.wav",
    "v6": SEP_DIR / "версия сведения 6" / "vocals.wav",
    "v7": SEP_DIR / "версия сведения 7" / "vocals.wav",
    "референс А": ROOT / "референс А" / "demucs_stems" / "vocals.wav",
    "референс Б": SEP_DIR / "референс Б - 27:4:2026, 18.58" / "vocals.wav",
    "внешний трек": SEP_DIR / "ЧёЗаУродыНаСцене - внешний трек" / "vocals.wav",
}
# референс А/внешний трек посчитаны 2026-08-17 (первый прогон, подтвердил гипотезу —
# см. lamp-dictionary.md). Здесь — полный бэкафилл на всё остальное,
# кроме demo (вокала там физически нет, см. докстринг выше). roughness_dw
# медленный (~10-15 мин/трек) — это самый долгий шаг во всём пайплайне.
REAL_PSYCHO_VERSIONS = {"v2", "v3", "v4", "v5", "v6", "v7", "референс А", "референс Б", "внешний трек"}

# путь к ЦЕЛОМУ МИКСУ (не вокалу) — нужен только чтобы взять его integrated_lufs
# и выровнять ОБЩУЮ громкость перед энергетическим гейтом. Демка немастерингована
# (-22 LUFS) против -9..-10 у инженера сведения — без этого гейт по абсолютному dBFS
# срезает почти всю демку (проверено: 4 окна из ~190 прошли без поправки).
# Метрики (band_frac/slope/skewness/warmth) от гейна не зависят, это только
# для честного порога энергии.
MIX_PATH_FOR_LUFS = {
    "demo": "основной трек/ТА/основной трек track out/демка_аранж_основной_трек.wav",
    "v2": "основной трек/-/версия сведения 2.wav",
    "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav",
    "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav",
    "v7": "основной трек/-/версия сведения 7.wav",
    "референс А": "референс А/+/1 референс А.mp3",
    "референс Б": "Песня в поддержку рака лёгких/+ но это моя грязная демка/референс Б - 27:4:2026, 18.58.wav",
    "внешний трек": "ЧёЗаУродыНаСцене - внешний трек.mp3",
}
TARGET_LUFS = -18.0


def lowpass(x, sr, cutoff=LOWPASS_HZ, order=4):
    b, a = butter(order, cutoff / (sr / 2), btype="low")
    return filtfilt(b, a, x)


def analyze_vocal_windows(mono, sr, psycho_frames=None):
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
        # LTAS/slope тоже только до LOWPASS_HZ — выше него сигнал уже обнулён
        # фильтром, но centers может доходить до sr/2
        keep = centers <= LOWPASS_HZ
        slope = spectral_slope_fn(centers[keep], levels[keep])
        centroid, spread, sk, ku = spectral_moments(mag, f)
        bands = named_band_energy_fraction(mag, f)
        warmth = warmth_ratio_fn(seg, sr)

        row = dict(
            t_start=t0, t_end=t1, rms_dbfs=rms_dbfs,
            band_frac_lowmid=float(np.median(bands["lowmid"])),
            band_frac_low=float(np.median(bands["low"])),
            band_frac_mud=float(np.median(bands["mud"])),
            band_frac_mid=float(np.median(bands["mid"])),
            band_frac_presence=float(np.median(bands["presence"])),
            skewness=float(np.median(sk)),
            spectral_slope=slope, warmth_ratio=warmth,
        )
        if psycho_frames is not None:
            for key, col in [("sharpness", "acum"), ("roughness", "asper"), ("loudness", "sone")]:
                fr = psycho_frames[key]
                sub = fr[(fr.t_s >= t0) & (fr.t_s < t1)][col]
                row[f"real_{key}"] = float(sub.median()) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    d1 = pd.read_parquet(ROOT / "_analysis" / "metrics" / "4_1_summary.parquet")

    all_rows = []
    for version, path in VOCAL_PATHS.items():
        if not path.exists():
            print(f"ПРОПУСК {version}: нет файла {path}")
            continue
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)

        mix_path = MIX_PATH_FOR_LUFS.get(version)
        mix_lufs = d1.loc[d1.path == mix_path, "integrated_lufs"]
        if len(mix_lufs):
            gain_db = TARGET_LUFS - float(mix_lufs.iloc[0])
            mono = mono * (10 ** (gain_db / 20))
        else:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: не нашёл integrated_lufs микса для {version}, гейн не применён")

        mono = lowpass(mono, sr)

        psycho_frames = None
        if version in REAL_PSYCHO_VERSIONS:
            print(f"  считаю настоящий sharpness/roughness (MoSQITo, медленно)...")
            _, psycho_frames = full_timevarying(mono, sr)

        wdf = analyze_vocal_windows(mono, sr, psycho_frames)
        wdf["version"] = version
        print(f"{version}: окон всего после гейта {len(wdf)}")
        all_rows.append(wdf)

    table = pd.concat(all_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "vocal_isolated_windows.parquet", index=False)
    table.to_csv(OUT_DIR / "vocal_isolated_windows.csv", index=False)
    print(f"\nСохранено: {len(table)} окон -> {OUT_DIR / 'vocal_isolated_windows.parquet'}")


if __name__ == "__main__":
    main()
