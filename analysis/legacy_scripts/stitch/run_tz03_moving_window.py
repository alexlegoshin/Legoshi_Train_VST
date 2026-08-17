"""ТЗ-03 (правки Opus по итогам ревью посекционного анализа): скользящее
окно 4с / шаг 1с с энергетическим гейтом — замена 15 неравномерных
"нарративных" секций (огрызки по 1-6с давали абсурдную дисперсию оценки)
единым, статистически сравнимым набором наблюдений (~190-200 на файл).

Гейт по вокалу для F0-метрик (вибрато/интонация): окно НЕ считается
вокальным, если pYIN-voiced доля в окне < порога, ИЛИ медианная confidence
< порога, ИЛИ (для «основного трека», где есть стем) вокальный стем в этом
же окне сам не помечен как voiced. Без вокала — NaN, не число. Это прямая
починка бага, пойманного на «переход 77с»/«пауза»: F0-трекер на полном
миксе цеплялся за шум/хвост реверба и давал абсурдные 42-91 цент
"вибрато" там, где живого голоса нет вообще (независимо перепроверено
руками на реальных данных, не только со слов ревью).

warmth_ratio и spectral_slope считаются здесь заново по каждому окну
напрямую из аудио (обе функции тривиально применимы к любому отрезку,
не только к целому файлу — psychoacoustic.warmth_ratio и
spectral.spectral_slope уже это умеют, просто раньше вызывались только
на полном файле)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf

from analysis.metrics.psychoacoustic import warmth_ratio as warmth_ratio_fn
from analysis.metrics.spectral import compute_stft, ltas, spectral_slope as spectral_slope_fn, \
    spectral_moments, named_band_energy_fraction

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

WIN_S = 4.0
HOP_S = 1.0
ENERGY_GATE_DBFS = -45.0
VOICED_FRAC_GATE = 0.5
STEM_VOICED_FRAC_GATE = 0.6
MAX_PLAUSIBLE_VIBRATO_CENTS = 100.0  # человеческое вибрато сюда не доходит; больше = сбой трекера
# ПОЙМАНО НА РЕАЛЬНЫХ ДАННЫХ: pYIN "prob" на полном миксе не различает
# хороший вокальный участок (медиана prob=0.015) и явно безголосый переход
# (медиана prob=0.023, ВЫШЕ!) — как критерий гейта бесполезен, снят.
# Основной сигнал — voiced_frac стема (независимая проверка по изолированной
# записи, не подвержена полифонии), с более строгим порогом, чем у микса.

FILES = {
    "demo": "основной трек/ТА/основной трек track out/демка_аранж_основной_трек.wav",
    "v2": "основной трек/-/версия сведения 2.wav",
    "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav",
    "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav",
    "v7": "основной трек/-/версия сведения 7.wav",
}
STEM_SAFE_NAME = "основной трек__ТА__Track Out__вокал основной.wav"
STEM_OFFSET_SAMPLES = -94  # из alignment.parquet


def load_f0(safe_name):
    f = METRICS_DIR / f"{safe_name}.4_6_f0.parquet"
    return pd.read_parquet(f) if f.exists() else None


def load_notes(safe_name):
    f = METRICS_DIR / f"{safe_name}.4_6_notes.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    return df[df.type == "note"] if len(df) else df


def hz_to_cents_arr(f0):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1200 * np.log2(f0 / 440.0)


def analyze_file_windows(path, mono, sr, f0_df, stem_f0_df, meter, notes_df=None, mono_for_lufs=None):
    n = len(mono)
    win_n, hop_n = int(WIN_S * sr), int(HOP_S * sr)
    starts = np.arange(0, n - win_n, hop_n)
    rows = []
    for s in starts:
        t0, t1 = s / sr, (s + win_n) / sr
        seg = mono[s:s + win_n]
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        if rms_dbfs < ENERGY_GATE_DBFS:
            continue

        f, t, mag = compute_stft(seg, sr)
        centers, levels = ltas(mag, f, bands_per_octave=3)
        slope = spectral_slope_fn(centers, levels)
        centroid, spread, sk, ku = spectral_moments(mag, f)
        bands = named_band_energy_fraction(mag, f)
        warmth = warmth_ratio_fn(seg, sr)
        lufs_seg = mono_for_lufs[s:s + win_n] if mono_for_lufs is not None else seg
        lufs = float(meter.integrated_loudness(lufs_seg)) if np.any(np.abs(lufs_seg) > 1e-9) else np.nan

        row = dict(
            t_start=t0, t_end=t1,
            band_frac_air=float(np.median(bands["air"])), band_frac_lowmid=float(np.median(bands["lowmid"])),
            band_frac_low=float(np.median(bands["low"])), skewness=float(np.median(sk)),
            spectral_slope=slope, warmth_ratio=warmth, lufs_normalized=lufs,
        )

        # --- гейт по вокалу для F0-метрик ---
        vibrato_val, voiced_frac_val = np.nan, np.nan
        if f0_df is not None:
            fmask = (f0_df.t_s >= t0) & (f0_df.t_s < t1)
            fsub = f0_df[fmask]
            if len(fsub):
                voiced_frac = float(fsub.voiced.mean())
                stem_ok = True
                if stem_f0_df is not None:
                    smask = (stem_f0_df.t_mix >= t0) & (stem_f0_df.t_mix < t1)
                    ssub = stem_f0_df[smask]
                    stem_ok = len(ssub) > 0 and float(ssub.voiced.mean()) >= STEM_VOICED_FRAC_GATE
                if voiced_frac >= VOICED_FRAC_GATE and stem_ok:
                    voiced_frac_val = voiced_frac
                    # вибрато НЕ пересчитываем с нуля по сырым кадрам окна (окно может
                    # захватить переход между нотами — глиссандо/легато, которых в этой
                    # песне много по жалобам автора — и тогда реальное мелодическое
                    # движение посчиталось бы как "вибрато", раздувая цифру; поймано на
                    # реальных данных: наивный пересчёт давал 470-870 центов).
                    # Вместо этого берём уже провалидированные per-note значения
                    # (границы нот честные, см. pitch_vocal.segment_notes) и агрегируем
                    # медианой те ноты, чей центр попадает в окно.
                    if notes_df is not None and len(notes_df):
                        note_mid = (notes_df.t_start + notes_df.t_end) / 2
                        nmask = (note_mid >= t0) & (note_mid < t1)
                        nsub = notes_df.loc[nmask, "vibrato_depth_cents"].dropna()
                        nsub = nsub[nsub <= MAX_PLAUSIBLE_VIBRATO_CENTS]
                        if len(nsub) >= 2:
                            vibrato_val = float(nsub.median())
        row["voiced_fraction"] = voiced_frac_val
        row["vibrato_depth_cents"] = vibrato_val
        rows.append(row)
    return pd.DataFrame(rows)


TARGET_LUFS = -18.0


def main():
    meter = pyln.Meter(44100)
    stem_f0 = load_f0(STEM_SAFE_NAME)
    if stem_f0 is not None:
        stem_f0 = stem_f0.copy()
        stem_f0["t_mix"] = stem_f0["t_s"] - STEM_OFFSET_SAMPLES / 44100

    d1 = pd.read_parquet(METRICS_DIR / "4_1_summary.parquet")

    all_rows = []
    for version, relpath in FILES.items():
        print(f"=== {version} ===")
        data, sr = sf.read(str(ROOT / relpath), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)
        safe_name = relpath.replace("/", "__")

        # приведение к -18 LUFS integrated ПЕРЕД расчётом покадрового lufs —
        # band_frac_*/spectral_slope/skewness/warmth_ratio доказанно
        # инвариантны к gain (см. run_tz02_task2_dynamics.py), пересчитывать
        # их на нормализованном сигнале бессмысленно; а вот сам lufs без
        # этого сравнивает разницу МАСТЕРИНГА между демкой и версиями
        # инженера сведения (там разница стабильно 15-18дБ), не разницу динамики.
        file_lufs = d1.loc[d1.path == relpath, "integrated_lufs"]
        if len(file_lufs):
            gain_db = TARGET_LUFS - float(file_lufs.iloc[0])
            mono_for_lufs = mono * (10 ** (gain_db / 20))
        else:
            mono_for_lufs = mono
            print(f"  ПРЕДУПРЕЖДЕНИЕ: не нашёл integrated_lufs для {relpath}, LUFS не нормализован")

        f0_df = load_f0(safe_name)
        notes_df = load_notes(safe_name)
        wdf = analyze_file_windows(relpath, mono, sr, f0_df, stem_f0, meter, notes_df, mono_for_lufs)
        wdf["version"] = version
        gated = wdf.vibrato_depth_cents.notna().sum()
        print(f"  окон всего: {len(wdf)}, прошли гейт по вокалу: {gated}")
        all_rows.append(wdf)

    table = pd.concat(all_rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / "moving_window_4s1s.parquet", index=False)
    table.to_csv(OUT / "moving_window_4s1s.csv", index=False)
    print(f"\nВсего окон (после энергетического гейта): {len(table)} -> {OUT / 'moving_window_4s1s.parquet'}")

    print("\n=== Проверка: вибрато на 'переход 77с' и 'пауза' теперь ===")
    for lo, hi, label in [(75, 81, "переход 77с"), (132, 137, "пауза")]:
        sub = table[(table.version == "v7") & (table.t_start >= lo) & (table.t_start < hi)]
        print(f"{label}: окон={len(sub)}, с вибрато(не NaN)={sub.vibrato_depth_cents.notna().sum()}, "
              f"значения={sub.vibrato_depth_cents.dropna().tolist()}")


if __name__ == "__main__":
    main()
