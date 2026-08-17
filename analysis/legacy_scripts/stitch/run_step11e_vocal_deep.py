"""Шаг 11e: максимально глубокий вокальный анализ — на этот раз по-настоящему
ПЕР-ВЕРСИОННЫЙ. run_tz03_moving_window.py брал вибрато/интонацию с сырого
трек-аута (одна запись на все версии инженера сведения — не может объяснить разницу
МЕЖДУ версиями). run_step11c/d брал спектральные метрики с Demucs-вокала
по версиям, но не питч. Здесь — pYIN, ноты, вибрато, интонация, признаки
тюна и форманты (LPC) СЧИТАЮТСЯ ЗАНОВО на каждом Demucs-вокале отдельно —
это и есть то, что инженер сведения реально сделал с голосом в каждой версии
(компрессия/тюн/де-эссинг меняют питч-трек и формантную картину, даже если
исходная нота одна и та же).

Формант-полоса (90-4000Гц) и питч-диапазон (65-1000Гц) целиком ниже отсечки
10кГц из §11c/d — LPC/pYIN не задевают зону, где Demucs шумит сильнее всего,
отдельного НЧ-фильтра здесь не требуется.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.metrics.pitch_vocal import extract_f0, segment_notes, vibrato_analysis, \
    detect_tune_artifacts, hz_to_cents, cents_to_nearest_semitone_deviation, HOP as PITCH_HOP
from analysis.metrics.vocal_texture import lpc_formants, _frame_starts, _voiced_lookup, \
    FRAME_LEN as FORMANT_FRAME_LEN, HOP as FORMANT_HOP
from analysis.metrics.psychoacoustic import harshness as harshness_fn

ROOT = Path(__file__).resolve().parents[4]
SEP_DIR = ROOT / "_analysis" / "separated" / "htdemucs_ft"
OUT_DIR = ROOT / "_analysis" / "diff"

WIN_S = 4.0
HOP_S = 1.0
MAX_PLAUSIBLE_VIBRATO_CENTS = 100.0  # тот же физический потолок, что в run_tz03

VOCAL_PATHS = {
    "v2": SEP_DIR / "версия сведения 2" / "vocals.wav",
    "v3": SEP_DIR / "версия сведения 3." / "vocals.wav",
    "v4": SEP_DIR / "версия сведения 4 " / "vocals.wav",
    "v5": SEP_DIR / "версия сведения 5" / "vocals.wav",
    "v6": SEP_DIR / "версия сведения 6" / "vocals.wav",
    "v7": SEP_DIR / "версия сведения 7" / "vocals.wav",
    "референс А": ROOT / "референс А" / "demucs_stems" / "vocals.wav",
    "референс Б": SEP_DIR / "референс Б - 27:4:2026, 18.58" / "vocals.wav",
    # demo сознательно исключена — см. run_step11c: Demucs-вокал демки это шум
    # (медиана RMS ~-100дБFS после выравнивания громкости), реального вокала
    # в этом файле практически нет.
}

MIX_PATH_FOR_LUFS = {
    "v2": "основной трек/-/версия сведения 2.wav", "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav", "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav", "v7": "основной трек/-/версия сведения 7.wav",
    "референс А": "референс А/+/1 референс А.mp3",
    "референс Б": "Песня в поддержку рака лёгких/+ но это моя грязная демка/референс Б - 27:4:2026, 18.58.wav",
}
TARGET_LUFS = -18.0


def analyze_version(mono, sr):
    """pYIN + ноты + вибрато + интонация + тюн-артефакты + форманты + harshness,
    всё агрегировано по 4с/1с скользящему окну."""
    t_f0, f0, voiced, prob = extract_f0(mono, sr)
    notes = segment_notes(t_f0, f0, voiced)
    notes_df = pd.DataFrame(notes)
    if len(notes_df):
        notes_df = notes_df[notes_df.type == "note"].copy()
        hop_s = float(np.median(np.diff(t_f0)))
        vib = notes_df.apply(lambda r: vibrato_analysis(t_f0, f0, r, hop_s), axis=1)
        notes_df["vibrato_rate_hz"] = vib.apply(lambda x: x[0])
        notes_df["vibrato_depth_cents"] = vib.apply(lambda x: x[1])
        # интонация: отклонение медианного питча ноты от ближайшего полутона
        mask_note = (t_f0[:, None] >= notes_df.t_start.to_numpy()) & (t_f0[:, None] <= notes_df.t_end.to_numpy())
        cents_all = hz_to_cents(f0)
        note_median_cents = [np.nanmedian(cents_all[mask_note[:, i]]) if mask_note[:, i].any() else np.nan
                              for i in range(len(notes_df))]
        notes_df["intonation_dev_cents"] = np.abs(cents_to_nearest_semitone_deviation(np.array(note_median_cents)))

    flats, jumps = detect_tune_artifacts(t_f0, f0, voiced)
    flats_df = pd.DataFrame(flats)
    jumps_df = pd.DataFrame(jumps)

    # форманты: свои кадры (FRAME_LEN/HOP из vocal_texture), voiced через merge_asof на f0_df
    f0_df = pd.DataFrame({"t_s": t_f0, "f0_hz": f0, "voiced": voiced})
    starts = _frame_starts(len(mono), FORMANT_FRAME_LEN, FORMANT_HOP)
    t_frames = (starts + FORMANT_FRAME_LEN / 2) / sr
    voiced_frames = _voiced_lookup(t_frames, f0_df)
    formant_rows = []
    for i, s in enumerate(starts):
        if not voiced_frames[i]:
            continue
        f1, f2, f3 = lpc_formants(mono[s:s + FORMANT_FRAME_LEN], sr)
        formant_rows.append(dict(t_s=float(t_frames[i]), f1_hz=f1, f2_hz=f2, f3_hz=f3))
    formants_df = pd.DataFrame(formant_rows)

    # --- агрегация по скользящему окну 4с/1с ---
    win_n = int(WIN_S * sr)
    n_hop_frames = int(HOP_S * sr)
    starts_win = np.arange(0, max(len(mono) - win_n, 0), n_hop_frames)
    rows = []
    for s in starts_win:
        t0, t1 = s / sr, (s + win_n) / sr
        row = dict(t_start=t0, t_end=t1)

        fmask = (t_f0 >= t0) & (t_f0 < t1)
        row["voiced_fraction"] = float(voiced[fmask].mean()) if fmask.any() else np.nan

        vibrato_val = np.nan
        intonation_val = np.nan
        if len(notes_df):
            note_mid = (notes_df.t_start + notes_df.t_end) / 2
            nmask = (note_mid >= t0) & (note_mid < t1)
            nsub_vib = notes_df.loc[nmask, "vibrato_depth_cents"].dropna()
            nsub_vib = nsub_vib[nsub_vib <= MAX_PLAUSIBLE_VIBRATO_CENTS]
            if len(nsub_vib) >= 2:
                vibrato_val = float(nsub_vib.median())
            nsub_int = notes_df.loc[nmask, "intonation_dev_cents"].dropna()
            if len(nsub_int) >= 2:
                intonation_val = float(nsub_int.median())
        row["vibrato_depth_cents"] = vibrato_val
        row["intonation_dev_cents"] = intonation_val

        n_flats = int(((flats_df.t_start >= t0) & (flats_df.t_start < t1)).sum()) if len(flats_df) else 0
        n_jumps = int(((jumps_df.t_s >= t0) & (jumps_df.t_s < t1)).sum()) if len(jumps_df) else 0
        row["n_tune_flats"] = n_flats
        row["n_tune_jumps"] = n_jumps

        if len(formants_df):
            fsub = formants_df[(formants_df.t_s >= t0) & (formants_df.t_s < t1)]
            row["formant_f1_hz"] = float(fsub.f1_hz.median()) if len(fsub) else np.nan
            row["formant_f2_hz"] = float(fsub.f2_hz.median()) if len(fsub) else np.nan
            row["formant_f3_hz"] = float(fsub.f3_hz.median()) if len(fsub) else np.nan
        else:
            row["formant_f1_hz"] = row["formant_f2_hz"] = row["formant_f3_hz"] = np.nan

        seg = mono[s:s + win_n]
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        row["harshness"] = harshness_fn(seg, sr) if rms_dbfs > -50 else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    d1 = pd.read_parquet(ROOT / "_analysis" / "metrics" / "4_1_summary.parquet")
    all_rows = []
    for version, path in VOCAL_PATHS.items():
        if not path.exists():
            print(f"ПРОПУСК {version}: нет файла {path}")
            continue
        print(f"=== {version} ===")
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)

        mix_path = MIX_PATH_FOR_LUFS.get(version)
        mix_lufs = d1.loc[d1.path == mix_path, "integrated_lufs"]
        if len(mix_lufs):
            gain_db = TARGET_LUFS - float(mix_lufs.iloc[0])
            mono = mono * (10 ** (gain_db / 20))

        wdf = analyze_version(mono, sr)
        wdf["version"] = version
        n_vib = wdf.vibrato_depth_cents.notna().sum()
        print(f"  окон: {len(wdf)}, с вибрато: {n_vib}, "
              f"tune_flats всего: {wdf.n_tune_flats.sum()}, tune_jumps всего: {wdf.n_tune_jumps.sum()}")
        all_rows.append(wdf)

    table = pd.concat(all_rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT_DIR / "vocal_deep_windows.parquet", index=False)
    table.to_csv(OUT_DIR / "vocal_deep_windows.csv", index=False)
    print(f"\nСохранено: {len(table)} окон -> {OUT_DIR / 'vocal_deep_windows.parquet'}")


if __name__ == "__main__":
    main()
