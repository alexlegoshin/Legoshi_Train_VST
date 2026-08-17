"""ТЗ-03/Шаг 10: то же скользящее окно 4с/1с, что и для «основного трека»
(run_tz03_moving_window.py), но для «референс А», референс Б и «контрольный трек» —
нужно для стратификации фрагментов опросника по всем 5 источникам.

Вокальный F0-источник для гейта/вибрато — НЕ полный микс (полифония
ненадёжна, см. задачу 5 ТЗ-02), а изолированный вокал:
- референс А: demucs vocals.wav (провалидировано в задаче 3 ТЗ-02)
- референс Б: demucs vocals.wav (посчитано только что)
- контрольный трек: реальный стем "Вокал (основной остальное)" — лучше
  demucs, там нет ошибки разделения вообще."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pyloudnorm as pyln
import soundfile as sf

from analysis.legacy_scripts.stitch.run_tz03_moving_window import (
    analyze_file_windows, load_f0, load_notes, TARGET_LUFS,
)

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

SOURCES = {
    "референс А": dict(
        mix="референс А/+/1 референс А.mp3",
        vocal_safe_name="референс А__demucs_stems__vocals.wav",
    ),
    "референс Б": dict(
        mix="Песня в поддержку рака лёгких/+ но это моя грязная демка/референс Б - 27:4:2026, 18.58.wav",
        vocal_safe_name="_analysis__separated__htdemucs_ft__референс Б - 27_4_2026, 18.58__vocals.wav",
    ),
    "контрольный трек": dict(
        mix="контрольный трек/ТА/финальная/фин.mp3",
        vocal_safe_name="контрольный трек__ТА__финальная__correct__Вокал (основной остальное).wav",
    ),
}


def get_integrated_lufs(relpath, mono, sr, meter):
    """Пытаемся найти уже посчитанный integrated_lufs в 4_1_summary; если
    файла там нет (напр. demucs-сепарация) — считаем на лету."""
    f = METRICS_DIR / "4_1_summary.parquet"
    if f.exists():
        d1 = pd.read_parquet(f)
        row = d1.loc[d1.path == relpath, "integrated_lufs"]
        if len(row):
            return float(row.iloc[0])
    return float(meter.integrated_loudness(mono))


def main():
    meter = pyln.Meter(44100)
    all_rows = []

    for song, cfg in SOURCES.items():
        print(f"=== {song} ===")
        data, sr = sf.read(str(ROOT / cfg["mix"]), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)

        file_lufs = get_integrated_lufs(cfg["mix"], mono, sr, meter)
        gain_db = TARGET_LUFS - file_lufs
        mono_for_lufs = mono * (10 ** (gain_db / 20))

        f0_df = load_f0(cfg["vocal_safe_name"])
        notes_df = load_notes(cfg["vocal_safe_name"])
        if f0_df is None:
            print(f"  ПРЕДУПРЕЖДЕНИЕ: нет F0 для {cfg['vocal_safe_name']}, вибрато/voiced будут NaN")

        wdf = analyze_file_windows(cfg["mix"], mono, sr, f0_df, None, meter, notes_df, mono_for_lufs)
        wdf["version"] = song
        gated = wdf.vibrato_depth_cents.notna().sum()
        print(f"  окон всего: {len(wdf)}, прошли гейт по вокалу: {gated}")
        all_rows.append(wdf)

    table = pd.concat(all_rows, ignore_index=True)
    existing = OUT / "moving_window_4s1s.parquet"
    if existing.exists():
        kp_table = pd.read_parquet(existing)
        kp_table = kp_table[~kp_table.version.isin(SOURCES.keys())]  # на случай повторного прогона
        table = pd.concat([kp_table, table], ignore_index=True)

    table.to_parquet(existing, index=False)
    table.to_csv(OUT / "moving_window_4s1s.csv", index=False)
    print(f"\nОбъединённая таблица: {len(table)} строк -> {existing}")
    print(table.groupby("version").size())


if __name__ == "__main__":
    main()
