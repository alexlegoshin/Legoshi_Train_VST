"""Скользящее окно (4с/1с) для «внешний трек» — по образцу
run_tz03_moving_window_other_songs.py. Вокальный F0-источник — Demucs
vocals.wav (реального мультитрека нет, как и у референс А/референс Б)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pyloudnorm as pyln
import soundfile as sf

from analysis.legacy_scripts.stitch.run_tz03_moving_window import analyze_file_windows, load_f0, load_notes, TARGET_LUFS

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

MIX_PATH = "ЧёЗаУродыНаСцене - внешний трек.mp3"
VOCAL_SAFE_NAME = "_analysis__separated__htdemucs_ft__ЧёЗаУродыНаСцене - внешний трек__vocals.wav"


def main():
    meter = pyln.Meter(44100)
    data, sr = sf.read(str(ROOT / MIX_PATH), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)

    d1 = pd.read_parquet(METRICS_DIR / "4_1_summary.parquet")
    file_lufs = float(d1.loc[d1.path == MIX_PATH, "integrated_lufs"].iloc[0])
    gain_db = TARGET_LUFS - file_lufs
    mono_for_lufs = mono * (10 ** (gain_db / 20))

    f0_df = load_f0(VOCAL_SAFE_NAME)
    notes_df = load_notes(VOCAL_SAFE_NAME)
    print(f"F0 найден: {f0_df is not None}, нот: {len(notes_df) if notes_df is not None else 0}")

    wdf = analyze_file_windows(MIX_PATH, mono, sr, f0_df, None, meter, notes_df, mono_for_lufs)
    wdf["version"] = "внешний трек"
    gated = wdf.vibrato_depth_cents.notna().sum()
    print(f"окон всего: {len(wdf)}, прошли гейт по вокалу: {gated}")

    existing = OUT / "moving_window_4s1s.parquet"
    table_old = pd.read_parquet(existing)
    table_old = table_old[table_old.version != "внешний трек"]
    table = pd.concat([table_old, wdf], ignore_index=True)
    table.to_parquet(existing, index=False)
    table.to_csv(OUT / "moving_window_4s1s.csv", index=False)
    print(f"Объединено: {len(table)} строк")


if __name__ == "__main__":
    main()
