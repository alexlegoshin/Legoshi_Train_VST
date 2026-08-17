"""Инструментал (bass/drums/other) для «внешний трек» — переиспользует
analyze_windows из run_step11g_instrumental_metrics.py, добавляет только
одну новую версию в уже накопленный instrumental_windows.parquet."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import soundfile as sf

from analysis.legacy_scripts.stitch.run_step11g_instrumental_metrics import (
    analyze_windows, lowpass, TARGET_LUFS, STEMS, SEP_DIR, OUT_DIR,
)

ROOT = Path(__file__).resolve().parents[4]
MIX_PATH = "ЧёЗаУродыНаСцене - внешний трек.mp3"
STEM_DIR = SEP_DIR / "ЧёЗаУродыНаСцене - внешний трек"


def main():
    d1 = pd.read_parquet(ROOT / "_analysis" / "metrics" / "4_1_summary.parquet")
    mix_lufs = float(d1.loc[d1.path == MIX_PATH, "integrated_lufs"].iloc[0])
    gain_db = TARGET_LUFS - mix_lufs

    rows = []
    for stem in STEMS:
        data, sr = sf.read(str(STEM_DIR / f"{stem}.wav"), dtype="float64", always_2d=True)
        mono = data.mean(axis=1) * (10 ** (gain_db / 20))
        mono = lowpass(mono, sr)
        wdf = analyze_windows(mono, sr)
        wdf["version"], wdf["stem"] = "внешний трек", stem
        print(f"внешний трек/{stem}: {len(wdf)} окон")
        rows.append(wdf)

    new_table = pd.concat(rows, ignore_index=True)
    existing = OUT_DIR / "instrumental_windows.parquet"
    old = pd.read_parquet(existing)
    old = old[old.version != "внешний трек"]
    table = pd.concat([old, new_table], ignore_index=True)
    table.to_parquet(existing, index=False)
    table.to_csv(OUT_DIR / "instrumental_windows.csv", index=False)
    print(f"Объединено: {len(table)} строк -> {existing}")


if __name__ == "__main__":
    main()
