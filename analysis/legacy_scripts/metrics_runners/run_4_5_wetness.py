"""Довесок к §4.5: «оценка мокрости через деконволюцию» — задача #25.
По ТЗ (§4.5, самая надёжная из оценок мокрости): энергия остатка §5
относительно прямого сигнала (= энергии микса, объяснённой стемами).

Есть только там, где посчитана деконволюция (§5): миксы инженера сведения «основной трек» (v2..v7) и контроль «контрольный трек» (финал). Для демки и «референс А»
(нет стемов) — честный NaN, не выдумываем."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
DECONV = ROOT / "_analysis" / "deconv"


def wetness_from_quality(quality_path, song):
    df = pd.read_parquet(quality_path)
    df["wetness_pct"] = ((df.residual_energy_L + df.residual_energy_R) /
                          (df.mix_energy_L + df.mix_energy_R) * 100)
    df["song"] = song
    return df[["song", "version", "wetness_pct"]]


def main():
    parts = []
    kp_q = DECONV / "kp_quality.parquet"
    if kp_q.exists():
        parts.append(wetness_from_quality(kp_q, "основной трек"))
    zt_q = DECONV / "zt_quality.parquet"
    if zt_q.exists():
        parts.append(wetness_from_quality(zt_q, "контрольный трек"))

    wetness = pd.concat(parts, ignore_index=True)
    print("=== Мокрость через остаток деконволюции ===")
    print(wetness.to_string(index=False))

    summary = pd.read_parquet(OUT / "4_5_summary.parquet")
    if "wetness_pct" in summary.columns:
        summary = summary.drop(columns=["wetness_pct"])
    merged = summary.merge(wetness, on=["song", "version"], how="left")
    merged.to_parquet(OUT / "4_5_summary.parquet", index=False)
    print(f"\nОбновлён -> {OUT / '4_5_summary.parquet'} ({len(merged)} строк, "
          f"{merged.wetness_pct.notna().sum()} с непустой мокростью)")

    print("\n=== основной трек: мокрость + остальные признаки реверба по версиям ===")
    kp = merged[merged.song == "основной трек"]
    cols = ["version", "wetness_pct", "rt60_s_median", "edt_s_median", "drr_db_median"]
    print(kp[cols].sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
