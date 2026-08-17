"""ТЗ-03: демо-контроль версии 2 — на скользящем окне (не на 15 неравномерных
секциях), с ОТНОСИТЕЛЬНЫМ расстоянием (не абсолютным — база метрики
меняется на порядки между тихими и громкими моментами, абсолютное
расстояние в основном повторяет разброс базового уровня, см. диагностику
Opus, Спирмен 0.40 для band_frac_air на старых данных).

lufs_normalized — уже посчитан на сигнале, приведённом к -18 LUFS
integrated ДО расчёта покадрового значения (см. run_tz03_moving_window.py);
остаточная разница между демкой и версиями инженера сведения после этого — реальная
разница динамики/баланса момента, не разница мастеринга."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "diff"

METRICS = ["band_frac_air", "band_frac_lowmid", "band_frac_low", "skewness",
           "spectral_slope", "warmth_ratio", "vibrato_depth_cents", "voiced_fraction",
           "lufs_normalized"]
VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]


def main():
    w = pd.read_parquet(OUT / "moving_window_4s1s.parquet")

    rows = []
    for metric in METRICS:
        piv = w.pivot_table(index="t_start", columns="version", values=metric, aggfunc="first")
        if "demo" not in piv.columns:
            continue
        for v in VERSIONS:
            if v not in piv.columns:
                continue
            both = piv[["demo", v]].dropna()
            if len(both) == 0:
                continue
            abs_dist = (both[v] - both["demo"]).abs()
            rel_dist = abs_dist / (both["demo"].abs() + 1e-9)
            for t_start, a, r in zip(both.index, abs_dist, rel_dist):
                rows.append(dict(metric=metric, version=v, t_start=t_start,
                                  abs_distance=a, rel_distance=r))

    long_df = pd.DataFrame(rows)
    long_df.to_parquet(OUT / "demo_control_moving_window.parquet", index=False)
    long_df.to_csv(OUT / "demo_control_moving_window.csv", index=False)
    print(f"Всего наблюдений: {len(long_df)} -> {OUT / 'demo_control_moving_window.parquet'}")

    print("\n=== Спирмен: ранги абсолютного vs относительного расстояния (v7, по числу окон) ===")
    for metric in METRICS:
        sub = long_df[(long_df.metric == metric) & (long_df.version == "v7")]
        if len(sub) < 5:
            continue
        corr = sub.abs_distance.corr(sub.rel_distance, method="spearman")
        print(f"  {metric}: n={len(sub)}, Спирмен={corr:.2f}")

    print("\n=== Медианное относительное расстояние по метрике (все версии v2-v7 вместе) ===")
    summary = long_df.groupby("metric").rel_distance.agg(["median", "mean", "count"])
    print(summary.to_string())

    print("\n=== Тренд относительного расстояния v2->v7 (растёт ли монотонно) ===")
    for metric in METRICS:
        vals = []
        for v in VERSIONS:
            sub = long_df[(long_df.metric == metric) & (long_df.version == v)]
            vals.append(sub.rel_distance.median() if len(sub) else np.nan)
        vals_clean = [x for x in vals if pd.notna(x)]
        if len(vals_clean) < 3:
            trend = "недостаточно данных"
        else:
            diffs = np.diff(vals_clean)
            trend = "растёт" if np.all(diffs >= 0) else ("убывает" if np.all(diffs <= 0) else "не монотонно")
        print(f"  {metric}: {[round(v, 4) if pd.notna(v) else None for v in vals]}  -> {trend}")


if __name__ == "__main__":
    main()
