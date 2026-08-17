"""ТЗ-02 Задача 1, пункт 4 (пропущенный ранее): демка-контроль ПО СЕКЦИЯМ,
не только по файлу целиком. Проверяем прямой вопрос автора: сильно ли
различаются метрики в разных местах трека, и совпадает ли расхождение
с ожиданием по правкам (куплеты/тихие места сильнее припевов)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "diff"

VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]
SECTION_METRICS = ["band_frac_air", "band_frac_lowmid", "skewness", "vibrato_depth_cents"]


def main():
    sec = pd.read_parquet(OUT / "kp_section_diff.parquet")
    sec = sec[sec.metric.isin(SECTION_METRICS)]

    rows = []
    for _, r in sec.iterrows():
        if pd.isna(r.get("demo")):
            continue
        for v in VERSIONS:
            if pd.notna(r.get(v)):
                rows.append(dict(metric=r.metric, section=r.section, start_s=r.start_s,
                                  version=v, dist_from_demo=abs(r[v] - r["demo"])))
    long_df = pd.DataFrame(rows)
    long_df.to_parquet(OUT / "demo_distance_by_section.parquet", index=False)
    long_df.to_csv(OUT / "demo_distance_by_section.csv", index=False)

    print("=== Расстояние до демки по секциям, v7 (последняя версия) ===")
    v7 = long_df[long_df.version == "v7"].sort_values(["metric", "start_s"])
    for metric in SECTION_METRICS:
        sub = v7[v7.metric == metric]
        print(f"\n--- {metric} ---")
        print(sub[["section", "dist_from_demo"]].to_string(index=False))
        print(f"  разброс по секциям: min={sub.dist_from_demo.min():.4f}  "
              f"max={sub.dist_from_demo.max():.4f}  "
              f"отношение max/min={sub.dist_from_demo.max()/max(sub.dist_from_demo.min(),1e-9):.1f}x")

    # проверка ожидания: припевы (Chorus) vs куплеты/тихие места
    print("\n=== Припевы vs остальное (v7) ===")
    v7["is_chorus"] = v7.section.str.contains("Chorus", case=False, na=False)
    print(v7.groupby(["metric", "is_chorus"]).dist_from_demo.mean().to_string())


if __name__ == "__main__":
    main()
