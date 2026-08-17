"""ТЗ-02 Задача 1: демка «основного трека» как внутренний контроль.

Межпесенное сравнение (референс А vs КП) тащит разницу аранжировок/жанра.
Внутри одной песни разница между демкой и версиями инженера сведения — чище:
та же песня, та же аранжировка, тот же вокал, единственная переменная —
обработка. |metric(vN) - metric(demo)| растёт монотонно v2->v7 или нет —
и на какую сторону от версий инженера сведения ложится «референс А» по тем же
метрикам, что и демка (два независимых "нравится" против версий)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]

TRAJECTORY_8 = [
    ("4.1", "sample_peak_dbfs"),
    ("4.2", "band_frac_air_median"),
    ("4.2", "band_frac_lowmid_median"),
    ("4.2", "skewness_median"),
    ("4.2", "spectral_slope_db_per_oct"),
    ("4.3", "warmth_ratio"),
    ("4.6", "vibrato_depth_cents_median"),
    ("4.6", "voiced_fraction"),
]


def classify_distance_trend(distances):
    vals = [d for d in distances if pd.notna(d)]
    if len(vals) < 4:
        return "недостаточно данных"
    diffs = np.diff(vals)
    if np.all(diffs >= -1e-9):
        return "уходит от демки"
    if np.all(diffs <= 1e-9):
        return "приближается к демке"
    return "не связано с демкой"


def load_reference_a_value(block, metric):
    files = {"4.1": "4_1_summary.parquet", "4.2": "4_2_summary.parquet",
             "4.3": "4_3_quick_summary.parquet", "4.4": "4_4_summary.parquet",
             "4.5": "4_5_summary.parquet", "4.6": "4_6_summary.parquet",
             "4.9": "4_9_summary.parquet", "4.10": "4_10_summary.parquet"}
    if block not in files:
        return None  # блок "5" (деконволюция) — нет данных для «референс А», без стемов
    f = METRICS_DIR / files[block]
    if not f.exists():
        return None
    df = pd.read_parquet(f)
    s = df[(df.song == "референс А") & (df.role == "reference")]
    return float(s.iloc[0][metric]) if len(s) and metric in s.columns else None


def main():
    diff = pd.read_parquet(OUT / "kp_version_diff.parquet")
    rows = []
    for _, r in diff.iterrows():
        block, metric = r["block"], r["metric"]
        demo = r["demo"]
        if pd.isna(demo):
            continue
        dist = {v: abs(r[v] - demo) if pd.notna(r[v]) else np.nan for v in VERSIONS}
        trend = classify_distance_trend([dist[v] for v in VERSIONS])
        ref_a = load_reference_a_value(block, metric)
        same_side = None
        if ref_a is not None and pd.notna(r["v7"]) and pd.notna(demo):
            v7_side = np.sign(r["v7"] - demo)
            ref_a_side = np.sign(ref_a - demo)
            same_side = bool(v7_side == ref_a_side) if v7_side != 0 else None
        rows.append(dict(block=block, metric=metric, demo=demo,
                          **{f"dist_{v}": dist[v] for v in VERSIONS},
                          класс=trend, референс_а=ref_a, референс_а_на_стороне_демки=same_side))

    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "КП_demo_reference.parquet", index=False)
    table.to_csv(OUT / "КП_demo_reference.csv", index=False)

    print(f"Всего метрик: {len(table)}")
    print(table["класс"].value_counts())

    print("\n=== 8 метрик Шага 6 — класс по демке-контролю ===")
    sub = table[table.apply(lambda r: (r.block, r.metric) in TRAJECTORY_8, axis=1)]
    print(sub[["block", "metric", "класс", "референс_а_на_стороне_демки"]].to_string(index=False))

    uhod = table[table["класс"] == "уходит от демки"]
    print(f"\n=== Метрики, уходящие от демки монотонно (кандидаты на 'что инженер сведения разрушал') ===")
    print(f"Всего: {len(uhod)}")
    print(uhod[["block", "metric", "референс_а_на_стороне_демки"]].to_string(index=False))


if __name__ == "__main__":
    main()
