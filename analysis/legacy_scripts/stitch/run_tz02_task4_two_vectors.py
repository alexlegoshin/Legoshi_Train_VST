"""ТЗ-02 Задача 4: разнести метрики Шага 6 на два вектора.

Вектор А (динамика, гипотеза "инженер сведения чинил реальную проблему"):
PLR, PSR, дисперсия short-term LUFS, pumping_score, LRA.
Вектор Б (тон, гипотеза "инженер сведения ломал характер"):
spectral_slope, warmth_ratio, band_frac_air, band_frac_lowmid,
band_frac_low, skewness.

Для каждого вектора: z-score каждой метрики (по демка+v2..v7+референс А+
референс Б вместе), ориентация знака по направлению v2->v7 (чтобы "больше" =
"дальше в ту сторону, куда двигался инженер сведения"), затем среднее по вектору —
это и есть координата точки на диаграмме А-vs-Б."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

VECTOR_A = [("4.1", "plr"), ("4.1", "lra"), ("4.1", "pumping_score"), ("4.1", "short_term_var")]
VECTOR_B = [("4.2", "spectral_slope_db_per_oct"), ("4.3", "warmth_ratio"),
            ("4.2", "band_frac_air_median"), ("4.2", "band_frac_lowmid_median"),
            ("4.2", "band_frac_low_median"), ("4.2", "skewness_median")]

TARGETS = [
    ("основной трек", "demo", None, "demo"),
    ("основной трек", "mix", "v2", "v2"),
    ("основной трек", "mix", "v3", "v3"),
    ("основной трек", "mix", "v4", "v4"),
    ("основной трек", "mix", "v5", "v5"),
    ("основной трек", "mix", "v6", "v6"),
    ("основной трек", "mix", "v7", "v7"),
    ("референс А", "reference", None, "референс А"),
    ("референс Б", "demo", None, "референс Б"),
]

FILES = {"4.1": "4_1_summary.parquet", "4.2": "4_2_summary.parquet", "4.3": "4_3_quick_summary.parquet"}

# psr не в summary — добавляем отдельно из dynamics_profile.csv (Задача 2)
PSR_LABELS = {"demo": "KP demo", "v2": "KP v2", "v3": "KP v3", "v4": "KP v4",
              "v5": "KP v5", "v6": "KP v6", "v7": "KP v7", "референс А": "референс А", "референс Б": "референс Б"}


def load_value(block, metric, song, role, version):
    df = pd.read_parquet(METRICS_DIR / FILES[block])
    s = df[(df.song == song) & (df.role == role)]
    if version is not None and "version" in s.columns:
        s = s[s.version == version]
    return float(s.iloc[0][metric]) if len(s) and metric in s.columns else None


def build_matrix(vector_specs):
    data = {}
    for block, metric in vector_specs:
        vals = {label: load_value(block, metric, song, role, ver) for song, role, ver, label in TARGETS}
        data[f"{block}::{metric}"] = vals
    return pd.DataFrame(data)


def score_vector(matrix):
    """z-score каждой колонки, знак по (v7-v2), среднее по строке = координата."""
    z = matrix.copy()
    for col in matrix.columns:
        v2, v7 = matrix.loc["v2", col], matrix.loc["v7", col]
        sign = 1.0 if (pd.notna(v7) and pd.notna(v2) and v7 >= v2) else -1.0
        mean, std = matrix[col].mean(), matrix[col].std()
        z[col] = ((matrix[col] - mean) / std * sign) if std > 0 else 0.0
    return z.mean(axis=1)


def main():
    psr = pd.read_csv(ROOT / "_analysis" / "diff" / "dynamics_profile.csv").set_index("label")["psr_median"]

    mat_a = build_matrix(VECTOR_A)
    mat_a.index = [t[3] for t in TARGETS]
    mat_a["psr_median"] = [psr.get(PSR_LABELS[label], np.nan) for label in mat_a.index]

    mat_b = build_matrix(VECTOR_B)
    mat_b.index = [t[3] for t in TARGETS]

    score_a = score_vector(mat_a)
    score_b = score_vector(mat_b)

    result = pd.DataFrame({"вектор_А_динамика": score_a, "вектор_Б_тон": score_b})
    result.to_parquet(OUT / "two_vectors.parquet")
    result.to_csv(OUT / "two_vectors.csv")

    print("=== Координаты на двух векторах (0 = среднее по выборке, знак = направление движения инженера сведения) ===")
    print(result.to_string())

    v2a, v7a = score_a["v2"], score_a["v7"]
    v2b, v7b = score_b["v2"], score_b["v7"]
    print(f"\nv2->v7 по вектору А: {v2a:.2f} -> {v7a:.2f} (движение {'+' if v7a > v2a else '-'})")
    print(f"v2->v7 по вектору Б: {v2b:.2f} -> {v7b:.2f} (движение {'+' if v7b > v2b else '-'})")

    print("\n=== Проверка ожидания: референс А/референс Б по ходу движения (А) и против движения (Б)? ===")
    for label in ["референс А", "референс Б"]:
        a_side = "по ходу (v7-сторона)" if (score_a[label] - v2a) * (v7a - v2a) > 0 else "против (v2-сторона)"
        b_side = "по ходу (v7-сторона)" if (score_b[label] - v2b) * (v7b - v2b) > 0 else "против (v2-сторона)"
        print(f"{label}: вектор А -> {a_side} | вектор Б -> {b_side}")


if __name__ == "__main__":
    main()
