"""Шаг 9 ТЗ-01: проверка находок Шага 6/8 на «референс А» и референс Б.

Критерий по ТЗ (§10, шаг 9): "подтверждаются ли найденные различия".

Метод: 8 метрик с монотонной траекторией v2->v7 внутри «основной трек»
(Шаг 6) + band_frac_low (значимая находка Шага 8) — смотрим, где по этим
метрикам располагаются «референс А» и референс Б (обе любимые автором записи,
но НЕ часть переписки с инженером сведения) относительно диапазона v2..v7.

Перед спектральными сравнениями — обязательная по ТЗ (§1.2) проверка
среза mp3-кодека «референс А»: обнаружен резкий обрыв ~20кГц (проверено
эмпирически, см. отчёт). Все используемые здесь полосы (air 8-16кГц,
spectral_slope 100Гц-10кГц) лежат ниже среза — коррекция не нужна."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "stitch"

# (block, column, KP-трактовка "лучше/хуже" по знаку v7-v2)
TRAJECTORY_METRICS = [
    ("4.1", "sample_peak_dbfs", None),
    ("4.2", "band_frac_air_median", None),
    ("4.2", "band_frac_lowmid_median", None),
    ("4.2", "skewness_median", None),
    ("4.2", "spectral_slope_db_per_oct", None),
    ("4.3", "warmth_ratio", None),
    ("4.6", "vibrato_depth_cents_median", None),
    ("4.6", "voiced_fraction", None),
    ("4.2", "band_frac_low_median", "Шаг 8: значимо выше в моментах-жалобах, чем в случайных (в пределах КП)"),
]

TARGETS = [
    ("основной трек", "demo", None, "KP demo"),
    ("основной трек", "mix", "v2", "KP v2"),
    ("основной трек", "mix", "v7", "KP v7"),
    ("референс А", "reference", None, "референс А"),
    ("референс Б", "demo", None, "референс Б"),
]

FILES_BY_BLOCK = {
    "4.1": "4_1_summary.parquet", "4.2": "4_2_summary.parquet",
    "4.3": "4_3_quick_summary.parquet", "4.6": "4_6_summary.parquet",
}


def load_value(block, column, song, role, version):
    df = pd.read_parquet(METRICS_DIR / FILES_BY_BLOCK[block])
    s = df[(df.song == song) & (df.role == role)]
    if version is not None and "version" in s.columns:
        s = s[s.version == version]
    if len(s) == 0 or column not in s.columns:
        return None
    return float(s.iloc[0][column])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for block, column, note in TRAJECTORY_METRICS:
        vals = {label: load_value(block, column, song, role, version)
                for song, role, version, label in TARGETS}
        v2, v7 = vals["KP v2"], vals["KP v7"]
        trend = None
        if v2 is not None and v7 is not None:
            trend = "растёт" if v7 > v2 else "убывает"
        rows.append(dict(block=block, metric=column, тренд_v2_v7=trend, note=note, **vals))

    table = pd.DataFrame(rows)

    # вердикт: лежит ли референс А/референс Б В диапазоне [min(v2,v7), max(v2,v7)]
    # или на "хорошей" стороне (в сторону v7 от v2), или полностью вне/против
    def verdict(row):
        v2, v7 = row["KP v2"], row["KP v7"]
        if v2 is None or v7 is None:
            return None
        lo, hi = min(v2, v7), max(v2, v7)
        out = {}
        for label in ["референс А", "референс Б"]:
            val = row[label]
            if val is None:
                out[label] = "нет данных"
            elif lo <= val <= hi:
                out[label] = "в диапазоне v2..v7"
            elif (val > hi and v7 > v2) or (val < lo and v7 < v2):
                out[label] = "дальше в сторону v7 (та же тенденция)"
            else:
                out[label] = "ПРОТИВОПОЛОЖНО тренду"
        return out

    verdicts = table.apply(verdict, axis=1)
    table["вердикт_референс А"] = [v["референс А"] if v else None for v in verdicts]
    table["вердикт_референс Б"] = [v["референс Б"] if v else None for v in verdicts]

    table.to_parquet(OUT / "crosscheck_radost_pvprk.parquet", index=False)
    table.to_csv(OUT / "crosscheck_radost_pvprk.csv", index=False)

    cols = ["metric", "тренд_v2_v7", "KP demo", "KP v2", "KP v7", "референс А", "референс Б",
            "вердикт_референс А", "вердикт_референс Б"]
    print(table[cols].to_string(index=False))

    consistent = sum(1 for v in table["вердикт_референс А"] if v and "ПРОТИВОПОЛОЖНО" not in v and v != "нет данных")
    total = table["вердикт_референс А"].notna().sum()
    print(f"\nреференс А согласуется с трендом КП: {consistent}/{total} метрик")
    consistent_p = sum(1 for v in table["вердикт_референс Б"] if v and "ПРОТИВОПОЛОЖНО" not in v and v != "нет данных")
    total_p = table["вердикт_референс Б"].notna().sum()
    print(f"референс Б согласуется с трендом КП: {consistent_p}/{total_p} метрик")


if __name__ == "__main__":
    main()
