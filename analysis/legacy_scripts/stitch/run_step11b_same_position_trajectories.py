"""Шаг 11b: сравнение одинаковых позиций времени across версий КП.

Мотивация автора: сравнивать не только точки, попавшие в опросник, а
одни и те же места трека между демкой и версиями инженера сведения систематически —
сетка времени общая для demo/v2..v7 (выровнена GCC-PHAT на Шаге 2, общий
шаг 1с), поэтому t_start=X в любой версии — один и тот же музыкальный
момент. Это не требует новой прослушки: работает на уже посчитанных
§4-метриках по всей песне, а не только на 94 оценённых фрагментах.

Что делает:
  1. Для каждой ключевой метрики находит моменты с наибольшим разбросом
     между версиями инженера сведения (v2..v7) за всю песню — где инженер сведения реально
     что-то поменял, независимо от того, что попало в опросник.
  2. Для каждого такого момента показывает полную траекторию
     demo -> v2 -> ... -> v7, чтобы видеть направление и монотонность.
  3. Отмечает, какие из уже оценённых автором точек (94 фрагмента)
     попадают в топ по разбросу — типичны они или случайны.

Не даёт корреляций с восприятием там, где восприятия нет (не выдумывать
рейтинг для неопрошенных моментов) — только объективная траектория метрик.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DIFF_DIR = ROOT / "_analysis" / "diff"
Q_DIR = ROOT / "_analysis" / "questionnaire"
OUT_DIR = ROOT / "_analysis" / "correlation"

KP_VERSIONS = ["demo", "v2", "v3", "v4", "v5", "v6", "v7"]
MARTIN_VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]
KEY_METRICS = ["warmth_ratio", "spectral_slope", "band_frac_air", "band_frac_lowmid",
               "band_frac_low", "skewness", "lufs_normalized"]
TOP_N = 10


def common_grid(kp):
    common = set(kp[kp.version == "demo"].t_start)
    for v in KP_VERSIONS[1:]:
        common &= set(kp[kp.version == v].t_start)
    return sorted(common)


def rated_kp_t_starts():
    """t_start фрагментов КП (demo+инженер сведения), реально попавших в ratings.csv."""
    frag = pd.read_parquet(Q_DIR / "fragments_selected.parquet")
    rated_ids = set(pd.read_csv(Q_DIR / "ratings.csv").fragment_id)
    kp = frag[frag.version.isin(KP_VERSIONS) & frag.fragment_id.isin(rated_ids)]
    return set(kp.t_start.round(0))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(DIFF_DIR / "moving_window_4s1s.parquet")
    kp = w[w.version.isin(KP_VERSIONS)]
    grid = common_grid(kp)
    print(f"Общая сетка времени (demo+v2..v7): {len(grid)} моментов, "
          f"{grid[0]}-{grid[-1]}с, шаг 1с")

    rated_ts = rated_kp_t_starts()
    print(f"Из них попало в опросник: {len(rated_ts & set(grid))}")

    all_hotspots = []
    for metric in KEY_METRICS:
        sub = kp[kp.t_start.isin(grid)]
        piv = sub.pivot(index="t_start", columns="version", values=metric)[KP_VERSIONS]
        martin = piv[MARTIN_VERSIONS]
        spread = martin.max(axis=1) - martin.min(axis=1)
        top = spread.sort_values(ascending=False).head(TOP_N)

        print(f"\n=== {metric}: топ-{TOP_N} моментов по разбросу между версиями инженера сведения ===")
        table = piv.loc[top.index].round(2).copy()
        table.insert(0, "spread_v2_v7", top.round(2))
        table["в_опроснике"] = table.index.map(lambda t: t in rated_ts)
        print(table.to_string())

        for t_start, row in table.iterrows():
            all_hotspots.append(dict(metric=metric, t_start=t_start,
                                      spread=row["spread_v2_v7"], in_survey=row["в_опроснике"]))

    hotspots_df = pd.DataFrame(all_hotspots)
    hotspots_df.to_csv(OUT_DIR / "same_position_hotspots.csv", index=False)

    print(f"\n=== Сводка: сколько топ-хотспотов реально попали в опросник ===")
    print(hotspots_df.groupby("metric").in_survey.agg(["sum", "count"]))
    print(f"\nСохранено в {OUT_DIR / 'same_position_hotspots.csv'}")


if __name__ == "__main__":
    main()
