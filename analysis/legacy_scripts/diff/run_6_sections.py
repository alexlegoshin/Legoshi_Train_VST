"""Шаг 6 ТЗ-01, вторая часть: разбивка метрика x версия x СЕКЦИЯ поверх
покадровых/пособытийных данных §4, которые уже посчитаны и сохранены как
per-file parquet (не пересчитываем ничего заново — только агрегируем).

Только «основной трек» — только там есть sections.csv (§3). Только миксы
инженера сведения (v2,v3,v4,v6,v7) + демка, где для неё есть соответствующие
покадровые файлы (у демки их меньше — она не входила в цели §4.3/§4.6/§4.9
по ролям, честный NaN, не подделываем).

Агрегация — медиана внутри секции (устойчива к выбросам, тот же принцип,
что и в whole-file summary во всех блоках §4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from analysis.legacy_scripts.diff.run_6 import VERSION_ORDER, NEIGHBOR_PAIRS, classify_trajectory

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

FILES = {
    "v2": "основной трек/-/версия сведения 2.wav",
    "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav",
    "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav",
    "v7": "основной трек/-/версия сведения 7.wav",
    "demo": "основной трек/ТА/основной трек track out/демка_аранж_основной_трек.wav",
}

# (тег блока, суффикс файла, колонка времени, [колонки-значения], переименование)
# ПОЙМАНО: 4_1_momentary и 4_1_short_term обе дают колонку "lufs" (разные
# окна интегрирования — 400мс и 3с по EBU R128, физически разные числа) —
# под одним именем метрики "lufs" они бы тихо перезаписывали друг друга
# при агрегации (и в build_section_table, и в Шаге 8). Переименовываем сразу
# при чтении, чтобы имя метрики было уникальным без потери источника.
CONTINUOUS_SPECS = [
    ("4.1", "4_1_momentary", "t_s", ["lufs_momentary"], {"lufs": "lufs_momentary"}),
    ("4.1", "4_1_short_term", "t_s", ["lufs_short_term", "psr"], {"lufs": "lufs_short_term"}),
    ("4.2", "4_2_moments", "t_s", ["centroid_hz", "spread_hz", "skewness", "kurtosis",
                                    "rolloff85_hz", "rolloff95_hz", "flatness", "flux",
                                    "band_frac_sub", "band_frac_low", "band_frac_lowmid",
                                    "band_frac_mud", "band_frac_mid", "band_frac_presence",
                                    "band_frac_sibilance", "band_frac_air"], {}),
    ("4.3", "4_3_loudness", "t_s", ["sone"], {}),
    ("4.3", "4_3_roughness", "t_s", ["asper"], {}),
    ("4.3", "4_3_sharpness", "t_s", ["acum"], {}),
    ("4.4", "4_4_blocks", "t_s", ["correlation", "ms_ratio"], {}),
    ("4.9", "4_9_curve", "t_s", ["sethares_dissonance", "vassilakis_roughness", "spectral_flux"], {}),
]

# то же самое, но по-событийные данные (не равномерная сетка) — колонка
# времени другая, агрегация та же (медиана по событиям, попавшим в секцию)
EVENT_SPECS = [
    ("4.5", "4_5_tails", "onset_s", ["rt60_s", "edt_s", "c50_db", "c80_db", "drr_db",
                                      "tail_spectral_tilt_db_per_oct", "predelay_s"], {}),
    ("4.6", "4_6_notes", "t_start", ["intonation_deviation_cents", "vibrato_rate_hz", "vibrato_depth_cents"], {}),
    ("4.2", "4_2_harmonics_notes", "t_start", ["thd", "even_odd_ratio"], {}),
]


def load_sections():
    sec = pd.read_csv(ROOT / "_analysis" / "sections.csv")
    return sec[sec.song == "основной трек"].sort_values("start_s").reset_index(drop=True)


def aggregate_by_section(df, time_col, value_cols, sections):
    rows = []
    for _, s in sections.iterrows():
        mask = (df[time_col] >= s.start_s) & (df[time_col] < s.end_s)
        sub = df[mask]
        for col in value_cols:
            if col not in sub.columns or len(sub) == 0:
                val = np.nan
            else:
                v = sub[col].dropna()
                val = float(v.median()) if len(v) else np.nan
            # "переход"/"пауза" встречаются в sections.csv несколько раз —
            # section САМ ПО СЕБЕ не уникальный ключ, сортируем и группируем
            # по start_s, имя оставляем только для читаемости
            rows.append(dict(start_s=float(s.start_s), section=s.section, metric=col, value=val, n=len(sub)))
    return pd.DataFrame(rows)


def build_section_table():
    sections = load_sections()
    all_rows = []

    for tag, suffix, time_col, cols, rename in CONTINUOUS_SPECS + EVENT_SPECS:
        for version, path in FILES.items():
            safe_name = path.replace("/", "__")
            f = METRICS_DIR / f"{safe_name}.{suffix}.parquet"
            if not f.exists():
                continue
            df = pd.read_parquet(f)
            if rename:
                df = df.rename(columns=rename)
            if time_col not in df.columns:
                continue
            agg = aggregate_by_section(df, time_col, cols, sections)
            agg["block"] = tag
            agg["version"] = version
            all_rows.append(agg)

    long_df = pd.concat(all_rows, ignore_index=True)
    wide = long_df.pivot_table(index=["block", "metric", "start_s", "section"], columns="version",
                                values="value", aggfunc="first")
    wide = wide.reindex(columns=[v for v in VERSION_ORDER if v in wide.columns])

    for a, b in NEIGHBOR_PAIRS:
        if a in wide.columns and b in wide.columns:
            wide[f"Δ{a}→{b}"] = wide[b] - wide[a]
    if "demo" in wide.columns:
        for v in VERSION_ORDER[1:]:
            if v in wide.columns:
                wide[f"Δdemo→{v}"] = wide[v] - wide["demo"]

    traj_cols = [v for v in ["v2", "v3", "v4", "v5", "v6", "v7"] if v in wide.columns]
    wide["траектория"] = wide[traj_cols].apply(lambda r: classify_trajectory(r.tolist()), axis=1)

    # порядок секций как в sections.csv (по start_s), не алфавитный
    wide = wide.reset_index().sort_values(["block", "metric", "start_s"])

    # аккорды по секциям (уже родные, без агрегации — просто подклеиваем как отдельный блок)
    chords_rows = []
    for version, path in FILES.items():
        safe_name = path.replace("/", "__")
        f = METRICS_DIR / f"{safe_name}.4_9_chords.parquet"
        if f.exists():
            cdf = pd.read_parquet(f)
            cdf["version"] = version
            chords_rows.append(cdf)
    chords = pd.concat(chords_rows, ignore_index=True) if chords_rows else pd.DataFrame()

    return wide, chords


if __name__ == "__main__":
    table, chords = build_section_table()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / "kp_section_diff.parquet", index=False)
    table.to_csv(OUT / "kp_section_diff.csv", index=False)
    print(f"Метрик x секций в таблице: {len(table)} -> {OUT / 'kp_section_diff.parquet'}")

    if len(chords):
        chords.to_parquet(OUT / "kp_section_chords.parquet", index=False)
        print(f"Аккорды по секциям: {len(chords)} строк -> {OUT / 'kp_section_chords.parquet'}")

    print("\n=== Пример: центроид спектра по секциям, v2 vs v7 ===")
    sub = table[(table.block == "4.2") & (table.metric == "centroid_hz")]
    cols_show = ["section", "v2", "v7", "Δv2→v7"] if "Δv2→v7" in sub.columns else ["section", "v2", "v7"]
    print(sub[[c for c in cols_show if c in sub.columns]].to_string(index=False))

    print(f"\nВсего строк метрика×секция: {len(table)}, "
          f"монотонных траекторий: {(table['траектория'].isin(['монотонно растёт', 'монотонно убывает'])).sum()}")
