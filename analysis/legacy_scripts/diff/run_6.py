"""Шаг 6 ТЗ-01: дифф версий. Таблица метрика x версия для «основной трек»
(демка, v2, v3, v4, v6, v7), дельты между соседними версиями, классификация
траектории (монотонно/скачет), демка как нулевая точка.

Сейчас — только уровень целого файла (то, что уже посчитано как summary по
каждому блоку §4 + качество/остаток §5). Разбивка по секциям — отдельно,
это следующий, более трудоёмкий проход поверх покадровых данных, которые
кэшированы не для всех блоков на уровне секций."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
DECONV_DIR = ROOT / "_analysis" / "deconv"
OUT = ROOT / "_analysis" / "diff"

VERSION_ORDER = ["demo", "v2", "v3", "v4", "v5", "v6", "v7"]
NEIGHBOR_PAIRS = [("v2", "v3"), ("v3", "v4"), ("v4", "v5"), ("v5", "v6"), ("v6", "v7")]

# метрики-идентификаторы/технические поля, не показатели качества звука —
# не нужны в диффе (испортят "монотонность" бессмысленными строками)
EXCLUDE_COLS = {
    "path", "song", "role", "version", "n_notes", "n_transitions",
    "n_flat_tune_segments", "n_fast_jumps", "n_isolated_tails", "n_anomalies",
    "n_resonances", "n_bands_mono_loss_gt3db", "n_persistent_narrowband",
    "top_resonance_freq_hz", "worst_mono_loss_freq_hz", "top_persistent_freq_hz",
    "goniometer_principal_angle_deg", "vibrato_rate_hz_median",
}


def load_summaries():
    frames = []

    f = METRICS_DIR / "4_1_summary.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        frames.append(("4.1", d[(d.song == "основной трек") & (d.role.isin(["mix", "demo"]))]))

    f = METRICS_DIR / "4_2_summary.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        frames.append(("4.2", d[(d.song == "основной трек") & (d.role.isin(["mix", "demo"]))]))

    f = METRICS_DIR / "4_3_quick_summary.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        frames.append(("4.3", d[(d.song == "основной трек") & (d.role.isin(["mix", "demo"]))]))

    for tag, fname in [("4.4", "4_4_summary.parquet"), ("4.5", "4_5_summary.parquet"),
                        ("4.9", "4_9_summary.parquet"), ("4.10", "4_10_summary.parquet")]:
        f = METRICS_DIR / fname
        if f.exists():
            d = pd.read_parquet(f)
            frames.append((tag, d[(d.song == "основной трек") & (d.role.isin(["mix", "demo"]))]))

    f = METRICS_DIR / "4_6_summary.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        # 4.6 считался и по вокальным стемам, и по миксам — тут нужны только миксы/демка
        frames.append(("4.6", d[(d.song == "основной трек") & (d.role.isin(["mix", "demo"]))]))

    f = DECONV_DIR / "kp_quality.parquet"
    if f.exists():
        d = pd.read_parquet(f).rename(columns={"version": "version"})
        d["residual_frac_pct"] = (d.residual_energy_L + d.residual_energy_R) / (d.mix_energy_L + d.mix_energy_R) * 100
        d["song"] = "основной трек"
        d["role"] = "mix"
        frames.append(("5", d))

    return frames


def version_label(row):
    v = row.get("version")
    if pd.notna(v) and v in VERSION_ORDER:
        return v
    if row.get("role") == "demo":
        return "demo"
    return None


def classify_trajectory(values):
    """values — список по v2..v7 (без демки), может содержать NaN."""
    vals = [v for v in values if pd.notna(v)]
    if len(vals) < 3:
        return "недостаточно данных"
    diffs = np.diff(vals)
    if np.all(diffs > 0):
        return "монотонно растёт"
    if np.all(diffs < 0):
        return "монотонно убывает"
    # почти монотонно (одно отклонение) отдельно не выделяем — честно "скачет"
    return "скачет"


def build_diff_table():
    frames = load_summaries()
    rows = []
    for block_tag, df in frames:
        df = df.copy()
        df["_version"] = df.apply(version_label, axis=1)
        df = df[df["_version"].notna()]
        for col in df.columns:
            if col in EXCLUDE_COLS or col == "_version" or col.startswith("_"):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            for _, r in df.iterrows():
                rows.append(dict(block=block_tag, metric=col, version=r["_version"], value=r[col]))

    long_df = pd.DataFrame(rows).drop_duplicates(subset=["block", "metric", "version"])
    wide = long_df.pivot_table(index=["block", "metric"], columns="version", values="value", aggfunc="first")
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

    return wide.reset_index()


if __name__ == "__main__":
    table = build_diff_table()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT / "kp_version_diff.parquet", index=False)
    table.to_csv(OUT / "kp_version_diff.csv", index=False)
    print(f"Метрик в таблице: {len(table)} -> {OUT / 'kp_version_diff.parquet'}")

    print("\n=== Метрики с монотонной траекторией v2..v7 ===")
    mono = table[table["траектория"].isin(["монотонно растёт", "монотонно убывает"])]
    cols_show = ["block", "metric", "траектория"] + [c for c in ["v2", "v3", "v4", "v5", "v6", "v7"] if c in table.columns]
    print(mono[cols_show].to_string(index=False))

    print(f"\nВсего метрик: {len(table)}, монотонных: {len(mono)}, "
          f"скачущих/недостаточно: {len(table) - len(mono)}")
