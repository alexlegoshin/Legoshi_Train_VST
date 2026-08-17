"""ТЗ-02 Задача 2: профиль динамики (LUFS/LRA/PLR/PSR/crest/DR) по демке,
всем версиям инженера сведения, «референс А» и референс Б.

Мат. проверка (см. отчёт): band_frac_*, spectral_slope, skewness,
centroid_hz УЖЕ инвариантны к громкости — постоянный gain сокращается в
отношениях энергий и в статистиках нормированного распределения
(эмпирически подтверждено на синтетике, разница ~1e-11..1e-17, машинный
ноль). Физическая перенормировка к -18 LUFS и пересчёт этих метрик дал
бы идентичные числа — не делаем лишнюю работу. То, что ДЕЙСТВИТЕЛЬНО
нужно — профиль динамики как таковой, и прямая проверка: коррелирует ли
разница spectral_slope между записями с разницей PLR (косвенный признак
разной степени лимитирования, которая MIGHT влиять на спектр нелинейно,
не через gain)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

TARGETS = [
    ("основной трек", "demo", None, "KP demo"),
    ("основной трек", "mix", "v2", "KP v2"),
    ("основной трек", "mix", "v3", "KP v3"),
    ("основной трек", "mix", "v4", "KP v4"),
    ("основной трек", "mix", "v5", "KP v5"),
    ("основной трек", "mix", "v6", "KP v6"),
    ("основной трек", "mix", "v7", "KP v7"),
    ("референс А", "reference", None, "референс А"),
    ("референс Б", "demo", None, "референс Б"),
]


def median_psr(path):
    safe_name = path.replace("/", "__")
    f = METRICS_DIR / f"{safe_name}.4_1_short_term.parquet"
    if not f.exists():
        return np.nan
    df = pd.read_parquet(f, columns=["psr"])
    return float(df.psr.median()) if len(df) else np.nan


def main():
    d1 = pd.read_parquet(METRICS_DIR / "4_1_summary.parquet")
    d2 = pd.read_parquet(METRICS_DIR / "4_2_summary.parquet")

    rows = []
    for song, role, ver, label in TARGETS:
        s = d1[(d1.song == song) & (d1.role == role)]
        if ver is not None:
            s = s[s.version == ver]
        if len(s) == 0:
            continue
        r = s.iloc[0]
        s2 = d2[(d2.song == song) & (d2.role == role)]
        if ver is not None and "version" in s2.columns:
            s2 = s2[s2.version == ver]
        slope = float(s2.iloc[0].spectral_slope_db_per_oct) if len(s2) else np.nan
        rows.append(dict(
            label=label, integrated_lufs=r.integrated_lufs, lra=r.lra, plr=r.plr,
            psr_median=median_psr(r.path), crest_factor_db=r.crest_factor_db,
            dr_tt=r.dr_tt, spectral_slope_db_per_oct=slope,
        ))

    profile = pd.DataFrame(rows)
    profile.to_csv(OUT / "dynamics_profile.csv", index=False)
    print(profile.to_string(index=False))

    kp7 = profile[profile.label == "KP v7"].iloc[0]
    others = profile[profile.label.isin(["референс А", "референс Б", "KP demo"])]
    print("\n=== Проверка конфаунда: Δspectral_slope vs ΔPLR относительно v7 ===")
    check = others.copy()
    check["d_slope"] = check.spectral_slope_db_per_oct - kp7.spectral_slope_db_per_oct
    check["d_plr"] = check.plr - kp7.plr
    print(check[["label", "d_slope", "d_plr"]].to_string(index=False))
    if len(check) >= 3:
        corr = check["d_slope"].corr(check["d_plr"])
        print(f"\nКорреляция Δslope vs ΔPLR (n={len(check)}): {corr:.3f}")


if __name__ == "__main__":
    main()
