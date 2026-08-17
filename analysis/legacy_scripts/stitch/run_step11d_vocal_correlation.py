"""Шаг 11d: корреляция метрик изолированного (Demucs) вокала с ratings.csv,
внутри группы «КП миксы инженера сведения» — единственная confound-free группа с
несколькими версиями одной песни. Демка исключена: Demucs-вокал демки
после выравнивания громкости даёт медиану RMS около -100дБFS (см. лог
run_step11c) — это не тихий вокал, а фактически его отсутствие в файле
(демка — «демка_аранж», судя по всему инструментальная/без сведённого
вокала), и 4 окна из ~190 не дают статистики. Не подделывать это включением.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
DIFF_DIR = ROOT / "_analysis" / "diff"
Q_DIR = ROOT / "_analysis" / "questionnaire"
OUT_DIR = ROOT / "_analysis" / "correlation"

VOCAL_METRICS = ["band_frac_lowmid", "band_frac_low", "band_frac_mud", "band_frac_mid",
                  "band_frac_presence", "skewness", "spectral_slope", "warmth_ratio"]
AXES = ["wet_dry", "slick_velvet", "warm_cold", "alive_sterile", "dense_empty",
        "close_far", "clear_smeared", "professional", "like_numeric"]


def main():
    df = pd.read_csv(OUT_DIR / "fragments_with_ratings_and_metrics.csv")
    sel = pd.read_parquet(Q_DIR / "fragments_selected.parquet")[["fragment_id", "t_start"]]
    df = df.merge(sel, on="fragment_id", how="left")

    voc = pd.read_parquet(DIFF_DIR / "vocal_isolated_windows.parquet")
    voc = voc[voc.version != "demo"]  # см. докстринг
    voc_r = voc.rename(columns={m: f"voc_{m}" for m in VOCAL_METRICS})
    vcols = [f"voc_{m}" for m in VOCAL_METRICS]

    merged = df.merge(voc_r[["version", "t_start"] + vcols], on=["version", "t_start"], how="left")
    merged.to_csv(OUT_DIR / "fragments_with_vocal_metrics.csv", index=False)

    kp = merged[merged.group == "КП миксы инженера сведения"].copy()
    print(f"КП миксы инженера сведения: {kp[vcols[0]].notna().sum()}/{len(kp)} фрагментов с вокальными метриками")

    ceiling = pd.read_csv(OUT_DIR / "reliability_ceiling.csv").set_index("axis")["rho_ceiling"]

    rows = []
    for feat in vcols:
        for ax in AXES:
            s = kp[[feat, ax]].dropna()
            if len(s) < 6:
                rows.append(dict(metric=feat, axis=ax, rho=np.nan, p=np.nan, n=len(s)))
                continue
            rho, p = stats.spearmanr(s[feat], s[ax])
            disatt = rho / np.sqrt(ceiling.get(ax, np.nan)) if ceiling.get(ax, 0) > 0 else np.nan
            disatt = float(np.clip(disatt, -1, 1)) if np.isfinite(disatt) else np.nan
            rows.append(dict(metric=feat, axis=ax, rho=rho, p=p, n=len(s), rho_disattenuated=disatt))
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "vocal_correlation_within_kp_martin.csv", index=False)

    print("\n=== Топ-3 вокальных метрики по |rho| для каждой оси (только КП миксы инженера сведения) ===")
    result["abs_rho"] = result.rho.abs()
    for ax in AXES:
        sub = result[result.axis == ax].sort_values("abs_rho", ascending=False)
        print(f"\n{ax}:")
        print(sub.head(3)[["metric", "rho", "rho_disattenuated", "p", "n"]].to_string(index=False))

    print(f"\nСохранено в {OUT_DIR}")


if __name__ == "__main__":
    main()
