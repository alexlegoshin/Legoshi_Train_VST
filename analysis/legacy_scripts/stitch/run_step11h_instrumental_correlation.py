"""Шаг 11h: корреляция инструментальных метрик (bass/drums/other, §11g) с
ratings.csv, внутри тех же трёх confound-free групп, что и вокал (§11f).
Демка ЗДЕСЬ участвует (в отличие от вокала) — инструментал в ней есть."""
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

INSTR_METRICS = ["band_frac_lowmid", "band_frac_low", "band_frac_mud", "band_frac_mid",
                  "band_frac_presence", "skewness", "spectral_slope", "warmth_ratio", "harshness"]
AXES = ["wet_dry", "slick_velvet", "warm_cold", "alive_sterile", "dense_empty",
        "close_far", "clear_smeared", "professional", "like_numeric"]
GROUPS = ["КП миксы инженера сведения", "КП демка", "референс А", "референс Б"]
STEMS = ["bass", "drums", "other"]


def main():
    df = pd.read_csv(OUT_DIR / "fragments_with_ratings_and_metrics.csv")
    sel = pd.read_parquet(Q_DIR / "fragments_selected.parquet")[["fragment_id", "t_start"]]
    df = df.merge(sel, on="fragment_id", how="left")

    instr = pd.read_parquet(DIFF_DIR / "instrumental_windows.parquet")
    ceiling = pd.read_csv(OUT_DIR / "reliability_ceiling.csv").set_index("axis")["rho_ceiling"]

    all_rows = []
    merged_out = df.copy()
    for stem in STEMS:
        stem_df = instr[instr.stem == stem][["version", "t_start"] + INSTR_METRICS].copy()
        stem_df = stem_df.rename(columns={m: f"{stem}_{m}" for m in INSTR_METRICS})
        merged_out = merged_out.merge(stem_df, on=["version", "t_start"], how="left")

    merged_out.to_csv(OUT_DIR / "fragments_with_instrumental_metrics.csv", index=False)

    for grp in GROUPS:
        sub_grp = merged_out[merged_out.group == grp]
        for stem in STEMS:
            mcols = [f"{stem}_{m}" for m in INSTR_METRICS]
            avail = sub_grp[mcols[0]].notna().sum() if mcols[0] in sub_grp else 0
            if avail < 6:
                continue
            for feat in mcols:
                for ax in AXES:
                    s = sub_grp[[feat, ax]].dropna()
                    if len(s) < 6 or s[feat].nunique() < 3:
                        continue
                    rho, p = stats.spearmanr(s[feat], s[ax])
                    disatt = rho / np.sqrt(ceiling.get(ax, np.nan)) if ceiling.get(ax, 0) > 0 else np.nan
                    disatt = float(np.clip(disatt, -1, 1)) if np.isfinite(disatt) else np.nan
                    all_rows.append(dict(group=grp, stem=stem, metric=feat, axis=ax,
                                          rho=rho, p=p, n=len(s), rho_disattenuated=disatt))

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_DIR / "instrumental_correlation.csv", index=False)
    result["abs_rho"] = result.rho.abs()

    for grp in GROUPS:
        print(f"\n{'='*20} {grp}: топ-10 по |rho| (bass/drums/other, все оси) {'='*20}")
        sub = result[result.group == grp].sort_values("abs_rho", ascending=False)
        print(sub.head(10)[["stem", "metric", "axis", "rho", "rho_disattenuated", "p", "n"]].to_string(index=False))

    sig = result[result.p < 0.05].sort_values("p")
    print(f"\n{'='*20} Всё p<0.05 (без поправки на множественные сравнения, n_tests={len(result)}) {'='*20}")
    print(sig[["group", "stem", "metric", "axis", "rho", "p", "n"]].to_string(index=False) if len(sig) else "ничего")

    print(f"\nСохранено в {OUT_DIR}")


if __name__ == "__main__":
    main()
