"""Шаг 11f: корреляция глубоких вокальных метрик (форманты, реальное
per-версийное вибрато/интонация, признаки тюна, harshness) с ratings.csv.

Внутри трёх групп отдельно — каждая confound-free сама по себе (одна
песня, версии/моменты времени различаются, песня не меняется):
  - «КП миксы инженера сведения» (n~20) — сравнение МЕЖДУ версиями одной песни.
  - «референс А» (n~15) и «референс Б» (n~12) — сравнение МЕЖДУ моментами
    внутри одной песни (нет вариации версий, зато нет и межпесенного
    confound'а вовсе).

n_tune_jumps исключён из выводов: на Demucs-разделённом вокале детектор
даёт ~1500-1900 "скачков" за 200-секундный файл — это заведомо шум
F0-трекера на артефактах разделения, а не реальный автотюн (физически
невозможная частота артефактов для живого пения). n_tune_flats оставлен —
на порядок меньше и правдоподобнее, но тоже не подтверждён независимо."""
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

DEEP_METRICS = ["voiced_fraction", "vibrato_depth_cents", "intonation_dev_cents",
                 "n_tune_flats", "formant_f1_hz", "formant_f2_hz", "formant_f3_hz", "harshness"]
AXES = ["wet_dry", "slick_velvet", "warm_cold", "alive_sterile", "dense_empty",
        "close_far", "clear_smeared", "professional", "like_numeric"]
GROUPS = ["КП миксы инженера сведения", "референс А", "референс Б"]


def main():
    df = pd.read_csv(OUT_DIR / "fragments_with_ratings_and_metrics.csv")
    sel = pd.read_parquet(Q_DIR / "fragments_selected.parquet")[["fragment_id", "t_start"]]
    df = df.merge(sel, on="fragment_id", how="left")

    deep = pd.read_parquet(DIFF_DIR / "vocal_deep_windows.parquet")
    deep_r = deep.rename(columns={m: f"deep_{m}" for m in DEEP_METRICS})
    dcols = [f"deep_{m}" for m in DEEP_METRICS]

    merged = df.merge(deep_r[["version", "t_start"] + dcols], on=["version", "t_start"], how="left")
    merged.to_csv(OUT_DIR / "fragments_with_vocal_deep_metrics.csv", index=False)

    ceiling = pd.read_csv(OUT_DIR / "reliability_ceiling.csv").set_index("axis")["rho_ceiling"]

    all_rows = []
    for grp in GROUPS:
        sub_grp = merged[merged.group == grp]
        print(f"\n{'='*20} {grp} (n={len(sub_grp)}, с формантами: {sub_grp[dcols[0]].notna().sum()}) {'='*20}")
        for feat in dcols:
            for ax in AXES:
                s = sub_grp[[feat, ax]].dropna()
                if len(s) < 6 or s[feat].nunique() < 3:
                    continue
                rho, p = stats.spearmanr(s[feat], s[ax])
                disatt = rho / np.sqrt(ceiling.get(ax, np.nan)) if ceiling.get(ax, 0) > 0 else np.nan
                disatt = float(np.clip(disatt, -1, 1)) if np.isfinite(disatt) else np.nan
                all_rows.append(dict(group=grp, metric=feat, axis=ax, rho=rho, p=p, n=len(s),
                                      rho_disattenuated=disatt))

    result = pd.DataFrame(all_rows)
    result.to_csv(OUT_DIR / "vocal_deep_correlation.csv", index=False)

    result["abs_rho"] = result.rho.abs()
    for grp in GROUPS:
        print(f"\n{'='*20} {grp}: топ-8 по |rho| (все оси, метрики) {'='*20}")
        sub = result[result.group == grp].sort_values("abs_rho", ascending=False)
        print(sub.head(8)[["metric", "axis", "rho", "rho_disattenuated", "p", "n"]].to_string(index=False))

    sig = result[result.p < 0.05].sort_values("p")
    print(f"\n{'='*20} Всё, что прошло p<0.05 (без поправки на множественные сравнения!) {'='*20}")
    print(sig[["group", "metric", "axis", "rho", "p", "n"]].to_string(index=False) if len(sig) else "ничего")

    print(f"\nСохранено в {OUT_DIR}")


if __name__ == "__main__":
    main()
