"""Прогон §4.7 на реальных парах вокальных дорожек «основной трек», с
использованием сдвигов из alignment.parquet (шаг 2) — сравнение на общей
временной оси, не "как есть в файлах"."""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from analysis.metrics.layering import analyze_pair

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    align = pd.read_parquet(ROOT / "_analysis" / "alignment.parquet")

    vocal_stems = reg[(reg.song == "основной трек") & (reg.role == "stem") &
                       reg.path.str.lower().str.contains("вокал")]
    print(f"Вокальные дорожки: {list(vocal_stems.path.str.split('/').str[-1])}")

    offsets = {}
    for _, r in vocal_stems.iterrows():
        a = align[align.path == r.path]
        offsets[r.path] = float(a.iloc[0].offset_s) if len(a) and a.iloc[0].aligned else 0.0

    rows = []
    pairs = list(itertools.combinations(vocal_stems.path.tolist(), 2))
    print(f"Пар: {len(pairs)}")
    for path_a, path_b in pairs:
        name_a, name_b = path_a.split("/")[-1], path_b.split("/")[-1]
        print(f"  {name_a}  x  {name_b} ...")
        summary = analyze_pair(ROOT / path_a, ROOT / path_b, offsets[path_a], offsets[path_b])
        summary.update(track_a=name_a, track_b=name_b)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_7_pairs.parquet", index=False)
    print(f"\nГотово -> {OUT / '4_7_pairs.parquet'}")
    cols = ["track_a", "track_b", "simultaneity_fraction", "pitch_divergence_cents_median",
            "time_divergence_ms_median", "comb_risk_upper_bound", "mutual_correlation"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
