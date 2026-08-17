"""Прогон форманты/дыхания/сибилянты/согл.-гласн. — задача #26. Те же
файлы, что и §4.6 (is_vocal — путь содержит "вокал"/"vocals"): полные
миксы сюда сознательно НЕ включены, LPC-форманты и детекторы дыхания на
полифонии дают мусор (см. докстринг vocal_texture.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.harmonics import is_vocal
from analysis.metrics.vocal_texture import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_6_texture"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
    targets = reg[reg.path.apply(is_vocal)]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors, skipped = [], [], []
    for _, row in tqdm(targets.iterrows(), total=len(targets)):
        safe_name = row.path.replace("/", "__")
        f0_path = OUT / f"{safe_name}.4_6_f0.parquet"
        if not f0_path.exists():
            skipped.append(row.path)
            continue
        cache_file = CACHE / f"{row.md5}.json"
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            f0_df = pd.read_parquet(f0_path)
            try:
                summary, frames = analyze_file(ROOT / row.path, f0_df)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            for key in ("formants", "breaths", "sibilants"):
                if len(frames[key]):
                    frames[key].to_parquet(OUT / f"{safe_name}.4_6_texture_{key}.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    new_cols = pd.DataFrame(rows)
    print(f"Посчитано: {len(new_cols)} файлов, пропущено (нет кэша §4.6 F0): {len(skipped)}")
    for p in skipped:
        print(f"  - пропущен: {p}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    new_cols.to_parquet(OUT / "4_6_texture_summary.parquet", index=False)
    print(f"\n-> {OUT / '4_6_texture_summary.parquet'} ({len(new_cols)} строк)")

    main_summary = pd.read_parquet(OUT / "4_6_summary.parquet")
    merge_cols = ["path", "formant_f1_hz_median", "formant_f2_hz_median", "formant_f3_hz_median",
                  "n_breath_events", "breath_total_duration_s", "n_sibilant_events",
                  "sibilant_rate_per_min", "consonant_vowel_energy_ratio"]
    for c in merge_cols:
        if c != "path" and c in main_summary.columns:
            main_summary = main_summary.drop(columns=[c])
    merged = main_summary.merge(new_cols[merge_cols], on="path", how="left")
    merged.to_parquet(OUT / "4_6_summary.parquet", index=False)
    print(f"Обновлён -> {OUT / '4_6_summary.parquet'} ({len(merged)} строк)")

    print("\n=== основной трек: вокальные стемы — форманты/дыхания/сибилянты ===")
    kp = merged[(merged.song == "основной трек") & merged.path.apply(is_vocal)]
    cols = ["path", "formant_f1_hz_median", "formant_f2_hz_median", "formant_f3_hz_median",
            "n_breath_events", "n_sibilant_events", "consonant_vowel_energy_ratio"]
    print(kp[cols].to_string(index=False))


if __name__ == "__main__":
    main()
