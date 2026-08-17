"""Прогон §4.4. Только настоящие стерео-файлы (mix/reference/demo) — стемы
дуал-моно, там нечего мерить."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.stereo_space import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_4"
STEREO_ROLES = {"mix", "reference", "demo"}


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna() & reg.role.isin(STEREO_ROLES)]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors, skipped = [], [], []
    for _, row in tqdm(reg.iterrows(), total=len(reg)):
        path = ROOT / row.path
        cache_file = CACHE / f"{row.md5}.json"
        safe_name = row.path.replace("/", "__")
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
            if summary is None:
                skipped.append(row.path)
                continue
        else:
            try:
                summary, frames = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, ensure_ascii=False) if summary else "null")
            if summary is None:
                skipped.append(row.path)
                continue
            frames["by_band"].to_parquet(OUT / f"{safe_name}.4_4_by_band.parquet", index=False)
            frames["blocks"].to_parquet(OUT / f"{safe_name}.4_4_blocks.parquet", index=False)

        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_4_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_4_summary.parquet'}")
    if skipped:
        print(f"Пропущено (не стерео): {skipped}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: стерео-картина по версиям ===")
    kp = df[df.song == "основной трек"]
    cols = ["version", "overall_correlation", "balance_db", "overall_ms_ratio",
            "n_bands_mono_loss_gt3db", "worst_mono_loss_db", "goniometer_axis_ratio"]
    print(kp[cols].sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
