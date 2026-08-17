"""Прогон §4.5. Начинаю с mix/reference/demo (сравнение версий) — стемы можно
довесить отдельным проходом, если понадобится "какой инструмент мокрый"."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.reverb import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_5"
ROLES = {"mix", "reference", "demo"}


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna() & reg.role.isin(ROLES)]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for _, row in tqdm(reg.iterrows(), total=len(reg)):
        path = ROOT / row.path
        cache_file = CACHE / f"{row.md5}.json"
        safe_name = row.path.replace("/", "__")
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            try:
                summary, df = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, ensure_ascii=False))
            df.to_parquet(OUT / f"{safe_name}.4_5_tails.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_5_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_5_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: реверб по версиям ===")
    kp = df[df.song == "основной трек"]
    cols = ["version", "n_isolated_tails", "rt60_s_median", "edt_s_median",
            "c50_db_median", "drr_db_median", "predelay_s_median"]
    print(kp[cols].sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
