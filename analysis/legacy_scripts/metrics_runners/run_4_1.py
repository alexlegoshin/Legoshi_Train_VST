"""Прогон §4.1 по всему реестру. Кэш по md5 — повторный прогон не пересчитывает
то, что не изменилось (§2 ТЗ: "тяжёлые промежуточные результаты в cache/")."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.loudness_dynamics import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_1"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]  # не считаем дубликаты дважды
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    summaries = []
    errors = []
    for _, row in tqdm(reg.iterrows(), total=len(reg)):
        path = ROOT / row.path
        cache_key = row.md5
        cache_file = CACHE / f"{cache_key}.json"
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            try:
                summary, frames = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            safe_name = row.path.replace("/", "__")
            frames["short_term"].to_parquet(OUT / f"{safe_name}.4_1_short_term.parquet", index=False)
            frames["momentary"].to_parquet(OUT / f"{safe_name}.4_1_momentary.parquet", index=False)
            frames["transients"].to_parquet(OUT / f"{safe_name}.4_1_transients.parquet", index=False)

        summary = dict(summary)
        summary.update(path=row.path, song=row.song, role=row.role, version=row.version)
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    df.to_parquet(OUT / "4_1_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_1_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: миксы инженера сведения, интегральная громкость и DR ===")
    kp = df[(df.song == "основной трек") & (df.role.isin(["mix", "demo"]))]
    print(kp[["path", "version", "integrated_lufs", "true_peak_dbfs", "lra", "dr_tt", "crest_factor_db"]]
          .sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
