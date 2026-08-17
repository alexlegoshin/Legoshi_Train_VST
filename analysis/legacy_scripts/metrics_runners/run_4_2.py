"""Прогон §4.2 по корпусу. Кэш по md5, как в 4.1."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.spectral import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_2"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    summaries, errors = [], []
    for _, row in tqdm(reg.iterrows(), total=len(reg)):
        path = ROOT / row.path
        cache_file = CACHE / f"{row.md5}.json"
        safe_name = row.path.replace("/", "__")
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            try:
                summary, frames = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            frames["ltas_13"].to_parquet(OUT / f"{safe_name}.4_2_ltas13.parquet", index=False)
            frames["ltas_16"].to_parquet(OUT / f"{safe_name}.4_2_ltas16.parquet", index=False)
            frames["resonances"].to_parquet(OUT / f"{safe_name}.4_2_resonances.parquet", index=False)
            frames["moments"].to_parquet(OUT / f"{safe_name}.4_2_moments.parquet", index=False)
            if len(frames["mfcc"]):
                frames["mfcc"].to_parquet(OUT / f"{safe_name}.4_2_mfcc.parquet", index=False)

        summary = dict(summary)
        summary.update(path=row.path, song=row.song, role=row.role, version=row.version)
        summaries.append(summary)

    df = pd.DataFrame(summaries)
    df.to_parquet(OUT / "4_2_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_2_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: миксы инженера сведения, спектральная форма ===")
    kp = df[(df.song == "основной трек") & (df.role.isin(["mix", "demo"]))]
    cols = ["path", "version", "spectral_slope_db_per_oct", "centroid_hz_median",
            "n_resonances", "top_resonance_freq_hz", "top_resonance_amp_db",
            "band_frac_mud_median", "band_frac_sibilance_median", "band_frac_air_median"]
    print(kp[cols].sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
