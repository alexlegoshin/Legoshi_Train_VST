"""Прогон §4.3. Quick (стационарные) — весь корпус. Full (покадровые) —
только mix/reference/demo (сравнение версий), не тейки/стемы — иначе часы."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from analysis.metrics.psychoacoustic import quick_metrics, full_timevarying

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE_Q = ROOT / "cache" / "4_3_quick"
CACHE_F = ROOT / "cache" / "4_3_full"
CORE_ROLES = {"mix", "reference", "demo"}


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_Q.mkdir(parents=True, exist_ok=True)
    CACHE_F.mkdir(parents=True, exist_ok=True)

    quick_rows, errors = [], []
    print("=== Quick (стационарные), весь корпус ===")
    for _, row in tqdm(reg.iterrows(), total=len(reg)):
        cache_file = CACHE_Q / f"{row.md5}.json"
        if cache_file.exists():
            m = json.loads(cache_file.read_text())
        else:
            try:
                data, sr = sf.read(str(ROOT / row.path), dtype="float64", always_2d=True)
                mono = data.mean(axis=1)
                m = quick_metrics(mono, sr)
            except Exception as e:
                errors.append((row.path, "quick", str(e)))
                continue
            cache_file.write_text(json.dumps(m, indent=2, ensure_ascii=False))
        m = dict(m, path=row.path, song=row.song, role=row.role, version=row.version)
        quick_rows.append(m)

    core = reg[reg.role.isin(CORE_ROLES)]
    print(f"\n=== Full (покадровые), только {CORE_ROLES}: {len(core)} файлов ===")
    full_rows = []
    for _, row in tqdm(core.iterrows(), total=len(core)):
        cache_file = CACHE_F / f"{row.md5}.json"
        safe_name = row.path.replace("/", "__")
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            try:
                data, sr = sf.read(str(ROOT / row.path), dtype="float64", always_2d=True)
                mono = data.mean(axis=1)
                summary, frames = full_timevarying(mono, sr)
            except Exception as e:
                errors.append((row.path, "full", str(e)))
                continue
            cache_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            for key, df in frames.items():
                df.to_parquet(OUT / f"{safe_name}.4_3_{key}.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        full_rows.append(summary)

    pd.DataFrame(quick_rows).to_parquet(OUT / "4_3_quick_summary.parquet", index=False)
    pd.DataFrame(full_rows).to_parquet(OUT / "4_3_full_summary.parquet", index=False)
    print(f"\nГотово: {len(quick_rows)} quick, {len(full_rows)} full")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, kind, e in errors:
            print(f"  - [{kind}] {p}: {e}")

    df = pd.DataFrame(full_rows)
    kp = df[(df.song == "основной трек")]
    print("\n=== основной трек: покадровая психоакустика по версиям ===")
    print(kp[["version", "sharpness_acum_median", "sharpness_acum_p95",
              "roughness_asper_median", "loudness_sone_median"]].sort_values("version").to_string(index=False))


if __name__ == "__main__":
    main()
