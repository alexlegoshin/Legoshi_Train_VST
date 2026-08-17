"""Прогон §4.6. Вокальные стемы (по имени файла) + миксы/референс (для
сравнения тюн-артефактов на полном миксе — менее надёжно из-за полифонии,
но это ровно то сравнение, которое просил автор).

НЕ включено в этот прогон (см. отдельные задачи в таск-трекере):
форманты (LPC), дыхания, сибилянты, отношение согласных/гласных."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.pitch_vocal import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_6"


def is_vocal(path):
    p = path.lower()
    return "вокал" in p or "vocals" in p  # demucs-стемы «референс А» назван по-английски


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
    targets = reg[reg.path.apply(is_vocal) | reg.role.isin(["mix", "reference", "demo"])]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for _, row in tqdm(targets.iterrows(), total=len(targets)):
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
            for key in ("f0", "notes", "transitions", "flats", "jumps"):
                if len(frames[key]):
                    frames[key].to_parquet(OUT / f"{safe_name}.4_6_{key}.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_6_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_6_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: вокал основной (стем) vs миксы — тюн-артефакты ===")
    kp = df[(df.song == "основной трек") &
            (df.path.str.contains("вокал основной") | df.role.isin(["mix"]))]
    cols = ["path", "version", "n_flat_tune_segments", "n_fast_jumps",
            "mean_abs_intonation_deviation_cents", "vibrato_depth_cents_median"]
    print(kp[cols].to_string(index=False))


if __name__ == "__main__":
    main()
