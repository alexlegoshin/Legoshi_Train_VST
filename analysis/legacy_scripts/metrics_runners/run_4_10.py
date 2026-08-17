"""Прогон §4.10 по всему корпусу — дёшево, без внешних библиотек.

Детектор наводок (find_persistent_narrowband) переделан под тихие STFT-
кадры, не весь файл (задача #29, закрыта). После починки на миксах
инженера сведения (v2-v7) устойчивых наводок по-прежнему НЕ находится — но теперь
это содержательный результат, не дыра в методе: 108Гц-пик, который нашёл
§4.2 (LTAS, средний уровень), скорее всего настоящий музыкальный/басовый
резонанс, а не сетевая наводка — будь это сеть, она была бы видна и в
тихих местах, а её там нет. На демке нашёлся один кандидат ~151Гц —
отдельно от 108Гц из §4.2, стоит перепроверить отдельно, не то же самое."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.noise import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_10"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
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
                summary, persistent_df = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, ensure_ascii=False))
            if len(persistent_df):
                persistent_df.to_parquet(OUT / f"{safe_name}.4_10_persistent.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_10_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_10_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    print("\n=== основной трек: наводки и SNR по всем ролям ===")
    kp = df[df.song == "основной трек"]
    cols = ["path", "role", "snr_db", "n_persistent_narrowband", "top_persistent_freq_hz", "top_persistent_stability"]
    print(kp[cols].sort_values("role").to_string(index=False))


if __name__ == "__main__":
    main()
