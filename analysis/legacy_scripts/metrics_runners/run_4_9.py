"""Прогон §4.9. Диссонанс/roughness/аномалии — на mix/reference/demo (сравнение
версий). Стабильность темпа по секциям — бесплатно переиспользует
beats.parquet + sections.csv из шага 3, без новой тяжёлой обработки.

Chroma/тональность/аккорды по секциям и расстройка вокал-бас считаются
отдельно в `harmony_key.py` / `run_4_9_key.py` (задача #28, закрыта) и
вмёржены в 4_9_summary.parquet колонками key_root/key_mode/key_correlation."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.metrics.harmony_dissonance import analyze_file

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_9"
ROLES = {"mix", "reference", "demo"}


def tempo_stability_by_section():
    beats_path = ROOT / "_analysis" / "beats.parquet"
    sections_path = ROOT / "_analysis" / "sections.csv"
    if not beats_path.exists() or not sections_path.exists():
        return pd.DataFrame()
    beats = pd.read_parquet(beats_path)
    sections = pd.read_csv(sections_path)
    rows = []
    for song in beats.song.unique():
        b = beats[beats.song == song]
        sec = sections[sections.song == song] if "song" in sections.columns else sections
        if len(sec) == 0:
            continue
        for _, s in sec.iterrows():
            mask = (b.time_s >= s.start_s) & (b.time_s < s.end_s)
            local = b[mask].local_bpm.dropna()
            if len(local) < 2:
                continue
            rows.append(dict(song=song, section=s.section, start_s=s.start_s, end_s=s.end_s,
                              n_beats=len(local), local_bpm_median=float(local.median()),
                              local_bpm_std=float(local.std())))
    return pd.DataFrame(rows)


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
                summary, frames = analyze_file(path)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, ensure_ascii=False))
            frames["curve"].to_parquet(OUT / f"{safe_name}.4_9_curve.parquet", index=False)
            frames["anomalies"].to_parquet(OUT / f"{safe_name}.4_9_anomalies.parquet", index=False)
        summary = dict(summary, path=row.path, song=row.song, role=row.role, version=row.version)
        rows.append(summary)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_9_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_9_summary.parquet'}")
    if errors:
        print(f"\nОШИБКИ: {errors}")

    print("\n=== основной трек: диссонанс/roughness/аномалии по версиям ===")
    kp = df[df.song == "основной трек"]
    print(kp[["version", "sethares_dissonance_median", "vassilakis_roughness_median", "n_anomalies"]]
          .sort_values("version").to_string(index=False))

    tempo_df = tempo_stability_by_section()
    if len(tempo_df):
        tempo_df.to_parquet(OUT / "4_9_tempo_stability.parquet", index=False)
        print("\n=== Стабильность темпа по секциям (основной трек) ===")
        kp_tempo = tempo_df[tempo_df.song == "основной трек"]
        print(kp_tempo[["section", "n_beats", "local_bpm_median", "local_bpm_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
