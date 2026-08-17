"""Прогон довеска THD/чёт-нечет к §4.2 (задача #24). Переиспользует
закэшированные ноты §4.6 (`_analysis/metrics/*.4_6_notes.parquet`) — те же
файлы, что и там: путь с "вокал" (монофонический источник, метрике можно
доверять напрямую) плюс миксы/референс инженера сведения (F0-трекер там в основном
ловит ведущий вокал, но на частоте k*f0 неизбежно есть чужая энергия —
то же ограничение, что и у тюн-артефактов в run_4_6.py, здесь размечено
явным столбцом is_monophonic_source, а не спрятано).

Результат мёржится в 4_2_summary.parquet по path (left join) — остальные
файлы корпуса (стемы драмсов/гитар/баса, где нот по F0 не искали) получают
NaN в новых столбцах, это честно, не пытаемся туда что-то досчитать.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
from tqdm import tqdm

from analysis.metrics.harmonics import analyze_file, is_vocal

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_2_harmonics"


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna()]
    targets = reg[reg.path.apply(is_vocal) | reg.role.isin(["mix", "reference"])]
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors, skipped = [], [], []
    for _, row in tqdm(targets.iterrows(), total=len(targets)):
        safe_name = row.path.replace("/", "__")
        notes_path = OUT / f"{safe_name}.4_6_notes.parquet"
        if not notes_path.exists():
            skipped.append(row.path)
            continue
        notes_df = pd.read_parquet(notes_path)
        if len(notes_df) == 0:
            summary = dict(n_notes_for_thd=0, thd_median=float("nan"),
                            thd_mean=float("nan"), even_odd_ratio_median=float("nan"))
        else:
            cache_file = CACHE / f"{row.md5}.json"
            if cache_file.exists():
                summary = json.loads(cache_file.read_text())
            else:
                try:
                    summary, per_note = analyze_file(ROOT / row.path, notes_df)
                except Exception as e:
                    errors.append((row.path, str(e)))
                    continue
                cache_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
                if len(per_note):
                    per_note.to_parquet(OUT / f"{safe_name}.4_2_harmonics_notes.parquet", index=False)

        summary = dict(summary, path=row.path, is_monophonic_source=bool(is_vocal(row.path)))
        rows.append(summary)

    new_cols = pd.DataFrame(rows)
    print(f"Посчитано: {len(new_cols)} файлов, пропущено (нет кэша §4.6): {len(skipped)}")
    if skipped:
        for p in skipped:
            print(f"  - пропущен: {p}")
    if errors:
        print(f"\nОШИБКИ ({len(errors)}):")
        for p, e in errors:
            print(f"  - {p}: {e}")

    main_summary = pd.read_parquet(OUT / "4_2_summary.parquet")
    merge_cols = ["path", "n_notes_for_thd", "thd_median", "thd_mean",
                  "even_odd_ratio_median", "is_monophonic_source"]
    for c in merge_cols:
        if c != "path" and c in main_summary.columns:
            main_summary = main_summary.drop(columns=[c])
    merged = main_summary.merge(new_cols[merge_cols], on="path", how="left")
    merged.to_parquet(OUT / "4_2_summary.parquet", index=False)
    print(f"\nОбновлён -> {OUT / '4_2_summary.parquet'} ({len(merged)} строк)")

    print("\n=== основной трек: THD/чёт-нечет по вокал основной + миксам ===")
    kp = merged[(merged.song == "основной трек") &
                (merged.path.str.contains("вокал основной") | merged.role.isin(["mix"]))]
    cols = ["path", "version", "is_monophonic_source", "n_notes_for_thd",
            "thd_median", "even_odd_ratio_median"]
    print(kp[cols].sort_values(["is_monophonic_source", "version"], ascending=[False, True]).to_string(index=False))

    print("\n=== «референс А» вокал: THD (открытый вопрос про финал трека) ===")
    rad = merged[merged.path.str.contains("референс А") & merged.path.str.contains("вокал")]
    print(rad[["path", "n_notes_for_thd", "thd_median", "even_odd_ratio_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
