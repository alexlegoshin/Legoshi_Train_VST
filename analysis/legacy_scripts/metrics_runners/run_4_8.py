"""Прогон §4.8. Вокальная группа (ТЗ просит её отдельно: "маскирование
самое злое ... жалобы «бэк не слышно»") + полная группа из всех
выровненных стемов «основной трек» (задача #27, было 4 вокальных, стало
9 из 12 — back high/соло/электро бэк остаются исключены, они не выровнены,
см. _analysis/alignment.parquet и печать ниже).

ИСПРАВЛЕНО по пути (см. докстринг masking_erb.analyze_group): раньше
дорожки грузились без выравнивания вообще — offset применяется здесь
через ту же pipeline.deconv.run_5.load_aligned_stems, что и в §5, не
дублируем логику."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from analysis.legacy_scripts.deconv.run_5 import load_aligned_stems
from analysis.metrics.masking_erb import analyze_group

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"


def run_and_save(track_mono, sr, tag):
    res = analyze_group(track_mono, sr)
    aud_df = pd.DataFrame([{"track": k, "audibility": v} for k, v in res["audibility"].items()])
    aud_df.to_parquet(OUT / f"4_8_{tag}_audibility.parquet", index=False)
    res["attribution"].to_parquet(OUT / f"4_8_{tag}_attribution.parquet", index=False)

    print(f"\n=== {tag}: audibility (доля время-частотных ячеек, где дорожка слышна) ===")
    print(aud_df.sort_values("audibility").to_string(index=False))
    print(f"\n=== {tag}: матрица атрибуции (кто кого маскирует, доля замаскированных клеток) ===")
    print(res["attribution"].sort_values("fraction_of_masked_cells", ascending=False).to_string(index=False))
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Загрузка и выравнивание всех стемов «основной трек» ===")
    stems_L, stems_R, sr = load_aligned_stems("основной трек")

    vocal_group = {k: v for k, v in stems_L.items() if "вокал" in k.lower()}
    print(f"\nВокальная группа: {list(vocal_group.keys())}")
    run_and_save(vocal_group, sr, "vocal")

    print(f"\nПолная группа (все выровненные стемы): {list(stems_L.keys())}")
    run_and_save(stems_L, sr, "full")


if __name__ == "__main__":
    main()
