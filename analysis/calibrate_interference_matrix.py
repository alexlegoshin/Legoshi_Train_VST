"""Блок 6 (калибровка, roadmap.md): сравнивает предсказанные дельты
(`presets/interference_matrix.json`) с РЕАЛЬНЫМИ сдвигами метрик между
версиями инженера сведения (Шаг 8, `_analysis/stitch/revisions_windows.parquet`
— вне публичного репо, путь передаётся аргументом).

ВАЖНАЯ ОГОВОРКА ПРОВЕНАНСА (найдена при построении этого модуля, не
предполагалась заранее — см. `analysis/legacy_scripts/stitch/run_8_revisions.py`,
`extract_window()` читает `FILES[version]` — путь к ЦЕЛОМУ МИКСУ версии,
не к стему). Колонка «объект» в revisions_windows — это то, о чём человек
ПИСАЛ в тексте правки (вокал/бас/барабаны/...), а НЕ то, какой источник
был измерен: метрика всегда с целого микса. Из-за этого валидное
сравнение возможно ТОЛЬКО для зон с source="mix" — `band_frac_lowmid`
(наша зона — drums), `band_frac_low` (other), `skewness` (bass) совпадают
по ИМЕНИ с легаси-метриками, но физически не сравнимы: легаси-значение
mix-level, наше предсказание — для конкретного стема. Сравнивать их
означало бы ошибку провенанса (mix vs stem), поэтому такие связки
намеренно исключены, а не молча собраны.

Также сознательно НЕ используются тип_претензии="динамика"/"пространство"/
"тюн"/"аранжировка": сам `run_8_revisions.py` (`DIRECTIONAL_PROXIES`) явно
решил, что для них в этом наборе нет чистой метрики-прокси, и не строил
по ним директивную проверку — «докладывать псевдо-точность хуже, чем
честно промолчать» (дословно из его докстринга). Наследуем ту же
дисциплину, не изобретаем за них.

Единственная валидная связка на сегодня: тип_претензии="тембр" +
metric="band_frac_air" (легаси) <-> mix::band_frac_air_median (наша зона)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

DEFAULT_MATRIX = Path(__file__).resolve().parents[1] / "presets" / "interference_matrix.json"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "presets" / "calibration_report.json"

# (legacy_metric -> (наша zone_metric, role)) — только пары, где легаси
# метрика физически измерена на ТОМ ЖЕ источнике (mix), что и наша зона.
# Не добавлять сюда band_frac_lowmid/band_frac_low/skewness — см. докстринг.
VALID_METRIC_LINKS = {
    "band_frac_air": ("band_frac_air_median", "mix"),
}
# run_8_revisions.py сам не строил directional_check для этих типов — та же дисциплина здесь
RELEVANT_CLAIM_TYPES = {"тембр"}
DIRECTION_SIGN = {"больше": 1, "меньше": -1}


def compare(revisions_windows: pd.DataFrame, valid_links: dict = VALID_METRIC_LINKS,
            claim_types: set = RELEVANT_CLAIM_TYPES) -> list:
    """Список валидных наблюдений (только реально сравнимые связки, см.
    докстринг модуля): {legacy_metric, zone_metric, role, revision_idx,
    направление, real_delta, real_sign, expected_sign, request_realized}.
    request_realized — сбылось ли то, что просил человек (сверка сама по
    себе не про Блок 6, но полезный побочный факт)."""
    rows = []
    for legacy_metric, (zone_metric, role) in valid_links.items():
        sub = revisions_windows[
            (revisions_windows["metric"] == legacy_metric)
            & (revisions_windows["тип_претензии"].isin(claim_types))
            & (revisions_windows["направление"].isin(DIRECTION_SIGN))
            & revisions_windows["delta"].notna()
        ]
        for _, r in sub.iterrows():
            expected_sign = DIRECTION_SIGN[r["направление"]]
            real_sign = int(np.sign(r["delta"]))
            rows.append(dict(
                legacy_metric=legacy_metric, zone_metric=zone_metric, role=role,
                revision_idx=int(r["revision_idx"]), направление=r["направление"],
                real_delta=float(r["delta"]), real_sign=real_sign, expected_sign=expected_sign,
                request_realized=(real_sign == expected_sign),
            ))
    return rows


def summarize(rows: list, interference_matrix: dict) -> dict:
    """{move: {agreement_fraction, n}} — доля наблюдений, где ЗНАК
    предсказанной Блоком 6 дельты совпал со знаком РЕАЛЬНОЙ дельты между
    версиями (не с тем, что человек просил — с тем, что реально
    произошло). n маленькое (Шаг 8 — ~40 наблюдений на одну песню) — это
    ориентир, не статистика с претензией на значимость."""
    by_move = {}
    for r in rows:
        key = f"{r['role']}::{r['zone_metric']}"
        for move, metrics in interference_matrix.items():
            pred = metrics.get(key)
            if pred is None:
                continue
            pred_sign = int(np.sign(pred["median_delta"]))
            by_move.setdefault(move, []).append(pred_sign == r["real_sign"])
    return {move: dict(agreement_fraction=float(np.mean(v)), n=len(v)) for move, v in by_move.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revisions-windows", required=True,
                     help="путь к revisions_windows.parquet (Шаг 8, вне публичного репо)")
    ap.add_argument("--interference-matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    revisions_windows = pd.read_parquet(args.revisions_windows)
    interference_matrix = json.loads(Path(args.interference_matrix).read_text(encoding="utf-8"))

    rows = compare(revisions_windows, VALID_METRIC_LINKS, RELEVANT_CLAIM_TYPES)
    summary = summarize(rows, interference_matrix)

    print(f"Валидных наблюдений (провенанс-безопасные связки, см. докстринг): {len(rows)}")
    for row in rows:
        print(f"  правка #{row['revision_idx']}: направление={row['направление']}, "
              f"реальная дельта {row['zone_metric']}={row['real_delta']:+.6f} "
              f"(запрошенное направление {'сбылось' if row['request_realized'] else 'НЕ сбылось'})")
    print()
    for move, s in sorted(summary.items(), key=lambda kv: -kv[1]["agreement_fraction"]):
        print(f"  {move:20s} совпадение по знаку с реальностью: {s['agreement_fraction']:.0%} (n={s['n']})")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(dict(rows=rows, summary=summary), indent=2, ensure_ascii=False, default=str),
                         encoding="utf-8")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
