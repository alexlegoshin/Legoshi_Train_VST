"""ТЗ-02 Задача 6: расширить §7.5 (Шаг 8) новыми прокси — пространство,
динамика, тюн — сверх исходных громкость/тембр. Цель: поднять число
проверяемых правок с 20 до ~35."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from analysis.legacy_scripts.diff.run_6_sections import FILES
from analysis.legacy_scripts.stitch.run_8_revisions import (
    WINDOW_S, extract_window, DIRECTIONAL_PROXIES as BASE_PROXIES,
)

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "stitch"

EXTENDED_PROXIES = dict(BASE_PROXIES)
EXTENDED_PROXIES["пространство"] = [("4.5", "drr_db", -1)]  # больше=wetter->DRR down
EXTENDED_PROXIES["динамика"] = [("4.1", "psr", +1)]          # больше punch -> PSR up


def count_flats_in_window(version, t_center, window=WINDOW_S):
    path = FILES.get(version)
    if not path:
        return np.nan
    f = METRICS_DIR / f"{path.replace('/', '__')}.4_6_flats.parquet"
    if not f.exists():
        return np.nan
    df = pd.read_parquet(f)
    mid = (df.t_start + df.t_end) / 2
    return int(((mid >= t_center - window) & (mid <= t_center + window)).sum())


def main():
    revisions = pd.read_csv(ROOT / "_analysis" / "revisions.csv")
    timed = revisions[revisions.таймкод_начало_с.notna()]

    rows = []
    for idx, r in timed.iterrows():
        typ, direction = r["тип_претензии"], r["направление"]
        if direction not in ("больше", "меньше"):
            continue
        want_sign = 1 if direction == "больше" else -1

        if typ in ("громкость", "тембр"):
            for block, metric, mult in EXTENDED_PROXIES[typ]:
                before = extract_window(r["версия"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
                after = extract_window(r["версия_после"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
                if before is None or after is None or pd.isna(before) or pd.isna(after):
                    continue
                delta = after - before
                moved = bool(np.sign(delta * mult) == want_sign) if delta != 0 else False
                rows.append(dict(revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                                  тип_претензии=typ, proxy=f"{block}::{metric}", delta=delta, moved_as_requested=moved))

        elif typ == "пространство":
            block, metric, mult = EXTENDED_PROXIES["пространство"][0]
            before = extract_window(r["версия"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
            after = extract_window(r["версия_после"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
            if before is None or after is None or pd.isna(before) or pd.isna(after):
                continue
            delta = after - before
            moved = bool(np.sign(delta * mult) == want_sign) if delta != 0 else False
            rows.append(dict(revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                              тип_претензии=typ, proxy=f"{block}::{metric}", delta=delta, moved_as_requested=moved))

        elif typ == "динамика":
            block, metric, mult = EXTENDED_PROXIES["динамика"][0]
            before = extract_window(r["версия"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
            after = extract_window(r["версия_после"], r["таймкод_начало_с"]).get(f"{block}::{metric}")
            if before is None or after is None or pd.isna(before) or pd.isna(after):
                continue
            delta = after - before
            moved = bool(np.sign(delta * mult) == want_sign) if delta != 0 else False
            rows.append(dict(revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                              тип_претензии=typ, proxy=f"{block}::{metric}", delta=delta, moved_as_requested=moved))

        elif typ == "тюн":
            before_n = count_flats_in_window(r["версия"], r["таймкод_начало_с"])
            after_n = count_flats_in_window(r["версия_после"], r["таймкод_начало_с"])
            if pd.isna(before_n) or pd.isna(after_n):
                continue
            delta = after_n - before_n
            # направление "меньше" тюна -> хотим МЕНЬШЕ плоских сегментов после
            want = -1 if direction == "меньше" else 1
            moved = bool(np.sign(delta) == want) if delta != 0 else False
            rows.append(dict(revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                              тип_претензии=typ, proxy="4.6::n_flats_in_window", delta=delta, moved_as_requested=moved))

    dc = pd.DataFrame(rows)
    dc.to_parquet(OUT / "directional_check_extended.parquet", index=False)
    dc.to_csv(OUT / "directional_check_extended.csv", index=False)

    print(f"Проверяемых правок: {len(dc)} (было 20)")
    print("\n=== По типу претензии ===")
    print(dc.groupby("тип_претензии").agg(n=("moved_as_requested", "size"),
                                           исправлено=("moved_as_requested", "sum")).to_string())
    print("\n=== По раундам ===")
    by_round = dc.groupby(["версия", "версия_после"]).agg(
        n=("moved_as_requested", "size"), исправлено=("moved_as_requested", "sum"))
    by_round["доля"] = (by_round["исправлено"] / by_round["n"]).round(2)
    print(by_round.to_string())
    print(f"\nИтого: {int(dc.moved_as_requested.sum())}/{len(dc)} "
          f"({dc.moved_as_requested.mean()*100:.0f}%)")


if __name__ == "__main__":
    main()
