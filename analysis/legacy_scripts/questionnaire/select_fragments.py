"""Шаг 10 §10.1-10.2: отбор фрагментов для опросника.

Состав корпуса (доля от ~90 уникальных фрагментов):
  КП v2-v7 ~30%, демка КП ~20%, референс А ~20%, референс Б ~15%, контрольный трек ~15%

Стратификация: внутри КАЖДОГО источника (не по всему корпусу разом —
иначе экстремумы по абсолютному уровню метрики просто съедутся к одному
источнику с самым широким диапазоном, а не дадут разнообразия внутри
каждой песни) — по своим верхним/нижним экстремумам 8 метрик Шага 6,
плюс все таймкоды правок (revisions.csv, только там, где есть готовая
метка "жалоба"/направление), плюс случайные контрольные точки.

PLR/DRR НЕ участвуют в пооконной стратификации — они не покадровые
величины в текущем наборе (PLR целофайловый, DRR событийный по onset'ам
§4.5), различие явно задокументировано, не скрыто."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DIFF_DIR = ROOT / "_analysis" / "diff"
OUT_DIR = ROOT / "_analysis" / "questionnaire"

RNG = np.random.default_rng(2026)

SOURCE_FILES = {
    "demo": "основной трек/ТА/основной трек track out/демка_аранж_основной_трек.wav",
    "v2": "основной трек/-/версия сведения 2.wav",
    "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav",
    "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav",
    "v7": "основной трек/-/версия сведения 7.wav",
    "референс А": "референс А/+/1 референс А.mp3",
    "референс Б": "Песня в поддержку рака лёгких/+ но это моя грязная демка/референс Б - 27:4:2026, 18.58.wav",
    "контрольный трек": "контрольный трек/ТА/финальная/фин.mp3",
}
KP_MIX_VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]

# группа источника -> версии внутри неё + целевая доля от общего бюджета
SOURCE_GROUPS = {
    "КП миксы инженера сведения": (KP_MIX_VERSIONS, 0.30),
    "КП демка": (["demo"], 0.20),
    "референс А": (["референс А"], 0.20),
    "референс Б": (["референс Б"], 0.15),
    "контрольный трек": (["контрольный трек"], 0.15),
}

METRICS = ["band_frac_air", "band_frac_lowmid", "band_frac_low", "skewness",
           "spectral_slope", "warmth_ratio", "vibrato_depth_cents", "voiced_fraction"]

TOTAL_UNIQUE_TARGET = 78  # + до ~15 правок + ~10 случайных = ~90-100 итого слотов на 3 сессии
N_RANDOM_TOTAL = 10
MAX_REVISION_BONUS = 15


def stratified_candidates_for_group(windows, versions, budget, metrics=METRICS):
    """Верхние/нижние экстремумы по каждой метрике внутри группы источников,
    плюс случайные — итеративно набираем до бюджета, чередуя метрики."""
    pool = windows[windows.version.isin(versions)].copy()
    if len(pool) == 0:
        return pd.DataFrame()

    picks = []
    picked_keys = set()

    def try_add(row, reason):
        key = (row["version"], round(row["t_start"], 1))
        if key in picked_keys:
            return False
        picked_keys.add(key)
        picks.append(dict(version=row["version"], t_start=row["t_start"], reason=reason))
        return True

    # раунд-робин по метрикам: top2/bottom2/mid1 за проход, пока не наберём бюджет
    per_metric_top = max(1, budget // (len(metrics) * 2))
    for metric in metrics:
        sub = pool[pool[metric].notna()].sort_values(metric)
        if len(sub) < 5:
            continue
        n = per_metric_top
        for _, row in sub.head(n).iterrows():
            try_add(row, f"{metric}: низ")
        for _, row in sub.tail(n).iterrows():
            try_add(row, f"{metric}: верх")
        mid_idx = len(sub) // 2
        try_add(sub.iloc[mid_idx], f"{metric}: середина")
        if len(picks) >= budget:
            break

    # добор случайными, если не набрали бюджет
    remaining = budget - len(picks)
    if remaining > 0:
        candidates = pool[~pool.apply(lambda r: (r["version"], round(r["t_start"], 1)) in picked_keys, axis=1)]
        if len(candidates):
            extra = candidates.sample(n=min(remaining, len(candidates)), random_state=RNG.integers(1e6))
            for _, row in extra.iterrows():
                try_add(row, "добор случайным")

    return pd.DataFrame(picks[:budget])


def revision_candidates():
    """Все timecoded правки из revisions.csv — обучающая выборка с уже
    известной меткой (объект/тип/направление)."""
    rev = pd.read_csv(ROOT / "_analysis" / "revisions.csv")
    rev = rev[rev.таймкод_начало_с.notna()]
    rows = []
    for _, r in rev.iterrows():
        rows.append(dict(version=r["версия"], t_start=float(r["таймкод_начало_с"]),
                          reason=f"правка: {r['объект']}/{r['тип_претензии']}/{r['направление']}"))
    return pd.DataFrame(rows)


def main():
    windows = pd.read_parquet(DIFF_DIR / "moving_window_4s1s.parquet")

    all_picks = []
    for group_name, (versions, share) in SOURCE_GROUPS.items():
        budget = round(TOTAL_UNIQUE_TARGET * share)
        cand = stratified_candidates_for_group(windows, versions, budget)
        cand["group"] = group_name
        all_picks.append(cand)
        print(f"{group_name}: бюджет {budget}, набрано {len(cand)}")

    fragments = pd.concat(all_picks, ignore_index=True)

    # правки КП — добавляем поверх бюджета как отдельную обучающую категорию,
    # но с потолком (иначе 42 правки раздувают корпус) — максимизируем
    # разнообразие объекта/типа претензии, не берём подряд
    rev_cand = revision_candidates()
    existing_keys = set(zip(fragments.version, fragments.t_start.round(1)))
    rev_cand = rev_cand[~rev_cand.apply(lambda r: (r["version"], round(r["t_start"], 1)) in existing_keys, axis=1)]
    if len(rev_cand) > MAX_REVISION_BONUS:
        rev_cand = rev_cand.sample(n=MAX_REVISION_BONUS, random_state=RNG.integers(1e6))
    rev_cand["group"] = "правка (обучающая)"
    fragments = pd.concat([fragments, rev_cand], ignore_index=True)

    # случайные контрольные — пропорционально долям источников
    random_rows = []
    for group_name, (versions, share) in SOURCE_GROUPS.items():
        n = round(N_RANDOM_TOTAL * share)
        pool = windows[windows.version.isin(versions)]
        if len(pool) == 0 or n == 0:
            continue
        sample = pool.sample(n=min(n, len(pool)), random_state=RNG.integers(1e6))
        for _, row in sample.iterrows():
            random_rows.append(dict(version=row["version"], t_start=row["t_start"],
                                     reason="случайный контроль", group="случайный контроль"))
    fragments = pd.concat([fragments, pd.DataFrame(random_rows)], ignore_index=True)

    fragments["file"] = fragments.version.map(SOURCE_FILES)
    fragments = fragments.drop_duplicates(subset=["version", "t_start"]).reset_index(drop=True)
    fragments["fragment_id"] = [f"F{idx:03d}" for idx in range(len(fragments))]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fragments.to_parquet(OUT_DIR / "fragments_selected.parquet", index=False)
    fragments.to_csv(OUT_DIR / "fragments_selected.csv", index=False)

    print(f"\nВсего уникальных фрагментов: {len(fragments)}")
    print(fragments.groupby("group").size().to_string())
    print("\nПо источнику (version):")
    print(fragments.version.value_counts().to_string())


if __name__ == "__main__":
    main()
