"""Шаг 8 ТЗ-01 (§7 в документе): сшивка метрик с текстовыми правками.

1. На каждую правку из revisions.csv (с таймкодом) — окно ±2с, метрики
   версии "до" и версии "после".
2. Таблица правка -> метрики до -> метрики после -> дельта.
3. Тест Манна-Уитни: отличаются ли окна с жалобами от случайных окон той
   же длины (по тем же файлам) — с поправкой Бенджамини-Хохберга и
   размером эффекта (rank-biserial), не только p-value.
4. По раундам (v2->v3, v3->v4, ...): для типов претензий, где есть чёткая
   метрика-прокси (громкость -> LUFS, тембр -> spectral centroid),
   проверяем, поменялось ли в запрошенную сторону. Остальные типы
   (пространство/тюн/динамика/аранжировка) намеренно НЕ получают вердикт
   "исправлено/нет" — прямой метрики-прокси под них в этом наборе нет,
   докладывать псевдо-точность хуже, чем честно промолчать.

Ограничение (обязательно к отчёту, п. ТЗ): ~40 наблюдений на одной песне,
всё найденное — гипотеза, требует проверки на «референс А»/референс Б (Шаг 9)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.legacy_scripts.diff.run_6_sections import CONTINUOUS_SPECS, EVENT_SPECS, FILES

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "stitch"
WINDOW_S = 2.0
RNG = np.random.default_rng(42)
N_RANDOM_PER_FILE = 200  # для статистической мощности; длина окна та же (WINDOW_S), не размер выборки

# тип_претензии -> (тег блока, имя метрики, множитель для "направление=больше")
# множитель +1 значит "больше по тексту" = "метрика должна вырасти";
# -1 значит обратная зависимость (например, тембр "больше бархата/тепла"
# физически означает, что spectral centroid должен УПАСТЬ)
DIRECTIONAL_PROXIES = {
    "громкость": [("4.1", "lufs_short_term", +1)],
    "тембр": [("4.2", "centroid_hz", -1)],
}


def file_duration(version):
    path = FILES.get(version)
    if not path:
        return None
    f = METRICS_DIR / f"{path.replace('/', '__')}.4_2_moments.parquet"
    if not f.exists():
        return None
    return float(pd.read_parquet(f, columns=["t_s"]).t_s.max())


def extract_window(version, t_center, window=WINDOW_S):
    """Все метрики (continuous+event) в окне [t-window, t+window] для версии."""
    path = FILES.get(version)
    if not path:
        return {}
    safe_name = path.replace("/", "__")
    out = {}
    for tag, suffix, time_col, cols, rename in CONTINUOUS_SPECS + EVENT_SPECS:
        f = METRICS_DIR / f"{safe_name}.{suffix}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if rename:
            df = df.rename(columns=rename)
        if time_col not in df.columns:
            continue
        mask = (df[time_col] >= t_center - window) & (df[time_col] <= t_center + window)
        sub = df[mask]
        for col in cols:
            if col not in sub.columns:
                continue
            v = sub[col].dropna()
            out[f"{tag}::{col}"] = float(v.median()) if len(v) else np.nan
    return out


def build_windows_table(revisions):
    rows = []
    timed = revisions[revisions.таймкод_начало_с.notna()].copy()
    for idx, r in timed.iterrows():
        before = extract_window(r["версия"], r["таймкод_начало_с"])
        after = extract_window(r["версия_после"], r["таймкод_начало_с"])
        all_keys = set(before) | set(after)
        for key in all_keys:
            block, metric = key.split("::", 1)
            v_before, v_after = before.get(key), after.get(key)
            delta = (v_after - v_before) if (v_before is not None and v_after is not None
                                              and pd.notna(v_before) and pd.notna(v_after)) else np.nan
            rows.append(dict(
                revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                таймкод_с=r["таймкод_начало_с"], объект=r["объект"], тип_претензии=r["тип_претензии"],
                направление=r["направление"], block=block, metric=metric,
                value_before=v_before, value_after=v_after, delta=delta,
            ))
    return pd.DataFrame(rows)


def random_windows_for_metric(version, block, metric, n, dur):
    """n случайных окон той же длины (WINDOW_S) на этой версии для одной метрики."""
    if dur is None or dur <= 2 * WINDOW_S + 1:
        return np.array([])
    path = FILES.get(version)
    safe_name = path.replace("/", "__")
    spec = next((s for s in CONTINUOUS_SPECS + EVENT_SPECS if s[0] == block and metric in s[3]), None)
    if spec is None:
        return np.array([])
    _, suffix, time_col, cols, rename = spec
    f = METRICS_DIR / f"{safe_name}.{suffix}.parquet"
    if not f.exists():
        return np.array([])
    df = pd.read_parquet(f)
    if rename:
        df = df.rename(columns=rename)
    ts = RNG.uniform(WINDOW_S, dur - WINDOW_S, size=n)
    vals = []
    for t in ts:
        mask = (df[time_col] >= t - WINDOW_S) & (df[time_col] <= t + WINDOW_S)
        v = df.loc[mask, metric].dropna()
        if len(v):
            vals.append(float(v.median()))
    return np.array(vals)


def rank_biserial(u_stat, n1, n2):
    return float(1 - 2 * u_stat / (n1 * n2))


def benjamini_hochberg(pvals):
    """Поправка Бенджамини-Хохберга. Возвращает скорректированные p-value."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    return out


def complained_vs_random(windows_table, durations):
    """§7.4: сравнить распределение метрики в окнах-с-жалобами против
    случайных окон той же длины на тех же файлах."""
    results = []
    for (block, metric), g in windows_table.groupby(["block", "metric"]):
        complained = g["value_before"].dropna().to_numpy()
        if len(complained) < 5:
            continue
        random_vals = []
        for version in g["версия"].unique():
            dur = durations.get(version)
            random_vals.append(random_windows_for_metric(version, block, metric, N_RANDOM_PER_FILE, dur))
        random_vals = np.concatenate([v for v in random_vals if len(v)]) if random_vals else np.array([])
        if len(random_vals) < 20:
            continue
        try:
            u_stat, p = mannwhitneyu(complained, random_vals, alternative="two-sided")
        except ValueError:
            continue
        eff = rank_biserial(u_stat, len(complained), len(random_vals))
        results.append(dict(
            block=block, metric=metric, n_complained=len(complained), n_random=len(random_vals),
            median_complained=float(np.median(complained)), median_random=float(np.median(random_vals)),
            u_stat=float(u_stat), p_value=float(p), effect_size_rank_biserial=eff,
        ))
    df = pd.DataFrame(results)
    if len(df):
        df["p_adj_bh"] = benjamini_hochberg(df.p_value.to_numpy())
        df = df.sort_values("p_adj_bh")
    return df


def directional_check(windows_table, revisions):
    """§7.3/7.5: для громкость/тембр — сместилась ли метрика в запрошенную
    сторону. Остальные типы претензий пропускаем — нет чистой метрики-прокси."""
    rows = []
    for idx, r in revisions[revisions.таймкод_начало_с.notna()].iterrows():
        typ, direction = r["тип_претензии"], r["направление"]
        if typ not in DIRECTIONAL_PROXIES or direction not in ("больше", "меньше"):
            continue
        want_sign = 1 if direction == "больше" else -1
        for block, metric, mult in DIRECTIONAL_PROXIES[typ]:
            sub = windows_table[(windows_table.revision_idx == idx) &
                                 (windows_table.block == block) & (windows_table.metric == metric)]
            if len(sub) == 0 or pd.isna(sub.delta.iloc[0]):
                continue
            delta = sub.delta.iloc[0]
            moved_as_requested = bool(np.sign(delta * mult) == want_sign) if delta != 0 else False
            rows.append(dict(
                revision_idx=idx, версия=r["версия"], версия_после=r["версия_после"],
                объект=r["объект"], тип_претензии=typ, направление=direction,
                block=block, metric=metric, delta=delta, moved_as_requested=moved_as_requested,
            ))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    revisions = pd.read_csv(ROOT / "_analysis" / "revisions.csv")

    print(f"Правок всего: {len(revisions)}, с таймкодом: {revisions.таймкод_начало_с.notna().sum()}")

    windows_table = build_windows_table(revisions)
    windows_table.to_parquet(OUT / "revisions_windows.parquet", index=False)
    windows_table.to_csv(OUT / "revisions_windows.csv", index=False)
    print(f"Таблица правка->метрики: {len(windows_table)} строк -> {OUT / 'revisions_windows.parquet'}")

    durations = {v: file_duration(v) for v in FILES}

    print("\n=== §7.4: жалобы vs случайные окна (Манн-Уитни + Бенджамини-Хохберг) ===")
    cvr = complained_vs_random(windows_table, durations)
    cvr.to_parquet(OUT / "complained_vs_random.parquet", index=False)
    cvr.to_csv(OUT / "complained_vs_random.csv", index=False)
    sig = cvr[cvr.p_adj_bh < 0.05]
    print(f"Метрик проверено: {len(cvr)}, значимых после поправки (p_adj<0.05): {len(sig)}")
    if len(sig):
        print(sig[["block", "metric", "n_complained", "n_random", "median_complained",
                    "median_random", "p_adj_bh", "effect_size_rank_biserial"]].to_string(index=False))
    else:
        print("Ни одна метрика не прошла поправку на множественные сравнения — "
              "см. полную таблицу без фильтра для метрик с наименьшим p_adj (гипотезы, не выводы).")
        print(cvr.sort_values("p_adj_bh").head(10)[
            ["block", "metric", "n_complained", "p_value", "p_adj_bh", "effect_size_rank_biserial"]
        ].to_string(index=False))

    print("\n=== §7.5: по раундам — сдвинулось ли в запрошенную сторону (только громкость/тембр) ===")
    dc = directional_check(windows_table, revisions)
    dc.to_parquet(OUT / "directional_check.parquet", index=False)
    dc.to_csv(OUT / "directional_check.csv", index=False)
    if len(dc):
        by_round = dc.groupby(["версия", "версия_после"]).agg(
            n=("moved_as_requested", "size"), исправлено=("moved_as_requested", "sum"))
        by_round["доля"] = (by_round["исправлено"] / by_round["n"]).round(2)
        print(by_round.to_string())
        print(f"\nВсего проверяемых (громкость/тембр): {len(dc)}, "
              f"исправлено в нужную сторону: {int(dc.moved_as_requested.sum())} "
              f"({dc.moved_as_requested.mean() * 100:.0f}%)")
    else:
        print("Нет строк для директивной проверки.")


if __name__ == "__main__":
    main()
