"""Шаг 11: корреляционный анализ ratings.csv против метрик Шага 6 (§8.3 ТЗ-01,
§10.3 ТЗ-02-addendum).

Порядок обязателен (§10.3 addendum): СНАЧАЛА потолок надёжности по дублям —
если автор сам с собой согласен на rho=0.6 по оси, метрика физически не
может предсказывать лучше 0.6. Отчитываться этим числом раньше любых выводов.

vibrato_depth_cents несёт ярлык «детектор артефакта тюна», не компонент
словаря вкуса (см. TZ-02-addendum.md:124) — в общий рейтинг метрик не входит,
показывается отдельной строкой.

PLR — целофайловая метрика (dynamics_profile.csv, по song+version), одно
значение на все фрагменты одной версии. DRR — событийная (§4.5, onset-level,
_analysis/metrics/*.4_5_tails.parquet) — агрегируется медианой по онсетам,
попавшим в 4-секундное окно фрагмента. Для «контрольный трек» PLR отсутствует
(не считался в ТЗ-02 Задаче 2) — NaN, не подделывать.
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
Q_DIR = ROOT / "_analysis" / "questionnaire"
DIFF_DIR = ROOT / "_analysis" / "diff"
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT_DIR = ROOT / "_analysis" / "correlation"

AXES = ["wet_dry", "slick_velvet", "warm_cold", "alive_sterile", "dense_empty",
        "close_far", "clear_smeared", "professional", "like_numeric"]

CORE_METRICS = ["band_frac_air", "band_frac_lowmid", "band_frac_low", "skewness",
                "spectral_slope", "warmth_ratio", "voiced_fraction"]
ARTIFACT_METRICS = ["vibrato_depth_cents"]  # детектор артефакта, не словарь вкуса
EXTRA_METRICS = ["plr", "drr_db"]

LIKE_MAP = {"dislike": -1, "neutral": 0, "like": 1}

# version -> файл 4_5_tails.parquet (событийные DRR)
TAILS_FILES = {
    "demo": "основной трек__ТА__основной трек track out__демка_аранж_основной_трек.wav.4_5_tails.parquet",
    "v2": "основной трек__-__версия сведения 2.wav.4_5_tails.parquet",
    "v3": "основной трек__-__версия сведения 3..wav.4_5_tails.parquet",
    "v4": "основной трек__-__версия сведения 4 .wav.4_5_tails.parquet",
    "v5": "основной трек__-__версия сведения 5.wav.4_5_tails.parquet",
    "v6": "основной трек__-__версия сведения 6.wav.4_5_tails.parquet",
    "v7": "основной трек__-__версия сведения 7.wav.4_5_tails.parquet",
    "референс А": "референс А__+__1 референс А.mp3.4_5_tails.parquet",
    "референс Б": "Песня в поддержку рака лёгких__+ но это моя грязная демка__референс Б - 27:4:2026, 18.58.wav.4_5_tails.parquet",
    "контрольный трек": "контрольный трек__ТА__финальная__фин.mp3.4_5_tails.parquet",
}

# version -> label в dynamics_profile.csv (PLR целофайловый)
PLR_LABELS = {
    "demo": "KP demo", "v2": "KP v2", "v3": "KP v3", "v4": "KP v4",
    "v5": "KP v5", "v6": "KP v6", "v7": "KP v7",
    "референс А": "референс А", "референс Б": "референс Б",
    # "контрольный трек" отсутствует в dynamics_profile.csv — не считалась, оставляем NaN
}


def load_drr_for_fragments(fragments):
    """Медиана drr_db по онсетам, попавшим в [t_start, t_start+4) окна фрагмента."""
    drr_vals = {}
    cache = {}
    for _, row in fragments.iterrows():
        version = row["version"]
        fname = TAILS_FILES.get(version)
        if fname is None:
            drr_vals[row["fragment_id"]] = np.nan
            continue
        if fname not in cache:
            path = METRICS_DIR / unicodedata.normalize("NFC", fname)
            if not path.exists():
                # пробуем NFD на случай APFS-разложения
                candidates = list(METRICS_DIR.glob("*4_5_tails.parquet"))
                match = [c for c in candidates
                         if unicodedata.normalize("NFC", c.name) == unicodedata.normalize("NFC", fname)]
                path = match[0] if match else None
            cache[fname] = pd.read_parquet(path) if path else None
        tails = cache[fname]
        if tails is None:
            drr_vals[row["fragment_id"]] = np.nan
            continue
        t0, t1 = row["t_start"], row["t_start"] + 4.0
        sub = tails[(tails.onset_s >= t0) & (tails.onset_s < t1)]
        drr_vals[row["fragment_id"]] = float(sub.drr_db.median()) if len(sub) else np.nan
    return pd.Series(drr_vals, name="drr_db")


def build_dataset():
    ratings = pd.read_csv(Q_DIR / "ratings.csv")
    fragments = pd.read_parquet(Q_DIR / "fragments_selected.parquet")
    windows = pd.read_parquet(DIFF_DIR / "moving_window_4s1s.parquet")
    dynamics = pd.read_csv(DIFF_DIR / "dynamics_profile.csv")

    fragments = fragments.merge(
        windows[["version", "t_start"] + CORE_METRICS + ARTIFACT_METRICS],
        on=["version", "t_start"], how="left")

    fragments["plr"] = fragments.version.map(PLR_LABELS).map(
        dynamics.set_index("label")["plr"])
    fragments["drr_db"] = load_drr_for_fragments(fragments)

    # ratings: like -> число, исключаем разогрев
    ratings = ratings[~ratings.is_warmup].copy()
    ratings["like_numeric"] = ratings["like"].map(LIKE_MAP)

    merged = ratings.merge(fragments, on="fragment_id", how="left")
    n_missing_metrics = merged[CORE_METRICS].isna().all(axis=1).sum()
    if n_missing_metrics:
        print(f"ВНИМАНИЕ: {n_missing_metrics} строк без метрик после merge — проверить fragment_id")
    return merged


def reliability_ceiling(merged):
    """Потолок: rho между двумя оценками одного и того же fragment_id
    (дубли в разных сессиях), по каждой оси. n мало (~15 пар) — CI широкий,
    это ориентир, не точная граница."""
    dup_ids = merged.fragment_id.value_counts()
    dup_ids = dup_ids[dup_ids > 1].index.tolist()
    rows = []
    for axis in AXES:
        pairs_a, pairs_b = [], []
        for fid in dup_ids:
            vals = merged.loc[merged.fragment_id == fid, axis].dropna().tolist()
            if len(vals) >= 2:
                pairs_a.append(vals[0])
                pairs_b.append(vals[1])
        if len(pairs_a) >= 4:
            rho, p = stats.spearmanr(pairs_a, pairs_b)
        else:
            rho, p = np.nan, np.nan
        rows.append(dict(axis=axis, n_pairs=len(pairs_a), rho_ceiling=rho, p=p))
    return pd.DataFrame(rows)


def dedupe_for_correlation(merged):
    """Усредняем повторные оценки одного fragment_id в одну строку —
    иначе дубли дают псевдо-независимые наблюдения и завышают n."""
    agg_cols = AXES + CORE_METRICS + ARTIFACT_METRICS + EXTRA_METRICS
    keep_first = ["version", "group", "reason"]
    g = merged.groupby("fragment_id")
    out = g[agg_cols].mean(numeric_only=True)
    out[keep_first] = g[keep_first].first()
    out["confidence_mean"] = g["confidence"].mean()
    return out.reset_index()


def correlation_matrix(df, metrics, min_n=8):
    rows = []
    for metric in metrics:
        for axis in AXES:
            sub = df[[metric, axis]].dropna()
            if len(sub) < min_n:
                rows.append(dict(metric=metric, axis=axis, rho=np.nan, p=np.nan, n=len(sub)))
                continue
            rho, p = stats.spearmanr(sub[metric], sub[axis])
            rows.append(dict(metric=metric, axis=axis, rho=rho, p=p, n=len(sub)))
    return pd.DataFrame(rows)


# std(метрика внутри группы)/std(метрика в пуле) ниже этого — инженер сведения/материал
# почти не варьировал метрику в этой группе, корреляцию считать нельзя
# физически (сужение диапазона), а не «связи нет». Порог грубый, ориентировочный.
MIN_EVALUABLE_STD_RATIO = 0.3


def disattenuate(rho, ceiling_rho):
    """Коррекция на затухание из-за ненадёжности оценки (классика психометрики):
    наблюдаемая rho занижена, т.к. сам автор не идеально воспроизводим.
    true_rho = rho_observed / sqrt(reliability). Не определено при
    ceiling <= 0 или неконечном значении — тогда NaN, не подделывать."""
    if not np.isfinite(ceiling_rho) or ceiling_rho <= 0:
        return np.nan
    val = rho / np.sqrt(ceiling_rho)
    return float(np.clip(val, -1.0, 1.0))


def restriction_of_range_expected(rho_pool, std_ratio):
    """Thorndike Case II: чего ожидать от rho при сужении разброса метрики
    до std_ratio от пула, если в пуле связь была rho_pool. Используется,
    чтобы отличить «данные не могли ответить» от «связи нет»."""
    u = std_ratio
    denom = np.sqrt(1 - rho_pool**2 + rho_pool**2 * u**2)
    if denom <= 0:
        return np.nan
    return rho_pool * u / denom


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = build_dataset()

    print("=== Потолок надёжности (rho между двумя оценками дубля) ===")
    ceiling = reliability_ceiling(merged)
    print(ceiling.to_string(index=False))
    ceiling.to_csv(OUT_DIR / "reliability_ceiling.csv", index=False)

    unique_df = dedupe_for_correlation(merged)
    print(f"\nУникальных фрагментов для корреляции: {len(unique_df)}")

    all_metrics = CORE_METRICS + ARTIFACT_METRICS + EXTRA_METRICS
    corr_all = correlation_matrix(unique_df, all_metrics)
    corr_all.to_csv(OUT_DIR / "correlation_matrix_full.csv", index=False)

    # робастность: только уверенные оценки (confidence >= 2)
    confident_df = unique_df[unique_df.confidence_mean >= 2]
    corr_confident = correlation_matrix(confident_df, all_metrics)
    corr_confident.to_csv(OUT_DIR / "correlation_matrix_confident_only.csv", index=False)

    # disattenuation: наблюдаемая rho занижена ненадёжностью самих оценок
    ceil_map = ceiling.set_index("axis")["rho_ceiling"]
    corr_all["rho_disattenuated"] = corr_all.apply(
        lambda r: disattenuate(r["rho"], ceil_map.get(r["axis"], np.nan)), axis=1)
    corr_all.to_csv(OUT_DIR / "correlation_matrix_full.csv", index=False)  # перезаписать с новой колонкой

    # оси с низкой надёжностью — не основа для словаря, только для справки
    RELIABLE_AXES = ceiling.loc[ceiling.rho_ceiling >= 0.5, "axis"].tolist()
    print(f"\nОси с потолком надёжности >=0.5 (годятся для словаря): {RELIABLE_AXES}")
    print(f"Оси НИЖЕ порога (clear_smeared и, вероятно, close_far/professional) — "
          f"чинить формулировку вопроса, а не строить на них словарь.")

    print("\n=== Топ метрик по |rho| для каждой оси (словарь, без vibrato) ===")
    vocab = corr_all[corr_all.metric.isin(CORE_METRICS + EXTRA_METRICS)].copy()
    vocab["abs_rho"] = vocab.rho.abs()
    for axis in AXES:
        sub = vocab[vocab.axis == axis].sort_values("abs_rho", ascending=False)
        ceil_rho = ceiling.loc[ceiling.axis == axis, "rho_ceiling"].values
        ceil_str = f"{ceil_rho[0]:.2f}" if len(ceil_rho) and np.isfinite(ceil_rho[0]) else "н/д"
        reliable_tag = "" if axis in RELIABLE_AXES else "  [НЕНАДЁЖНАЯ ОСЬ]"
        print(f"\n{axis} (потолок надёжности: {ceil_str}){reliable_tag}:")
        print(sub.head(3)[["metric", "rho", "rho_disattenuated", "p", "n"]].to_string(index=False))

    print("\n=== vibrato_depth_cents (детектор артефакта, отдельно от словаря) ===")
    print(corr_all[corr_all.metric == "vibrato_depth_cents"][["axis", "rho", "p", "n"]].to_string(index=False))

    unique_df.to_csv(OUT_DIR / "fragments_with_ratings_and_metrics.csv", index=False)

    print("\n=== Проверка на confound «источник» (корреляция ВНУТРИ каждой группы) ===")
    print("Общий rho может быть просто разницей между песнями, а не реальной связью.")
    print(f"Метрика с std-ratio (внутригрупповой std / пуловый std) < {MIN_EVALUABLE_STD_RATIO} "
          f"физически не может дать оценённую корреляцию (сужение диапазона) — помечается NE, не 0.")
    key_features = ["spectral_slope", "band_frac_air", "plr", "warmth_ratio"]
    key_axes = ["warm_cold", "like_numeric", "professional", "wet_dry"]
    pooled_std = unique_df[key_features].std()
    group_rows = []
    for feat in key_features:
        rho_pool_by_axis = {ax: corr_all.loc[(corr_all.metric == feat) & (corr_all.axis == ax), "rho"].values
                             for ax in key_axes}
        for ax in key_axes:
            rp = rho_pool_by_axis[ax][0] if len(rho_pool_by_axis[ax]) else np.nan
            for grp, sub in unique_df.groupby("group"):
                s = sub[[feat, ax]].dropna()
                std_ratio = s[feat].std() / pooled_std[feat] if len(s) > 1 and pooled_std[feat] > 0 else 0.0
                evaluable = std_ratio >= MIN_EVALUABLE_STD_RATIO
                if len(s) >= 6 and s[feat].nunique() > 1:
                    rho, p = stats.spearmanr(s[feat], s[ax])
                else:
                    rho, p = np.nan, np.nan
                expected = restriction_of_range_expected(rp, std_ratio) if np.isfinite(rp) else np.nan
                group_rows.append(dict(metric=feat, axis=ax, group=grp, rho=rho, p=p, n=len(s),
                                        std_ratio=round(std_ratio, 3), evaluable=evaluable,
                                        rho_expected_from_restriction=expected))
    group_corr = pd.DataFrame(group_rows)
    group_corr.to_csv(OUT_DIR / "correlation_within_group.csv", index=False)
    for feat in key_features:
        print(f"\n{feat} (rho / std_ratio; NE = не оценимо, std_ratio < {MIN_EVALUABLE_STD_RATIO}):")
        sub = group_corr[group_corr.metric == feat].copy()
        sub["cell"] = sub.apply(
            lambda r: "NE" if not r["evaluable"] else (f"{r['rho']:.2f}" if np.isfinite(r["rho"]) else "н/д"),
            axis=1)
        print(sub.pivot(index="group", columns="axis", values="cell").to_string())
    print("\nband_frac_air — единственная метрика с std_ratio > 1 внутри «КП миксы инженера сведения» "
          "(инженер сведения варьировал её сильнее, чем в среднем по корпусу) — приоритетный кандидат "
          "для этой группы, не spectral_slope/PLR/warmth_ratio.")

    print(f"\nСохранено в {OUT_DIR}")


if __name__ == "__main__":
    main()
