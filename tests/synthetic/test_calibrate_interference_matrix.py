"""Блок 6 (калибровка): пламбинг compare()/summarize() на синтетике — не
трогает реальный revisions_windows.parquet (тот вне публичного репо, см.
docstring analysis/calibrate_interference_matrix.py про провенанс)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from analysis.calibrate_interference_matrix import compare, summarize


def _revisions_df(rows):
    cols = ["revision_idx", "metric", "тип_претензии", "направление", "delta"]
    return pd.DataFrame(rows, columns=cols)


def test_compare_only_keeps_valid_links_and_claim_types():
    df = _revisions_df([
        [1, "band_frac_air", "тембр", "больше", 0.001],       # валидно
        [2, "band_frac_lowmid", "тембр", "больше", 0.01],     # metric не в VALID_METRIC_LINKS — исключён
        [3, "band_frac_air", "динамика", "больше", 0.001],    # тип не в RELEVANT_CLAIM_TYPES — исключён
        [4, "band_frac_air", "тембр", "неприменимо", 0.001],  # направление не больше/меньше — исключён
        [5, "band_frac_air", "тембр", "меньше", None],        # NaN delta — исключён
    ])
    rows = compare(df)
    assert len(rows) == 1
    assert rows[0]["revision_idx"] == 1
    assert rows[0]["zone_metric"] == "band_frac_air_median"
    assert rows[0]["role"] == "mix"


def test_compare_computes_signs_and_realized_flag():
    df = _revisions_df([
        [1, "band_frac_air", "тембр", "больше", 0.002],   # ожидали +, реально + -> сбылось
        [2, "band_frac_air", "тембр", "больше", -0.002],  # ожидали +, реально - -> НЕ сбылось
    ])
    rows = compare(df)
    assert rows[0]["expected_sign"] == 1 and rows[0]["real_sign"] == 1 and rows[0]["request_realized"]
    assert rows[1]["expected_sign"] == 1 and rows[1]["real_sign"] == -1 and not rows[1]["request_realized"]


def test_summarize_agreement_fraction():
    rows = [
        dict(role="mix", zone_metric="band_frac_air_median", real_sign=1),
        dict(role="mix", zone_metric="band_frac_air_median", real_sign=-1),
    ]
    matrix = {
        "shelf_air_boost": {"mix::band_frac_air_median": {"median_delta": 0.001, "n": 2}},  # предсказывает +
        "shelf_air_cut": {"mix::band_frac_air_median": {"median_delta": -0.001, "n": 2}},    # предсказывает -
    }
    summary = summarize(rows, matrix)
    assert summary["shelf_air_boost"]["agreement_fraction"] == 0.5  # совпал только с первой строкой
    assert summary["shelf_air_boost"]["n"] == 2
    assert summary["shelf_air_cut"]["agreement_fraction"] == 0.5  # совпал только со второй


def test_summarize_skips_moves_without_prediction_for_metric():
    rows = [dict(role="mix", zone_metric="band_frac_air_median", real_sign=1)]
    matrix = {"bell_cut_lowmid": {"vocals::harshness": {"median_delta": -0.1, "n": 1}}}
    summary = summarize(rows, matrix)
    assert summary == {}


if __name__ == "__main__":
    test_compare_only_keeps_valid_links_and_claim_types()
    test_compare_computes_signs_and_realized_flag()
    test_summarize_agreement_fraction()
    test_summarize_skips_moves_without_prediction_for_metric()
    print("Все тесты пламбинга калибровки (Блок 6) прошли.")
