"""Блок 7 («с выбором»): вкусовые рекомендации — от Verdict вне зоны до
Recommendation с ходом Блока 6, атрибуцией по секции и честными
конфликтами. Синтетика, не требует реального interference_matrix.json."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.recommendations import (
    _best_covered_move, _flag_opposing_conflicts, _layering_recommendation,
    all_taste_recommendations, taste_recommendation_for_verdict,
)
from analysis.verdict import MetricZone, Reliability, Status, Verdict, evaluate


def _zone(**kw):
    defaults = dict(metric="m", source="mix", granularity="track_avg", axis="ось",
                     liked_lo=0.0, liked_hi=1.0)
    defaults.update(kw)
    return MetricZone(**defaults)


def test_best_covered_move_picks_largest_correct_sign_among_candidates():
    matrix = {
        "bell_cut_lowmid": {"mix::warmth_ratio": {"median_delta": -5.5, "n": 2}},
        "shelf_low_cut": {"mix::warmth_ratio": {"median_delta": -1.0, "n": 2}},
        "bell_boost_lowmid": {"mix::warmth_ratio": {"median_delta": 10.2, "n": 2}},  # неверный знак для "above"
    }
    best = _best_covered_move("mix", "warmth_ratio", "above", matrix)
    assert best[0] == "bell_cut_lowmid"  # -5.5 больше по модулю, чем -1.0
    assert best[1] == -5.5


def test_best_covered_move_none_when_no_correct_sign_candidate():
    matrix = {"bell_cut_lowmid": {"mix::warmth_ratio": {"median_delta": +2.0, "n": 1}}}  # знак не тот
    assert _best_covered_move("mix", "warmth_ratio", "above", matrix) is None


def test_best_covered_move_none_for_uncovered_metric():
    assert _best_covered_move("bass", "skewness", "above", {}) is None


def test_taste_recommendation_none_for_in_zone_and_no_data():
    zone = _zone(metric="warmth_ratio")
    verdicts = evaluate({("warmth_ratio", "mix"): 0.5}, preset=[zone])  # внутри зоны
    assert taste_recommendation_for_verdict(verdicts[0], {}, {}) is None
    verdicts_no_data = evaluate({}, preset=[zone])
    assert taste_recommendation_for_verdict(verdicts_no_data[0], {}, {}) is None


def test_taste_recommendation_none_for_direction_only_and_no_zone():
    dz = _zone(metric="skewness", liked_lo=None, liked_hi=None, direction_only=True)
    v = evaluate({("skewness", "mix"): 5.0}, preset=[dz])[0]
    assert taste_recommendation_for_verdict(v, {}, {}) is None

    nz = _zone(metric="overall_ms_ratio", reliability=Reliability.NO_ZONE)
    v2 = evaluate({("overall_ms_ratio", "mix"): 5.0}, preset=[nz])[0]
    assert taste_recommendation_for_verdict(v2, {}, {}) is None


def test_taste_recommendation_builds_move_backed_recommendation():
    zone = _zone(metric="warmth_ratio", liked_lo=0.0, liked_hi=1.0,
                  disliked_lo=2.0, disliked_hi=3.0)
    v = evaluate({("warmth_ratio", "mix"): 2.5}, preset=[zone])[0]  # выше зоны -> direction "above"
    matrix = {"bell_cut_lowmid": {"mix::warmth_ratio": {"median_delta": -5.5, "n": 2}}}
    rec = taste_recommendation_for_verdict(v, {}, matrix)
    assert rec is not None
    assert rec.category == "bell_cut_lowmid"
    assert rec.stage == "вычитающий EQ"
    assert rec.source == "mix"
    assert rec.section is None  # track_avg-зона, секций не считалось
    assert "колокол" in rec.text and "300Гц" in rec.text  # MOVE_DESCRIPTIONS, не голое имя хода
    assert 0.0 < rec.confidence <= 1.0


def test_taste_recommendation_no_move_when_uncovered():
    zone = _zone(metric="skewness", source="bass", liked_lo=0.0, liked_hi=1.0,
                  disliked_lo=2.0, disliked_hi=3.0)
    v = evaluate({("skewness", "bass"): 2.5}, preset=[zone])[0]
    rec = taste_recommendation_for_verdict(v, {}, {})  # skewness нет в TASTE_MOVE_CANDIDATES
    assert rec is not None
    assert rec.category == "taste_no_move"
    assert rec.confidence == 0.0
    assert "не покрыто" in rec.text or "экспериментально" in rec.text


def test_taste_recommendation_attributes_window_zone_to_worst_section():
    zone = _zone(metric="harshness", source="vocals", granularity="window",
                  liked_lo=0.0, liked_hi=1.0, disliked_lo=2.0, disliked_hi=3.0)
    v = evaluate({("harshness", "vocals"): 2.5}, preset=[zone])[0]
    diagnostics = {"vocals": {"section_medians": {"harshness": {"куплет": 1.1, "припев": 2.6}}}}
    matrix = {"bell_cut_presence": {"vocals::harshness": {"median_delta": -0.5, "n": 1}}}
    rec = taste_recommendation_for_verdict(v, diagnostics, matrix)
    assert rec.section == "припев"
    assert "припев" in rec.text


def test_layering_recommendation_skips_error_pairs():
    assert _layering_recommendation("vocals", {"pair": ["a", "b"], "error": "oops"}) is None
    pair = dict(pair=["a.wav", "b.wav"], time_divergence_ms_median=3.2,
                pitch_divergence_cents_median=12.3, comb_risk_upper_bound=0.6)
    rec = _layering_recommendation("vocals", pair)
    assert rec.category == "layering_choice"
    assert rec.stage == "наложение дублей"
    assert "a.wav" in rec.text and "b.wav" in rec.text


def test_flag_opposing_conflicts_marks_both_sides_same_role_only():
    from analysis.recommendations import Recommendation
    r1 = Recommendation(category="bell_cut_lowmid", source="mix", location_s=None,
                         params={}, confidence=0.7, text="cut")
    r2 = Recommendation(category="bell_boost_lowmid", source="mix", location_s=None,
                         params={}, confidence=0.7, text="boost")
    r3 = Recommendation(category="bell_boost_lowmid", source="bass", location_s=None,
                         params={}, confidence=0.7, text="boost bass")  # другая роль — не конфликт
    recs = [r1, r2, r3]
    _flag_opposing_conflicts(recs)
    assert r1.params.get("conflict_with") == "bell_boost_lowmid"
    assert r2.params.get("conflict_with") == "bell_cut_lowmid"
    assert "КОНФЛИКТ" in r1.text and "КОНФЛИКТ" in r2.text
    assert "conflict_with" not in r3.params


def test_all_taste_recommendations_sorts_by_section_then_confidence():
    zone_a = _zone(metric="warmth_ratio", liked_lo=0.0, liked_hi=1.0, disliked_lo=2.0, disliked_hi=3.0)
    zone_b = _zone(metric="band_frac_air_median", liked_lo=0.0, liked_hi=1.0, disliked_lo=2.0, disliked_hi=3.0)
    verdicts = evaluate({("warmth_ratio", "mix"): 2.5, ("band_frac_air_median", "mix"): 2.5},
                         preset=[zone_a, zone_b])
    matrix = {
        "bell_cut_lowmid": {"mix::warmth_ratio": {"median_delta": -5.5, "n": 2}},
        "shelf_air_cut": {"mix::band_frac_air_median": {"median_delta": -0.001, "n": 2}},
    }
    recs = all_taste_recommendations(verdicts, {}, matrix)
    assert len(recs) == 2
    # оба track_avg (section=None) -> сортировка по убыванию уверенности
    assert recs[0].confidence >= recs[1].confidence


if __name__ == "__main__":
    test_best_covered_move_picks_largest_correct_sign_among_candidates()
    test_best_covered_move_none_when_no_correct_sign_candidate()
    test_best_covered_move_none_for_uncovered_metric()
    test_taste_recommendation_none_for_in_zone_and_no_data()
    test_taste_recommendation_none_for_direction_only_and_no_zone()
    test_taste_recommendation_builds_move_backed_recommendation()
    test_taste_recommendation_no_move_when_uncovered()
    test_taste_recommendation_attributes_window_zone_to_worst_section()
    test_layering_recommendation_skips_error_pairs()
    test_flag_opposing_conflicts_marks_both_sides_same_role_only()
    test_all_taste_recommendations_sorts_by_section_then_confidence()
    print("Все тесты вкусовых рекомендаций (Блок 7) прошли.")
