"""ТЗ-05 Блок 5: delta_to_zone — численная дельта до зоны «нравится», не
только статус вне/внутри (roadmap.md, Блок 5, «для каждой метрики вне
зоны — численная дельта до цели»)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.verdict import MetricZone, Reliability, Status, delta_to_zone, evaluate


def _zone(**kw):
    defaults = dict(metric="m", source="mix", granularity="track_avg", axis="ось",
                     liked_lo=None, liked_hi=None)
    defaults.update(kw)
    return MetricZone(**defaults)


def test_inside_zone_delta_is_zero():
    z = _zone(liked_lo=0.0, liked_hi=10.0)
    assert delta_to_zone(5.0, z) == 0.0


def test_below_zone_positive_delta():
    z = _zone(liked_lo=0.0, liked_hi=10.0)
    assert delta_to_zone(-3.0, z) == 3.0  # нужно поднять на 3, чтобы войти в зону


def test_above_zone_negative_delta():
    z = _zone(liked_lo=0.0, liked_hi=10.0)
    assert delta_to_zone(14.0, z) == -4.0  # нужно опустить на 4


def test_direction_only_zone_has_no_delta():
    z = _zone(liked_lo=None, liked_hi=None, direction_only=True)
    assert delta_to_zone(5.0, z) is None


def test_no_zone_reliability_has_no_delta():
    z = _zone(liked_lo=0.0, liked_hi=10.0, reliability=Reliability.NO_ZONE)
    # delta_to_zone сама по себе не смотрит на reliability — проверяем
    # именно через evaluate(), где NO_ZONE обрабатывается раньше классификации
    verdicts = evaluate({("m", "mix"): 14.0}, preset=[z])
    assert verdicts[0].status is Status.NO_ZONE
    assert verdicts[0].delta_to_zone is None


def test_evaluate_populates_delta_for_out_of_zone():
    z = _zone(liked_lo=0.0, liked_hi=10.0, disliked_lo=20.0, disliked_hi=30.0)
    verdicts = evaluate({("m", "mix"): 25.0}, preset=[z])
    assert verdicts[0].status is Status.OUT_OF_ZONE
    assert verdicts[0].delta_to_zone == -15.0


if __name__ == "__main__":
    test_inside_zone_delta_is_zero()
    test_below_zone_positive_delta()
    test_above_zone_negative_delta()
    test_direction_only_zone_has_no_delta()
    test_no_zone_reliability_has_no_delta()
    test_evaluate_populates_delta_for_out_of_zone()
    print("Все тесты delta_to_zone (Блок 5) прошли.")
