"""ТЗ-05 Блок 5: атрибуция window-метрик по (роль, секция), не по
таймкоду — переиспользует уже посчитанные window-метрики (wdf) и границы
секций с mix (section_profile), см. roadmap.md, Блок 5."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from analysis.section_attribution import attribute_by_section, worst_section
from analysis.verdict import MetricZone


def _section_profile():
    return pd.DataFrame([
        dict(start_s=0.0, end_s=10.0, section="куплет", lufs=-20.0, level_rel_db=-2.0),
        dict(start_s=10.0, end_s=20.0, section="припев", lufs=-14.0, level_rel_db=4.0),
    ])


def test_windows_grouped_by_section():
    wdf = pd.DataFrame([
        dict(t_start=1.0, t_end=5.0, rms_dbfs=-20, warmth_ratio=1.0),
        dict(t_start=3.0, t_end=7.0, rms_dbfs=-20, warmth_ratio=1.2),
        dict(t_start=12.0, t_end=16.0, rms_dbfs=-14, warmth_ratio=3.0),
        dict(t_start=14.0, t_end=18.0, rms_dbfs=-14, warmth_ratio=3.4),
    ])
    out = attribute_by_section(wdf, _section_profile())
    assert out["warmth_ratio"]["куплет"] == 1.1  # медиана 1.0/1.2
    assert out["warmth_ratio"]["припев"] == 3.2  # медиана 3.0/3.4
    assert "rms_dbfs" not in out  # служебная колонка, исключена так же, как в measurements


def test_windows_outside_any_section_are_dropped():
    wdf = pd.DataFrame([
        dict(t_start=1.0, t_end=5.0, rms_dbfs=-20, warmth_ratio=1.0),
        dict(t_start=25.0, t_end=29.0, rms_dbfs=-20, warmth_ratio=9.9),  # за пределами profile
    ])
    out = attribute_by_section(wdf, _section_profile())
    assert list(out["warmth_ratio"].keys()) == ["куплет"]


def test_empty_inputs_return_empty_dict():
    assert attribute_by_section(pd.DataFrame(), _section_profile()) == {}
    assert attribute_by_section(pd.DataFrame([dict(t_start=1.0, x=1.0)]), pd.DataFrame()) == {}
    assert attribute_by_section(None, _section_profile()) == {}


def test_worst_section_picks_farthest_from_zone():
    zone = MetricZone(metric="warmth_ratio", source="vocals", granularity="window",
                       axis="тепло/холодно", liked_lo=0.5, liked_hi=1.5)
    section_medians = {"куплет": 1.1, "припев": 3.2}
    label, val, delta = worst_section(section_medians, zone)
    assert label == "припев"
    assert val == 3.2
    assert abs(delta - (-1.7)) < 1e-9  # 1.5 - 3.2


def test_worst_section_none_for_direction_only_zone():
    zone = MetricZone(metric="m", source="mix", granularity="window", axis="ось",
                       liked_lo=None, liked_hi=None, direction_only=True)
    label, val, delta = worst_section({"куплет": 1.0}, zone)
    assert (label, val, delta) == (None, None, None)


def test_worst_section_empty_medians():
    zone = MetricZone(metric="m", source="mix", granularity="window", axis="ось",
                       liked_lo=0.0, liked_hi=1.0)
    assert worst_section({}, zone) == (None, None, None)


def test_worst_section_none_when_every_section_actually_in_zone():
    """Регрессия (найдена код-ревью): если ВСЕ секции по отдельности
    внутри зоны (delta_to_zone==0.0 у каждой), возвращать первую попавшуюся
    как "худшую" — вводить в заблуждение (общий Verdict мог быть
    OUT_OF_ZONE только из-за иной агрегации в verdict.py, не потому что
    какая-то конкретная секция реально плохая)."""
    zone = MetricZone(metric="m", source="mix", granularity="window", axis="ось",
                       liked_lo=0.0, liked_hi=1.0)
    section_medians = {"куплет": 0.5, "припев": 0.8}  # обе строго внутри [0.0, 1.0]
    assert worst_section(section_medians, zone) == (None, None, None)


def test_worst_section_ignores_in_zone_sections_when_picking_worst():
    """Смежный случай: часть секций реально в зоне (delta=0), часть нет —
    "худшей" должна выбираться только среди тех, что реально вне зоны, а
    не случайно среди всех (в частности, не первая по порядку, если она
    внутри зоны)."""
    zone = MetricZone(metric="m", source="mix", granularity="window", axis="ось",
                       liked_lo=0.0, liked_hi=1.0)
    section_medians = {"куплет": 0.5, "припев": 2.5}  # куплет в зоне, припев далеко вне
    label, val, delta = worst_section(section_medians, zone)
    assert label == "припев"
    assert val == 2.5


if __name__ == "__main__":
    test_windows_grouped_by_section()
    test_windows_outside_any_section_are_dropped()
    test_empty_inputs_return_empty_dict()
    test_worst_section_picks_farthest_from_zone()
    test_worst_section_none_for_direction_only_zone()
    test_worst_section_empty_medians()
    test_worst_section_none_when_every_section_actually_in_zone()
    test_worst_section_ignores_in_zone_sections_when_picking_worst()
    print("Все тесты атрибуции по секциям (Блок 5) прошли.")
