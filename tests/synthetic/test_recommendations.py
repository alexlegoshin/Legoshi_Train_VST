"""Блок 2 (Этап 1): диагностика -> рекомендации. Главный принцип
(roadmap.md) — программа не применяет фикс, только называет место,
параметры и категорию для будущего каталога плагинов (Блок 8)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.recommendations import (
    Recommendation, restoration_recommendations, all_restoration_recommendations,
)


def test_no_recommendations_on_clean_diagnostics():
    assert restoration_recommendations({}, "vocals") == []
    assert restoration_recommendations({"clipped": False}, "vocals") == []


def test_clipping_produces_one_recommendation_per_region():
    diagnostics = {
        "clipping_detail": {
            "clipped": True, "n_clipped_runs": 2, "clipped_fraction": 0.001,
            "clipped_regions_s": [
                dict(start_s=1.0, end_s=1.05, duration_s=0.05, channels=[0, 1], severity="sustained"),
                dict(start_s=10.2, end_s=10.3, duration_s=0.1, channels=[0], severity="sustained"),
            ],
        },
    }
    recs = restoration_recommendations(diagnostics, "bass")
    assert len(recs) == 2
    assert all(r.category == "declip_sustained" for r in recs)
    assert all(r.source == "bass" for r in recs)
    assert recs[0].location_s == (1.0, 1.05)
    assert recs[1].location_s == (10.2, 10.3)
    assert recs[1].params["channels"] == [0]
    assert "declip" in recs[0].text
    assert "bass" in recs[0].text


def test_clipping_category_reflects_severity():
    """category разделяет click/sustained — разные режимы восстановления у
    реального declip-инструмента, Блок 8 будет искать по этому тегу."""
    diagnostics = {
        "clipping_detail": {
            "clipped": True, "n_clipped_runs": 1, "clipped_fraction": 0.0001,
            "clipped_regions_s": [dict(start_s=1.0, end_s=1.001, duration_s=0.001,
                                        channels=[0, 1], severity="click")],
        },
    }
    recs = restoration_recommendations(diagnostics, "vocals")
    assert recs[0].category == "declip_click"


def test_hum_recommendation_uses_refined_frequency_when_available():
    diagnostics = {
        "hum_candidates": [
            {"freq_hz": 55.0, "freq_hz_refined": 49.97, "stability_score": 8.5},
        ],
    }
    recs = restoration_recommendations(diagnostics, "vocals")
    assert len(recs) == 1
    assert recs[0].category == "dehum"
    assert recs[0].params["freq_hz"] == 50.0  # округлено от 49.97
    assert "49.97" in recs[0].text or "50.0" in recs[0].text
    assert recs[0].location_s is None  # гул по всей дорожке, не отрезок


def test_hum_recommendation_falls_back_to_raw_frequency():
    """Если по каким-то причинам уточнение не попало в diagnostics —
    рекомендация всё равно строится, просто на сырой частоте, не падает."""
    diagnostics = {"hum_candidates": [{"freq_hz": 50.0, "stability_score": 5.0}]}
    recs = restoration_recommendations(diagnostics, "vocals")
    assert len(recs) == 1
    assert recs[0].params["freq_hz"] == 50.0


def test_confidence_clamped_to_one():
    diagnostics = {"hum_candidates": [{"freq_hz": 50.0, "stability_score": 999.0}]}
    recs = restoration_recommendations(diagnostics, "vocals")
    assert recs[0].confidence == 1.0


def test_sorting_by_timeline_then_confidence():
    """Двойная сортировка: сначала по позиции на таймлайне (гул без отрезка
    считается "в конце"), внутри той же позиции — по убыванию уверенности."""
    def _clip_region(start_s, end_s):
        return dict(start_s=start_s, end_s=end_s, duration_s=round(end_s - start_s, 3),
                    channels=[0, 1], severity="click")

    all_diag = {
        "vocals": {
            "clipping_detail": {"clipped": True, "n_clipped_runs": 1, "clipped_fraction": 0.0001,
                                 "clipped_regions_s": [_clip_region(20.0, 20.01)]},
        },
        "bass": {
            "clipping_detail": {"clipped": True, "n_clipped_runs": 1, "clipped_fraction": 0.0001,
                                 "clipped_regions_s": [_clip_region(5.0, 5.01)]},
            "hum_candidates": [{"freq_hz": 50.0, "stability_score": 5.0}],
        },
        "_run": {"preset_name": "legoshi_amber"},  # служебная запись, должна быть пропущена
    }
    recs = all_restoration_recommendations(all_diag)
    # ожидание: клиппинг на 5с (bass) -> клиппинг на 20с (vocals) -> гул (без отрезка, в конце)
    assert recs[0].location_s == (5.0, 5.01)
    assert recs[1].location_s == (20.0, 20.01)
    assert recs[2].location_s is None
    assert recs[2].category == "dehum"


if __name__ == "__main__":
    test_no_recommendations_on_clean_diagnostics()
    test_clipping_produces_one_recommendation_per_region()
    test_clipping_category_reflects_severity()
    test_hum_recommendation_uses_refined_frequency_when_available()
    test_hum_recommendation_falls_back_to_raw_frequency()
    test_confidence_clamped_to_one()
    test_sorting_by_timeline_then_confidence()
    print("Все тесты формата рекомендаций (Блок 2) прошли.")
