"""Блок 8: пламбинг каталога плагинов (presets/plugins.json) — загрузка,
матчинг по canonical_interventions, обогащение Recommendation. Не проверяет
музыкальную правильность связок плагин->ход (это личный каталог автора,
см. caveat в самом JSON), только что код вокруг него не ломается."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analysis.plugin_catalog import PluginEntry, enrich_with_plugins, load_catalog, match
from analysis.recommendations import Recommendation

ALL_CATEGORIES = {"declip_click", "declip_sustained", "dehum",
                   "bell_cut_lowmid", "bell_boost_lowmid", "bell_cut_presence", "bell_boost_presence",
                   "shelf_air_boost", "shelf_air_cut", "shelf_low_boost", "shelf_low_cut",
                   "compressor_soft", "saturation_soft"}


def _entry(name, canonical_interventions=(), mapping_confidence=None,
           excluded_from_corrective=False, requires_confirmation=False, kind="third_party", note=""):
    return PluginEntry(id=name, name=name, vendor="", kind=kind, category_tags=(), mode="insert",
                        stage=(), canonical_interventions=tuple(canonical_interventions),
                        canonical_op=(), mapping_confidence=mapping_confidence,
                        excluded_from_corrective=excluded_from_corrective,
                        requires_confirmation=requires_confirmation, artistic_only=False,
                        last_resort=False, min_live_version=None, min_edition=None,
                        parameter_map=None, note=note)


def test_load_catalog_missing_file_returns_empty():
    assert load_catalog("_no_such_catalog_") == []


def test_real_catalog_loads_without_error():
    catalog = load_catalog()
    assert len(catalog) > 0
    for p in catalog:
        assert isinstance(p, PluginEntry)
        assert p.name
        assert p.mapping_confidence in ("exact", "approx", "opaque", None)


def test_real_catalog_declip_and_dehum_match_rx11():
    catalog = load_catalog()
    for category in ("declip_click", "declip_sustained", "dehum"):
        hits = match(category, catalog)
        assert any(p.name == "iZotope RX 11" for p in hits), (
            f"{category} должен матчиться на RX 11 в реальном каталоге")


def test_real_catalog_masking_fix_present_and_unmatched_today():
    """abl.compressor.sidechain_ducking (Блок 8, известный ключевой фикс
    маскирования) должен существовать в каталоге, но БЕЗ canonical_interventions
    — Recommendation.category под маскирование ещё не заведён в recommendations.py,
    честно не матчим, пока измерять нечем."""
    catalog = load_catalog()
    hits = [p for p in catalog if p.id == "abl.compressor.sidechain_ducking"]
    assert len(hits) == 1
    assert hits[0].canonical_interventions == ()
    assert "masking_fix" in hits[0].category_tags


def test_real_catalog_never_matches_arrangement_plugins():
    """Splice INSTRUMENT/Surge XT/Vital помечены excluded_from_corrective —
    ни один Recommendation.category (реальный ход Блока 6 + категории
    Блока 2) не должен их вернуть, даже случайно."""
    catalog = load_catalog()
    excluded_names = {p.name for p in catalog if p.excluded_from_corrective}
    assert excluded_names, "ожидали хотя бы один аранжировочный плагин в реальном каталоге"
    for category in ALL_CATEGORIES:
        hits = match(category, catalog)
        hit_names = {p.name for p in hits}
        assert not (hit_names & excluded_names), (
            f"{category} матчится на аранжировочный плагин {hit_names & excluded_names}")


def test_real_catalog_no_duplicate_ids():
    catalog = load_catalog()
    ids = [p.id for p in catalog]
    assert len(ids) == len(set(ids)), "дублирующиеся id в presets/plugins.json"


def test_match_ranks_by_mapping_confidence():
    catalog = [
        _entry("Opaque", canonical_interventions=("compressor_soft",), mapping_confidence="opaque"),
        _entry("Exact", canonical_interventions=("compressor_soft",), mapping_confidence="exact"),
        _entry("Approx", canonical_interventions=("compressor_soft",), mapping_confidence="approx"),
    ]
    hits = match("compressor_soft", catalog)
    assert [p.name for p in hits] == ["Exact", "Approx", "Opaque"]


def test_match_excludes_flagged_plugin_even_if_tagged():
    """Защита на уровне кода, не только данных (см. docstring match())."""
    catalog = [_entry("Синт", canonical_interventions=("saturation_soft",),
                       mapping_confidence="exact", excluded_from_corrective=True)]
    assert match("saturation_soft", catalog) == []


def test_enrich_with_plugins_no_catalog_leaves_recs_untouched():
    recs = [Recommendation(category="dehum", source="mix", location_s=None, params={},
                            confidence=1.0, text="[mix]: гул")]
    out = enrich_with_plugins(recs, catalog=[])
    assert out[0].plugin_suggestion is None
    assert out[0].text == "[mix]: гул"


def test_enrich_with_plugins_appends_matched_name():
    catalog = [_entry("Тестовый деклиппер", canonical_interventions=("declip_click",),
                       mapping_confidence="exact")]
    recs = [Recommendation(category="declip_click", source="vocals", location_s=(1.0, 1.1), params={},
                            confidence=1.0, text="[vocals] 1.00-1.10с: клиппинг")]
    out = enrich_with_plugins(recs, catalog=catalog)
    assert out[0].plugin_suggestion == "Тестовый деклиппер"
    assert out[0].text.endswith("— подходит: Тестовый деклиппер")


def test_enrich_with_plugins_no_match_leaves_suggestion_none():
    catalog = [_entry("Тестовый деклиппер", canonical_interventions=("declip_click",),
                       mapping_confidence="exact")]
    recs = [Recommendation(category="dehum", source="mix", location_s=None, params={},
                            confidence=1.0, text="[mix]: гул")]
    out = enrich_with_plugins(recs, catalog=catalog)
    assert out[0].plugin_suggestion is None


def test_enrich_with_plugins_flags_requires_confirmation_in_text():
    catalog = [_entry("Gate-подобный", canonical_interventions=("dehum",),
                       mapping_confidence="approx", requires_confirmation=True,
                       note="съест шум, который считается частью атмосферы")]
    recs = [Recommendation(category="dehum", source="mix", location_s=None, params={},
                            confidence=1.0, text="[mix]: гул")]
    out = enrich_with_plugins(recs, catalog=catalog)
    assert out[0].plugin_suggestion == "Gate-подобный"
    assert "подтверди" in out[0].text


if __name__ == "__main__":
    test_load_catalog_missing_file_returns_empty()
    test_real_catalog_loads_without_error()
    test_real_catalog_declip_and_dehum_match_rx11()
    test_real_catalog_masking_fix_present_and_unmatched_today()
    test_real_catalog_never_matches_arrangement_plugins()
    test_real_catalog_no_duplicate_ids()
    test_match_ranks_by_mapping_confidence()
    test_match_excludes_flagged_plugin_even_if_tagged()
    test_enrich_with_plugins_no_catalog_leaves_recs_untouched()
    test_enrich_with_plugins_appends_matched_name()
    test_enrich_with_plugins_no_match_leaves_suggestion_none()
    test_enrich_with_plugins_flags_requires_confirmation_in_text()
    print("Все тесты каталога плагинов (Блок 8) прошли.")
