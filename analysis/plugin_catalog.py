"""Блок 8 (roadmap.md): каталог плагинов + штатных устройств Ableton пользователя
— presets/plugins.json, персональный (не общий), не обязателен для работы
рекомендателя (Блоки 2/7 работают и без него — просто без имени плагина в
тексте). Схема расширена 02.09.2026 (_notes/ableton-stock-catalog.md) —
kind/stage/mapping_confidence/canonical_op появились из-за стоковых устройств
Ableton, которых сильно больше и разнообразнее, чем сторонних плагинов.

Матчинг — пересечение тегов + ранжирование, тот же принцип, что verdict.py:
`canonical_interventions` записи сопоставляется с `Recommendation.category`
напрямую (это один и тот же словарь ключей — declip_click/declip_sustained/
dehum из Блока 2, имя канонического хода из analysis.interventions.MOVES из
Блока 7). Это ЕДИНСТВЕННОЕ поле, которое реально участвует в матчинге сегодня.

`canonical_op` — более широкий словарь DSP-примитивов (bell/shelf/gain/
duck_sidechain/harmonics_even/... — см. presets/plugins.json, "caveat" и сам
JSON), заготовка на будущее: когда interventions.py/Блок 6 обзаведётся новыми
ходами, эти теги уже на месте, но СЕГОДНЯ они не проверены Блоком 6 и не
участвуют в match() — не гадаем, откуда взялась связь, если её нельзя
измерить (тот же принцип, что verdict.py: без эмпирики/структурного
обоснования связь не заявляем)."""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
DEFAULT_CATALOG = "plugins"
MAPPING_CONFIDENCE_ORDER = {"exact": 0, "approx": 1, "opaque": 2, None: 3}


@dataclass(frozen=True)
class PluginEntry:
    id: str
    name: str
    vendor: str
    kind: str  # "stock_device" | "third_party" | "daw_operation"
    category_tags: tuple
    mode: Optional[str]  # "insert" | "offline" | "monitor" — None для daw_operation
                          # (усиление клипа/автоматизация — не "вставка", это правка данных)
    stage: tuple  # подмножество {"0".."5","M"} — восстановление/уровни и баланс/
                   # вычитающий EQ/динамика/добавляющий EQ и цвет/пространство/измерение
                   # (см. presets/plugins.json -> "stage_labels"); может быть пустым
                   # (мета-записи вроде Audio Effect Rack)
    canonical_interventions: tuple  # ключи Recommendation.category — единственное
                                      # поле, которое реально участвует в матчинге
    canonical_op: tuple  # словарь DSP-примитивов на будущее, см. docstring модуля
    mapping_confidence: Optional[str]  # "exact" | "approx" | "opaque" | None
    excluded_from_corrective: bool  # аранжировочные инструменты — никогда не должны
                                      # попасть в маппинг «чего не хватает -> что добавить»
    requires_confirmation: bool  # трогает то, что пользователь может считать фичей
                                    # (шум как атмосфера, глиссандо) — спросить, не молчать
    artistic_only: bool  # только под словесный запрос, не под числовой диагноз
    last_resort: bool
    min_live_version: Optional[str]
    min_edition: Optional[str]
    parameter_map: Optional[dict]
    note: str


def load_catalog(name: str = DEFAULT_CATALOG) -> list[PluginEntry]:
    """Грузит каталог по имени (без .json) из Project/presets/, либо по
    прямому пути. Файла нет (личный каталог не собран/не нужен) — пустой
    список, не исключение: Блоки 2/7 работают без Блока 8."""
    path = Path(name)
    if not path.suffix:
        path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for p in data.get("plugins", []):
        out.append(PluginEntry(
            id=p.get("id", p["name"]), name=p["name"], vendor=p.get("vendor") or "",
            kind=p.get("kind", "third_party"),
            category_tags=tuple(p.get("category_tags", [])),
            mode=p.get("mode"),
            stage=tuple(p.get("stage") or []),
            canonical_interventions=tuple(p.get("canonical_interventions", [])),
            canonical_op=tuple(p.get("canonical_op", [])),
            mapping_confidence=p.get("mapping_confidence"),
            excluded_from_corrective=bool(p.get("excluded_from_corrective", False)),
            requires_confirmation=bool(p.get("requires_confirmation", False)),
            artistic_only=bool(p.get("artistic_only", False)),
            last_resort=bool(p.get("last_resort", False)),
            min_live_version=p.get("min_live_version"),
            min_edition=p.get("min_edition"),
            parameter_map=p.get("parameter_map"),
            note=p.get("note", ""),
        ))
    return out


def match(category: str, catalog: list[PluginEntry]) -> list[PluginEntry]:
    """Записи, чей canonical_interventions содержит `category`.
    Ранжирование: mapping_confidence exact < approx < opaque < None (сначала
    структурно чистые связки), порядок внутри группы как в каталоге (личный
    список, не независимая оценка качества). excluded_from_corrective
    отфильтровывается явно — защита от будущей правки JSON, которая случайно
    заполнит теги аранжировочному инструменту."""
    hits = [p for p in catalog if category in p.canonical_interventions and not p.excluded_from_corrective]
    hits.sort(key=lambda p: MAPPING_CONFIDENCE_ORDER.get(p.mapping_confidence, 3))
    return hits


def enrich_with_plugins(recs: list, catalog: Optional[list[PluginEntry]] = None) -> list:
    """Мутирует список Recommendation на месте: где для r.category есть
    матч в каталоге, дописывает имя(имена) в r.text и заполняет
    r.plugin_suggestion. Записи с requires_confirmation=True помечаются
    отдельно в тексте — молчать про них нельзя (см. docstring PluginEntry).
    catalog=None — грузит presets/plugins.json по умолчанию; каталога нет —
    recs возвращается без изменений (Блок 8 необязателен). Топ-2 совпадения
    на рекомендацию, не весь список — текст рекомендации, не каталог сам по
    себе."""
    if catalog is None:
        catalog = load_catalog()
    if not catalog:
        return recs
    for r in recs:
        hits = match(r.category, catalog)
        if not hits:
            continue
        top = hits[:2]
        names = ", ".join(
            f"{p.name} (подтверди — {p.note.splitlines()[0][:60]}...)" if p.requires_confirmation else p.name
            for p in top
        )
        r.plugin_suggestion = ", ".join(p.name for p in top)
        r.text += f" — подходит: {names}"
    return recs
