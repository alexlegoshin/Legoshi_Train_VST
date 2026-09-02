"""Блок 2 (Этап 1, «без выбора») + Блок 7 («с выбором»): превращает сырую
диагностику в рекомендации человекочитаемым текстом. Единая точка
унификации для ВСЕХ категорий (roadmap.md) — Recommendation один и тот же
класс для restoration (Блок 2) и taste (Блок 7).

Главный принцип (roadmap.md): программа никогда не применяет фикс сама —
только называет точное место, точные параметры и категорию инструмента.
Пока Блок 8 (каталог плагинов) не собран, категория — родовое имя
("declip", "dehum", имя канонического хода Блока 6), не конкретный
плагин; когда каталог появится, `category` — это и есть ключ, по которому
Блок 8 подставит конкретное имя (см. `documentation/roadmap.md`, Блок 8).

«Без выбора» (клиппинг, гул) — однозначный фикс, не требует матрицы
интерференции, таймкоды. «С выбором» (вкусовые EQ/динамика, наложение
дублей) — использует matrix интерференции (Блок 6) и секции, не таймкоды
(см. roadmap.md: восстановление — точный момент, вкусовая правка —
секция, скользящее окно не даёт чёткой границы события)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recommendation:
    category: str          # "declip" | "dehum" | имя хода Блока 6 — ключ для Блока 8, не текст для показа
    source: str              # роль дорожки: "mix" | "vocals" | "bass" | ...
    location_s: tuple         # (start_s, end_s) — точный отрезок, не секция;
                                # None для «по всей дорожке» и для taste-рекомендаций
                                # (см. roadmap.md: восстановление — таймкоды,
                                # вкусовые правки — секции, разная природа)
    params: dict                # параметры фикса: freq_hz, q и т.п. — то, что
                                  # понадобится Блоку 8, чтобы сформулировать
                                  # действие под конкретный плагин
    confidence: float             # 0..1, из детектора (stability_score/аналог)
                                    # или из _confidence() для taste-рекомендаций
    text: str                      # готовая строка для отчёта
    section: Optional[str] = None   # Блок 7: секция (не таймкод) для window-зон,
                                      # None для track_avg-зон (весь трек) и для Блока 2
    stage: Optional[str] = None      # Блок 7: стадия сведения хода (см. MOVE_STAGES) —
                                       # только для группировки вывода, НЕ заявка на
                                       # проверенный совместный эффект нескольких
                                       # ходов сразу — Блок 6 мерил каждый ход
                                       # отдельно, комбинации не тестировались
    plugin_suggestion: Optional[str] = None  # Блок 8: имя(имена) плагина(ов) из
                                               # presets/plugins.json, чей
                                               # canonical_interventions содержит
                                               # category — заполняется отдельным
                                               # вызовом plugin_catalog.enrich_with_plugins,
                                               # не здесь (каталог персональный,
                                               # необязательный для работы рекомендателя)


def _clipping_recommendations(source: str, clipping_detail: dict) -> list:
    if not clipping_detail or not clipping_detail.get("clipped"):
        return []
    regions = clipping_detail["clipped_regions_s"]
    fraction_pct = clipping_detail["clipped_fraction"] * 100
    out = []
    for r in regions:
        start_s, end_s = r["start_s"], r["end_s"]
        # категория — не просто "declip": short click vs sustained run
        # обычно требуют разного режима восстановления (см. engine.py,
        # CLICK_MAX_S) — разные записи каталога Блока 8, не один параметр
        category = f"declip_{r['severity']}"
        ch_str = "оба канала" if len(r["channels"]) > 1 else (
            f"канал {r['channels'][0]}" if r["channels"] else "?")
        out.append(Recommendation(
            category=category,
            source=source,
            location_s=(start_s, end_s),
            params=dict(duration_s=r["duration_s"], channels=r["channels"], severity=r["severity"]),
            confidence=1.0,  # клиппинг — измерение, не оценка, порог фиксирован
            text=(f"[{source}] {start_s:.2f}-{end_s:.2f}с: клиппинг "
                  f"({r['duration_s']:.3f}с, {r['severity']}, {ch_str}) — нужен declip"),
        ))
    if len(regions) > 1:
        out[0].text += f" (всего клиппинга по дорожке: {fraction_pct:.2f}% длительности)"
    return out


def _hum_recommendations(source: str, hum_candidates: list) -> list:
    if not hum_candidates:
        return []
    out = []
    for c in hum_candidates:
        freq = c.get("freq_hz_refined", c["freq_hz"])
        # stability_score не ограничен [0,1] по конструкции (prominence/std) —
        # переводим в грубую уверенность для показа, не для отсева (отсев уже
        # сделан в engine.py порогом >=3.0 до попадания сюда)
        confidence = min(1.0, c["stability_score"] / 10.0)
        out.append(Recommendation(
            category="dehum",
            source=source,
            location_s=None,  # гул присутствует по всей дорожке — не точечный отрезок
            params=dict(freq_hz=round(freq, 1)),
            confidence=confidence,
            text=f"[{source}] по всей дорожке: гул на {freq:.1f}Гц — нужен dehum (notch-фильтр)",
        ))
    return out


def restoration_recommendations(diagnostics: dict, source: str) -> list:
    """diagnostics — словарь ОДНОГО источника (то, что track_avg_metrics
    кладёт в diagnostics для этой роли), не весь diagnostics прогона."""
    out = []
    out.extend(_clipping_recommendations(source, diagnostics.get("clipping_detail")))
    out.extend(_hum_recommendations(source, diagnostics.get("hum_candidates")))
    return out


def all_restoration_recommendations(all_diagnostics: dict) -> list:
    """all_diagnostics — {role: diagnostics}, как строит analyze_all_sources.
    Сортировка — двойная, как договорились в roadmap.md: сначала по позиции
    на таймлайне (без отрезка — гул по всей дорожке — в конец), поверх — по
    убыванию уверенности внутри той же позиции."""
    out = []
    for role, d in all_diagnostics.items():
        if role.startswith("_"):  # "_run", "_trackout" — служебные, не источник
            continue
        out.extend(restoration_recommendations(d, role))
    out.sort(key=lambda r: (r.location_s[0] if r.location_s else float("inf"), -r.confidence))
    return out


# --------------------------------------------------------------------------
# Блок 7 (Этап 2, «с выбором»): вкусовые рекомендации из verdict.evaluate()
# --------------------------------------------------------------------------
# Кандидат-ход по (metric, source) при выходе за зону — черновик из
# roadmap.md, Блок 7 (таблица «ось зоны -> кандидат-ход»), заполнен
# заранее по DSP-смыслу оси. Финальный выбор хода при генерации
# рекомендации ВСЁ РАВНО идёт по реальным числам interference_matrix.json
# (см. _best_covered_move) — эта таблица только сужает круг до
# a priori осмысленных кандидатов, не позволяя случайной корреляции на
# n=1-2 треках выдумать несуществующую связь (см. roadmap.md, находка на
# калибровке: bell_boost_lowmid эмпирически поднимал mix::plr, хотя по
# DSP-смыслу это не про то же самое явление — совпадение на маленьком
# корпусе, не реальный эффект; такие связи отсекаются тем, что их вообще
# нет в этой таблице).
TASTE_MOVE_CANDIDATES = {
    ("spectral_slope_db_per_oct", "mix"): {"above": ["shelf_air_cut", "bell_cut_presence"],
                                             "below": ["shelf_air_boost", "bell_boost_presence"]},
    ("warmth_ratio", "mix"): {"above": ["bell_cut_lowmid", "shelf_low_cut"], "below": ["shelf_low_boost"]},
    ("band_frac_air_median", "mix"): {"above": ["shelf_air_cut"], "below": ["shelf_air_boost"]},
    ("plr", "mix"): {"above": ["compressor_soft"], "below": None},
    ("harshness", "vocals"): {"above": ["bell_cut_presence"], "below": ["bell_boost_presence"]},
    ("real_roughness", "vocals"): {"above": ["bell_cut_presence"], "below": None},
    ("real_sharpness", "vocals"): {"above": ["bell_cut_presence", "shelf_air_cut"], "below": None},
    ("spectral_slope", "bass"): {"above": ["bell_cut_presence"], "below": ["bell_boost_presence", "shelf_air_boost"]},
    ("spectral_slope", "other"): {"above": ["bell_cut_presence"], "below": ["bell_boost_presence", "shelf_air_boost"]},
    ("band_frac_lowmid", "drums"): {"above": ["bell_cut_lowmid"], "below": ["bell_boost_lowmid"]},
    ("band_frac_low", "other"): {"above": ["shelf_low_cut"], "below": ["shelf_low_boost"]},
}
# (metric, source, direction) — связки, где сам DSP-смысл оси не строго
# спектральный (roughness/sharpness психоакустические, не чисто EQ) —
# рекомендация всё равно строится, но с пометкой и пониженной уверенностью
EXPERIMENTAL_TASTE_ZONES = {
    ("harshness", "vocals", "below"),
    ("real_roughness", "vocals", "above"),
    ("real_sharpness", "vocals", "above"),
}

# Стадия хода — только для группировки вывода (roadmap.md: уровни ->
# вычитающий EQ -> компрессия -> добавляющий EQ -> пространство).
# ВАЖНО: Блок 6 мерил каждый ход ОТДЕЛЬНО, не комбинациями — стадия НЕ
# заявка на проверенный совместный эффект нескольких ходов внутри одной
# стадии, только читаемая группировка по классическому порядку сведения
MOVE_STAGES = {
    "bell_cut_lowmid": "вычитающий EQ", "bell_cut_presence": "вычитающий EQ",
    "shelf_air_cut": "вычитающий EQ", "shelf_low_cut": "вычитающий EQ",
    "bell_boost_lowmid": "добавляющий EQ", "bell_boost_presence": "добавляющий EQ",
    "shelf_air_boost": "добавляющий EQ", "shelf_low_boost": "добавляющий EQ",
    "compressor_soft": "компрессия",
    "saturation_soft": "сатурация",
}
# один и тот же физический параметр (полоса), противоположные направления —
# если оба попали в вывод для одной роли, это реальный, а не выдуманный
# конфликт («честные неразрешимые конфликты», roadmap.md, Блок 7)
OPPOSING_MOVE_PAIRS = [
    ("bell_cut_lowmid", "bell_boost_lowmid"),
    ("bell_cut_presence", "bell_boost_presence"),
    ("shelf_air_boost", "shelf_air_cut"),
    ("shelf_low_boost", "shelf_low_cut"),
]


def _best_covered_move(role: str, metric: str, direction: str, interference_matrix: dict):
    """Среди a priori осмысленных кандидатов (TASTE_MOVE_CANDIDATES) —
    тот, чья РЕАЛЬНО ИЗМЕРЕННАЯ (Блок 6) дельта для (role, metric)
    совпадает по знаку с нужным направлением и максимальна по модулю.
    Не гадаем: кандидат без эмпирического подтверждения правильного
    знака не рекомендуется, даже если он в таблице. Возвращает
    (move_name, predicted_delta, n) либо None."""
    candidates = TASTE_MOVE_CANDIDATES.get((metric, role), {}).get(direction)
    if not candidates:
        return None
    wanted_sign = 1 if direction == "below" else -1
    key = f"{role}::{metric}"
    scored = []
    for move in candidates:
        pred = interference_matrix.get(move, {}).get(key)
        if pred is None or pred["median_delta"] == 0:
            continue
        sign = 1 if pred["median_delta"] > 0 else -1
        if sign != wanted_sign:
            continue
        scored.append((move, pred["median_delta"], pred["n"]))
    if not scored:
        return None
    scored.sort(key=lambda c: -abs(c[1]))
    return scored[0]


def _taste_confidence(experimental: bool, n: int) -> float:
    """Уверенность — не ложная точность из формулы с n=1-2, честная
    грубая шкала: a priori осмысленный ход vs экспериментальная связка,
    подрезано, если эмпирика опирается на один трек корпуса."""
    base = 0.4 if experimental else 0.7
    if n <= 1:
        base *= 0.7
    return round(base, 2)


def taste_recommendation_for_verdict(v, diagnostics: dict, interference_matrix: dict):
    """v — Verdict из verdict.evaluate(). diagnostics — ПОЛНЫЙ словарь
    прогона (не одной роли — нужен diagnostics['mix']['section_profile']
    для атрибуции по секции). Возвращает Recommendation либо None (в
    зоне/direction_only/NO_ZONE/нет измерения — рекомендовать нечего)."""
    from analysis import section_attribution
    from analysis.verdict import Reliability, Status

    if v.zone.reliability is Reliability.NO_ZONE:
        return None
    if v.status not in (Status.OUT_OF_ZONE, Status.BORDERLINE):
        return None
    if not v.delta_to_zone:  # None (direction_only) или 0.0 (уже в зоне)
        return None

    direction = "below" if v.delta_to_zone > 0 else "above"
    role, metric = v.source, v.metric

    section_label = None
    if v.granularity == "window":
        sec_medians = ((diagnostics.get(role) or {}).get("section_medians") or {}).get(metric)
        if sec_medians:
            label, _val, _delta = section_attribution.worst_section(sec_medians, v.zone)
            section_label = label

    where = f" ({section_label})" if section_label else ""
    best = _best_covered_move(role, metric, direction, interference_matrix)
    if best is None:
        return Recommendation(
            category="taste_no_move", source=role, location_s=None, section=section_label, stage=None,
            params=dict(metric=metric, direction=direction, delta_to_zone=v.delta_to_zone),
            confidence=0.0,
            text=(f"[{role}]{where}: {v.axis} вне зоны (дельта {v.delta_to_zone:+.2g}) — "
                  f"конкретного хода из Блока 6 для этой оси нет, экспериментально/не покрыто"),
        )

    move_name, predicted_delta, n = best
    stage = MOVE_STAGES.get(move_name)
    experimental = (metric, role, direction) in EXPERIMENTAL_TASTE_ZONES
    confidence = _taste_confidence(experimental, n)
    from analysis.interventions import MOVE_DESCRIPTIONS
    description = MOVE_DESCRIPTIONS.get(move_name, move_name)

    return Recommendation(
        category=move_name, source=role, location_s=None, section=section_label, stage=stage,
        params=dict(metric=metric, direction=direction, delta_to_zone=v.delta_to_zone,
                    predicted_delta=predicted_delta, interference_n=n, experimental=experimental),
        confidence=confidence,
        text=(f"[{role}]{where}: {v.axis} — {description} (стадия: {stage})"
              f"{', экспериментальная связка' if experimental else ''}"),
    )


def _layering_recommendation(role: str, pair: dict):
    """Наложение дублей (Блок 3, измерение) -> рекомендация Блока 7:
    художественный выбор, не однозначный фикс (roadmap.md) — программа
    называет измерение и предлагает решить руками, не диктует."""
    if "error" in pair:
        return None
    a, b = pair["pair"]
    return Recommendation(
        category="layering_choice", source=role, location_s=None, section=None, stage="наложение дублей",
        params=dict(pair=[a, b], time_divergence_ms=pair["time_divergence_ms_median"],
                    pitch_divergence_cents=pair["pitch_divergence_cents_median"],
                    comb_risk_upper_bound=pair["comb_risk_upper_bound"]),
        confidence=0.5,  # измерение, но фикс — художественный выбор, не однозначный (см. roadmap.md)
        text=(f"[{role}] дубли {a} vs {b}: расхождение по времени "
              f"{pair['time_divergence_ms_median']:.1f}мс, по питчу {pair['pitch_divergence_cents_median']:.1f}¢, "
              f"верхняя оценка риска гребёнки {pair['comb_risk_upper_bound']:.0%} — "
              f"реши сам: свести в ноль (чисто), развести (эффект хоруса) или только по панораме"),
    )


def _flag_opposing_conflicts(recs: list) -> None:
    """Один и тот же физический параметр, противоположные направления,
    одна роль — реальный конфликт (не выдуманный, не требует непротестированных
    данных о комбинациях ходов), помечаем прямо в тексте и params, не молчим
    («честные неразрешимые конфликты», roadmap.md, Блок 7). Мутирует recs на месте."""
    by_role = {}
    for r in recs:
        by_role.setdefault(r.source, []).append(r)
    for role, role_recs in by_role.items():
        categories = {r.category for r in role_recs}
        for move_a, move_b in OPPOSING_MOVE_PAIRS:
            if move_a in categories and move_b in categories:
                for r in role_recs:
                    if r.category in (move_a, move_b):
                        r.params["conflict_with"] = move_b if r.category == move_a else move_a
                        r.text += f" — КОНФЛИКТ с рекомендацией «{r.params['conflict_with']}» для этой же роли, реши сам"


def all_taste_recommendations(verdicts: list, diagnostics: dict, interference_matrix: dict,
                               section_profile=None) -> list:
    """Полный список за один запуск (roadmap.md, Блок 7) — не одна правка
    за раз. Двойная сортировка: сначала по позиции на таймлайне (по
    старту секции, без секции — в конец), поверх — по убыванию
    уверенности. Плюс наложение дублей (Блок 3) — тот же список,
    художественный выбор.

    section_profile — необязательно передать явно (DataFrame секций
    mix). Если не передан, ищем в diagnostics['mix']['section_profile'] —
    НО вызывающий код (orchestrate.write_report) может к этому моменту
    уже вынуть этот ключ из diagnostics['mix'] через pop() для отдельной
    JSON-сериализации (тот же объект, не копия — mutable). БАГ (найден
    код-ревью, исправлен): раньше это было единственным способом получить
    section_profile здесь, и он молча возвращал {} после pop() — двойная
    сортировка по таймлайну вырождалась в сортировку только по
    уверенности. Явный параметр — самый надёжный способ, не полагается на
    то, что кто-то ещё не тронул diagnostics."""
    recs = []
    for v in verdicts:
        r = taste_recommendation_for_verdict(v, diagnostics, interference_matrix)
        if r is not None:
            recs.append(r)

    for role, d in (diagnostics or {}).items():
        if role.startswith("_"):
            continue
        for pair in (d.get("layering_pairs") or []):
            r = _layering_recommendation(role, pair)
            if r is not None:
                recs.append(r)

    _flag_opposing_conflicts(recs)

    section_starts = {}
    if section_profile is None:
        section_profile = ((diagnostics or {}).get("mix") or {}).get("section_profile")
    if section_profile is not None and len(section_profile):
        for _, row in section_profile.iterrows():
            label = row["section"] or f"{row['start_s']:.0f}-{row['end_s']:.0f}с"
            section_starts[label] = float(row["start_s"])

    recs.sort(key=lambda r: (section_starts.get(r.section, float("inf")), -r.confidence))
    return recs
