"""Блок 2 (Этап 1, «без выбора»): превращает сырую диагностику
(track_avg_metrics diagnostics) в рекомендации человекочитаемым текстом.

Главный принцип (roadmap.md): программа никогда не применяет фикс сама —
только называет точное место, точные параметры и категорию инструмента.
Пока Блок 8 (каталог плагинов) не собран, категория — родовое имя
("declip", "dehum"), не конкретный плагин; когда каталог появится,
`category` — это и есть ключ, по которому Блок 8 подставит конкретное имя
(см. `documentation/roadmap.md`, Блок 8).

Только категория «без выбора» (клиппинг, гул) — однозначный фикс, не
требует матрицы интерференции. «С выбором» (наложение дублей, вкусовые
EQ/динамика) — отдельный, более поздний путь (Блок 7-8), сюда не входит."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Recommendation:
    category: str          # "declip" | "dehum" — ключ для Блока 8, не текст для показа
    source: str              # роль дорожки: "mix" | "vocals" | "bass" | ...
    location_s: tuple         # (start_s, end_s) — точный отрезок, не секция
                                # (см. roadmap.md: восстановление — таймкоды,
                                # вкусовые правки — секции, разная природа)
    params: dict                # параметры фикса: freq_hz, q и т.п. — то, что
                                  # понадобится Блоку 8, чтобы сформулировать
                                  # действие под конкретный плагин
    confidence: float             # 0..1, из детектора (stability_score/аналог)
    text: str                      # готовая строка для отчёта


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
