"""ТЗ-03 §6 / lamp-dictionary.md TODO #3: формализация словаря «слово →
метрика» в код. Числа живут в JSON-пресете (`Project/presets/*.json`,
по умолчанию `amber.json`) — ОДИН источник правды, не дублируется здесь
литералом. Человекочитаемое обоснование того же словаря с p-value —
`_notes/lamp-dictionary.md`; правишь диапазон — правь пресет, коммент в
lamp-dictionary.md держи в синхроне отдельно (это объяснение, не данные).

Использование:
    from analysis.verdict import evaluate, load_preset
    preset = load_preset()  # amber.json по умолчанию
    measurements = {
        ("spectral_slope", "mix"): -5.55,                      # целый трек — число
        ("real_roughness", "vocals"): window_series,            # окна — pandas.Series
        ...
    }
    verdicts = evaluate(measurements, preset)
    for v in verdicts:
        print(v.axis, v.status, v.measured)
"""
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
DEFAULT_PRESET = "amber"


class Reliability(Enum):
    HIGH = "высокая"
    MEDIUM = "средняя"
    LOW = "низкая"
    PENDING = "не хватает данных"
    NO_ZONE = "проверено — единой зоны нет"

    @property
    def weight(self) -> float:
        """Вес в сводном счёте write_report'а — low-reliability метрика
        (напр. harshness, списана в пользу real_roughness) должна реально
        меньше влиять на итог, не только быть помечена как слабая."""
        return {
            Reliability.HIGH: 1.0, Reliability.MEDIUM: 0.6,
            Reliability.LOW: 0.25, Reliability.PENDING: 0.0, Reliability.NO_ZONE: 0.0,
        }[self]


class Status(Enum):
    IN_ZONE = "в зоне «нравится»"
    OUT_OF_ZONE = "вне зоны — не любимая сторона"
    BORDERLINE = "на границе зон"
    UNKNOWN = "диапазон не установлен"
    NO_DATA = "нет измерения"
    NO_ZONE = "единой зоны нет (подтверждено, не измеримо в принципе)"


@dataclass(frozen=True)
class MetricZone:
    metric: str        # имя колонки в данных, напр. "spectral_slope"
    source: str         # "mix" | "vocals" | "bass" | "drums" | "other"
    granularity: str     # "track_avg" | "window" | "section" (ТЗ-05 Б9 —
                          # между окном 4с и целым треком, драматургическая
                          # арка и т.п.)
    axis: str             # ось словаря, человекочитаемо
    liked_lo: Optional[float]
    liked_hi: Optional[float]
    disliked_lo: Optional[float] = None
    disliked_hi: Optional[float] = None
    reliability: Reliability = Reliability.MEDIUM
    note: str = ""
    # ТЗ-05 Б1: как получено число — между разными песнями (риск конфаунда:
    # могла бы объясняться любой другой особенностью тех же 3 треков, не
    # только этой осью — см. Шаг 11), внутри одной песни по времени (разные
    # фрагменты/окна одного трека) или между разными сведёнными версиями
    # одной песни (КП миксы инженера сведения). Обосновывает, почему reliability
    # именно такая — не участвует в счёте отдельно от reliability.
    evidence_type: str = ""  # "between_songs" | "within_song" | "within_mix_versions" | ""
    # ТЗ-05 Б4: на чём калибровалась зона — "mix" (целый микс, не стем),
    # "demucs" (ML-разделённый стем) или "real_stems" (настоящая
    # многодорожечная запись). Зона, откалиброванная на Demucs-стемах,
    # может не переноситься на реальные трек-ауты (bleed/артефакты
    # разделения отсутствуют или другие) — используется для предупреждения
    # в write_report, когда режим входа не совпадает с learned_on.
    learned_on: str = ""
    # ТЗ-05 Б5: несколько зон могут измерять по сути одно и то же свойство
    # разными числами (напр. spectral_slope/warmth_ratio/band_frac_air на
    # миксе — все три про тональный наклон "ярко/тускло"). Без группировки
    # такая зона в сумме голосует 3 раза за одну и ту же ось, а не даёт
    # 3 независимых свидетельства. Зоны с одинаковым cluster усредняются
    # ПЕРЕД суммированием в write_report; пустая строка — зона в кластере
    # ровно с собой (не группируется).
    cluster: str = ""
    # для метрик, где "нравится" — это конкретное направление (выше/ниже),
    # а не диапазон между двумя числами (напр. bass_skewness — просто "чем
    # выше тем лучше", а не "между X и Y")
    direction_only: bool = False
    higher_is_liked: bool = True


_RELIABILITY_FROM_JSON = {
    "high": Reliability.HIGH, "medium": Reliability.MEDIUM,
    "low": Reliability.LOW, "pending": Reliability.PENDING, "no_zone": Reliability.NO_ZONE,
}


def load_preset(name: str = DEFAULT_PRESET) -> list[MetricZone]:
    """Грузит пресет по имени (без .json) из Project/presets/, либо по
    прямому пути, если передан путь к файлу."""
    path = Path(name)
    if not path.suffix:
        path = PRESETS_DIR / f"{name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = []
    for z in data["zones"]:
        zones.append(MetricZone(
            metric=z["metric"], source=z["source"], granularity=z["granularity"], axis=z["axis"],
            liked_lo=z.get("liked_lo"), liked_hi=z.get("liked_hi"),
            disliked_lo=z.get("disliked_lo"), disliked_hi=z.get("disliked_hi"),
            reliability=_RELIABILITY_FROM_JSON[z.get("reliability", "medium")],
            note=z.get("note", ""),
            direction_only=z.get("direction_only", False),
            higher_is_liked=z.get("higher_is_liked", True),
            evidence_type=z.get("evidence_type", ""),
            learned_on=z.get("learned_on", ""),
            cluster=z.get("cluster", ""),
        ))
    return zones


# словарь по умолчанию — загружается один раз при импорте модуля
DICTIONARY: list[MetricZone] = load_preset()


@dataclass
class Verdict:
    metric: str
    source: str
    axis: str
    granularity: str
    measured: Optional[float]
    zone: MetricZone
    status: Status
    fraction_in_zone: Optional[float] = None  # только для window-измерений
    closer_to: Optional[str] = None            # "нравится"/"не нравится" — при BORDERLINE


def _as_scalar_and_fraction(value, zone: MetricZone):
    """Число -> (число, None). pandas.Series -> (медиана, доля окон в зоне
    «нравится», если зона определена)."""
    if pd is not None and isinstance(value, pd.Series):
        value = value.dropna()
        if len(value) == 0:
            return None, None
        median = float(value.median())
        frac = None
        if zone.liked_lo is not None and zone.liked_hi is not None:
            frac = float(((value >= zone.liked_lo) & (value <= zone.liked_hi)).mean())
        return median, frac
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    return float(value), None


def _midpoint(lo, hi):
    return (lo + hi) / 2


def _classify(measured: float, zone: MetricZone) -> tuple[Status, Optional[str]]:
    if zone.reliability is Reliability.NO_ZONE:
        # проверено и подтверждено, что единой зоны «нравится» не существует
        # (напр. overall_ms_ratio — референс А широкая, референс Б почти моно, обе
        # любимые) — не гадаем по числу, честно говорим «не измеримо»
        return Status.NO_ZONE, None
    if zone.direction_only:
        # нет числовой зоны — статус не определяем количественно, только
        # сообщаем измеренное значение и направление, которое считается лучше
        return Status.UNKNOWN, None
    if zone.liked_lo is None or zone.liked_hi is None:
        return Status.UNKNOWN, None
    lo, hi = zone.liked_lo, zone.liked_hi
    if lo <= measured <= hi:
        return Status.IN_ZONE, None
    if zone.disliked_lo is not None and zone.disliked_hi is not None:
        dlo, dhi = zone.disliked_lo, zone.disliked_hi
        if dlo <= measured <= dhi:
            return Status.OUT_OF_ZONE, None
        # не попал ни в одну зону строго — но можно сказать, к какой ближе
        # по расстоянию до середины соответствующей зоны
        d_liked = abs(measured - _midpoint(lo, hi))
        d_disliked = abs(measured - _midpoint(dlo, dhi))
        closer = "нравится" if d_liked < d_disliked else "не нравится"
        return Status.BORDERLINE, closer
    return Status.BORDERLINE, None


def evaluate(measurements: dict, preset: Optional[list[MetricZone]] = None) -> list[Verdict]:
    """measurements: {(metric, source): значение}. Значение — число
    (среднее по треку) или pandas.Series с сырыми значениями по окнам
    (для window-метрик — тогда берём медиану + долю окон в зоне).
    preset: список MetricZone (см. load_preset) — по умолчанию DICTIONARY (amber.json)."""
    out = []
    for zone in (preset if preset is not None else DICTIONARY):
        key = (zone.metric, zone.source)
        if zone.reliability is Reliability.NO_ZONE:
            # не зависит от того, есть ли измерение — метрика в принципе не
            # даёт единой зоны, это не проблема нехватки данных
            measured, _ = _as_scalar_and_fraction(measurements.get(key), zone)
            out.append(Verdict(zone.metric, zone.source, zone.axis, zone.granularity,
                                measured, zone, Status.NO_ZONE))
            continue
        if key not in measurements:
            out.append(Verdict(zone.metric, zone.source, zone.axis, zone.granularity,
                                None, zone, Status.NO_DATA))
            continue
        measured, frac = _as_scalar_and_fraction(measurements[key], zone)
        if measured is None:
            out.append(Verdict(zone.metric, zone.source, zone.axis, zone.granularity,
                                None, zone, Status.NO_DATA))
            continue
        status, closer_to = _classify(measured, zone)
        out.append(Verdict(zone.metric, zone.source, zone.axis, zone.granularity,
                            measured, zone, status, frac, closer_to))
    return out


def format_report(verdicts: list[Verdict]) -> str:
    """Человекочитаемая таблица для консоли/лога."""
    lines = [f"{'источник':<10}{'метрика':<28}{'ось':<38}{'значение':>12}  статус  (надёжность, тип свидетельства)"]
    for v in verdicts:
        val = f"{v.measured:.4f}" if v.measured is not None else "—"
        rel = v.zone.reliability.value
        ev = v.zone.evidence_type or "?"
        status_txt = v.status.value
        if v.status is Status.BORDERLINE and v.closer_to:
            status_txt += f" (ближе к «{v.closer_to}»)"
        lines.append(f"{v.source:<10}{v.metric:<28}{v.axis:<38}{val:>12}  {status_txt}  ({rel}, {ev})")
    return "\n".join(lines)


if __name__ == "__main__":
    # маленький самопроверочный прогон на числах «внешний трек» из отчёта в чате
    demo_measurements = {
        ("spectral_slope_db_per_oct", "mix"): -5.55,
        ("warmth_ratio", "mix"): 3.02,
        ("band_frac_air_median", "mix"): 0.00256,
        ("plr", "mix"): 9.33,
        ("harshness", "vocals"): 0.072,
    }
    print(format_report(evaluate(demo_measurements)))
