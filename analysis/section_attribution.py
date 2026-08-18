"""ТЗ-05 Блок 5: атрибуция window-метрик по (роль, СЕКЦИЯ), не по
таймкоду. Скользящее окно (engine.WIN_S/HOP_S) не даёт чётких границ
события — «в припевах ярче на 1.8дБ» честно, «с 1:23 до 1:27» — ложная
точность, порождённая размером окна, а не реальной границей проблемы
(roadmap.md, Блок 5).

Секции размечены ТОЛЬКО на миксе (sections.analyze), но применимы к любой
роли: обе схемы источников (Demucs, трек-аут) уже дают общую временную ось
с миксом — тот же факт, на который опирается Блок 4 (masking_erb)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


def attribute_by_section(wdf: pd.DataFrame, section_profile: pd.DataFrame) -> dict:
    """wdf — DataFrame из engine.window_metrics (со столбцами t_start,
    t_end и метриками). section_profile — DataFrame(section,start_s,end_s,
    ...) из sections.analyze. Возвращает {metric_col: {section_label:
    медиана}} — окно относится к секции по t_start (не по середине окна:
    при типичных длинах секций/окон разница на границе — доли окна, не
    стоит усложнения).

    Окна, чей t_start не попал ни в одну размеченную секцию (напр. между
    последней секцией и концом файла, или в "переход"/"пауза", исключённые
    из profile при ручной разметке) — просто выпадают, не искажают
    остальные секции выдумкой."""
    if section_profile is None or not len(section_profile) or wdf is None or not len(wdf):
        return {}

    starts = section_profile["start_s"].to_numpy()
    ends = section_profile["end_s"].to_numpy()
    labels = section_profile["section"].to_numpy()

    def _label_for(t):
        idx = np.where((starts <= t) & (t < ends))[0]
        if len(idx) == 0:
            return None
        lbl = labels[idx[0]]
        return lbl if lbl else f"{starts[idx[0]]:.0f}-{ends[idx[0]]:.0f}с"

    section_col = wdf["t_start"].map(_label_for)

    out = {}
    for col in wdf.columns:
        if col in ("t_start", "t_end", "rms_dbfs"):
            continue
        if wdf[col].notna().sum() == 0:
            continue
        grouped = wdf.assign(_section=section_col).dropna(subset=[col])
        grouped = grouped[grouped["_section"].notna()]
        if not len(grouped):
            continue
        medians = grouped.groupby("_section")[col].median()
        if len(medians):
            out[col] = {k: float(v) for k, v in medians.items()}
    return out


def worst_section(section_medians: dict, zone) -> tuple:
    """Среди медиан по секциям для одной метрики — какая секция дальше
    всего от зоны «нравится» (см. verdict.delta_to_zone). Возвращает
    (label, median, delta) либо (None, None, None), если зона без
    числового диапазона, секций нет, или (БАГ, найден код-ревью,
    исправлен) КАЖДАЯ секция по отдельности формально внутри зоны
    (delta_to_zone == 0.0 у всех) — раньше в этом случае функция всё
    равно возвращала первую по порядку секцию как «худшую» с delta=0.0
    (worst_delta инициализировался None и первый d==0.0 проходил
    проверку `worst_delta is None or ...`), хотя «худшей» на деле нет:
    verdict.py считает медиану по ВСЕМ сырым window-значениям для
    итогового статуса зоны, а эта функция группирует окна СНАЧАЛА по
    секциям и берёт медиану ВНУТРИ каждой — две агрегации могут
    расходиться, и общий вердикт может быть OUT_OF_ZONE/BORDERLINE, даже
    когда каждая секция по отдельности в норме. Указывать в такой
    ситуации конкретную секцию как источник проблемы — вводит в
    заблуждение."""
    from analysis.verdict import delta_to_zone

    if not section_medians:
        return None, None, None
    worst_label, worst_val, worst_delta = None, None, None
    for label, val in section_medians.items():
        d = delta_to_zone(val, zone)
        if d is None:
            return None, None, None  # зона без диапазона — дельта не определена ни для одной секции
        if d == 0.0:
            continue  # эта секция сама по себе в зоне — не кандидат в "худшие"
        if worst_delta is None or abs(d) > abs(worst_delta):
            worst_label, worst_val, worst_delta = label, val, d
    return worst_label, worst_val, worst_delta
