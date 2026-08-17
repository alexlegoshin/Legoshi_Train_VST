"""ТЗ-05 Б9: третий уровень гранулярности между окном (4с) и целым треком —
секции (куплет/припев/бридж и т.д.). Без него не видна "драматургическая
арка" — разница громкости между началом и кульминацией трека: window-
метрики усредняют/медианят весь трек в одно распределение, track_avg — тоже
одно число, обе одинаково слепы к МАКРО-профилю "тихо в начале, громко в
кульминации". Пример находки, которую без этого уровня нельзя было увидеть
и потерявшейся при реорганизации в продукт: у демки контраст между интро и
кульминацией был ~8-9дБ, у сведений инженера сведения — вдвое меньше, ~4дБ (см.
_notes/TZ-05-audit.md, Б9) — трек стал заметно более "выровненным" по
динамике на макро-уровне, что не видно ни на overall LUFS (обе версии
нормализованы к одному integrated loudness), ни на medians окон.

Границы секций — либо ручная разметка (_analysis/sections.csv, есть для
"основной трек" и "референс А"), либо автоматическая сегментация по
self-similarity+novelty (Foote, 2000) для любого нового трека, где ручной
разметки нет — продукт должен работать не только на изученных треках."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln

HOP = 2048  # ~46мс при 44.1к — масштаб структуры, не транзиентов
MIN_SECTION_S = 5.0
SECTIONS_CSV = Path(__file__).resolve().parents[2] / "_analysis" / "sections.csv"
# "переход"/"пауза" — не музыкальные секции, а связки между ними; при
# автосегментации таких меток не бывает, при ручной — исключаем, чтобы
# короткая тихая связка не попала в "2 самые ранние"/"2 самые громкие"
# случайно, а не по драматургии
NON_MUSICAL_LABELS = {"переход", "пауза"}


def foote_novelty(ssm, kernel_size):
    """Шахматное (checkerboard) ядро Foote вдоль диагонали SSM."""
    half = kernel_size // 2
    ax = np.arange(-half, half)
    sign = np.sign(np.outer(ax, ax))
    gauss = np.outer(np.exp(-ax**2 / (2 * (half / 2)**2)), np.exp(-ax**2 / (2 * (half / 2)**2)))
    kernel = sign * gauss

    n = ssm.shape[0]
    novelty = np.zeros(n)
    for i in range(half, n - half):
        block = ssm[i - half:i + half, i - half:i + half]
        novelty[i] = np.sum(block * kernel)
    novelty -= novelty.min()
    novelty /= max(novelty.max(), 1e-9)
    return novelty


def _pick_peaks(novelty, frame_rate, min_sep_s=MIN_SECTION_S):
    min_sep = int(min_sep_s * frame_rate)
    idx = np.where((novelty[1:-1] > novelty[:-2]) & (novelty[1:-1] > novelty[2:]))[0] + 1
    idx = idx[novelty[idx] > 0.15]
    idx = sorted(idx, key=lambda i: -novelty[i])
    kept = []
    for i in idx:
        if all(abs(i - k) > min_sep for k in kept):
            kept.append(i)
    return sorted(kept)


def auto_segment_boundaries(mono, sr) -> list:
    """Автоматические границы секций по MFCC+chroma self-similarity —
    работает на ЛЮБОМ треке, не только на изученных с ручной разметкой.
    mono — сигнал ЦЕЛОГО трека (секции по определению крупнее окна 4с)."""
    import librosa
    y = mono.astype(np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    feat = np.vstack([mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-9),
                       chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)])
    ssm = feat.T @ feat
    frame_rate = sr / HOP
    novelty = foote_novelty(ssm, kernel_size=int(8 * frame_rate))
    peaks = _pick_peaks(novelty, frame_rate)
    boundary_times = [0.0] + [p / frame_rate for p in peaks] + [len(mono) / sr]
    return sorted(set(round(t, 1) for t in boundary_times))


def load_manual_sections(song: str, sections_csv: Path = SECTIONS_CSV):
    """DataFrame с колонками start_s/end_s/section для `song`, либо None,
    если ручной разметки для этой песни нет (тогда — auto_segment_boundaries)."""
    if not sections_csv.exists():
        return None
    df = pd.read_csv(sections_csv)
    sub = df[df.song == song]
    return sub[["start_s", "end_s", "section"]].reset_index(drop=True) if len(sub) else None


def section_loudness_profile(mono, sr, sections) -> pd.DataFrame:
    """sections: DataFrame(start_s,end_s[,section]) ИЛИ список границ
    [t0,t1,...,tN] (тогда секции — соседние пары, безымянные).
    Возвращает start_s/end_s/section/lufs/level_rel_db (относительно
    среднего integrated loudness ВСЕГО трека — так секции сопоставимы
    между версиями с разной абсолютной громкостью мастеринга)."""
    if isinstance(sections, (list, tuple, np.ndarray)):
        bounds = sorted(sections)
        sections = pd.DataFrame({"start_s": bounds[:-1], "end_s": bounds[1:],
                                  "section": [f"секция {i+1}" for i in range(len(bounds) - 1)]})

    meter = pyln.Meter(sr)
    track_lufs = meter.integrated_loudness(mono)

    rows = []
    for _, r in sections.iterrows():
        s, e = int(r.start_s * sr), min(int(r.end_s * sr), len(mono))
        seg = mono[s:e]
        if len(seg) < int(0.5 * sr):  # короче 0.5с — не измерение
            continue
        try:
            lufs = meter.integrated_loudness(seg)
        except Exception:
            lufs = float("nan")
        if not np.isfinite(lufs):
            rms = np.sqrt(np.mean(seg ** 2)) + 1e-12
            lufs = 20 * np.log10(rms)  # тихая/короткая секция — RMS-фолбэк, не строгий LUFS
        rows.append(dict(start_s=r.start_s, end_s=r.end_s, section=r.get("section", ""),
                          lufs=float(lufs), level_rel_db=float(lufs - track_lufs)))
    return pd.DataFrame(rows)


def dramaturgical_arc(profile: pd.DataFrame, exclude_labels=NON_MUSICAL_LABELS) -> dict:
    """arc_db = среднее 2 самых громких секций минус среднее 2 самых ранних
    по времени (не самых тихих — именно ранних, это и есть "как трек
    открывается против того, куда он приходит"). Плюс разброс между
    секциями (spread=std, range=max-min) — сам факт наличия макро-динамики,
    не только направление."""
    p = profile[~profile["section"].isin(exclude_labels)].sort_values("start_s")
    if len(p) < 2:
        return dict(arc_db=float("nan"), section_spread_db=float("nan"), section_range_db=float("nan"))
    levels = p["level_rel_db"].values
    earliest2 = levels[:2].mean()
    loudest2 = np.sort(levels)[-2:].mean()
    return dict(arc_db=float(loudest2 - earliest2),
                section_spread_db=float(np.std(levels)),
                section_range_db=float(levels.max() - levels.min()))


def analyze(mono, sr, song: str = None) -> tuple:
    """Собирает всё вместе: ручная разметка, если есть для `song`, иначе
    автосегментация. Возвращает (profile_df, arc_dict, source) — source
    ∈ {"manual","auto"} для отчёта (Б4-style — надо знать, откуда границы)."""
    sections = load_manual_sections(song) if song else None
    source = "manual"
    if sections is None:
        sections = auto_segment_boundaries(mono, sr)
        source = "auto"
    profile = section_loudness_profile(mono, sr, sections)
    arc = dramaturgical_arc(profile)
    return profile, arc, source
