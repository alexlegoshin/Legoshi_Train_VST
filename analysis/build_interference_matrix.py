"""Блок 6 (roadmap.md): прогоняет ~8-10 канонических ходов
(analysis.interventions.MOVES) по корпусу локальных треков, меряет сдвиг
каждой зоны пресета ДО/ПОСЛЕ хода — только быстрым путём метрик (без
--deep-psychoacoustics, без реверба — Блок 6 сознательно не тянет тяжёлый
психоакустический путь на десятки прогонов), таблица «ход -> вектор дельт
метрик» пишется рядом с пресетом.

Дженерик и без персональных данных: пути к трекам — только аргументы
командной строки, ни один реальный путь/название песни не хранится в
самом файле (см. документацию про анонимизацию публичного репо) —
вызывающий передаёт свой корпус сам, скрипт ничего не предполагает про то,
чей это материал.

Запуск (пример):
    .venv/bin/python analysis/build_interference_matrix.py \\
        --mix /path/to/track1_mix.wav /path/to/track2_mix.wav \\
        --vocals /path/to/track1_vocals.wav \\
        --out presets/interference_matrix.json
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from analysis import engine
from analysis.interventions import MOVES
from analysis.verdict import load_preset

ROLES = ("mix", "vocals", "bass", "drums", "other")
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "presets" / "interference_matrix.json"


def _measure(path: Path, role: str) -> dict:
    """Тот же набор ключей (metric, source), что ждёт verdict.evaluate() —
    track_avg-метрики как есть + медиана по окнам для window-метрик.
    allow_reverb=False и do_real_psychoacoustics=False намеренно — только
    быстрый путь (roadmap.md, Блок 6)."""
    m_avg, vocal_frames, _diag = engine.track_avg_metrics(
        path, role, is_stereo_capable=(role == "mix"), allow_reverb=False)
    mono, sr, _ = engine.load_mono(path)
    wdf = engine.window_metrics(mono, sr, role, do_real_psychoacoustics=False)
    out = dict(m_avg)
    for col in wdf.columns:
        if col in ("t_start", "t_end", "rms_dbfs"):
            continue
        if wdf[col].notna().sum() == 0:
            continue
        out[(col, role)] = float(wdf[col].median())
    return out


def build_matrix(track_paths_by_role: dict, work_dir: Path, zone_keys: set, verbose: bool = False) -> list:
    """track_paths_by_role: {role: [Path, ...]}. zone_keys: {(metric, source)}
    из пресета — считаем дельту только там, где есть зона, незачем мерить
    остальное. Возвращает плоский список строк
    {move, role, track, metric, delta} — агрегация (summarize) отдельно,
    сырые строки полезны сами по себе для последующей калибровки.

    verbose=True печатает прогресс по (роль, трек, ход) — прогон на
    реальном корпусе занимает десятки минут, без построчного лога всё
    время выглядит как чёрный ящик до самого конца (см. roadmap.md,
    найдено на первом реальном прогоне)."""
    total_sources = sum(len(paths) for role, paths in track_paths_by_role.items()
                         if any(k[1] == role for k in zone_keys))
    done_sources = 0
    rows = []
    for role, paths in track_paths_by_role.items():
        role_keys = {k for k in zone_keys if k[1] == role}
        if not role_keys:
            continue
        for path in paths:
            if verbose:
                print(f"[{done_sources + 1}/{total_sources}] {role}: {path.name} — базовое измерение...")
            data, sr = sf.read(str(path), dtype="float64", always_2d=True)
            baseline = _measure(path, role)
            for i, (move_name, fn) in enumerate(MOVES.items(), 1):
                if verbose:
                    print(f"    [{i}/{len(MOVES)}] {move_name}...")
                modified = fn(data, sr)
                tmp_path = work_dir / f"_{move_name}_{Path(path).stem}.wav"
                sf.write(str(tmp_path), modified, sr, subtype="FLOAT")
                after = _measure(tmp_path, role)
                for key in role_keys:
                    if key in baseline and key in after:
                        rows.append(dict(move=move_name, role=role, track=Path(path).stem,
                                          metric=key[0], delta=after[key] - baseline[key]))
            done_sources += 1
    return rows


def summarize(rows: list) -> dict:
    """{move: {"role::metric": {median_delta, n}}} — медиана по трекам
    корпуса на один (ход, метрика), не среднее — устойчивее к одному
    треку-выбросу при малом корпусе."""
    by_move = {}
    for r in rows:
        by_move.setdefault(r["move"], {}).setdefault(f"{r['role']}::{r['metric']}", []).append(r["delta"])
    return {move: {k: dict(median_delta=float(np.median(v)), n=len(v)) for k, v in metrics.items()}
            for move, metrics in by_move.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for role in ROLES:
        ap.add_argument(f"--{role}", nargs="*", default=[], help=f"пути к файлам роли {role}")
    ap.add_argument("--preset", default="amber")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    track_paths_by_role = {role: [Path(p) for p in getattr(args, role)] for role in ROLES}
    track_paths_by_role = {k: v for k, v in track_paths_by_role.items() if v}
    if not track_paths_by_role:
        print("Не передано ни одного файла (--mix/--vocals/--bass/--drums/--other) — нечего анализировать")
        return

    zones = load_preset(args.preset)
    zone_keys = {(z.metric, z.source) for z in zones}

    with tempfile.TemporaryDirectory(prefix="interference_matrix_") as tmp:
        rows = build_matrix(track_paths_by_role, Path(tmp), zone_keys, verbose=True)

    result = summarize(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out_path} ({len(rows)} измерений, {len(result)} ходов)")


if __name__ == "__main__":
    main()
