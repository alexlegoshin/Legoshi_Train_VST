#!/usr/bin/env python3
"""Legoshi Train — консольный оркестратор. Кладёшь в Project/import/:

  * ОДИН аудиофайл — полный микс, отдельных дорожек нет.
    Запускается Demucs (htdemucs_ft), дальше метрики на миксе +
    выделенных вокале/басе/барабанах/остальном.

  * ПАПКУ с несколькими файлами — трек-аут. Один файл — главный
    референс-микс (имя содержит main/mix/master/full), остальные —
    отдельные дорожки. Роль дорожки определяется по английскому
    ключевому слову в имени файла (vocal/vox/bass/drum/kick/snare/...).
    Дорожек одной роли может быть несколько (например два вокальных
    дубля) — они суммируются в один сигнал на роль. Настоящие стемы
    точнее Demucs, деление не нужно.

Результат — в Project/output/<имя>_<таймстемп>/: report.txt (читаемый
вердикт по пресету Legoshi Amber — оригинальный пресет разработчика,
presets/legoshi_amber.json) и measurements.json (сырые числа).

Запуск:
    .venv/bin/python Project/orchestrate.py                  # все новые файлы/папки в import/
    .venv/bin/python Project/orchestrate.py "мой трек.wav"    # конкретный файл из import/
    .venv/bin/python Project/orchestrate.py --preset legoshi_amber --deep-psychoacoustics
"""
import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from analysis import alignment, engine, recommendations, section_attribution, sections
from analysis.metrics import layering, masking_erb
from analysis.verdict import evaluate, format_report, load_preset, Reliability, Status

ROOT = Path(__file__).resolve().parent
IMPORT_DIR = ROOT / "import"
OUTPUT_DIR = ROOT / "output"
PRESETS_DIR = ROOT / "presets"
AUDIO_EXT = {".wav", ".mp3", ".flac", ".aiff", ".aif"}


def load_interference_matrix() -> dict:
    """Блок 7: пусто, если матрица ещё не построена (Блок 6 не прогнан) —
    не падать, taste-рекомендации просто не строятся (см.
    recommendations.all_taste_recommendations, отсутствие данных в
    interference_matrix означает "нет хода", не ошибку)."""
    path = PRESETS_DIR / "interference_matrix.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

# --- определение роли дорожки в трек-ауте ---
# ТЗ-05 А10: английские ключевые слова — по явному запросу пользователя,
# остаются основными. Русские добавлены НАРЯДУ (не вместо) — реальный
# трек-аут пользователя называет дорожки по-русски (вокал основной.wav,
# драмсы.wav и т.п.), без них инструмент не читает свой же профильный ввод.
MAIN_KEYWORDS = ["main", "mix", "master", "full", "микс", "мастер", "общ", "итог"]
VOCAL_KEYWORDS = ["vocal", "vox", "voice", "choir", "bgv",
                    "вокал", "голос", "бэк", "хор"]
BASS_KEYWORDS = ["bass", "бас"]
DRUMS_KEYWORDS = ["drum", "kick", "snare", "hihat", "hi-hat", "hat", "overhead",
                   "tom", "cymbal", "perc", "percussion",
                   "драм", "удар", "бочк", "малый", "хай", "том", "тарелк", "перкусс"]


def _norm(s: str) -> str:
    """ТЗ-05 А10: NFC-нормализация перед сравнением — macOS APFS отдаёт
    имена файлов в NFD (кириллица разложена на буква+диакритика), а
    ключевые слова в коде — в NFC. Без нормализации 'вокал' в коде и
    'вокал' в NFD-имени файла byte-for-byte не совпадают (тот же баг,
    что уже ловили в исследовательском коде на «й» — см. lamp-dictionary
    §5, run_tz02_task3). unicodedata.normalize делает сравнение byte-safe
    независимо от того, в какой форме пришло имя файла."""
    import unicodedata
    return unicodedata.normalize("NFC", s).lower()


def classify_role(filename: str) -> str:
    name = _norm(filename)
    if any(_norm(k) in name for k in VOCAL_KEYWORDS):
        return "vocals"
    if any(_norm(k) in name for k in BASS_KEYWORDS):
        return "bass"
    if any(_norm(k) in name for k in DRUMS_KEYWORDS):
        return "drums"
    return "other"


def is_main(filename: str) -> bool:
    name = _norm(filename)
    return any(_norm(k) in name for k in MAIN_KEYWORDS)


def _describe_format(path: Path) -> dict:
    info = sf.info(str(path))
    return dict(samplerate=info.samplerate, channels=info.channels,
                subtype=info.subtype, format=info.format,
                duration_s=round(info.frames / info.samplerate, 2))


# --------------------------------------------------------------------------
# Режим 1: целый трек -> Demucs -> 4 источника
# --------------------------------------------------------------------------
DEMUCS_MODEL = "htdemucs_ft"


def run_demucs(mix_path: Path, work_dir: Path) -> tuple[Path, dict]:
    print(f"  Demucs ({DEMUCS_MODEL}) на {mix_path.name} — займёт минуты, пропорционально длине трека...")
    # ТЗ-05 А6: analyze_file() во всех analysis/metrics/*.py жёстко ждёт
    # 44100Гц (assert, не мягкая деградация) — привести вход к этому sr
    # ДО Demucs, не полагаться на то, что модель сама всё сделает верно.
    mix_path = engine.ensure_sr(mix_path, engine.PIPELINE_SR, work_dir)
    venv_demucs = ROOT.parent / ".venv" / "bin" / "demucs"
    demucs_bin = str(venv_demucs) if venv_demucs.exists() else "demucs"
    # ТЗ-05 А11: --shifts по умолчанию РАВЕН 1, не 0 — "random shifts for
    # equivariant stabilization" реально даёт разный результат от запуска к
    # запуску (проверено эмпирически: max|diff|~0.05-0.08 на идентичном
    # входе). --shifts 0 отключает эту аугментацию целиком — прогон
    # становится побайтово детерминированным (тоже проверено эмпирически),
    # ценой небольшого потенциального качества разделения. Для
    # инструмента, чей смысл — воспроизводимое измерение, а не
    # максимальное качество разделения, детерминированность важнее.
    subprocess.run([demucs_bin, "-n", DEMUCS_MODEL, "--shifts", "0", "-o", str(work_dir), str(mix_path)],
                    check=True, capture_output=True, text=True)
    stem_dir = work_dir / DEMUCS_MODEL / mix_path.stem
    stems = {}
    for role, fname in [("vocals", "vocals.wav"), ("bass", "bass.wav"),
                         ("drums", "drums.wav"), ("other", "other.wav")]:
        p = stem_dir / fname
        if p.exists():
            stems[role] = p
    return mix_path, stems


# --------------------------------------------------------------------------
# Режим 2: трек-аут -> суммирование дорожек одной роли
# --------------------------------------------------------------------------
def _pairwise_layering(aligned_meta: list, sr: int) -> list:
    """Блок 3 (измерение, «наложение дублей»): пока дорожки одной роли не
    слиты в сумму, у нас есть последний момент, когда каждый дубль ещё
    отдельный сигнал СО СВОИМ офсетом относительно общей оси (тот же
    offset, что уже посчитан GCC-PHAT против главного микса выше) — именно
    это и нужно layering.analyze_pair: относительный сдвиг между дублями
    без гадания заново, они уже в одной системе координат. После этой
    функции дубли суммируются и по отдельности не восстановить — если не
    посчитать здесь, негде будет взять. Не рекомендация (только измерение
    расхождения по времени/питчу и верхняя оценка риска гребёнки) —
    рекомендация по фиксу ждёт Блок 8 («с выбором», см. roadmap.md)."""
    pairs = []
    for (path_a, off_a), (path_b, off_b) in itertools.combinations(aligned_meta, 2):
        try:
            summary = layering.analyze_pair(path_a, path_b, off_a, off_b, sr_expected=sr)
            pairs.append(dict(pair=[path_a.name, path_b.name], **summary))
        except Exception as e:
            pairs.append(dict(pair=[path_a.name, path_b.name], error=str(e)))
    return pairs


def align_and_sum_tracks(paths: list[Path], ref_mono: np.ndarray, ref_sr: int,
                          work_dir: Path, out_name: str,
                          max_shift_s: float = 2.0) -> tuple[Path, list[str], list]:
    """ТЗ-05 А4: каждая дорожка перед суммированием в роль проверяется
    GCC-PHAT против главного микса (критерий уверенности — ТЗ-01 §3.2:
    confidence>1.3 и z>8). Выровненные — сдвигаются на найденный офсет.
    Невыровненные — ИСКЛЮЧАЮТСЯ из суммы, а не складываются вслепую —
    несинхронная дорожка не усиливает роль, а портит её шумом фазовых
    biений. Возвращает (путь к сумме, список исключённых с причиной,
    список попарных измерений наложения дублей — Блок 3, пусто, если
    дубль в роли один)."""
    excluded = []
    aligned_signals = []
    aligned_meta = []  # (путь, офсет_в_секундах) — для _pairwise_layering
    sr_ref = None
    for p in paths:
        data, sr = sf.read(str(p), dtype="float64", always_2d=True)
        mono = data.mean(axis=1)
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"{p.name}: sr={sr}, ожидался {sr_ref} — привести все дорожки трек-аута "
                              f"к одной частоте дискретизации перед прогоном")

        cmp_ref = ref_mono if sr == ref_sr else librosa.resample(ref_mono, orig_sr=ref_sr, target_sr=sr)
        shift, confidence, _, _, z = alignment.gcc_phat(mono, cmp_ref, sr, max_shift_s=max_shift_s)
        if alignment.is_confident(confidence, z):
            shifted = mono[shift:] if shift >= 0 else np.concatenate([np.zeros(-shift), mono])
            aligned_signals.append(shifted)
            aligned_meta.append((p, shift / sr))
            print(f"      {p.name}: выровнено, сдвиг {shift/sr*1000:+.1f}мс (conf={confidence:.2f}, z={z:.1f})")
        else:
            excluded.append(f"{p.name} (conf={confidence:.2f}, z={z:.1f} — ниже порога 1.3/8)")
            print(f"      {p.name}: НЕ выровнено, исключена из суммы (conf={confidence:.2f}, z={z:.1f})")

    if not aligned_signals:
        raise ValueError(f"{out_name}: ни одна дорожка не прошла проверку синхронности против главного микса")

    layering_pairs = _pairwise_layering(aligned_meta, sr_ref) if len(aligned_meta) >= 2 else []

    max_len = max(len(s) for s in aligned_signals)
    summed = np.zeros(max_len)
    for s in aligned_signals:
        summed[:len(s)] += s
    out_path = work_dir / f"{out_name}.wav"
    sf.write(str(out_path), summed, sr_ref)
    return out_path, excluded, layering_pairs


def classify_trackout(folder: Path, work_dir: Path) -> tuple[Path, dict, dict, dict, dict]:
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_EXT]
    if not files:
        raise ValueError(f"{folder}: нет аудиофайлов")

    # ТЗ-05 А6: исходные параметры — до любого ресемплинга, для отчёта
    input_formats = {}
    for p in files:
        info = sf.info(str(p))
        input_formats[p.name] = dict(samplerate=info.samplerate, channels=info.channels,
                                       subtype=info.subtype, format=info.format,
                                       duration_s=round(info.frames / info.samplerate, 2))
    # разные sr/битность в одной папке — привести к единой частоте ДО
    # любого анализа (analyze_file жёстко ждёт 44100, см. engine.PIPELINE_SR)
    files = [engine.ensure_sr(p, engine.PIPELINE_SR, work_dir) for p in files]

    main_candidates = [p for p in files if is_main(p.name)]
    if len(main_candidates) == 1:
        main_path = main_candidates[0]
        stem_files = [p for p in files if p != main_path]
    elif len(main_candidates) == 0:
        raise ValueError(f"{folder}: не нашёл главный референс-микс (имя должно содержать "
                          f"main/mix/master/full) — без него не с чем сравнивать роли")
    else:
        raise ValueError(f"{folder}: больше одного файла похоже на главный микс "
                          f"({[p.name for p in main_candidates]}) — оставь один")

    by_role: dict[str, list[Path]] = {}
    unrecognized = []
    for p in stem_files:
        role = classify_role(p.name)
        by_role.setdefault(role, []).append(p)
        if role == "other":
            unrecognized.append(p.name)

    print("  роли дорожек трек-аута:")
    for role, ps in by_role.items():
        print(f"    {role}: {[p.name for p in ps]}")
    if unrecognized:
        # ТЗ-05 А10: "спросить пользователя" в неинтерактивном батч-режиме
        # оркестратора буквально нереализуемо (нет диалога посреди прогона
        # по папке import/) — ближайший честный эквивалент: не молчать,
        # явно и заметно предупредить, что роль угадана по умолчанию, а не
        # распознана по ключевому слову, и куда конкретно она попала.
        print(f"  ВНИМАНИЕ: роль не распознана ни по одному ключевому слову, "
              f"отправлено в «other» по умолчанию: {unrecognized}")
        print(f"  Если это не гитары/клавиши/синтезаторы — переименуй файл или "
              f"добавь ключевое слово в orchestrate.py (VOCAL_KEYWORDS/BASS_KEYWORDS/DRUMS_KEYWORDS)")
        input_formats["_unrecognized_roles"] = unrecognized

    ref_data, ref_sr = sf.read(str(main_path), dtype="float64", always_2d=True)
    ref_mono = ref_data.mean(axis=1)

    stems = {}
    excluded_all = {}
    layering_all = {}
    for role, ps in by_role.items():
        print(f"    выравниваю {role} против главного микса...")
        stems[role], excluded, layering_pairs = align_and_sum_tracks(ps, ref_mono, ref_sr, work_dir, role)
        if excluded:
            excluded_all[role] = excluded
        if layering_pairs:
            layering_all[role] = layering_pairs
    if excluded_all:
        print("  дорожки, исключённые из-за рассинхрона с главным миксом:")
        for role, items in excluded_all.items():
            for item in items:
                print(f"    {role}: {item}")
    return main_path, stems, excluded_all, input_formats, layering_all


# --------------------------------------------------------------------------
# Общая часть: метрики + вердикт
# --------------------------------------------------------------------------
def analyze_all_sources(mix_path: Path, stems: dict, deep_psychoacoustics: bool,
                         is_ml_separated: bool, track_name: str = None) -> tuple[dict, dict]:
    """is_ml_separated=True — режим 1 (Demucs): реверб считается ТОЛЬКО на
    миксе, на разделённых источниках — null с причиной (ТЗ-05 А9,
    разделение размазывает хвосты). Режим 2 (реальные дорожки) — реверб
    можно считать на любом источнике, они не ML-приближение."""
    measurements = {}
    all_diagnostics = {}
    mix_gain_db = engine.get_mix_gain_db(mix_path)
    track_mono_by_role = {}  # Блок 4: {role: mono} без mix — вход masking_erb.analyze_group
    sr_masking = None
    section_profile_for_attribution = None  # Блок 5: границы секций с mix, применимы к любой роли

    sources = {"mix": mix_path, **stems}
    for role, path in sources.items():
        print(f"  метрики: {role} ({path.name})")
        allow_reverb = (role == "mix") or not is_ml_separated
        m_avg, vocal_frames, diag = engine.track_avg_metrics(
            path, role, is_stereo_capable=(role == "mix"), allow_reverb=allow_reverb)
        measurements.update(m_avg)
        all_diagnostics[role] = diag

        if role == "mix":
            # ТЗ-05 Б9: драматургическая арка — третий уровень
            # гранулярности (секции), не окно и не целый трек. Только на
            # миксе: у стемов отдельно "макро-профиль" не так осмыслен, а
            # автосегментация на изолированном стеме (особенно
            # бас/барабаны) менее надёжна, чем на полном миксе.
            try:
                mono_mix, sr_mix, _ = engine.load_mono(path)
                profile, arc, section_source = sections.analyze(mono_mix, sr_mix, song=track_name)
                measurements[("arc_db", "mix")] = arc["arc_db"]
                measurements[("section_spread_db", "mix")] = arc["section_spread_db"]
                measurements[("section_range_db", "mix")] = arc["section_range_db"]
                diag["section_profile"] = profile
                diag["section_source"] = section_source
                section_profile_for_attribution = profile
            except Exception as e:
                diag["section_analysis_error"] = str(e)

        mono, sr, _ = engine.load_mono(path)
        if role != "mix":
            # Блок 4: mix — сумма всего, маскирование её самой собой не
            # осмыслено, нужны только отдельные источники. mono уже
            # загружен здесь для window_metrics — переиспользуем, не грузим
            # файл ещё раз.
            track_mono_by_role[role] = mono
            sr_masking = sr
        f0_df = vocal_frames.get("f0") if role == "vocals" else None
        notes_df = vocal_frames.get("notes") if role == "vocals" else None
        wdf = engine.window_metrics(mono, sr, role, mix_gain_db=mix_gain_db,
                                     f0_df=f0_df, notes_df=notes_df,
                                     do_real_psychoacoustics=(deep_psychoacoustics and role == "vocals"))
        for col in wdf.columns:
            if col in ("t_start", "t_end", "rms_dbfs"):
                continue
            if wdf[col].notna().sum() == 0:
                continue
            measurements[(col, role)] = wdf[col]

        # Блок 5: атрибуция по (роль, секция) — переиспользует уже
        # посчитанные wdf (window-метрики этой роли) и section_profile
        # (границы с mix, общая временная ось). Только window-метрики —
        # track_avg метрики уже одно число на весь трек, атрибутировать
        # по секциям нечего.
        if section_profile_for_attribution is not None and len(section_profile_for_attribution):
            sec_medians = section_attribution.attribute_by_section(wdf, section_profile_for_attribution)
            if sec_medians:
                diag["section_medians"] = sec_medians

        # ТЗ-05 Б8: formant_f3_hz — зона в пресете (vocals/window), но
        # раньше нигде не вызывалась — обнаружено тестом на соответствие
        # ключей пресета выходу engine.py. formant_series даёт F3 по
        # voiced-кадрам LPC, отдельно от 4с-окон window_metrics.
        if role == "vocals" and f0_df is not None and len(f0_df):
            fdf = engine.formant_series(mono, sr, f0_df)
            if len(fdf) and fdf["f3_hz"].notna().sum() > 0:
                measurements[("formant_f3_hz", role)] = fdf["f3_hz"]

    # Блок 4 (частотные конфликты, измерение): masking_erb.analyze_group уже
    # принимает произвольное число дорожек — просто интеграция. Требует
    # общей временной оси (см. docstring analyze_group) — оба режима это
    # дают: Demucs (режим 1) разделяет на месте без сдвига, трек-аут
    # (режим 2) уже выровнен по главному миксу в align_and_sum_tracks.
    if len(track_mono_by_role) >= 2:
        try:
            masking_result = masking_erb.analyze_group(track_mono_by_role, sr_masking)
            for role, aud in masking_result["audibility"].items():
                all_diagnostics.setdefault(role, {})["audibility"] = aud
            attribution_df = masking_result["attribution"]
            if len(attribution_df):
                all_diagnostics["_masking"] = {"attribution": attribution_df.to_dict(orient="records")}
        except Exception as e:
            all_diagnostics["_masking"] = {"error": str(e)}

    return measurements, all_diagnostics


def write_report(out_dir: Path, track_name: str, measurements: dict, verdicts, diagnostics: dict = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = format_report(verdicts)

    n_liked = sum(1 for v in verdicts if v.status is Status.IN_ZONE)
    n_disliked = sum(1 for v in verdicts if v.status is Status.OUT_OF_ZONE)
    n_border_liked = sum(1 for v in verdicts if v.status is Status.BORDERLINE and v.closer_to == "нравится")
    n_border_disliked = sum(1 for v in verdicts if v.status is Status.BORDERLINE and v.closer_to == "не нравится")

    # взвешенный итог: low-reliability метрика (напр. списанная harshness)
    # реально должна влиять слабо, а не наравне с high-reliability — вес
    # берётся из Reliability.weight, не только показывается в отчёте.
    # ТЗ-05 Б5: зоны с одинаковым zone.cluster измеряют по сути одну и ту же
    # ось (напр. 3 разных числа про тональный наклон микса) — усредняются
    # ВНУТРИ кластера до contribution в [-1..+1], и только сумма кластеров
    # входит в итог, иначе одна ось голосует несколько раз.
    def _contribution(v):
        if v.status is Status.IN_ZONE:
            return 1.0
        if v.status is Status.OUT_OF_ZONE:
            return -1.0
        if v.status is Status.BORDERLINE and v.closer_to == "нравится":
            return 0.5
        if v.status is Status.BORDERLINE and v.closer_to == "не нравится":
            return -0.5
        return None  # NO_DATA / UNKNOWN / NO_ZONE — не голосует

    clusters = {}
    for v in verdicts:
        key = v.zone.cluster or (v.zone.metric, v.zone.source)
        clusters.setdefault(key, []).append(v)

    score, max_score = 0.0, 0.0
    for members in clusters.values():
        contribs, weights = [], []
        for v in members:
            c = _contribution(v)
            if c is not None:
                contribs.append(c)
                weights.append(v.zone.reliability.weight)
        if not contribs:
            continue
        cluster_weight = sum(weights) / len(weights)
        score += cluster_weight * (sum(contribs) / len(contribs))
        max_score += cluster_weight
    weighted_pct = 100 * score / max_score if max_score > 0 else float("nan")

    # ТЗ-05 Б7: единый процент вводит в заблуждение, если измерена меньше
    # половины реально применимых зон (pending/no_zone исключены — они и не
    # должны иметь измерения). Тогда печатаем честное "данных недостаточно"
    # вместо числа, которое на деле посчитано по 2-3 зонам из 12.
    measurable = [v for v in verdicts if v.zone.reliability not in (Reliability.PENDING, Reliability.NO_ZONE)]
    with_data = [v for v in measurable if v.status is not Status.NO_DATA]
    enough_data = len(measurable) == 0 or len(with_data) >= len(measurable) / 2

    run_meta = (diagnostics or {}).get("_run", {})
    run_line = ""
    if run_meta.get("preset_name"):
        run_line += f"Пресет: {run_meta['preset_name']}.json, sha256={run_meta.get('preset_hash', '?')}\n"
    if run_meta.get("demucs_model"):
        run_line += (f"Demucs: модель={run_meta['demucs_model']}, "
                     f"version={run_meta.get('demucs_version', '?')}, "
                     f"shifts={run_meta.get('demucs_shifts')} "
                     f"(ТЗ-05 А11: shifts=0 — прогон детерминирован, побайтово воспроизводим)\n")
    if run_line:
        run_line += "\n"

    if enough_data:
        score_line = (f"Взвешенный итог: {weighted_pct:+.0f}%  (+100% = все измеренные метрики в зоне «нравится», "
                       f"вес по надёжности: high=1.0, medium=0.6, low=0.25, pending/no_zone=0 — не влияют)\n\n")
    else:
        score_line = (f"Взвешенный итог: данных недостаточно для суждения "
                       f"(измерено {len(with_data)} из {len(measurable)} применимых зон, меньше половины — "
                       f"единый процент был бы обманчив, ТЗ-05 Б7)\n\n")

    header = (f"Legoshi Train — анализ «{track_name}»\n{datetime.now().isoformat(timespec='seconds')}\n\n"
              f"{run_line}"
              f"{score_line}"
              f"Зон с измерением: {len(with_data)}/{len(measurable)} применимых "
              f"({len(verdicts) - len(measurable)} pending/no_zone не считаются)\n\n"
              f"Сырые счётчики (без веса, для справки):\n"
              f"В зоне «нравится»: {n_liked}   На стороне «нравится» (не строго в зоне): {n_border_liked}\n"
              f"В зоне «не нравится»: {n_disliked}   На стороне «не нравится»: {n_border_disliked}\n\n")

    # ТЗ-05 Б4: зона, откалиброванная на Demucs-стемах, применена к реальному
    # трек-ауту (или наоборот) — предупредить, не молчать
    is_ml = run_meta.get("is_ml_separated")
    learned_on_warnings = []
    if is_ml is not None:
        actual = "demucs" if is_ml else "real_stems"
        for v in measurable:
            if v.status is Status.NO_DATA:
                continue
            lo = v.zone.learned_on
            if lo and lo != "mix" and lo != actual:
                learned_on_warnings.append(
                    f"  {v.source}/{v.metric}: зона откалибрована на «{lo}», сейчас применена к «{actual}» — "
                    f"возможен систематический сдвиг (bleed/артефакты разделения другие или отсутствуют)")
    learned_on_block = ""
    if learned_on_warnings:
        learned_on_block = "ВНИМАНИЕ (ТЗ-05 Б4), несовпадение learned_on:\n" + "\n".join(learned_on_warnings) + "\n\n"

    diag_lines = []
    # ТЗ-05 Д: _run/_trackout раньше выкидывались после того, как из них
    # брали текст для report.txt — то, что попадало в консоль, не попадало
    # в measurements.json. Сохраняем, чтобы прокинуть в JSON ниже (см.
    # output_json) — воспроизводимость должна быть в структурированном
    # виде, не только в человекочитаемом тексте.
    run_meta_full = (diagnostics or {}).pop("_run", None)
    trackout_diag = (diagnostics or {}).pop("_trackout", {})
    masking_diag = (diagnostics or {}).pop("_masking", {})
    excluded_tracks = trackout_diag.get("excluded_unaligned_tracks", {})
    input_formats = trackout_diag.get("input_formats", {})
    if input_formats:
        diag_lines.append("Исходные параметры входных файлов (до ресемплинга к 44100Гц):")
        for fname, fmt in input_formats.items():
            if fname == "_unrecognized_roles":
                continue
            diag_lines.append(f"  {fname}: {fmt['samplerate']}Гц, {fmt['channels']}ch, "
                               f"{fmt['subtype']}, {fmt['duration_s']}с")
        if input_formats.get("_unrecognized_roles"):
            diag_lines.append(f"  ВНИМАНИЕ: роль не распознана ни по одному ключевому слову, "
                               f"отправлено в «other» по умолчанию: {input_formats['_unrecognized_roles']}")
        diag_lines.append("")
    if excluded_tracks:
        diag_lines.append("Дорожки, исключённые из-за рассинхрона с главным миксом (ТЗ-05 А4):")
        for role, items in excluded_tracks.items():
            for item_desc in items:
                diag_lines.append(f"  {role}: {item_desc}")
        diag_lines.append("")

    # ТЗ-05 Б9: таблица "секция -> уровень относительно среднего по треку" —
    # драматургическая арка не читается из одного числа arc_db без картины
    # целиком, особенно когда она пограничная или структура трека странная
    mix_diag = (diagnostics or {}).get("mix", {})
    section_profile = mix_diag.pop("section_profile", None)
    section_source = mix_diag.pop("section_source", None)
    if section_profile is not None and len(section_profile):
        src_label = {"manual": "ручная разметка _analysis/sections.csv",
                     "auto": "автосегментация (self-similarity+novelty)"}.get(section_source, section_source)
        diag_lines.append(f"Драматургическая арка по секциям (ТЗ-05 Б9, границы: {src_label}):")
        for _, r in section_profile.iterrows():
            bar_len = max(0, round(r["level_rel_db"])) if r["level_rel_db"] > 0 else 0
            label = r["section"] or f"{r['start_s']:.0f}-{r['end_s']:.0f}с"
            diag_lines.append(f"  {label:<32}{r['start_s']:>6.0f}-{r['end_s']:<6.0f}с  "
                               f"{r['level_rel_db']:+6.1f}дБ отн. среднего трека")
        arc_val = measurements.get(("arc_db", "mix"))
        if arc_val is not None:
            diag_lines.append(f"  arc_db (2 громких - 2 ранних секции) = {arc_val:+.2f}дБ, "
                               f"section_spread_db = {measurements.get(('section_spread_db','mix'), float('nan')):.2f}дБ, "
                               f"section_range_db = {measurements.get(('section_range_db','mix'), float('nan')):.2f}дБ")
        diag_lines.append("")

    # Блок 4 (частотные конфликты, измерение): кто кого маскирует чаще всего
    # — не рекомендация (см. roadmap.md, Блок 7/8), просто факт по паре
    if masking_diag.get("error"):
        diag_lines.append(f"Маскирование (Блок 4): не удалось измерить ({masking_diag['error']})\n")
    elif masking_diag.get("attribution"):
        diag_lines.append("Маскирование между дорожками (Блок 4, ERB, доля замаскированных "
                            "TF-клеток по паре, только пары с долей >5%):")
        for row in sorted(masking_diag["attribution"], key=lambda r: -r["fraction_of_masked_cells"]):
            if row["fraction_of_masked_cells"] <= 0.05:
                continue
            diag_lines.append(f"  {row['masker']} маскирует {row['masked']}: "
                               f"{row['fraction_of_masked_cells']:.0%} замаскированных клеток")
        diag_lines.append("")

    if diagnostics:
        diag_lines.append("Что обнаружено по источникам (не метрики, факты о входе):")
        for role, d in diagnostics.items():
            facts = []
            if d.get("codec_cutoff_hz"):
                facts.append(f"срез спектра на {d['codec_cutoff_hz']:.0f}Гц (lossy-кодек)")
            if d.get("codec_cutoff_invalidated_keys"):
                facts.append(f"инвалидированы полосы выше среза: {', '.join(d['codec_cutoff_invalidated_keys'])}")
            if d.get("dual_mono"):
                facts.append("дуал-моно (каналы идентичны — не настоящее стерео)")
            if "audibility" in d:
                facts.append(f"audibility {d['audibility']:.0%} "
                             f"(Блок 4, доля TF-клеток, где источник слышен поверх остальных)")
            if d.get("clipped"):
                cd = d.get("clipping_detail")
                if cd:
                    regions_str = ", ".join(f"{r['start_s']:.2f}-{r['end_s']:.2f}с ({r['severity']})"
                                            for r in cd["clipped_regions_s"][:8])
                    more = f" +ещё {len(cd['clipped_regions_s']) - 8}" if len(cd["clipped_regions_s"]) > 8 else ""
                    facts.append(f"клиппинг: {cd['n_clipped_runs']} отрезков, "
                                 f"{cd['clipped_fraction']*100:.2f}% трека — {regions_str}{more}")
                else:
                    facts.append("клиппинг (серия сэмплов на полной шкале)")
            if d.get("hum_candidates"):
                hum_str = ", ".join(f"{c.get('freq_hz_refined', c['freq_hz']):.1f}Гц "
                                    f"(score={c['stability_score']:.1f})"
                                    for c in d["hum_candidates"][:5])
                facts.append(f"возможная наводка: {hum_str}")
            if d.get("hum_windowed"):
                hw = d["hum_windowed"]
                facts.append(f"наводка присутствует не по всей дорожке, а в {len(hw)} "
                             f"интервал(ах) по ~{engine.DIAG_WINDOW_S:.0f}с (детали — measurements.json)")
            if d.get("reverb_windowed"):
                facts.append(f"реверб-диагностика по {len(d['reverb_windowed'])} интервал(ам) "
                             f"по ~{engine.DIAG_WINDOW_S:.0f}с (детали — measurements.json, "
                             f"для отделения бликида/вставки от общей акустики — следующий шаг Блока 2)")
            if d.get("reverb_skipped_reason"):
                facts.append(f"реверб не считался: {d['reverb_skipped_reason']}")
            if d.get("vocal_analysis_error"):
                facts.append(f"ошибка вокального анализа: {d['vocal_analysis_error']}")
            if d.get("layering_pairs"):
                # Блок 3 (измерение, не рекомендация — фикс ждёт Блок 8, "с
                # выбором", наложение дублей не однозначная ошибка): дубли
                # уже суммированы в единый сигнал role, это последний
                # диагностический след того, КАК они соотносились до суммы
                for lp in d["layering_pairs"]:
                    a_name, b_name = lp["pair"]
                    if "error" in lp:
                        facts.append(f"наложение дублей {a_name} vs {b_name}: не удалось измерить ({lp['error']})")
                        continue
                    facts.append(
                        f"наложение дублей {a_name} vs {b_name}: расхождение по времени "
                        f"{lp['time_divergence_ms_median']:.1f}мс, по питчу {lp['pitch_divergence_cents_median']:.1f}¢ "
                        f"(медианы), верхняя оценка риска гребёнки {lp['comb_risk_upper_bound']:.0%} "
                        f"(измерение, не рекомендация — см. roadmap.md Блок 3/8)")
            if facts:
                diag_lines.append(f"  {role}: " + "; ".join(facts))
        if len(diag_lines) == 1:
            diag_lines.append("  без замечаний")
        diag_lines.append("")

    # Блок 2 (Этап 1, «без выбора»): рекомендации по очистке — программа
    # только называет место, параметры и категорию, никогда не применяет
    # сама (см. roadmap.md, главный принцип)
    restore_recs = recommendations.all_restoration_recommendations(diagnostics or {})
    if restore_recs:
        diag_lines.append("Рекомендации по очистке (Блок 2, без выбора — не применяется автоматически):")
        for r in restore_recs:
            diag_lines.append(f"  {r.text} (уверенность {r.confidence:.0%})")
        diag_lines.append("")

    # Блок 5: для метрик вне зоны — какая секция сильнее всего тянет от
    # зоны, не таймкод (roadmap.md, Блок 5 — окно не даёт чёткой границы
    # события). Только window-granularity зоны — track_avg метрики уже
    # одно число на весь трек, секций там не считалось.
    attribution_lines = []
    for v in verdicts:
        if v.status not in (Status.OUT_OF_ZONE, Status.BORDERLINE) or v.granularity != "window":
            continue
        sec_medians = (diagnostics or {}).get(v.source, {}).get("section_medians", {}).get(v.metric)
        if not sec_medians:
            continue
        label, val, delta = section_attribution.worst_section(sec_medians, v.zone)
        if label is None:
            continue
        attribution_lines.append(f"  {v.source}/{v.metric}: хуже всего в «{label}» "
                                  f"({val:.3g}, дельта {delta:+.3g})")
    if attribution_lines:
        diag_lines.append("Атрибуция по секциям (Блок 5, не таймкод — окно не даёт чёткой границы):")
        diag_lines.extend(attribution_lines)
        diag_lines.append("")

    # Блок 7 («с выбором»): полный список за один запуск, сгруппированный
    # по стадиям сведения (roadmap.md) — стадия только для читаемости
    # вывода, НЕ заявка на проверенный совместный эффект (Блок 6 мерил
    # каждый ход отдельно, не комбинациями). Программа не применяет сама.
    interference_matrix = load_interference_matrix()
    taste_recs = recommendations.all_taste_recommendations(verdicts, diagnostics or {}, interference_matrix)
    if taste_recs:
        diag_lines.append("Рекомендации по вкусовым правкам (Блок 7, с выбором — "
                            "не применяется автоматически, стадии только для читаемости):")
        # классический порядок сведения (roadmap.md) — не порядок появления
        # на таймлайне, тот уже задан сортировкой ВНУТРИ каждой стадии
        stage_order = ["вычитающий EQ", "компрессия", "добавляющий EQ", "сатурация",
                        "наложение дублей", "без стадии/не покрыто"]
        by_stage = {}
        for r in taste_recs:
            by_stage.setdefault(r.stage or "без стадии/не покрыто", []).append(r)
        for stage in sorted(by_stage, key=lambda s: stage_order.index(s) if s in stage_order else len(stage_order)):
            stage_recs = by_stage[stage]
            diag_lines.append(f"  -- {stage} --")
            for r in stage_recs:
                diag_lines.append(f"    {r.text} (уверенность {r.confidence:.0%})")
        diag_lines.append("")

    # ТЗ-05 Д: явный итоговый блок "что не измерено и почему" — часть
    # структурная (всегда так, не зависит от конкретного трека), часть
    # собрана из диагностики этого конкретного прогона
    not_measured = []
    non_mix_roles = [r for r in (diagnostics or {}) if r not in ("mix", "_run", "_trackout")]
    if non_mix_roles:
        not_measured.append(
            f"  band_frac_air — только на mix, NaN на {', '.join(sorted(non_mix_roles))} "
            f"(структурно: 'воздух' 12-20кГц на разделённом стеме — артефакт разделения, не тембр)")
        not_measured.append(
            "  вибрато/интонация/форманты (F0-метрики) — только на источнике vocals, "
            "не считаются на остальных ролях (структурно)")
    reverb_skipped = [role for role, d in (diagnostics or {}).items()
                       if isinstance(d, dict) and d.get("reverb_skipped_reason")]
    if reverb_skipped:
        not_measured.append(f"  реверб (RT60/EDT/DRR/C50/C80) — пропущен на {', '.join(sorted(reverb_skipped))}: "
                             f"{diagnostics[reverb_skipped[0]]['reverb_skipped_reason']}")
    cutoff_invalidated = [role for role, d in (diagnostics or {}).items()
                          if isinstance(d, dict) and d.get("codec_cutoff_invalidated_keys")]
    if cutoff_invalidated:
        not_measured.append(f"  спектральные полосы выше среза lossy-кодека — инвалидированы на "
                            f"{', '.join(sorted(cutoff_invalidated))} (детали выше, в фактах по источникам)")
    vocal_errors = [role for role, d in (diagnostics or {}).items()
                     if isinstance(d, dict) and d.get("vocal_analysis_error")]
    if vocal_errors:
        not_measured.append(f"  вокальный анализ (F0/вибрато/форманты) — упал с ошибкой на {', '.join(vocal_errors)}")
    if not_measured:
        diag_lines.append("Что НЕ измерено в этом прогоне и почему (ТЗ-05 Д):")
        diag_lines.extend(not_measured)
        diag_lines.append("")

    (out_dir / "report.txt").write_text(header + learned_on_block + "\n".join(diag_lines) + "\n" + report, encoding="utf-8")

    serializable = {}
    for (metric, source), val in measurements.items():
        key = f"{source}::{metric}"
        if isinstance(val, pd.Series):
            serializable[key] = {"median": float(val.median()), "n_windows": int(val.notna().sum())}
        else:
            serializable[key] = float(val)
    output_json = {"measurements": serializable, "diagnostics": diagnostics or {}}
    if run_meta_full:
        output_json["run"] = run_meta_full
    if trackout_diag:
        output_json["trackout"] = trackout_diag
    if masking_diag:
        output_json["masking"] = masking_diag
    if section_profile is not None and len(section_profile):
        output_json["section_profile"] = {
            "source": section_source,
            "sections": section_profile.to_dict(orient="records"),
        }
    if restore_recs:
        output_json["restoration_recommendations"] = [
            dict(category=r.category, source=r.source, location_s=r.location_s,
                 params=r.params, confidence=r.confidence, text=r.text)
            for r in restore_recs
        ]
    if taste_recs:
        output_json["taste_recommendations"] = [
            dict(category=r.category, source=r.source, section=r.section, stage=r.stage,
                 params=r.params, confidence=r.confidence, text=r.text)
            for r in taste_recs
        ]
    (out_dir / "measurements.json").write_text(json.dumps(output_json, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")
    print(f"\n  -> {out_dir / 'report.txt'}")
    print(f"  -> {out_dir / 'measurements.json'}")


def process_item(item: Path, preset, deep_psychoacoustics: bool, preset_name: str = "legoshi_amber"):
    print(f"\n=== {item.name} ===")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / f"{item.stem}_{ts}"

    with tempfile.TemporaryDirectory(prefix="legoshi_train_") as tmp:
        work_dir = Path(tmp)
        excluded_tracks = {}
        input_formats = {}
        layering_all = {}
        # ТЗ-05 Д: хэш файла пресета в отчёте — какой именно набор зон
        # (включая формулировки/диапазоны) использовался в этом прогоне,
        # без хэша не отличить "legoshi_amber.json на момент прогона" от
        # более поздней правки с тем же именем
        preset_path = PRESETS_DIR / f"{preset_name}.json"
        preset_hash = hashlib.sha256(preset_path.read_bytes()).hexdigest()[:12] if preset_path.exists() else "?"
        run_metadata = dict(demucs_model=None, demucs_shifts=None,
                             preset_name=preset_name, preset_hash=preset_hash)
        if item.is_file():
            print("  режим: целый трек (Demucs)")
            input_formats[item.name] = _describe_format(item)
            mix_path, stems = run_demucs(item, work_dir)
            is_ml_separated = True
            try:
                import importlib.metadata as _im
                demucs_version = _im.version("demucs")
            except Exception:
                demucs_version = "неизвестно"
            run_metadata.update(demucs_model=DEMUCS_MODEL, demucs_shifts=0, demucs_version=demucs_version,
                                 is_ml_separated=True)
        elif item.is_dir():
            print("  режим: трек-аут (реальные дорожки)")
            mix_path, stems, excluded_tracks, input_formats, layering_all = classify_trackout(item, work_dir)
            is_ml_separated = False
            run_metadata["is_ml_separated"] = False
        else:
            print(f"  пропуск: не файл и не папка")
            return

        measurements, diagnostics = analyze_all_sources(
            mix_path, stems, deep_psychoacoustics, is_ml_separated, track_name=item.stem)
        if excluded_tracks:
            diagnostics.setdefault("_trackout", {})["excluded_unaligned_tracks"] = excluded_tracks
        if input_formats:
            diagnostics.setdefault("_trackout", {})["input_formats"] = input_formats
        for role, pairs in layering_all.items():
            # роль уже есть в diagnostics — track_avg_metrics вызывался на
            # СУММЕ дублей этой роли (analyze_all_sources), layering же
            # посчитан ДО суммирования, на отдельных дублях (см.
            # _pairwise_layering) — два разных факта об одной роли, не
            # конфликтуют, просто разные ключи одного диагностического словаря
            diagnostics.setdefault(role, {})["layering_pairs"] = pairs
        diagnostics["_run"] = run_metadata
        verdicts = evaluate(measurements, preset)
        write_report(out_dir, item.stem, measurements, verdicts, diagnostics)
        print(format_report(verdicts))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", help="имя файла/папки внутри Project/import/ (по умолчанию — все)")
    ap.add_argument("--preset", default="legoshi_amber", help="имя пресета в Project/presets/ (без .json)")
    ap.add_argument("--deep-psychoacoustics", action="store_true",
                     help="настоящий sharpness/roughness (MoSQITo) на вокале — медленно, ~10-15 мин/трек")
    args = ap.parse_args()

    preset = load_preset(args.preset)
    IMPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.target:
        items = [IMPORT_DIR / args.target]
    else:
        items = [p for p in sorted(IMPORT_DIR.iterdir()) if p.name != ".gitkeep"]

    if not items:
        print(f"Пусто в {IMPORT_DIR} — положи трек или папку трек-аута и запусти снова.")
        return

    for item in items:
        if not item.exists():
            print(f"Не найдено: {item}")
            continue
        process_item(item, preset, args.deep_psychoacoustics, preset_name=args.preset)


if __name__ == "__main__":
    main()
