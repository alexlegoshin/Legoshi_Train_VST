"""Прогон chroma/тональность/аккорды + расстройка вокал-бас — задача #28.

Аккорды по секциям — только «основной трек» (только там есть sections.csv).
Тональность (без секций) считаем на всех mix/reference/demo, как и
остальной §4.9.

Расстройка вокал-бас: «основной трек» (вокал основной + бас, уже
выровненные тем же способом, что и в §5/§4.8 — иначе "общие ноты"
сравнивались бы вслепую) и «референс А» (demucs vocals+bass — оговорка:
бас-стем там с посторонней дребезжащей окраской, см. ANALYSIS.md,
доверять с осторожностью, не как первичному источнику).

ПОДТВЕРЖДЕНО АВТОРОМ (16.08.2026): «основной трек» — Am, он уверен.
Автодетектор тональности на ПОЛНОМ МИКСЕ стабильно давал G major
(корр. 0.60-0.76) на всех версиях и демке — разошлось. автор сам назвал
вероятную причину: дисторшн электрогитары плюс смена последнего аккорда
квадрата (Dm/G). Проверено run_key_on_clean_stems() ниже: на чистых
источниках (акустика микро, вокал основной) автодетектор САМ находит
A minor как лучший вариант, а на "электро риф" (дисторшн) — C major с
корреляцией 0.87, самой сильной из всех проверенных дорожек. Гипотеза
подтверждена данными, не только словами: дисторшн размазывает энергию по
чужим питч-классам в chroma и тянет агрегированную оценку по всему миксу
в сторону, амплитудно устойчивую тонику Am топит менее заметный, но
искажённый рифф. Ключ песни для отчёта — Am, G major по всему миксу
задокументирован как известное ограничение автодетектора на
дисторшн-материале, не как альтернативный факт."""
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from analysis.legacy_scripts.deconv.run_5 import load_aligned_stems
from analysis.metrics.harmony_key import key_and_chords, vocal_bass_detuning

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "metrics"
CACHE = ROOT / "cache" / "4_9_key"
ROLES = {"mix", "reference", "demo"}


def run_key_chords():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    reg = reg[reg.duplicate_of.isna() & reg.role.isin(ROLES)]
    sections = pd.read_csv(ROOT / "_analysis" / "sections.csv")
    CACHE.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for _, row in tqdm(reg.iterrows(), total=len(reg), desc="key/chords"):
        cache_file = CACHE / f"{row.md5}.json"
        safe_name = row.path.replace("/", "__")
        sec = sections[sections.song == row.song] if row.song in sections.song.unique() else None
        if cache_file.exists():
            summary = json.loads(cache_file.read_text())
        else:
            try:
                summary, frames = key_and_chords(ROOT / row.path, sections=sec)
            except Exception as e:
                errors.append((row.path, str(e)))
                continue
            cache_file.write_text(json.dumps(summary, ensure_ascii=False))
            if len(frames["chords"]):
                frames["chords"].to_parquet(OUT / f"{safe_name}.4_9_chords.parquet", index=False)
        rows.append(dict(summary, path=row.path, song=row.song, role=row.role, version=row.version))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "4_9_key_summary.parquet", index=False)
    print(f"Готово: {len(df)} файлов -> {OUT / '4_9_key_summary.parquet'}")
    if errors:
        print(f"ОШИБКИ: {errors}")

    print("\n=== основной трек: тональность по версиям ===")
    kp = df[df.song == "основной трек"]
    print(kp[["path", "version", "key_root", "key_mode", "key_correlation"]]
          .sort_values("version").to_string(index=False))

    chords_path = OUT / "основной трек__-__версия сведения 7.wav.4_9_chords.parquet"
    if chords_path.exists():
        print("\n=== Аккорды по секциям (v7, финальный микс инженера сведения) ===")
        print(pd.read_parquet(chords_path).to_string(index=False))

    main_summary_path = OUT / "4_9_summary.parquet"
    if main_summary_path.exists():
        main_summary = pd.read_parquet(main_summary_path)
        merge_cols = ["path", "key_root", "key_mode", "key_correlation"]
        for c in merge_cols:
            if c != "path" and c in main_summary.columns:
                main_summary = main_summary.drop(columns=[c])
        merged = main_summary.merge(df[merge_cols], on="path", how="left")
        merged.to_parquet(main_summary_path, index=False)
        print(f"\nВмёржено в -> {main_summary_path} ({len(merged)} строк)")


def run_key_on_clean_stems():
    """Проверка гипотезы автора: дисторшн электрогитары тянет
    автодетектор тональности всего микса в сторону от настоящей Am.
    Считаем тональность отдельно по стемам разной "чистоты" — если
    гипотеза верна, чистые источники (акустика, вокал) должны сами
    находить A minor лучше, чем искажённые (электро риф/бэк)."""
    stems = ["акустика микро", "акустика пьеза", "вокал основной", "бас", "электро риф", "электро бэк"]
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    kp_stems = reg[(reg.song == "основной трек") & (reg.role == "stem")]

    rows = []
    for _, row in kp_stems.iterrows():
        name = unicodedata.normalize("NFC", Path(row.path).stem)
        if name not in stems:
            continue
        summary, frames = key_and_chords(ROOT / row.path)
        rows.append(dict(stem=name, **summary))

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "kp_key_by_stem.parquet", index=False)
    print("\n=== Проверка: тональность по отдельным стемам (чистый источник vs дисторшн) ===")
    print(df.sort_values("key_correlation", ascending=False).to_string(index=False))
    print("\nA minor у чистых источников (акустика/вокал), не искажённый электро риф "
          "тянет весь микс к другой оценке -> подтверждает объяснение автора.")


def run_detuning():
    rows = []

    print("\n=== Расстройка вокал-бас: основной трек ===")
    stems_L, stems_R, sr = load_aligned_stems("основной трек")
    # ПОЙМАНО: имена файлов на APFS приходят в NFD (разложенный Unicode),
    # литералы в коде — NFC; расходятся ровно на "й" (composed vs "и"+combining
    # breve). Обычный "in dict" по буквальной строке с "й" тихо не находит
    # ключ — нормализуем обе стороны перед сравнением, не полагаемся на
    # визуальное сходство.
    stems_L = {unicodedata.normalize("NFC", k): v for k, v in stems_L.items()}
    if "вокал основной" in stems_L and "бас" in stems_L:
        summary, df = vocal_bass_detuning(stems_L["вокал основной"], stems_L["бас"], sr)
        summary.update(song="основной трек", source="stems (§5, выровнено)")
        rows.append(summary)
        if len(df):
            df.to_parquet(OUT / "kp__vocal_bass_detuning.parquet", index=False)
        print(summary)
    else:
        print("вокал основной/бас не выровнены или отсутствуют — пропуск")

    print("\n=== Расстройка вокал-бас: референс А (demucs, бас-стем с оговоркой) ===")
    rad_dir = ROOT / "референс А" / "demucs_stems"
    v_path, b_path = rad_dir / "vocals.wav", rad_dir / "bass.wav"
    if v_path.exists() and b_path.exists():
        v_data, sr_v = sf.read(str(v_path), dtype="float64", always_2d=True)
        b_data, sr_b = sf.read(str(b_path), dtype="float64", always_2d=True)
        assert sr_v == sr_b
        n = min(len(v_data), len(b_data))
        summary, df = vocal_bass_detuning(v_data[:n].mean(axis=1), b_data[:n].mean(axis=1), sr_v)
        summary.update(song="референс А", source="demucs (одноразовое ML-разделение, бас с артефактом)")
        rows.append(summary)
        if len(df):
            df.to_parquet(OUT / "radost__vocal_bass_detuning.parquet", index=False)
        print(summary)

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT / "4_9_vocal_bass_detuning_summary.parquet", index=False)
    print(f"\n-> {OUT / '4_9_vocal_bass_detuning_summary.parquet'}")


if __name__ == "__main__":
    run_key_chords()
    run_key_on_clean_stems()
    run_detuning()
