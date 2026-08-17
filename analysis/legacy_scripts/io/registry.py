"""Реестр источников, §3.1 ТЗ-01. Ручной список путей по §1 ТЗ — не глобим
наугад, берём ровно то, что перечислено (плюс явные исключения §1.5)."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]


def _sources():
    """(path, song, role, version) — по §1.1-1.4 ТЗ."""
    S = []
    kp = ROOT / "основной трек"
    to = kp / "ТА" / "Track Out"
    for p in sorted(to.glob("*.wav")):
        S.append((p, "основной трек", "stem", None))
    S.append((kp / "ТА" / "основной трек track out" / "демка_аранж_основной_трек.wav",
               "основной трек", "demo", None))
    martin_dir = kp / "-"
    for ver, fname in [("v2", "версия сведения 2.wav"), ("v3", "версия сведения 3..wav"),
                        ("v4", "версия сведения 4 .wav"), ("v5", "версия сведения 5.wav"),
                        ("v6", "версия сведения 6.wav"), ("v7", "версия сведения 7.wav")]:
        S.append((martin_dir / fname, "основной трек", "mix", ver))
    S.append((martin_dir / "Итог сведения.wav", "основной трек", "mix", "v7_итог"))
    takes_dir = kp / "ТА" / "основной трек track out" / "Project 3.band" / "Media" / "Audio Files"
    for p in sorted(takes_dir.glob("*.wav")):
        S.append((p, "основной трек", "take", None))

    S.append((ROOT / "референс А" / "+" / "1 референс А.mp3", "референс А", "reference", None))
    demucs_dir = ROOT / "референс А" / "demucs_stems"
    for stem_name in ["vocals", "drums", "bass", "other"]:
        S.append((demucs_dir / f"{stem_name}.wav", "референс А", "demucs_stem", stem_name))

    pv = ROOT / "Песня в поддержку рака лёгких"
    S.append((pv / "+ но это моя грязная демка" / "референс Б - 27:4:2026, 18.58.wav",
               "референс Б", "demo", None))
    pv_takes = pv / "проект" / "референс Б.band" / "Media" / "Audio Files"
    for p in sorted(pv_takes.glob("*.wav")):
        S.append((p, "референс Б", "take", None))

    zt = ROOT / "контрольный трек"
    for p in sorted((zt / "ТА" / "финальная" / "correct").glob("*.wav")):
        S.append((p, "контрольный трек", "stem", None))
    S.append((zt / "ТА" / "финальная" / "фин.mp3", "контрольный трек", "mix", "финал"))

    # Внешний трек для проверки словаря (не автора) — та же схема, что «референс А»:
    # референс-микс + Demucs-стемы, посчитанные для теста на 2026-08-17.
    S.append((ROOT / "ЧёЗаУродыНаСцене - внешний трек.mp3", "внешний трек", "reference", None))
    svet_demucs = ROOT / "_analysis" / "separated" / "htdemucs_ft" / "ЧёЗаУродыНаСцене - внешний трек"
    for stem_name in ["vocals", "drums", "bass", "other"]:
        S.append((svet_demucs / f"{stem_name}.wav", "внешний трек", "demucs_stem", stem_name))

    return [(p, song, role, ver) for p, song, role, ver in S if p.exists()]


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _bits_from_subtype(subtype):
    m = re.search(r"(\d+)", subtype or "")
    return int(m.group(1)) if m else None


def _stream_stats(path, block_frames=1 << 16, clip_thresh=0.999, clip_run=3):
    """Потоково: dual_mono, peak_dbfs, is_clipped — без загрузки файла целиком."""
    peak = 0.0
    dual_mono = True
    clipped = False
    run_len = 0
    checked_dual_mono = False
    with sf.SoundFile(str(path)) as f:
        n_ch = f.channels
        while True:
            block = f.read(block_frames, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            peak = max(peak, float(np.max(np.abs(block))))
            if n_ch == 2 and not checked_dual_mono:
                if not np.array_equal(block[:, 0], block[:, 1]):
                    dual_mono = False
                    checked_dual_mono = True
            over = np.any(np.abs(block) >= clip_thresh, axis=1) if n_ch > 1 else (np.abs(block[:, 0]) >= clip_thresh)
            for v in over:
                run_len = run_len + 1 if v else 0
                if run_len >= clip_run:
                    clipped = True
    peak_dbfs = 20 * np.log10(max(peak, 1e-12))
    return dual_mono if n_ch == 2 else False, peak_dbfs, clipped


def build_registry():
    rows = []
    anomalies = []
    seen_md5 = {}
    for path, song, role, version in _sources():
        try:
            info = sf.info(str(path))
        except Exception as e:
            anomalies.append(f"НЕ ЧИТАЕТСЯ: {path.relative_to(ROOT)} — {e}")
            continue
        md5 = _md5(path)
        dup_of = None
        if md5 in seen_md5:
            dup_of = seen_md5[md5]
            anomalies.append(f"ДУБЛИКАТ по md5: {path.relative_to(ROOT)} идентичен {dup_of}")
        else:
            seen_md5[md5] = str(path.relative_to(ROOT))

        dual_mono, peak_dbfs, is_clipped = _stream_stats(path)
        if is_clipped:
            anomalies.append(f"КЛИППИНГ: {path.relative_to(ROOT)} (peak {peak_dbfs:.1f} dBFS)")

        rows.append(dict(
            path=str(path.relative_to(ROOT)), song=song, role=role, version=version,
            sr=info.samplerate, bits=_bits_from_subtype(info.subtype), channels=info.channels,
            duration_s=info.frames / info.samplerate, n_samples=info.frames,
            md5=md5, dual_mono=dual_mono, peak_dbfs=round(peak_dbfs, 2), is_clipped=is_clipped,
            mtime=path.stat().st_mtime, duplicate_of=dup_of,
        ))

    df = pd.DataFrame(rows)

    for song, g in df.groupby("song"):
        stems = g[g.role == "stem"]
        if len(stems) == 0:
            continue
        # Эталонная длина — статистическая мода среди самих стемов (с
        # округлением до 0.1с), а не длина первого попавшегося микса: у
        # разных версий инженера сведения легитимно разная длина (v2 короче), и брать
        # "первую версию" как эталон — ошибка проверки, а не факт о данных.
        rounded = stems.duration_s.round(1)
        ref_dur = rounded.mode().iloc[0]
        off = stems[(stems.duration_s - ref_dur).abs() > 0.5]
        if len(off):
            for _, r in off.iterrows():
                anomalies.append(f"ДЛИНА СТЕМА ОТЛИЧАЕТСЯ ОТ БОЛЬШИНСТВА: {r.path} = {r.duration_s:.2f}с, "
                                  f"типичная длина стемов «{song}» = {ref_dur:.2f}с")
        mixes = g[g.role.isin(["mix", "reference"])]
        for _, r in mixes.iterrows():
            if abs(r.duration_s - ref_dur) > 0.5:
                anomalies.append(f"ДЛИНА МИКСА ОТЛИЧАЕТСЯ ОТ СТЕМОВ (может быть легитимно — другая версия/черновик): "
                                  f"{r.path} = {r.duration_s:.2f}с, стемы «{song}» = {ref_dur:.2f}с")

    return df, anomalies


if __name__ == "__main__":
    df, anomalies = build_registry()
    out = ROOT / "_analysis" / "registry.parquet"
    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"registry.parquet: {len(df)} файлов -> {out}")
    print(df.groupby(["song", "role"]).size())
    print(f"\n=== АНОМАЛИИ ({len(anomalies)}) ===")
    for a in anomalies:
        print(" -", a)
