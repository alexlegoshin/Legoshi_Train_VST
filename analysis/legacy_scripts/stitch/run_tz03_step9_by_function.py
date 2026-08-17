"""ТЗ-03 задача 6 (Шаг 9 пересчёт): сравнение КП и «референс А» по секциям
СОПОСТАВИМОЙ ФУНКЦИИ (куплет-куплет, кульминация-кульминация...), не
файл целиком. Сопоставление подтверждено автором 17.08.2026, с уточнением
эскалации по трём припевам/кульминациям (не общая группировка):

  Chorus 1                    <-> Куплет 1 (локальная кульминация)
  Chorus 2 (кульминация)      <-> Кульминация
  Chorus 3                    <-> Эхо кульминации

Метрики для «референс А» считаются заново по её собственному аудио на
границах секций (band_frac_*/skewness/spectral_slope/warmth_ratio —
тривиально на любом отрезке) и по demucs-вокалу (vibrato_depth_cents,
voiced_fraction — валидировано в задаче 3 ТЗ-02, ошибка ~0.2-2%, надёжнее
полного микса)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.metrics.psychoacoustic import warmth_ratio as warmth_ratio_fn
from analysis.metrics.spectral import compute_stft, ltas, spectral_slope as spectral_slope_fn, \
    spectral_moments, named_band_energy_fraction

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

VERSIONS = ["v2", "v3", "v4", "v5", "v6", "v7"]
MAX_PLAUSIBLE_VIBRATO_CENTS = 100.0

PAIRS = [
    ("Тихое/интро", ["Verse 1 (интро)"], ["Вступление"]),
    ("Куплеты/развитие", ["Verse 1 (развитие)", "Verse 2"], ["Куплет 1 (развитие)"]),
    ("Chorus1 <-> локальная кульминация", ["Chorus 1"], ["Куплет 1 (локальная кульминация)"]),
    ("Chorus2 <-> кульминация", ["Chorus 2 (кульминация припева)"], ["Кульминация"]),
    ("Chorus3 <-> эхо кульминации", ["Chorus 3"], ["Эхо кульминации"]),
    ("Бридж", ["Bridge/Verse 3 (пик)", "Bridge/Verse 3 (спад)"], ["Бридж"]),
    ("Соло/проигрыши", ["Соло", "Соло (выход)"], ["Проигрыш (бас входит)", "Проигрыш 2 (база перед кульминацией)"]),
    ("Паузы/затишья", ["пауза"], ["Минизатишье перед разъёбом"]),
    ("Финал", ["Финал"], ["Возврат к теме (пустота)"]),
]

METRICS_SPECTRAL = ["band_frac_air", "band_frac_lowmid", "band_frac_low", "skewness", "spectral_slope", "warmth_ratio"]


def radost_section_metrics(mono, sr, start_s, end_s):
    seg = mono[int(start_s * sr):int(end_s * sr)]
    if len(seg) < sr // 2:
        return {m: np.nan for m in METRICS_SPECTRAL}
    f, t, mag = compute_stft(seg, sr)
    if mag.shape[1] == 0:
        return {m: np.nan for m in METRICS_SPECTRAL}
    centers, levels = ltas(mag, f, bands_per_octave=3)
    slope = spectral_slope_fn(centers, levels)
    _, _, sk, _ = spectral_moments(mag, f)
    bands = named_band_energy_fraction(mag, f)
    warmth = warmth_ratio_fn(seg, sr)
    return dict(
        band_frac_air=float(np.median(bands["air"])), band_frac_lowmid=float(np.median(bands["lowmid"])),
        band_frac_low=float(np.median(bands["low"])), skewness=float(np.median(sk)),
        spectral_slope=slope, warmth_ratio=warmth,
    )


def radost_vocal_metrics(notes_df, f0_df, start_s, end_s):
    out = dict(vibrato_depth_cents=np.nan, voiced_fraction=np.nan)
    if f0_df is not None:
        fsub = f0_df[(f0_df.t_s >= start_s) & (f0_df.t_s < end_s)]
        if len(fsub):
            out["voiced_fraction"] = float(fsub.voiced.mean())
    if notes_df is not None:
        mid = (notes_df.t_start + notes_df.t_end) / 2
        nsub = notes_df.loc[(mid >= start_s) & (mid < end_s), "vibrato_depth_cents"].dropna()
        nsub = nsub[nsub <= MAX_PLAUSIBLE_VIBRATO_CENTS]
        if len(nsub) >= 2:
            out["vibrato_depth_cents"] = float(nsub.median())
    return out


def kp_windows_by_section(kp_windows, kp_sections, section_labels):
    """Окна скользящего окна КП, чьи t_start попадают в объединение секций
    из section_labels (по границам sections.csv). Используем kp_windows
    (moving_window_4s1s.parquet), а не старый kp_section_diff.parquet —
    там не было spectral_slope/warmth_ratio, единообразие с референсом А важнее."""
    mask = np.zeros(len(kp_windows), dtype=bool)
    for lbl in section_labels:
        row = kp_sections[kp_sections.section == lbl]
        if len(row) == 0:
            continue
        s, e = float(row.iloc[0].start_s), float(row.iloc[0].end_s)
        mask |= (kp_windows.t_start >= s) & (kp_windows.t_start < e)
    return kp_windows[mask]


def main():
    sec = pd.read_csv(ROOT / "_analysis" / "sections.csv")
    kp_sections = sec[sec.song == "основной трек"]
    kp_windows = pd.read_parquet(OUT / "moving_window_4s1s.parquet")
    radost_sec = sec[sec.song == "референс А"].set_index("section")

    data, sr = sf.read(str(ROOT / "референс А" / "+" / "1 референс А.mp3"), dtype="float64", always_2d=True)
    mono = data.mean(axis=1)

    notes_df = pd.read_parquet(METRICS_DIR / "референс А__demucs_stems__vocals.wav.4_6_notes.parquet")
    notes_df = notes_df[notes_df.type == "note"] if "type" in notes_df.columns else notes_df
    f0_df = pd.read_parquet(METRICS_DIR / "референс А__demucs_stems__vocals.wav.4_6_f0.parquet")

    rows = []
    for label, kp_labels, rad_labels in PAIRS:
        # --- КП: медиана по версии по окнам скользящего окна, попавшим в секции группы ---
        kp_group_windows = kp_windows_by_section(kp_windows, kp_sections, kp_labels)
        kp_vals = {}
        for metric in METRICS_SPECTRAL + ["vibrato_depth_cents", "voiced_fraction"]:
            kp_vals[metric] = {}
            for v in ["demo"] + VERSIONS:
                sub = kp_group_windows[(kp_group_windows.version == v)][metric].dropna()
                kp_vals[metric][v] = float(sub.median()) if len(sub) else np.nan

        # --- референс А: медиана по всем секциям группы ---
        rad_spectral = []
        rad_vocal = []
        for rl in rad_labels:
            if rl not in radost_sec.index:
                continue
            s, e = radost_sec.loc[rl, ["start_s", "end_s"]]
            rad_spectral.append(radost_section_metrics(mono, sr, s, e))
            rad_vocal.append(radost_vocal_metrics(notes_df, f0_df, s, e))
        rad_vals = {}
        for metric in METRICS_SPECTRAL:
            vals = [d[metric] for d in rad_spectral if pd.notna(d.get(metric))]
            rad_vals[metric] = float(np.median(vals)) if vals else np.nan
        for metric in ["vibrato_depth_cents", "voiced_fraction"]:
            vals = [d[metric] for d in rad_vocal if pd.notna(d.get(metric))]
            rad_vals[metric] = float(np.median(vals)) if vals else np.nan

        for metric in METRICS_SPECTRAL + ["vibrato_depth_cents", "voiced_fraction"]:
            v2, v7 = kp_vals[metric].get("v2"), kp_vals[metric].get("v7")
            rad = rad_vals[metric]
            verdict = None
            if pd.notna(v2) and pd.notna(v7) and pd.notna(rad):
                lo, hi = min(v2, v7), max(v2, v7)
                if lo <= rad <= hi:
                    verdict = "в диапазоне v2..v7"
                elif (rad > hi and v7 > v2) or (rad < lo and v7 < v2):
                    verdict = "дальше в сторону v7 (по ходу)"
                else:
                    verdict = "ПРОТИВОПОЛОЖНО (в сторону v2 или дальше)"
            rows.append(dict(pair=label, metric=metric, kp_v2=v2, kp_v7=v7, radost=rad, вердикт=verdict))

    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "step9_by_function.parquet", index=False)
    table.to_csv(OUT / "step9_by_function.csv", index=False)

    pd.set_option("display.width", 160)
    print(table.to_string(index=False))

    valid = table[table.вердикт.notna()]
    matched = valid[valid.вердикт != "ПРОТИВОПОЛОЖНО (в сторону v2 или дальше)"]
    print(f"\nСогласуется с траекторией КП: {len(matched)}/{len(valid)}")
    print("\n=== По метрике ===")
    print(valid.groupby("metric").apply(
        lambda g: f"{(g.вердикт != 'ПРОТИВОПОЛОЖНО (в сторону v2 или дальше)').sum()}/{len(g)}"))


if __name__ == "__main__":
    main()
