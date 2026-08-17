"""Шаг 10 §10.2/§10.5: нарезка отобранных фрагментов в реальные WAV-клипы.

Границы — не жёсткая сетка, а ближайшие onset'ы (librosa.onset.onset_detect)
вокруг точки интереса, длина 8-12с. Нормализация к integrated LUFS -18,
фейды 20мс, единый формат (WAV 44.1кГц), без дизера, без пиковой
нормализации поверх LUFS."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import librosa

ROOT = Path(__file__).resolve().parents[4]
Q_DIR = ROOT / "_analysis" / "questionnaire"
CLIPS_DIR = Q_DIR / "clips"

TARGET_LUFS = -18.0
FADE_MS = 20
MIN_LEN_S, MAX_LEN_S, DEFAULT_LEN_S = 8.0, 12.0, 10.0
SR_OUT = 44100


def find_clip_bounds(onsets, center_s, min_len=MIN_LEN_S, max_len=MAX_LEN_S, default_len=DEFAULT_LEN_S):
    target_start = center_s - default_len / 2
    start_candidates = onsets[onsets <= target_start]
    start = float(start_candidates[-1]) if len(start_candidates) else max(0.0, target_start)

    end_candidates = onsets[(onsets >= start + min_len) & (onsets <= start + max_len)]
    if len(end_candidates):
        # ближайший к дефолтной длине
        end = float(end_candidates[np.argmin(np.abs(end_candidates - (start + default_len)))])
    else:
        end = start + default_len
    return start, end


def apply_fades(x, sr, fade_ms=FADE_MS):
    n_fade = int(fade_ms / 1000 * sr)
    n_fade = min(n_fade, len(x) // 4)
    if n_fade <= 0:
        return x
    fade_in = np.linspace(0, 1, n_fade)
    fade_out = np.linspace(1, 0, n_fade)
    x = x.copy()
    x[:n_fade] *= fade_in
    x[-n_fade:] *= fade_out
    return x


TRUE_PEAK_CEILING_DBFS = -1.0  # запас перед клиппингом/интерсэмпл-перегрузом


def normalize_lufs(x, sr, meter, target=TARGET_LUFS, peak_ceiling_dbfs=TRUE_PEAK_CEILING_DBFS):
    """ПОЙМАНО НА РЕАЛЬНЫХ КЛИПАХ: приведение к -18 LUFS без оглядки на пик
    даёт клиппинг там, где у источника (особенно у тихой демки) громкие
    транзиенты при низкой средней громкости — 21 из 102 клипов клиппинговали
    (peak=0dBFS) при первом прогоне. Теперь gain дополнительно ограничен
    так, чтобы пик после усиления не превышал peak_ceiling_dbfs."""
    if not np.any(np.abs(x) > 1e-9):
        return x
    current = meter.integrated_loudness(x)
    if not np.isfinite(current):
        return x
    gain_db = target - current
    peak = np.max(np.abs(x))
    peak_dbfs = 20 * np.log10(peak + 1e-12)
    max_gain_db = peak_ceiling_dbfs - peak_dbfs
    peak_limited = gain_db > max_gain_db
    gain_db = min(gain_db, max_gain_db)
    return x * (10 ** (gain_db / 20)), peak_limited


def main():
    fragments = pd.read_parquet(Q_DIR / "fragments_selected.parquet")
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    meter = pyln.Meter(SR_OUT)

    audio_cache = {}
    onset_cache = {}
    rows = []
    errors = []

    for _, frag in fragments.iterrows():
        file_path = ROOT / frag["file"]
        key = str(file_path)
        if key not in audio_cache:
            data, sr = sf.read(str(file_path), dtype="float64", always_2d=True)
            mono = data.mean(axis=1)
            if sr != SR_OUT:
                mono = librosa.resample(mono, orig_sr=sr, target_sr=SR_OUT)
            audio_cache[key] = mono
            onset_frames = librosa.onset.onset_detect(y=mono, sr=SR_OUT, units="time", backtrack=True)
            onset_cache[key] = np.asarray(onset_frames)
            print(f"загружен {frag['file']}: {len(mono)/SR_OUT:.1f}с, onset'ов: {len(onset_frames)}")

        mono = audio_cache[key]
        onsets = onset_cache[key]
        center = frag["t_start"] + 2.0  # середина 4с окна метрики
        dur_total = len(mono) / SR_OUT
        center = min(center, dur_total - MIN_LEN_S / 2)

        start_s, end_s = find_clip_bounds(onsets, center)
        end_s = min(end_s, dur_total)
        if end_s - start_s < MIN_LEN_S - 0.5:
            errors.append((frag["fragment_id"], "клип короче минимума после привязки к onset"))
            continue

        clip = mono[int(start_s * SR_OUT):int(end_s * SR_OUT)]
        clip, peak_limited = normalize_lufs(clip, SR_OUT, meter)
        clip = apply_fades(clip, SR_OUT)
        achieved_peak_dbfs = 20 * np.log10(np.max(np.abs(clip)) + 1e-12)
        clip = np.clip(clip, -1.0, 1.0)

        out_path = CLIPS_DIR / f"{frag['fragment_id']}.wav"
        sf.write(str(out_path), clip, SR_OUT, subtype="PCM_16")

        rows.append(dict(fragment_id=frag["fragment_id"], version=frag["version"], group=frag["group"],
                          reason=frag["reason"], file=frag["file"], start_s=round(start_s, 3),
                          end_s=round(end_s, 3), duration_s=round(end_s - start_s, 3),
                          clip_path=str(out_path.relative_to(ROOT)),
                          peak_limited=bool(peak_limited), achieved_peak_dbfs=round(achieved_peak_dbfs, 2)))

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(Q_DIR / "clips_manifest.parquet", index=False)
    manifest.to_csv(Q_DIR / "clips_manifest.csv", index=False)
    print(f"\nНарезано клипов: {len(manifest)} -> {Q_DIR / 'clips_manifest.parquet'}")
    print(f"Средняя длина: {manifest.duration_s.mean():.2f}с, диапазон: "
          f"{manifest.duration_s.min():.2f}-{manifest.duration_s.max():.2f}с")
    print(f"Ограничены по пику (не дотянули точно до -18 LUFS, чтобы не клиппинговать): "
          f"{int(manifest.peak_limited.sum())}/{len(manifest)}")
    if errors:
        print(f"\nОшибок: {len(errors)}")
        for fid, msg in errors:
            print(f"  {fid}: {msg}")


if __name__ == "__main__":
    main()
