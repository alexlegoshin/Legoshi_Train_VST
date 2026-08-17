"""ТЗ-02 Задача 3: валидация Demucs. Прогоняем htdemucs_ft на v7 «основного трека», где реальные вокальные стемы известны (уже выровнены в §5), и
меряем, насколько разделённый вокал отличается от суммы настоящих
вокальных дорожек. Даёт числовую погрешность НА ЭТОМ КОНКРЕТНОМ
материале, не по чужому бенчмарку — и калибрует, насколько можно
доверять §4.6-метрикам (вибрато, F0), посчитанным на demucs-стеме
«референс А», где настоящих стемов нет вообще."""
import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.legacy_scripts.deconv.run_5 import load_aligned_stems
from analysis.legacy_scripts.align.gcc_phat import gcc_phat

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "separated"
SEP_DIR = ROOT / "_analysis" / "separated" / "htdemucs_ft" / "версия сведения 7"

VOCAL_STEM_NAMES = ["вокал основной", "вокал дублирующий основной",
                     "вокал дублирующий на октаву ниже", "вокал бэк в соло"]


def main():
    print("=== Загрузка реальных вокальных стемов (выровненных, §5) ===")
    stems_L, stems_R, sr = load_aligned_stems("основной трек")
    stems_L = {unicodedata.normalize("NFC", k): v for k, v in stems_L.items()}
    present = [n for n in VOCAL_STEM_NAMES if n in stems_L]
    print(f"Вокальных стемов в сумме: {present}")
    n = min(len(stems_L[n]) for n in present)
    true_vocal = np.sum([stems_L[name][:n] for name in present], axis=0)

    demucs_vocal_path = SEP_DIR / "vocals.wav"
    if not demucs_vocal_path.exists():
        print(f"ЖДЁМ: {demucs_vocal_path} ещё не готов")
        return
    data, sr_d = sf.read(str(demucs_vocal_path), dtype="float64", always_2d=True)
    assert sr_d == sr
    demucs_vocal = data.mean(axis=1)
    m = min(len(demucs_vocal), n)
    true_vocal, demucs_vocal = true_vocal[:m], demucs_vocal[:m]

    print("\n=== Выравнивание (Demucs может давать сдвиг в несколько мс) ===")
    shift, conf, _, _, z = gcc_phat(demucs_vocal, true_vocal, sr, max_shift_s=0.05)
    print(f"сдвиг: {shift/sr*1000:.2f}мс, confidence={conf:.2f}, z={z:.1f}")
    shift_samples = int(round(shift))
    if shift_samples > 0:
        demucs_aligned = demucs_vocal[shift_samples:]
        true_aligned = true_vocal[:len(demucs_aligned)]
    elif shift_samples < 0:
        true_aligned = true_vocal[-shift_samples:]
        demucs_aligned = demucs_vocal[:len(true_aligned)]
    else:
        demucs_aligned, true_aligned = demucs_vocal, true_vocal

    # нормализация по RMS перед сравнением формы (Demucs даёт свой уровень)
    true_rms = np.sqrt(np.mean(true_aligned ** 2))
    demucs_rms = np.sqrt(np.mean(demucs_aligned ** 2))
    demucs_scaled = demucs_aligned * (true_rms / max(demucs_rms, 1e-12))

    corr = float(np.corrcoef(demucs_scaled, true_aligned)[0, 1])
    residual = demucs_scaled - true_aligned
    residual_energy = float(np.sum(residual ** 2))
    true_energy = float(np.sum(true_aligned ** 2))
    explained_fraction = 1 - residual_energy / max(true_energy, 1e-12)

    # спектральное сходство: корреляция LTAS
    from analysis.metrics.spectral import compute_stft, ltas
    f1, t1, mag1 = compute_stft(true_aligned, sr)
    f2, t2, mag2 = compute_stft(demucs_scaled, sr)
    c1, l1 = ltas(mag1, f1, bands_per_octave=6)
    c2, l2 = ltas(mag2, f2, bands_per_octave=6)
    spectral_corr = float(np.corrcoef(l1, l2)[0, 1])

    # F0/вибрато: сравнить §4.6-метрики на true_vocal (стемы) и на demucs vocal
    from analysis.metrics.pitch_vocal import analyze_file
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p_true = Path(td) / "true.wav"
        p_dem = Path(td) / "demucs.wav"
        sf.write(str(p_true), true_aligned, sr)
        sf.write(str(p_dem), demucs_scaled, sr)
        s_true, _ = analyze_file(p_true)
        s_dem, _ = analyze_file(p_dem)

    result = dict(
        shift_ms=shift / sr * 1000, gcc_confidence=conf,
        waveform_correlation=corr, explained_fraction=explained_fraction,
        residual_energy=residual_energy, true_energy=true_energy,
        spectral_ltas_correlation=spectral_corr,
        true_vibrato_depth_cents=s_true.get("vibrato_depth_cents_median"),
        demucs_vibrato_depth_cents=s_dem.get("vibrato_depth_cents_median"),
        true_voiced_fraction=s_true.get("voiced_fraction"),
        demucs_voiced_fraction=s_dem.get("voiced_fraction"),
        true_f0_median=s_true.get("f0_hz_median"),
        demucs_f0_median=s_dem.get("f0_hz_median"),
    )

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "validation.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print("\n=== Результат валидации (v7, demucs vocal vs сумма настоящих вокальных стемов) ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"\n-> {OUT / 'validation.json'}")


if __name__ == "__main__":
    main()
