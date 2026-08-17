"""Синтетический тест на pipeline/deconv/model.py — стемы с известными
EQ/громкостью/панорамой, проверяем что деконволюция их действительно
восстанавливает, прежде чем доверять ей на реальных 16+ дорожках.

Важное свойство модели, которое тест обязан учитывать, а не игнорировать:
ранг-1 разложение a_i[band,block] = G_i[t] (x) H_i[f] определено только с
точностью до масштаба на стем (G_i*c и H_i/c дают тот же продукт). Поэтому
сравниваем НЕ абсолютные значения, а форму — корреляцию с истинной кривой.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from analysis.legacy_scripts.deconv.model import deconvolve_channel

SR = 22050  # ниже, чем 44.1к — синтетика короткая, экономим время теста
DUR = 12.0


def band_limited_noise(sr, dur, lo, hi, seed):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    X[~mask] = 0
    return np.fft.irfft(X, n=n)


def apply_freq_gaussian_eq(x, sr, center, width, boost):
    """Точно заданная EQ-кривая H(f) = 1 + boost*exp(-((f-center)/width)^2),
    применённая прямым домножением в частотной области — истинная H(f)
    известна аналитически, без приближений IIR-фильтра."""
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    H = 1.0 + boost * np.exp(-((freqs - center) / width) ** 2)
    return np.fft.irfft(X * H, n=n), H, freqs


def true_h_at(freqs_query, center, width, boost):
    return 1.0 + boost * np.exp(-((freqs_query - center) / width) ** 2)


def test_deconv_recovers_known_gain_eq_pan():
    t = np.arange(int(SR * DUR)) / SR

    stems_spec = {
        # name: (band lo, band hi, gain-env freq Hz, gain-env phase, EQ center, EQ width, EQ boost, pan_L, pan_R)
        "bass": dict(lo=40, hi=250, genv_f=0.08, genv_ph=0.0, eq_c=120, eq_w=60, eq_boost=1.2, panL=1.0, panR=0.3),
        "mid":  dict(lo=300, hi=3000, genv_f=0.05, genv_ph=1.5, eq_c=1200, eq_w=400, eq_boost=0.8, panL=0.6, panR=0.6),
        "high": dict(lo=4000, hi=9000, genv_f=0.12, genv_ph=3.0, eq_c=6000, eq_w=1500, eq_boost=1.5, panL=0.25, panR=1.0),
    }

    raw = {name: band_limited_noise(SR, DUR, s["lo"], s["hi"], seed=hash(name) % 1000)
           for name, s in stems_spec.items()}

    true_genv = {}
    true_H_fn = {}
    processed = {}
    for name, s in stems_spec.items():
        env = 0.6 + 0.4 * np.sin(2 * np.pi * s["genv_f"] * t + s["genv_ph"])  # строго >0
        eqd, H_full, freqs_full = apply_freq_gaussian_eq(raw[name], SR, s["eq_c"], s["eq_w"], s["eq_boost"])
        processed[name] = env * eqd
        true_genv[name] = env
        true_H_fn[name] = lambda fq, s=s: true_h_at(fq, s["eq_c"], s["eq_w"], s["eq_boost"])

    mix_L = sum(stems_spec[n]["panL"] * processed[n] for n in stems_spec)
    mix_R = sum(stems_spec[n]["panR"] * processed[n] for n in stems_spec)

    stems_L = {n: raw[n] for n in stems_spec}  # деконволюция получает СЫРЫЕ стемы, как в реальности
    stems_R = stems_L  # дуал-моно контейнеры: тот же сигнал что и "левый" вход стема

    res_L = deconvolve_channel(mix_L, stems_L, SR, n_fft=2048, hop=256,
                                f_lo=50, f_hi=10000, bands_per_octave=4, block_s=0.5)
    res_R = deconvolve_channel(mix_R, stems_R, SR, n_fft=2048, hop=256,
                                f_lo=50, f_hi=10000, bands_per_octave=4, block_s=0.5)

    print(f"explained_fraction L={res_L.explained_fraction:.3f} R={res_R.explained_fraction:.3f}")
    assert res_L.explained_fraction > 0.85, f"L: модель плохо объясняет чистый (без остатка) микс: {res_L.explained_fraction}"
    assert res_R.explained_fraction > 0.85, f"R: модель плохо объясняет чистый (без остатка) микс: {res_R.explained_fraction}"

    for name, s in stems_spec.items():
        # --- форма огибающей громкости (корреляция, не абсолютное значение) ---
        true_g_at_blocks = np.interp(res_L.block_times, t, true_genv[name])
        rec_g = res_L.G[name]
        corr_g = np.corrcoef(true_g_at_blocks, rec_g)[0, 1]
        print(f"[{name}] gain-envelope corr = {corr_g:.3f}")
        assert corr_g > 0.7, f"{name}: восстановленная огибающая громкости не похожа на истинную (corr={corr_g:.3f})"

        # --- форма EQ-кривой (корреляция по полосам, не абсолютное значение) ---
        true_h_at_bands = true_H_fn[name](res_L.band_centers)
        rec_h = res_L.H[name]
        mask = rec_h > 0
        corr_h = np.corrcoef(true_h_at_bands[mask], rec_h[mask])[0, 1]
        print(f"[{name}] EQ-curve corr = {corr_h:.3f}")
        assert corr_h > 0.5, f"{name}: восстановленная EQ-кривая не похожа на истинную (corr={corr_h:.3f})"

        # --- панорама: отношение суммарной энергии L/R должно отражать pan^2 ---
        energy_L = np.sum(res_L.a_raw[name])
        energy_R = np.sum(res_R.a_raw[name])
        true_ratio = (s["panL"] ** 2) / (s["panR"] ** 2)
        rec_ratio = energy_L / max(energy_R, 1e-9)
        rel_err = abs(np.log(rec_ratio / true_ratio))
        print(f"[{name}] pan energy ratio true={true_ratio:.3f} recovered={rec_ratio:.3f} log-err={rel_err:.3f}")
        assert rel_err < 0.5, f"{name}: панорама восстановлена неверно (log-err={rel_err:.3f})"


def test_deconv_quality_drops_with_unexplained_residual():
    """Контроль качества §5.4: если в миксе есть контент, которого нет ни в
    одном стеме (реверб/добавленный слой), explained_fraction обязан упасть,
    а не остаться ложно высоким."""
    t = np.arange(int(SR * DUR)) / SR
    a = band_limited_noise(SR, DUR, 200, 2000, seed=1)
    b = band_limited_noise(SR, DUR, 3000, 8000, seed=2)
    stems = {"a": a, "b": b}

    mix_clean = a + b
    unexplained = band_limited_noise(SR, DUR, 500, 1500, seed=99) * 0.9  # "чужой" источник
    mix_with_residual = mix_clean + unexplained

    res_clean = deconvolve_channel(mix_clean, stems, SR, n_fft=2048, hop=256,
                                    f_lo=100, f_hi=9000, bands_per_octave=4, block_s=0.5)
    res_dirty = deconvolve_channel(mix_with_residual, stems, SR, n_fft=2048, hop=256,
                                    f_lo=100, f_hi=9000, bands_per_octave=4, block_s=0.5)

    print(f"explained_fraction clean={res_clean.explained_fraction:.3f} "
          f"with_unexplained_source={res_dirty.explained_fraction:.3f}")
    assert res_clean.explained_fraction > 0.9
    assert res_dirty.explained_fraction < res_clean.explained_fraction - 0.1, \
        "неучтённый источник обязан заметно просадить explained_fraction"


if __name__ == "__main__":
    test_deconv_recovers_known_gain_eq_pan()
    test_deconv_quality_drops_with_unexplained_residual()
    print("ALL SYNTHETIC DECONV TESTS PASSED")
