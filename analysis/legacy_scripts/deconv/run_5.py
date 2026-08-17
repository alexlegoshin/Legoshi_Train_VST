"""Шаг 5 ТЗ-01: деконволюция микса по стемам. «основной трек» (все 5 версий
инженера сведения) + «контрольный трек» для контроля. Только выровненные стемы (§3.2) —
невыровненные (back high, соло, электро бэк для «основной трек») исключены
и об этом явно сказано, а не забыто: их энергия уйдёт в остаток и завысит
оценку «мокрости», это надо держать в голове при интерпретации residual."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf

from analysis.legacy_scripts.deconv.model import deconvolve_channel

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "_analysis" / "deconv"

MARTIN_VERSIONS = {
    "v2": "основной трек/-/версия сведения 2.wav",
    "v3": "основной трек/-/версия сведения 3..wav",
    "v4": "основной трек/-/версия сведения 4 .wav",
    "v5": "основной трек/-/версия сведения 5.wav",
    "v6": "основной трек/-/версия сведения 6.wav",
    "v7": "основной трек/-/версия сведения 7.wav",
}
ZT_MIX = "контрольный трек/ТА/финальная/фин.mp3"


def align_stem(stem, offset_samples):
    """stem_aligned[n] = stem[n + offset_samples] (см. вывод знака в docstring
    run_alignment.py): + означает стем отстаёт от референса."""
    offset_samples = int(round(offset_samples))
    n = len(stem)
    if offset_samples == 0:
        return stem.copy()
    if offset_samples > 0:
        aligned = np.concatenate([stem[offset_samples:], np.zeros(offset_samples)])
    else:
        aligned = np.concatenate([np.zeros(-offset_samples), stem[:offset_samples]])
    return aligned[:n]


def load_aligned_stems(song, root=ROOT):
    reg = pd.read_parquet(root / "_analysis" / "registry.parquet")
    align = pd.read_parquet(root / "_analysis" / "alignment.parquet")
    stems = reg[(reg.song == song) & (reg.role == "stem")]
    merged = stems.merge(align[["path", "aligned", "offset_samples"]], on="path", how="left")

    excluded = merged[~merged.aligned.fillna(False)]
    included = merged[merged.aligned.fillna(False)]
    print(f"[{song}] стемов всего: {len(merged)}, включено: {len(included)}, "
          f"исключено (не выровнены): {list(excluded.path.str.split('/').str[-1])}")

    stems_L, stems_R = {}, {}
    for _, row in included.iterrows():
        name = row.path.split("/")[-1].replace(".wav", "")
        data, sr = sf.read(str(root / row.path), dtype="float64", always_2d=True)
        mono = data[:, 0] if data.shape[1] == 1 or np.array_equal(data[:, 0], data[:, 1]) else data.mean(axis=1)
        aligned = align_stem(mono, row.offset_samples)
        stems_L[name] = aligned
        stems_R[name] = aligned  # дуал-моно источник — тот же сигнал в обе деконволюции L/R
    return stems_L, stems_R, sr


def run_one(mix_path, stems_L, stems_R, sr_expected, label):
    data, sr = sf.read(str(ROOT / mix_path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mix_L, mix_R = data[:, 0], data[:, 1]

    n = min(len(mix_L), *[len(s) for s in stems_L.values()])
    mix_L, mix_R = mix_L[:n], mix_R[:n]
    stems_L_cut = {k: v[:n] for k, v in stems_L.items()}
    stems_R_cut = {k: v[:n] for k, v in stems_R.items()}

    print(f"  [{label}] деконволюция L...")
    res_L = deconvolve_channel(mix_L, stems_L_cut, sr)
    print(f"  [{label}] деконволюция R...")
    res_R = deconvolve_channel(mix_R, stems_R_cut, sr)

    quality = dict(
        version=label, explained_fraction_L=res_L.explained_fraction, explained_fraction_R=res_R.explained_fraction,
        residual_energy_L=res_L.residual_energy, residual_energy_R=res_R.residual_energy,
        mix_energy_L=res_L.mix_energy, mix_energy_R=res_R.mix_energy,
    )
    ok = res_L.explained_fraction >= 0.70 and res_R.explained_fraction >= 0.70
    print(f"  [{label}] explained_fraction L={res_L.explained_fraction:.3f} R={res_R.explained_fraction:.3f} "
          f"{'OK' if ok else '!!! НИЖЕ 70%, РАЗБИРАТЬСЯ, НЕ ДОКЛАДЫВАТЬ ВЫВОДЫ'}")

    gains_rows, eq_rows, pan_rows = [], [], []
    for name in stems_L_cut:
        g_l, g_r = res_L.G[name], res_R.G[name]
        h_l, h_r = res_L.H[name], res_R.H[name]
        energy_l, energy_r = res_L.a_raw[name].sum(), res_R.a_raw[name].sum()
        pan_ratio = energy_l / max(energy_r, 1e-12)

        for k, t in enumerate(res_L.block_times):
            gains_rows.append(dict(version=label, stem=name, t_s=float(t),
                                    gain_L=float(g_l[k]) if k < len(g_l) else np.nan,
                                    gain_R=float(g_r[k]) if k < len(g_r) else np.nan))
        for b, f in enumerate(res_L.band_centers):
            eq_rows.append(dict(version=label, stem=name, freq_hz=float(f),
                                 eq_L=float(h_l[b]) if b < len(h_l) else np.nan,
                                 eq_R=float(h_r[b]) if b < len(h_r) else np.nan))
        pan_rows.append(dict(version=label, stem=name, energy_L=float(energy_l),
                              energy_R=float(energy_r), pan_ratio_L_over_R=float(pan_ratio)))

    return quality, pd.DataFrame(gains_rows), pd.DataFrame(eq_rows), pd.DataFrame(pan_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== Загрузка и выравнивание стемов «основной трек» ===")
    stems_L, stems_R, sr = load_aligned_stems("основной трек")

    all_quality, all_gains, all_eq, all_pan = [], [], [], []
    for label, path in MARTIN_VERSIONS.items():
        print(f"\n=== основной трек {label} ===")
        q, g, e, p = run_one(path, stems_L, stems_R, sr, label)
        all_quality.append(q)
        all_gains.append(g)
        all_eq.append(e)
        all_pan.append(p)

    pd.DataFrame(all_quality).to_parquet(OUT / "kp_quality.parquet", index=False)
    pd.concat(all_gains).to_parquet(OUT / "kp_gains.parquet", index=False)
    pd.concat(all_eq).to_parquet(OUT / "kp_eq.parquet", index=False)
    pd.concat(all_pan).to_parquet(OUT / "kp_pan.parquet", index=False)
    print(f"\nСохранено -> {OUT}")

    print("\n=== Контроль: «контрольный трек» ===")
    zt_stems_L, zt_stems_R, zt_sr = load_aligned_stems("контрольный трек")
    q_zt, g_zt, e_zt, p_zt = run_one(ZT_MIX, zt_stems_L, zt_stems_R, zt_sr, "финал")
    pd.DataFrame([q_zt]).to_parquet(OUT / "zt_quality.parquet", index=False)
    g_zt.to_parquet(OUT / "zt_gains.parquet", index=False)
    e_zt.to_parquet(OUT / "zt_eq.parquet", index=False)
    p_zt.to_parquet(OUT / "zt_pan.parquet", index=False)

    print("\n=== ИТОГ: качество по всем версиям ===")
    print(pd.DataFrame(all_quality + [q_zt]).to_string(index=False))


if __name__ == "__main__":
    main()
