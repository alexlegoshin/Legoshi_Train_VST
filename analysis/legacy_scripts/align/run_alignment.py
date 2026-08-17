"""Шаг 2 ТЗ-01: выравнивание реального корпуса. §3.2 целиком:
полоса 200-4000 Гц -> огибающая 10мс -> log -> вычесть среднее -> GCC-PHAT
(грубо) -> уточнение до сэмпла сырой корреляцией в окне +-50мс.
Особый случай периодики (§3.2, активность <15% кадров) -> тактовая гипотеза.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

from analysis.legacy_scripts.align.gcc_phat import gcc_phat, bar_hypothesis_shift, activity_fraction

ROOT = Path(__file__).resolve().parents[4]

BPM = {"основной трек": 72.0, "контрольный трек": 73.0}
REF_MIX = {  # финальный микс инженера сведения на песню, §3.2: "референс — финальный микс"
    "основной трек": "основной трек/-/Итог сведения.wav",
    "контрольный трек": "контрольный трек/ТА/финальная/фин.mp3",
}
COARSE_HOP_MS = 10
COARSE_MAX_SHIFT_S = 8.0  # трекаут заведомо выровнен грубо, большой запас не нужен
REFINE_WINDOW_S = 0.05


def bandpass_envelope(x, sr, lo=200, hi=4000, hop_ms=COARSE_HOP_MS):
    sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    xf = sosfilt(sos, x)
    hop = int(sr * hop_ms / 1000)
    n = len(xf) // hop
    env = np.array([np.sqrt(np.mean(xf[i*hop:(i+1)*hop] ** 2) + 1e-20) for i in range(n)])
    log_env = np.log(env + 1e-9)
    return log_env - log_env.mean()


def _loudest_window_start(x, sr, win_len, hop_s=0.25):
    """Начало самого энергичного окна длиной win_len — якорь для уточнения.
    Фикс второго бага: раньше окно сравнения всегда бралось от начала файла
    (позиция 0), а дорожки типа баса/соло часто вступают не с первой
    секунды — там просто тишина, норма = 0, всё уточнение молча ломалось."""
    hop = int(hop_s * sr)
    if hop <= 0 or len(x) <= win_len:
        return 0
    n_steps = max(1, (len(x) - win_len) // hop)
    energies = [np.sum(x[i*hop:i*hop+win_len] ** 2) for i in range(n_steps)]
    return int(np.argmax(energies)) * hop


def refine_shift(sig, ref, sr, coarse_shift_samples, window_s=REFINE_WINDOW_S):
    """Кросс-корреляция в узком окне вокруг грубой оценки — до сэмпла.

    Два бага, пойманных именно на реальных файлах, не на синтетике:
    1) Сырой dot product без нормировки тянется туда, где сравниваемый
       кусок громче, а не туда, где сигналы реально совпадают по времени —
       нормируем на нормы окон (косинусная корреляция).
    2) Окно сравнения бралось от начала файла (позиция 0) — а бас/соло/
       поздний вокал там часто просто молчат, норма 0, всё уточнение
       молча проваливалось. Теперь окно — вокруг самого энергичного
       участка сигнала, а не вокруг первой секунды."""
    win = int(window_s * sr)
    cmp_len = min(sr * 2, len(sig) // 2)
    anchor = _loudest_window_start(sig, sr, cmp_len)

    best_shift, best_score = coarse_shift_samples, -np.inf
    for d in range(coarse_shift_samples - win, coarse_shift_samples + win + 1):
        ref_start = anchor - d
        if ref_start < 0 or ref_start + cmp_len > len(ref) or anchor + cmp_len > len(sig):
            continue
        a_m = sig[anchor:anchor + cmp_len]
        b_m = ref[ref_start:ref_start + cmp_len]
        denom = np.linalg.norm(a_m) * np.linalg.norm(b_m)
        if denom < 1e-9:
            continue
        score = np.dot(a_m, b_m) / denom
        if score > best_score:
            best_score, best_shift = score, d
    return best_shift, best_score


def align_song(song, stem_rows, ref_path, sr_expected=44100):
    ref, sr_ref = sf.read(str(ROOT / ref_path), dtype="float32", always_2d=True)
    ref = ref.mean(axis=1)  # моно-сумма референса для выравнивания достаточно
    assert sr_ref == sr_expected, f"неожиданный sr референса {song}: {sr_ref}"
    ref_env = bandpass_envelope(ref, sr_ref)

    results = []
    for _, row in stem_rows.iterrows():
        p = ROOT / row.path
        sig, sr_sig = sf.read(str(p), dtype="float32", always_2d=True)
        sig_mono = sig[:, 0] if sig.shape[1] == 1 or _is_dual_mono(sig) else sig.mean(axis=1)
        assert sr_sig == sr_expected, f"неожиданный sr у {row.path}: {sr_sig}"

        sig_env = bandpass_envelope(sig_mono, sr_sig)
        shift_frames, confidence, p1, p2, z = gcc_phat(sig_env, ref_env, sr=int(1000 / COARSE_HOP_MS),
                                                          max_shift_s=COARSE_MAX_SHIFT_S)
        coarse_shift_samples = int(round(shift_frames * sr_sig * COARSE_HOP_MS / 1000))
        confident = (confidence > 1.3) and (z > 8)
        method = "gcc_phat_envelope"
        ambiguous_ties = None

        act = activity_fraction(sig_mono, sr_sig)
        if not confident and act < 0.15 and song in BPM:
            period_s = 60.0 / BPM[song]
            bar_s = period_s * 4
            bshift_s, bscore, ambiguous, ties = bar_hypothesis_shift(
                bar_s, max_bars=int(COARSE_MAX_SHIFT_S / bar_s) + 1, sig=sig_mono, ref=ref, sr=sr_sig)
            method = "bar_hypothesis"
            if not ambiguous:
                coarse_shift_samples = int(round(bshift_s * sr_sig))
                confident = True
            else:
                ambiguous_ties = ties

        aligned = confident
        offset_samples, refine_score = None, None
        refine_hit_boundary = False
        if aligned:
            win = int(REFINE_WINDOW_S * sr_sig)
            offset_samples, refine_score = refine_shift(sig_mono, ref, sr_sig, coarse_shift_samples)
            # Если уточнение легло ровно на край окна поиска — это не находка,
            # это сигнал, что окно узкое или уточнение неустойчиво. Не тащим
            # такой результат как будто он надёжен — метим отдельно.
            if abs(offset_samples - coarse_shift_samples) >= win:
                refine_hit_boundary = True

        results.append(dict(
            path=row.path, song=song, role=row.role, activity=round(act, 3),
            coarse_offset_samples=coarse_shift_samples,
            offset_samples=offset_samples, offset_s=(offset_samples / sr_sig) if offset_samples is not None else None,
            refine_score=round(float(refine_score), 4) if refine_score is not None else None,
            refine_hit_boundary=refine_hit_boundary,
            confidence=round(float(confidence), 3), z_score=round(float(z), 2),
            method=method, aligned=aligned, ambiguous_ties=str(ambiguous_ties) if ambiguous_ties else None,
        ))
        boundary_flag = " [!УПЁРЛОСЬ В ГРАНИЦУ ОКНА]" if refine_hit_boundary else ""
        print(f"  {row.path.split('/')[-1]:45s} aligned={aligned!s:5} "
              f"offset={results[-1]['offset_s']} conf={confidence:.2f} z={z:.1f} method={method}{boundary_flag}")
    return results


def _is_dual_mono(sig):
    return sig.shape[1] == 2 and np.array_equal(sig[:, 0], sig[:, 1])


def main():
    reg = pd.read_parquet(ROOT / "_analysis" / "registry.parquet")
    all_results = []
    for song, ref_rel in REF_MIX.items():
        stems = reg[(reg.song == song) & (reg.role.isin(["stem", "demo"]))]
        print(f"\n=== {song}: {len(stems)} файлов, референс {ref_rel} ===")
        all_results += align_song(song, stems, ref_rel)

    df = pd.DataFrame(all_results)
    out = ROOT / "_analysis" / "alignment.parquet"
    df.to_parquet(out, index=False)

    print(f"\n=== ИТОГ: {out} ===")
    not_aligned = df[~df.aligned]
    print(f"Не выровнено: {len(not_aligned)} из {len(df)}")
    for _, r in not_aligned.iterrows():
        print(f"  - {r.path}: conf={r.confidence} z={r.z_score} method={r.method} "
              f"activity={r.activity} ties={r.ambiguous_ties}")

    hit_boundary = df[df.aligned & df.refine_hit_boundary]
    if len(hit_boundary):
        print(f"\nВыровнено, но уточнение легло на край окна ({REFINE_WINDOW_S*1000:.0f}мс) — "
              f"доверять офсету осторожно, окно узкое для этих файлов: {len(hit_boundary)}")
        for _, r in hit_boundary.iterrows():
            print(f"  - {r.path}: offset={r.offset_s}s refine_score={r.refine_score}")


if __name__ == "__main__":
    main()
