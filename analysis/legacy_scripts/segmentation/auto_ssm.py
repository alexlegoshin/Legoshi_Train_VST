"""Автоматическая сегментация, §3.4 п.2: self-similarity matrix по MFCC +
chroma, novelty curve (Foote, 2000), пики -> границы. Сравнение с ручной."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[4]
HOP = 2048  # ~46мс при 44.1к — достаточно для масштаба структуры, не для транзиентов


def foote_novelty(ssm, kernel_size=32):
    """Шахматный (checkerboard) ядро Foote вдоль диагонали SSM."""
    half = kernel_size // 2
    ax = np.arange(-half, half)
    sign = np.sign(np.outer(ax, ax))
    gauss = np.outer(np.exp(-ax**2 / (2 * (half/2)**2)), np.exp(-ax**2 / (2 * (half/2)**2)))
    kernel = sign * gauss

    n = ssm.shape[0]
    novelty = np.zeros(n)
    for i in range(half, n - half):
        block = ssm[i-half:i+half, i-half:i+half]
        novelty[i] = np.sum(block * kernel)
    novelty -= novelty.min()
    novelty /= max(novelty.max(), 1e-9)
    return novelty


def pick_peaks(novelty, sr_frames, min_sep_s=5.0):
    min_sep = int(min_sep_s * sr_frames)
    idx = np.where((novelty[1:-1] > novelty[:-2]) & (novelty[1:-1] > novelty[2:]))[0] + 1
    idx = idx[novelty[idx] > 0.15]
    idx = sorted(idx, key=lambda i: -novelty[i])
    kept = []
    for i in idx:
        if all(abs(i - k) > min_sep for k in kept):
            kept.append(i)
    return sorted(kept)


def analyze(path, song):
    y, sr = sf.read(str(ROOT / path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP)
    feat = np.vstack([mfcc / (np.linalg.norm(mfcc, axis=0, keepdims=True) + 1e-9),
                       chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)])

    ssm = feat.T @ feat
    frame_rate = sr / HOP
    novelty = foote_novelty(ssm, kernel_size=int(8 * frame_rate))  # ~8с окно
    peaks = pick_peaks(novelty, frame_rate, min_sep_s=5.0)
    boundary_times = [0.0] + [p / frame_rate for p in peaks] + [len(y) / sr]
    return sorted(set(round(t, 1) for t in boundary_times))


def compare_to_manual(auto_times, song):
    sec = pd.read_csv(ROOT / "_analysis" / "sections.csv")
    sec = sec[sec.song == song]
    manual_times = sorted(set(sec.start_s.tolist() + sec.end_s.tolist()))

    print(f"\nАвто-границы ({len(auto_times)}): {auto_times}")
    print(f"Ручные границы ({len(manual_times)}): {manual_times}")

    matched_manual, unmatched_auto = set(), []
    for a in auto_times:
        nearest = min(manual_times, key=lambda m: abs(m - a))
        if abs(nearest - a) <= 3.0:
            matched_manual.add(nearest)
        else:
            unmatched_auto.append(a)

    print(f"\nСовпало с ручными (в пределах 3с): {sorted(matched_manual)} из {manual_times}")
    print(f"Авто нашёл, чего нет в ручной (в пределах 3с): {unmatched_auto}")
    print(f"Ручные, которые авто не подтвердил: {sorted(set(manual_times) - matched_manual)}")


if __name__ == "__main__":
    times = analyze("основной трек/-/Итог сведения.wav", "основной трек")
    compare_to_manual(times, "основной трек")
