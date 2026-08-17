"""§4.9 довесок: chroma/HPCP, тональность (Krumhansl-Schmuckler), аккорды
по секциям, расстройка вокал-бас — задача #28.

Профили Krumhansl-Kessler (Krumhansl & Kessler, 1982) — опубликованные
психоакустические константы, как формулы Sethares/Plomp-Levelt в
harmony_dissonance.py, не переоткрываем, только реализуем.

Аккорды — только мажор/минор трезвучия (без септаккордов и обращений):
для задачи "определить тональность по секциям", а не полный
автоматический транскрайбер, этого достаточно."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

N_FFT, HOP = 4096, 512
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CHORD_TEMPLATES = {}
for _root in range(12):
    _maj = np.zeros(12); _maj[[_root, (_root + 4) % 12, (_root + 7) % 12]] = 1.0
    _minr = np.zeros(12); _minr[[_root, (_root + 3) % 12, (_root + 7) % 12]] = 1.0
    CHORD_TEMPLATES[NOTE_NAMES[_root]] = _maj
    CHORD_TEMPLATES[NOTE_NAMES[_root] + "m"] = _minr


def chroma_stft_frames(mono, sr, n_fft=N_FFT, hop=HOP):
    import librosa
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=n_fft, hop_length=hop)
    t = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    return t, chroma


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def estimate_key(chroma_mean):
    """Krumhansl-Schmuckler: корреляция усреднённого chroma с 24 профилями
    (12 мажор + 12 минор, все повороты). Возвращает (нота, лад, корреляция)."""
    best = None
    for mode, profile in [("major", KK_MAJOR), ("minor", KK_MINOR)]:
        for root in range(12):
            rotated = np.roll(profile, root)
            corr = _corr(chroma_mean, rotated)
            if best is None or corr > best[2]:
                best = (NOTE_NAMES[root], mode, corr)
    return best


def estimate_chord(chroma_mean):
    best_label, best_corr = None, -np.inf
    for label, template in CHORD_TEMPLATES.items():
        corr = _corr(chroma_mean, template)
        if corr > best_corr:
            best_label, best_corr = label, corr
    return best_label, best_corr


def key_and_chords(path, sections=None, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)
    t, chroma = chroma_stft_frames(mono, sr)

    key_root, key_mode, key_corr = estimate_key(chroma.mean(axis=1))

    chord_rows = []
    if sections is not None and len(sections):
        for _, s in sections.iterrows():
            mask = (t >= s.start_s) & (t < s.end_s)
            if mask.sum() < 3:
                continue
            label, corr = estimate_chord(chroma[:, mask].mean(axis=1))
            chord_rows.append(dict(section=s.section, start_s=float(s.start_s), end_s=float(s.end_s),
                                    chord=label, correlation=corr))
    chords_df = pd.DataFrame(chord_rows)

    summary = dict(key_root=key_root, key_mode=key_mode, key_correlation=key_corr)
    chroma_df = pd.DataFrame(chroma.T, columns=NOTE_NAMES)
    chroma_df.insert(0, "t_s", t)
    frames = dict(chroma=chroma_df, chords=chords_df)
    return summary, frames


def vocal_bass_detuning(vocal_mono, bass_mono, sr, vocal_fmin=65, vocal_fmax=1000,
                         bass_fmin=30, bass_fmax=400, hop=HOP,
                         vocal_frame_length=2048, bass_frame_length=4096, capture_cents=50):
    """§4.9: расстройка между источниками. Вокал и бас должны быть уже
    выровнены по времени тем же способом, что и в §5/§4.8 (общие сэмплы —
    общая временная ось), иначе "общие ноты" сравнивались бы вслепую.

    Сравниваем не абсолютную высоту (вокал и бас обычно на разных октавах),
    а отклонение от БЛИЖАЙШЕЙ октавной кратности — именно это TZ называет
    расстройкой: "стабильное расхождение больше 10 центов — источник
    мутности, который эквалайзером не лечится".

    РАЗНЫЕ frame_length для вокала (2048, как в §4.6) и баса (4096 — нужен
    под низкий fmin=30Гц). ПОЙМАНО НА РЕАЛЬНОМ ФАЙЛЕ (не на синтетике —
    короткие тестовые клипы этого не показали): на полном 200-секундном
    файле сочетание frame_length=4096 с мелким resolution давало Viterbi
    pYIN сойти с ума и залипнуть на fmin (медиана 66Гц вместо настоящих
    ~250Гц, sanity-проверено сравнением с уже провалидированным кэшем
    §4.6 на том же файле с frame_length=2048). По отдельности ни большой
    frame_length, ни мелкий resolution не ломали — только их сочетание на
    длинном файле. Отсюда: resolution дефолтный (0.1), frame_length разный
    под вокал/бас, а не один "универсальный" под оба.

    ВТОРОЙ БАГ, найденный уже на реальных данных (не на синтетике — там
    вокал и бас всегда пелись "в унисон" по построению теста): нельзя
    считать "расстройкой" любую пару одновременно озвученных кадров.
    Вокал и бас в норме поют РАЗНЫЕ ноты одного аккорда (терция, квинта) —
    это гармония, а не рассогласование. Медиана по ВСЕМ одновременно
    звучащим кадрам на реальном файле дала 321 цент (это треть, обычный
    гармонический интервал, не брак). Поэтому сначала фильтруем кадры-
    кандидаты на унисон/октаву (|отклонение| <= capture_cents=50 —
    четверть тона, разумный захват для "видимо одна и та же нота"), и
    только по НИМ считаем медиану расстройки. TZ и просит именно так:
    "на общих нотах", не на любых одновременно звучащих."""
    import librosa
    f0_v, voiced_v, _ = librosa.pyin(vocal_mono, fmin=vocal_fmin, fmax=vocal_fmax, sr=sr,
                                      frame_length=vocal_frame_length, hop_length=hop, fill_na=np.nan)
    f0_b, voiced_b, _ = librosa.pyin(bass_mono, fmin=bass_fmin, fmax=bass_fmax, sr=sr,
                                      frame_length=bass_frame_length, hop_length=hop, fill_na=np.nan)
    n = min(len(f0_v), len(f0_b))
    f0_v, voiced_v = f0_v[:n], voiced_v[:n]
    f0_b, voiced_b = f0_b[:n], voiced_b[:n]
    both = voiced_v & voiced_b & np.isfinite(f0_v) & np.isfinite(f0_b)
    n_common_voiced = int(both.sum())

    if n_common_voiced < 5:
        return dict(n_common_voiced_frames=n_common_voiced, n_candidate_unison_frames=0,
                     detuning_cents_median=np.nan, frac_gt_10cents=np.nan), pd.DataFrame()

    cents = 1200 * np.log2(f0_v[both] / f0_b[both])
    dev = ((cents + 600) % 1200) - 600  # отклонение от ближайшей октавной кратности, [-600,600]
    dev_abs = np.abs(dev)
    t = librosa.frames_to_time(np.where(both)[0], sr=sr, hop_length=hop)

    is_candidate = dev_abs <= capture_cents
    n_candidate = int(is_candidate.sum())
    if n_candidate < 5:
        summary = dict(n_common_voiced_frames=n_common_voiced, n_candidate_unison_frames=n_candidate,
                        detuning_cents_median=np.nan, frac_gt_10cents=np.nan)
    else:
        cand_dev = dev_abs[is_candidate]
        summary = dict(
            n_common_voiced_frames=n_common_voiced,
            n_candidate_unison_frames=n_candidate,
            detuning_cents_median=float(np.median(cand_dev)),
            frac_gt_10cents=float(np.mean(cand_dev > 10)),
        )
    df = pd.DataFrame({"t_s": t, "f0_vocal_hz": f0_v[both], "f0_bass_hz": f0_b[both],
                        "detuning_cents": dev, "is_candidate_unison": is_candidate})
    return summary, df
