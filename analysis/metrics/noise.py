"""§4.10 ТЗ-01: шум и «грязь». Наводки ищем не по громкости пика (это уже
есть в §4.2 find_resonances), а по УСТОЙЧИВОСТИ ВО ВРЕМЕНИ — музыкальный
резонанс то звучит, то нет, наводка (сеть/Останкино) присутствует всегда,
и это единственный надёжный признак, который отличает одно от другого."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.signal import stft

N_FFT, HOP = 4096, 512


def noise_floor_and_snr(mono, sr, quiet_percentile=10):
    hop = int(0.1 * sr)
    n = len(mono) // hop
    frame_rms = np.array([np.sqrt(np.mean(mono[i*hop:(i+1)*hop] ** 2) + 1e-20) for i in range(n)])
    thresh = np.percentile(frame_rms, quiet_percentile)
    quiet_mask_frames = frame_rms <= thresh

    quiet_samples = np.concatenate([mono[i*hop:(i+1)*hop] for i in range(n) if quiet_mask_frames[i]])
    if len(quiet_samples) < sr // 10:
        return dict(noise_floor_dbfs=np.nan, snr_db=np.nan, noise_spectral_slope=np.nan), None

    noise_floor_rms = np.sqrt(np.mean(quiet_samples ** 2) + 1e-20)
    noise_floor_dbfs = 20 * np.log10(noise_floor_rms + 1e-20)
    signal_rms = np.sqrt(np.mean(mono ** 2))
    snr_db = 20 * np.log10((signal_rms + 1e-20) / (noise_floor_rms + 1e-20))

    X = np.abs(np.fft.rfft(quiet_samples))
    freqs = np.fft.rfftfreq(len(quiet_samples), 1 / sr)
    mask = (freqs >= 100) & (freqs <= 8000) & (X > 0)
    slope = np.nan
    if mask.sum() > 5:
        slope, _ = np.polyfit(np.log2(freqs[mask]), 20 * np.log10(X[mask] + 1e-20), 1)

    return dict(noise_floor_dbfs=float(noise_floor_dbfs), snr_db=float(snr_db),
                noise_spectral_slope=float(slope) if np.isfinite(slope) else np.nan), (freqs, X)


def find_persistent_narrowband(mono, sr, f_lo=30, f_hi=1000, top_n=10, quiet_percentile=20):
    """Наводки: узкополосные пики с НИЗКОЙ дисперсией уровня во времени —
    в отличие от музыкальных резонансов, которые то звучат, то нет.

    ИСПРАВЛЕНО (задача #29): раньше mean/std считались по ВСЕМ кадрам
    файла целиком. На синтетике/сыром дубле работало, но на занятых миксах
    находило НОЛЬ наводок там, где §4.2 (find_resonances, порог по
    амплитуде, не по стабильности) уже нашёл настоящий устойчивый
    108Гц-резонанс во всех версиях подряд — реальный сигнал, не мусор
    детектора. Диагноз: громкая музыка в тех же частотных бинах, что и
    наводка, раздувает std_level наводки настолько, что она перестаёт
    выглядеть "стабильной" на фоне остального шума. Наводка (сеть/Останкино)
    присутствует ВСЕГДА, значит она видна и в паузах — искать нужно там,
    где музыка не перекрывает: только по тихим STFT-кадрам (нижние
    quiet_percentile% по широкополосному уровню кадра), не по всему файлу."""
    f, t, Z = stft(mono, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    mag = np.abs(Z)
    mag_db = 20 * np.log10(mag + 1e-12)

    broadband_level = np.sqrt(np.mean(mag ** 2, axis=0))
    thresh = np.percentile(broadband_level, quiet_percentile)
    quiet_frames = broadband_level <= thresh
    if quiet_frames.sum() < 10:
        quiet_frames = np.ones(mag.shape[1], dtype=bool)  # файл целиком "громкий" — не сужать до пустоты

    band = (f >= f_lo) & (f <= f_hi)
    f_band = f[band]
    mag_band = mag_db[np.ix_(band, quiet_frames)]

    mean_level = mag_band.mean(axis=1)
    std_level = mag_band.std(axis=1)
    smoothed = np.convolve(mean_level, np.ones(9) / 9, mode="same")
    prominence = mean_level - smoothed

    candidates = []
    for i in range(2, len(f_band) - 2):
        if (prominence[i] > 3.0 and prominence[i] > prominence[i-1] and prominence[i] > prominence[i+1]
                and std_level[i] < np.median(std_level) * 0.7):
            candidates.append(dict(freq_hz=float(f_band[i]), mean_level_db=float(mean_level[i]),
                                    prominence_db=float(prominence[i]), std_db=float(std_level[i]),
                                    stability_score=float(prominence[i] / max(std_level[i], 0.1))))
    candidates.sort(key=lambda c: -c["stability_score"])
    return candidates[:top_n]


def find_persistent_narrowband_windowed(mono, sr, win_s=15.0, f_lo=30, f_hi=1000,
                                          top_n=5, quiet_percentile=20):
    """Блок 2 (среднее окно): та же логика find_persistent_narrowband, но по
    неперекрывающимся окнам ~win_s секунд, а не по всему треку разом.
    Настоящая сетевая наводка присутствует ВЕЗДЕ — если кандидат стабилен
    только в части окон, это, скорее, монтажная вставка/артефакт одного
    участка, не общая наводка дорожки (см. roadmap.md, Блок 2). win_s — то
    же самое, что CLICK_MAX_S у клиппинга: эвристический выбор, не
    откалиброван на реальных данных."""
    n = len(mono)
    win_n = int(win_s * sr)
    windows = []
    for start in range(0, n, win_n):
        end = min(start + win_n, n)
        if end - start < sr:  # меньше секунды — хвост, не окно
            continue
        seg = mono[start:end]
        candidates = find_persistent_narrowband(seg, sr, f_lo=f_lo, f_hi=f_hi,
                                                  top_n=top_n, quiet_percentile=quiet_percentile)
        windows.append(dict(t_start=round(start / sr, 2), t_end=round(end / sr, 2),
                             candidates=candidates))
    return windows


def refine_narrowband_freq(mono, sr, f_approx, search_hz=15.0, n_fft=1 << 18):
    """Блок 2 (Этап 1, устранение гула): find_persistent_narrowband даёт
    частоту с точностью STFT-бина детектора (~11Гц при N_FFT=4096) — для
    notch-фильтра этого мало: узкий notch на 5Гц мимо цели снимает гул на
    единицы дБ вместо десятков (проверено эмпирически). Уточняем: FFT с
    нулевым дополнением (интерполированное разрешение) вокруг приближённой
    частоты + параболическая интерполяция по трём соседним бинам вокруг
    пика — стандартный дешёвый приём, даёт точность в доли Гц."""
    # БАГ (найден код-ревью, исправлен): np.fft.rfft(a, n=n) при len(a)>n
    # ОБРЕЗАЕТ вход до первых n сэмплов, а не дополняет нулями — для
    # реального трека длиннее ~n_fft/sr (~5.94с при n_fft по умолчанию)
    # уточнение считалось только по первым секундам файла, не по всему
    # треку, где наводка реально стабильна. n_fft — нижняя граница
    # разрешения, не верхняя: если сигнал длиннее, берём len(mono).
    n_fft = max(n_fft, len(mono))
    spec = np.abs(np.fft.rfft(mono, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mask = (freqs >= f_approx - search_hz) & (freqs <= f_approx + search_hz)
    idx = np.where(mask)[0]
    if len(idx) < 3:
        return float(f_approx)
    k = idx[np.argmax(spec[idx])]
    if k == 0 or k == len(spec) - 1:
        return float(freqs[k])
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    denom = a - 2 * b + c
    delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
    return float(freqs[k] + delta * (freqs[1] - freqs[0]))


def remove_narrowband_hum(mono, sr, candidates, q=10, min_stability=3.0):
    """Notch-фильтр (zero-phase, filtfilt) на каждой узкополосной наводке,
    независимо детектированной find_persistent_narrowband — снимаем то, что
    реально обнаружено как устойчивое, не гадаем заранее про 50/60Гц и
    гармоники (см. ресёрч в documentation/roadmap.md: для цифровых дорожек
    из DAW статичный notch-банк — стандартный и достаточный подход, гул не
    "плывёт" по частоте, адаптивный вариант не нужен).

    min_stability — отдельный, более строгий порог для УДАЛЕНИЯ, чем для
    отчёта: детекция — недорогая операция, вырезание частоты — необратимо
    портит сигнал при ложном срабатывании, порог стоит держать строже."""
    from scipy.signal import iirnotch, filtfilt
    out = mono.copy()
    removed = []
    for c in candidates:
        if c["stability_score"] < min_stability:
            continue
        f_approx = c["freq_hz"]
        if f_approx <= 0 or f_approx >= sr / 2:
            continue
        f_refined = refine_narrowband_freq(mono, sr, f_approx)
        b, a = iirnotch(f_refined, q, sr)
        out = filtfilt(b, a, out)
        removed.append(dict(freq_hz_detected=f_approx, freq_hz_refined=f_refined,
                             stability_score=c["stability_score"]))
    return out, removed


def analyze_file(path, sr_expected=44100):
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    assert sr == sr_expected
    mono = data.mean(axis=1)

    noise_summary, _ = noise_floor_and_snr(mono, sr)
    persistent = find_persistent_narrowband(mono, sr)

    summary = dict(**noise_summary, n_persistent_narrowband=len(persistent),
                    top_persistent_freq_hz=(persistent[0]["freq_hz"] if persistent else np.nan),
                    top_persistent_stability=(persistent[0]["stability_score"] if persistent else np.nan))
    return summary, pd.DataFrame(persistent)
