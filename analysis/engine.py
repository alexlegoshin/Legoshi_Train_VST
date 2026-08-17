"""Ядро оркестратора: считает метрики (среднее по треку + скользящее окно)
для одного источника (mix/vocals/bass/drums/other). Переиспользует только
модули из analysis/metrics/ — никакой зависимости от legacy_scripts.

Формат вывода track_avg и window согласован с ключами, которые ждёт
analysis.verdict.evaluate(): {(metric, source): значение_или_Series}."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import butter, filtfilt

from analysis.metrics import loudness_dynamics, spectral, stereo_space, psychoacoustic, reverb
from analysis.metrics import pitch_vocal, vocal_texture

WIN_S = 4.0
HOP_S = 1.0
ENERGY_GATE_DBFS = -45.0
LOWPASS_HZ = 10000.0  # НЧ-фильтр перед спектральными метриками на разделённых источниках —
                       # у Demucs (и у грубо сведённых сумм стемов) шум/артефакты сильнее всего наверху
TARGET_LUFS = -18.0
MAX_PLAUSIBLE_VIBRATO_CENTS = 100.0
VOICED_FRAC_GATE = 0.5  # ТЗ-05 А8: доля voiced-кадров pYIN в окне, ниже
                          # которой окно не считается вокальным — F0-метрики NaN


def load_mono(path):
    data, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return data.mean(axis=1), sr, data


PIPELINE_SR = 44100  # ТЗ-05 А6: analyze_file() во всех analysis/metrics/*.py
                      # жёстко ждёт этот sr (assert sr == sr_expected) —
                      # не мягкая деградация, падение с исключением. Любой
                      # вход, который может иметь другой sr (трек-аут
                      # пользователя, произвольный залитый файл), приводится
                      # к этому значению ДО того, как попасть в analyze_file.


def ensure_sr(path: Path, target_sr: int, work_dir: Path) -> Path:
    """Ресемплинг к target_sr, если исходный файл на другой частоте —
    возвращает исходный путь без изменений, если частота уже совпадает
    (не плодит лишние копии на диске)."""
    import soundfile as sf_module
    info = sf_module.info(str(path))
    if info.samplerate == target_sr:
        return path
    data, sr = sf_module.read(str(path), dtype="float64", always_2d=True)
    import librosa as librosa_module
    resampled = np.stack([librosa_module.resample(data[:, ch], orig_sr=sr, target_sr=target_sr)
                           for ch in range(data.shape[1])], axis=1)
    out_path = work_dir / f"_resampled_{path.stem}.wav"
    sf_module.write(str(out_path), resampled, target_sr, subtype="FLOAT")
    return out_path


def _lowpass(x, sr, cutoff=LOWPASS_HZ, order=4):
    b, a = butter(order, cutoff / (sr / 2), btype="low")
    return filtfilt(b, a, x)


def _invalidate_bands_above_cutoff(d, cutoff_hz, key_fmt="band_frac_{name}_median"):
    """ТЗ-05 А3: band_frac_X, чья полоса частично или полностью лежит выше
    среза кодека, измеряет тишину, не звук — заменяем на NaN, не оставляем
    молчаливо неверное число. Полоса считается задетой, если её ВЕРХНИЙ
    край выше cutoff (частичное перекрытие тоже контаминирует энергетическую
    долю — не только полное). Возвращает список инвалидированных ключей —
    для diagnostics, не пихаем строку в числовой словарь измерений."""
    if cutoff_hz is None:
        return []
    invalidated = []
    for name, lo, hi in spectral.NAMED_BANDS:
        if hi > cutoff_hz:
            key = key_fmt.format(name=name)
            if key in d:
                d[key] = np.nan
                invalidated.append(key)
    return invalidated


def detect_dual_mono(data, tol=1e-9):
    """ТЗ-05 А5: стерео-контейнер с идентичными каналами — не настоящее
    стерео. Побайтовое (числовое) сравнение, не эвристика по корреляции."""
    if data.shape[1] != 2:
        return False
    return bool(np.max(np.abs(data[:, 0] - data[:, 1])) < tol)


def detect_clipping(data, run_len=3, thresh=0.999):
    """ТЗ-05 А7: серии >= run_len сэмплов на полной шкале на любом канале."""
    over = np.any(np.abs(data) >= thresh, axis=1)
    run = 0
    for v in over:
        run = run + 1 if v else 0
        if run >= run_len:
            return True
    return False


def track_avg_metrics(path, role, is_stereo_capable=True, allow_reverb=True):
    """Среднее по треку. role: 'mix'|'vocals'|'bass'|'drums'|'other'.

    Возвращает (measurements, vocal_frames, diagnostics):
    - measurements — {(metric, role): значение}, готово для verdict.evaluate()
    - vocal_frames — {} для не-вокала, иначе {"f0": DataFrame, "notes": DataFrame}
      — переиспользуется в window_metrics, чтобы не гонять pYIN дважды
    - diagnostics — не числа, факты для отчёта (ТЗ-05 блок Д): срез кодека,
      дуал-моно, клиппинг, инвалидированные полосы

    allow_reverb=False (ТЗ-05 А9) — для источников, разделённых ML (Demucs):
    разделение размазывает хвосты, RT60/EDT/DRR/C50/C80 на таком источнике
    не измерение, а шум — реверб-блок пропускается, не считается вовсе."""
    out = {}
    diagnostics = {}
    d1, _ = loudness_dynamics.analyze_file(path)
    out.update({k: d1[k] for k in ("integrated_lufs", "lra", "plr", "crest_factor_db",
                                     "dr_tt", "true_peak_dbfs", "sample_peak_dbfs") if k in d1})

    d2, _ = spectral.analyze_file(path)
    out.update({k: d2[k] for k in d2 if k not in ("path",)})

    mono, sr, data = load_mono(path)

    cutoff_hz = spectral.detect_codec_cutoff(mono, sr)
    diagnostics["codec_cutoff_hz"] = cutoff_hz
    invalidated = _invalidate_bands_above_cutoff(out, cutoff_hz)
    if invalidated:
        diagnostics["codec_cutoff_invalidated_keys"] = invalidated

    diagnostics["dual_mono"] = detect_dual_mono(data)
    diagnostics["clipped"] = detect_clipping(data)

    # ТЗ-05 А1: psychoacoustic.quick_metrics считает Zwicker loudness/DIN
    # sharpness/tonality — эти величины частично зависят от абсолютного
    # уровня входного сигнала (не чистые тембровые отношения, как
    # warmth_ratio/harshness в том же вызове). Без нормализации метрика
    # сравнивает мастеринг, а не тембр — гоняем на сигнале, приведённом к
    # -18 LUFS (та же цель, что уже была у window_metrics), не на сыром.
    gain_db = TARGET_LUFS - float(out.get("integrated_lufs", d1.get("integrated_lufs", 0.0)))
    mono_for_psycho = mono * (10 ** (gain_db / 20))
    out.update(psychoacoustic.quick_metrics(mono_for_psycho, sr))

    if is_stereo_capable and data.shape[1] == 2 and not diagnostics["dual_mono"]:
        try:
            d4, _ = stereo_space.analyze_file(path)
            out.update({k: d4[k] for k in d4 if k not in ("path",)})
        except Exception:
            pass
    elif diagnostics.get("dual_mono"):
        # ТЗ-05 А5: не шум около нуля — явный, честный ноль
        out["overall_correlation"] = 1.0
        out["overall_ms_ratio"] = 0.0

    if allow_reverb:
        try:
            d5, _ = reverb.analyze_file(path)
            out.update({k: d5[k] for k in d5 if k not in ("path",)})
        except Exception:
            pass
    else:
        diagnostics["reverb_skipped_reason"] = "ML-разделённый источник — хвосты размазаны разделением (ТЗ-05 А9)"

    vocal_frames = {}
    if role == "vocals":
        try:
            d6, frames6 = pitch_vocal.analyze_file(path)
            out.update({k: d6[k] for k in d6 if k not in ("path",)})
            f0_df = frames6.get("f0")
            if f0_df is not None and len(f0_df):
                vocal_frames = {"f0": f0_df, "notes": frames6.get("notes")}
                d7, _ = vocal_texture.analyze_file(path, f0_df)
                out.update({k: d7[k] for k in d7})
        except Exception as e:
            diagnostics["vocal_analysis_error"] = str(e)

    measurements = {(metric, role): val for metric, val in out.items()
                     if isinstance(val, (int, float, np.floating))}
    return measurements, vocal_frames, diagnostics


def window_metrics(mono, sr, role, mix_gain_db=0.0, f0_df=None, notes_df=None,
                    do_real_psychoacoustics=False):
    """Скользящее окно 4с/1с, энергетический гейт. mix_gain_db — гейн,
    приводящий ЦЕЛЫЙ МИКС к -18 LUFS (применяется и к разделённым
    источникам того же трека — честное сравнение порогов энергии между
    треками разной громкости мастеринга)."""
    x = mono * (10 ** (mix_gain_db / 20))
    x_lp = _lowpass(x, sr) if role != "mix" else x

    psycho_frames = None
    if do_real_psychoacoustics:
        _, psycho_frames = psychoacoustic.full_timevarying(x_lp, sr)

    win_n, hop_n = int(WIN_S * sr), int(HOP_S * sr)
    starts = np.arange(0, max(len(x_lp) - win_n, 0), hop_n)
    rows = []
    for s in starts:
        t0, t1 = s / sr, (s + win_n) / sr
        seg = x_lp[s:s + win_n]
        rms_dbfs = 20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-12)
        if rms_dbfs < ENERGY_GATE_DBFS:
            continue

        f, t, mag = spectral.compute_stft(seg, sr)
        centers, levels = spectral.ltas(mag, f, bands_per_octave=3)
        keep = centers <= LOWPASS_HZ
        slope = spectral.spectral_slope(centers[keep], levels[keep])
        _, _, sk, _ = spectral.spectral_moments(mag, f)
        bands = spectral.named_band_energy_fraction(mag, f)
        warmth = psychoacoustic.warmth_ratio(seg, sr)
        harsh = psychoacoustic.harshness(seg, sr)

        row = dict(
            t_start=t0, t_end=t1, rms_dbfs=rms_dbfs,
            band_frac_lowmid=float(np.median(bands["lowmid"])),
            band_frac_low=float(np.median(bands["low"])),
            band_frac_mud=float(np.median(bands["mud"])),
            band_frac_mid=float(np.median(bands["mid"])),
            band_frac_presence=float(np.median(bands["presence"])),
            band_frac_air=float(np.median(bands["air"])) if role == "mix" else np.nan,
            skewness=float(np.median(sk)), spectral_slope=slope,
            warmth_ratio=warmth, harshness=harsh,
        )

        if role == "vocals" and f0_df is not None:
            # ТЗ-05 А8: F0-метрики (вибрато/интонация) — ТОЛЬКО на окнах с
            # подтверждённым вокалом. Гейт — voiced-доля pYIN на этом окне,
            # НЕ pYIN "prob" (эмпирически в этом проекте не отличает вокал
            # от шума/хвоста реверба — см. documentation/methodology.md).
            # Ниже порога — все F0-метрики окна остаются NaN, не число.
            fmask = (f0_df.t_s >= t0) & (f0_df.t_s < t1)
            fsub = f0_df[fmask]
            voiced_frac = float(fsub.voiced.mean()) if len(fsub) else 0.0
            is_vocal_window = voiced_frac >= VOICED_FRAC_GATE
            row["voiced_fraction"] = voiced_frac if is_vocal_window else np.nan
            row["vibrato_depth_cents"] = np.nan
            row["intonation_dev_cents"] = np.nan
            if is_vocal_window and notes_df is not None and len(notes_df):
                note_mid = (notes_df.t_start + notes_df.t_end) / 2
                nmask = (note_mid >= t0) & (note_mid < t1)
                nsub_vib = notes_df.loc[nmask, "vibrato_depth_cents"].dropna()
                nsub_vib = nsub_vib[nsub_vib <= MAX_PLAUSIBLE_VIBRATO_CENTS]
                if len(nsub_vib) >= 2:
                    row["vibrato_depth_cents"] = float(nsub_vib.median())
                if "intonation_deviation_cents" in notes_df.columns:
                    nsub_int = notes_df.loc[nmask, "intonation_deviation_cents"].dropna()
                    if len(nsub_int) >= 2:
                        row["intonation_dev_cents"] = float(nsub_int.abs().median())

        if psycho_frames is not None:
            for key, col, out_name in [("sharpness", "acum", "real_sharpness"),
                                         ("roughness", "asper", "real_roughness"),
                                         ("loudness", "sone", "real_loudness")]:
                fr = psycho_frames[key]
                sub = fr[(fr.t_s >= t0) & (fr.t_s < t1)][col]
                row[out_name] = float(sub.median()) if len(sub) else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def get_mix_gain_db(mix_path):
    """Гейн до -18 LUFS integrated, посчитанный на ЦЕЛОМ МИКСЕ — применяется
    одинаково к миксу и ко всем разделённым источникам того же трека."""
    d1, _ = loudness_dynamics.analyze_file(mix_path)
    return TARGET_LUFS - float(d1["integrated_lufs"])


def formant_series(mono, sr, f0_df):
    """Форманты (LPC) по voiced-кадрам — для агрегации по окнам отдельно
    от track_avg-версии в vocal_texture.analyze_file (та даёт только summary)."""
    starts = vocal_texture._frame_starts(len(mono), vocal_texture.FRAME_LEN, vocal_texture.HOP)
    t_frames = (starts + vocal_texture.FRAME_LEN / 2) / sr
    voiced_frames = vocal_texture._voiced_lookup(t_frames, f0_df)
    rows = []
    for i, s in enumerate(starts):
        if not voiced_frames[i]:
            continue
        f1, f2, f3 = vocal_texture.lpc_formants(mono[s:s + vocal_texture.FRAME_LEN], sr)
        rows.append(dict(t_s=float(t_frames[i]), f1_hz=f1, f2_hz=f2, f3_hz=f3))
    return pd.DataFrame(rows)
