"""Блок 2 (Этап 1, устранение гула): полный цикл — find_persistent_narrowband
детектирует (грубо, с точностью STFT-бина), refine_narrowband_freq уточняет
(параболическая интерполяция), remove_narrowband_hum вырезает notch-фильтром.
Без уточнения частоты notch промахивается мимо цели почти полностью —
проверено эмпирически (см. docstring remove_narrowband_hum), это ядро теста."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from analysis.metrics.noise import (
    find_persistent_narrowband, refine_narrowband_freq, remove_narrowband_hum,
)

SR = 44100


def _band_energy(sig, f0, sr=SR, bw=3):
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1 / sr)
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    return np.sum(spec[mask] ** 2)


def _make_signal(hum_freq, dur_s=3.0, seed=0):
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    music = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 330 * t)
             + 0.02 * rng.standard_normal(n))
    hum = 0.05 * np.sin(2 * np.pi * hum_freq * t)
    return music + hum


def test_refine_freq_corrects_stft_bin_quantization():
    """N_FFT=4096 @ 44100Гц -> разрешение бина ~10.8Гц — детектор неизбежно
    даёт частоту с такой погрешностью. Уточнение должно сократить ошибку
    минимум на порядок."""
    x = _make_signal(hum_freq=50.0)
    coarse_guess = 55.0  # имитация грубой STFT-бин-оценки детектора, ошибка 5Гц
    refined = refine_narrowband_freq(x, SR, coarse_guess)
    assert abs(refined - 50.0) < 0.1, f"уточнённая частота {refined:.3f}Гц, ожидали ~50.0"


def test_notch_without_refinement_barely_touches_hum():
    """Контрольный отрицательный результат: notch на грубой (неуточнённой)
    частоте с реалистичной ошибкой детектора снимает гул на единицы дБ, не
    на десятки — обосновывает, зачем вообще нужен refine_narrowband_freq."""
    from scipy.signal import iirnotch, filtfilt
    x = _make_signal(hum_freq=50.0)
    e_before = _band_energy(x, 50.0)
    b, a = iirnotch(55.0, 10, SR)  # промах на 5Гц, без уточнения
    y = filtfilt(b, a, x)
    e_after = _band_energy(y, 50.0)
    reduction_db = 10 * np.log10(e_after / e_before)
    assert reduction_db > -15, f"ожидали слабое подавление без уточнения (>-15дБ), получили {reduction_db:.1f}дБ"


def test_full_pipeline_detect_refine_remove():
    """Детекция -> уточнение -> вырезание на реалистичном сигнале (гул +
    музыка + широкополосный шум): гул должен уйти минимум на 20дБ, музыка на
    удалённых частотах — практически не тронута."""
    x = _make_signal(hum_freq=50.0)
    e50_before = _band_energy(x, 50.0)
    e220_before = _band_energy(x, 220.0)

    candidates = find_persistent_narrowband(x, SR, f_lo=30, f_hi=200)
    assert len(candidates) > 0, "детектор должен найти гул на 50Гц в этом сигнале"
    top = candidates[0]
    assert abs(top["freq_hz"] - 50.0) < 15, f"грубая оценка {top['freq_hz']}Гц слишком далека от истинных 50Гц"

    cleaned, removed = remove_narrowband_hum(x, SR, candidates)
    assert len(removed) > 0, "должна была снять хотя бы одну наводку"
    assert abs(removed[0]["freq_hz_refined"] - 50.0) < 0.5

    e50_after = _band_energy(cleaned, 50.0)
    e220_after = _band_energy(cleaned, 220.0)
    reduction_db = 10 * np.log10(e50_after / e50_before)
    music_change_db = 10 * np.log10(e220_after / e220_before)
    assert reduction_db < -20, f"гул должен уйти минимум на 20дБ, получили {reduction_db:.1f}дБ"
    assert abs(music_change_db) < 0.1, f"музыка на 220Гц не должна пострадать, изменилась на {music_change_db:.3f}дБ"


def test_low_stability_candidate_not_removed():
    """min_stability по умолчанию отсекает слабые/нестабильные кандидаты —
    удаление необратимо, порог для него строже, чем для одной детекции."""
    x = _make_signal(hum_freq=50.0)
    fake_weak_candidate = [dict(freq_hz=50.0, mean_level_db=-40.0,
                                 prominence_db=3.5, std_db=2.0, stability_score=1.0)]
    _, removed = remove_narrowband_hum(x, SR, fake_weak_candidate, min_stability=3.0)
    assert removed == [], "кандидат со stability_score ниже порога не должен быть вырезан"


def test_engine_diagnostics_carries_refined_frequency(tmp_path):
    """Интеграционный тест на реальный стык: track_avg_metrics раньше клал
    в diagnostics["hum_candidates"] сырую, неуточнённую частоту детектора
    (точность STFT-бина ~11Гц) — refine_narrowband_freq был написан и
    протестирован изолированно, но не вызывался из engine.py. Проверяем
    именно то, что попадает в diagnostics, не саму функцию в отрыве."""
    import soundfile as sf
    from analysis import engine

    x = _make_signal(hum_freq=50.0, dur_s=5.0)
    path = tmp_path / "hum_test.wav"
    sf.write(str(path), np.column_stack([x, x]), SR, subtype="FLOAT")

    _, _, diagnostics = engine.track_avg_metrics(path, "mix")
    assert "hum_candidates" in diagnostics, "гул на 50Гц должен быть обнаружен"
    # синтетическая "музыка" в _make_signal — постоянные синусоиды 220/330Гц,
    # такие же стационарные, как и сам гул (в отличие от настоящей музыки с
    # нотами/динамикой) — на полном диапазоне детектора (f_hi=1000 по
    # умолчанию, engine.py его не сужает) они тоже попадают в кандидаты и
    # могут обогнать гул по stability_score. Ищем нужный по частоте, не по
    # индексу — тест проверяет наличие и точность freq_hz_refined, не ранжирование.
    candidates_near_50 = [c for c in diagnostics["hum_candidates"] if abs(c["freq_hz"] - 50.0) < 10]
    assert candidates_near_50, (
        f"среди кандидатов не нашлось ничего рядом с 50Гц: "
        f"{[c['freq_hz'] for c in diagnostics['hum_candidates']]}")
    top = candidates_near_50[0]
    assert "freq_hz_refined" in top, "diagnostics обязаны нести уточнённую частоту, не только сырую"
    assert abs(top["freq_hz_refined"] - 50.0) < 1.0, (
        f"уточнённая частота в diagnostics = {top['freq_hz_refined']:.2f}Гц, ожидали ~50.0")


def test_engine_diagnostics_hum_windowed_carries_refined_frequency(tmp_path):
    """Регрессия (найдена код-ревью): diagnostics["hum_windowed"] строился
    тем же фильтром stability_score>=3.0, что и diagnostics["hum_candidates"],
    но БЕЗ вызова refine_narrowband_freq — оконный путь (Блок 2, среднее
    окно), созданный именно для локализации гула, присутствующего лишь в
    части трека, нёс только грубую ~11Гц частоту STFT-бина. Сигнал длиной
    20с — заведомо больше DIAG_WINDOW_S=15с, чтобы реально получить хотя бы
    одно полное окно."""
    import soundfile as sf
    from analysis import engine

    x = _make_signal(hum_freq=50.0, dur_s=20.0)
    path = tmp_path / "hum_windowed_test.wav"
    sf.write(str(path), np.column_stack([x, x]), SR, subtype="FLOAT")

    _, _, diagnostics = engine.track_avg_metrics(path, "mix")
    assert "hum_windowed" in diagnostics, "гул на 50Гц должен быть виден и в оконной диагностике"
    found = False
    for window in diagnostics["hum_windowed"]:
        for c in window["candidates"]:
            if abs(c["freq_hz"] - 50.0) < 10:
                assert "freq_hz_refined" in c, "оконные кандидаты обязаны нести уточнённую частоту, не только сырую"
                assert abs(c["freq_hz_refined"] - 50.0) < 1.0, (
                    f"уточнённая частота в hum_windowed = {c['freq_hz_refined']:.2f}Гц, ожидали ~50.0")
                found = True
    assert found, "ни одно окно не нашло кандидата рядом с 50Гц"


def test_refine_freq_uses_whole_track_not_just_leading_prefix():
    """Регрессия (найдена код-ревью): np.fft.rfft(mono, n=n_fft) при
    len(mono) > n_fft ОБРЕЗАЕТ вход до первых n_fft сэмплов (~5.94с при
    дефолтном n_fft=2**18), а не дополняет нулями — для гула, реально
    присутствующего только в поздней части длинного трека, уточнение
    находило бы совсем не ту частоту, если бы искало по обрезанному
    префиксу. "Отвлекающий" тон 45Гц целиком лежит ДО границы обрезки
    (~5.94с), настоящая наводка 50Гц начинается сразу ПОСЛЕ и длится
    дольше отвлекающего — так что при старом баге (анализ только первых
    ~5.94с) единственное, что вообще видно детектору, это отвлекающий
    тон; при верном поведении (анализ всего трека) 50Гц перевешивает и по
    длительности, и по суммарной энергии."""
    hum_start_s, total_s = 6.0, 20.0  # 6.0с > n_fft/sr (~5.9457с) — граница обрезки строго внутри отвлекающего тона
    n = int(total_s * SR)
    t = np.arange(n) / SR
    x = np.zeros(n)
    before = t < hum_start_s
    x[before] += 0.05 * np.sin(2 * np.pi * 45.0 * t[before])
    after = ~before
    x[after] += 0.05 * np.sin(2 * np.pi * 50.0 * t[after])

    refined = refine_narrowband_freq(x, SR, f_approx=50.0, search_hz=15.0)
    assert abs(refined - 50.0) < 1.0, (
        f"должен найти настоящую наводку 50Гц по всему треку, получили {refined:.2f}Гц "
        f"(45.0 означало бы, что уточнение до сих пор смотрит только на обрезанный префикс)")


if __name__ == "__main__":
    test_refine_freq_corrects_stft_bin_quantization()
    test_notch_without_refinement_barely_touches_hum()
    test_full_pipeline_detect_refine_remove()
    test_low_stability_candidate_not_removed()
    test_refine_freq_uses_whole_track_not_just_leading_prefix()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_engine_diagnostics_carries_refined_frequency(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_engine_diagnostics_hum_windowed_carries_refined_frequency(Path(d))
    print("Все тесты устранения гула (Блок 2) прошли.")
