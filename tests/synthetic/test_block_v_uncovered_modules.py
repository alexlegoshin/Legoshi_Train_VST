"""ТЗ-05 Блок В (вторая часть): хотя бы один тест на синтетике с известным
ответом для модулей без покрытия. Приоритет по ТЗ: reverb (RT60 на
экспоненте с заданным временем), loudness_dynamics (LUFS на калибровочном
сигнале BS.1770), psychoacoustic (sharpness на узкополосном шуме — монотонно
растёт с центральной частотой, характерная кривая DIN 45692)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from analysis.metrics import loudness_dynamics, reverb
from analysis.metrics.harmony_dissonance import sethares_dissonance, vassilakis_roughness
from analysis.metrics.stereo_space import goniometer_stats

SR = 44100
MOSQITO_SR = 48000  # MoSQITo требует >=48к (см. предупреждение библиотеки)


def test_reverb_rt60_recovered_from_known_exponential_decay():
    """RT60 = время затухания на 60дБ. Синтезируем шум, огибающая которого —
    чистая экспонента с заданным RT60 (tau = 8.686*RT60/60, из определения
    dB(t) = -8.686*t/tau), и проверяем, что schroeder_edc+fit_decay
    восстанавливают именно это время, а не что-то произвольное."""
    rt60_true = 1.0
    tau = 8.686 * rt60_true / 60
    rng = np.random.default_rng(3)
    dur = 2.0
    n = int(dur * SR)
    t = np.arange(n) / SR
    decay = rng.standard_normal(n) * np.exp(-t / tau)

    edc = reverb.schroeder_edc(decay)
    rt60_est = reverb.fit_decay(edc, SR)
    assert abs(rt60_est - rt60_true) / rt60_true < 0.1, (
        f"RT60 оценённый={rt60_est:.3f}с, истинный={rt60_true}с — расхождение >10%")


def test_reverb_rt60_scales_with_decay_time():
    """Не только точное число, но и направление: вдвое более медленное
    затухание должно давать вдвое больший RT60 — ловит баги в единицах
    измерения (сэмплы vs секунды) и в знаке наклона регрессии."""
    def rt60_of(rt60_true, seed):
        tau = 8.686 * rt60_true / 60
        rng = np.random.default_rng(seed)
        n = int(3.0 * SR)
        t = np.arange(n) / SR
        decay = rng.standard_normal(n) * np.exp(-t / tau)
        return reverb.fit_decay(reverb.schroeder_edc(decay), SR)

    rt60_short = rt60_of(0.5, seed=10)
    rt60_long = rt60_of(1.5, seed=11)
    assert rt60_long > rt60_short * 2.0, (
        f"RT60=1.5с должен оцениваться существенно больше RT60=0.5с: "
        f"short={rt60_short:.3f}, long={rt60_long:.3f}")


def test_lufs_matches_bs1770_calibration_signal(tmp_path):
    """Калибровочный сигнал: полношкальный синус 1кГц В ОБОИХ каналах
    (L=R). Эмпирически подтверждённое значение pyloudnorm (референсная
    BS.1770-реализация) на таком сигнале ~0.0 LUFS — RMS синуса -3.01дБFS
    относительно пика (K-weighting на 1кГц даёт ~0дБ добавки), плюс +3.01дБ
    от суммирования мощности двух идентичных каналов по BS.1770 (важно
    держать это в тесте явно — то же самое МОНО даёт ~-3.0, не 0, лёгкая
    ловушка при проверке числа руками). analyze_file должен воспроизводить
    то же самое, не какое-то другое число (ловит баги вроде неверного
    усреднения каналов или порчи сигнала при записи/чтении)."""
    dur = 3.0
    n = int(dur * SR)
    sine = np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
    path = tmp_path / "calibration_1khz_fullscale.wav"
    sf.write(str(path), np.column_stack([sine, sine]), SR, subtype="FLOAT")

    summary, _ = loudness_dynamics.analyze_file(path)
    assert abs(summary["integrated_lufs"] - 0.0) < 0.5, (
        f"LUFS калибровочного сигнала = {summary['integrated_lufs']:.3f}, "
        f"ожидали ~0.0 (BS.1770 на полношкальном стерео-синусе 1кГц, L=R)")


def test_lufs_drops_10db_when_signal_gained_down_10db(tmp_path):
    """Простейшее свойство LUFS — линейность в дБ: сигнал, ослабленный на
    10дБ, должен давать integrated_lufs ровно на 10 меньше (в пределах
    погрешности гейтинга), не какое-то другое число."""
    dur = 3.0
    n = int(dur * SR)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n) * 0.3

    path0 = tmp_path / "control_0db.wav"
    sf.write(str(path0), np.column_stack([x, x]), SR, subtype="FLOAT")
    path_gained = tmp_path / "gained_-10db.wav"
    sf.write(str(path_gained), np.column_stack([x, x]) * (10 ** (-10 / 20)), SR, subtype="FLOAT")

    s0, _ = loudness_dynamics.analyze_file(path0)
    s_gained, _ = loudness_dynamics.analyze_file(path_gained)
    diff = s0["integrated_lufs"] - s_gained["integrated_lufs"]
    assert abs(diff - 10.0) < 0.1, f"ожидали разницу ровно 10дБ, получили {diff:.3f}"


def _narrowband_noise(center, bw, dur, sr, seed):
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    x = rng.standard_normal(n)
    sos = butter(4, [max(center - bw / 2, 20), center + bw / 2], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, x)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * 0.1  # фиксированный RMS — сравнение по тембру, не по громкости


def test_sharpness_increases_with_noise_band_center_frequency():
    """DIN 45692: sharpness — характеристика "яркости" звука, растёт с
    положением энергии по частоте. Узкополосный шум на высокой центральной
    частоте обязан давать заметно бОльший sharpness (акум), чем такой же
    по RMS шум на низкой частоте — фундаментальное свойство метрики,
    не деталь калибровки, которая могла бы отличаться между версиями
    MoSQITo."""
    from analysis.metrics.psychoacoustic import quick_metrics
    low = _narrowband_noise(500, 200, 1.0, MOSQITO_SR, seed=1)
    high = _narrowband_noise(5000, 1000, 1.0, MOSQITO_SR, seed=2)

    s_low = quick_metrics(low, MOSQITO_SR)["sharpness_acum_stationary"]
    s_high = quick_metrics(high, MOSQITO_SR)["sharpness_acum_stationary"]
    assert s_high > s_low * 2, (
        f"sharpness высокочастотного шума должен быть заметно выше низкочастотного: "
        f"500Гц={s_low:.3f}, 5000Гц={s_high:.3f}")


def test_dissonance_zero_at_unison_peaks_near_critical_bandwidth():
    """Кривая Plomp-Levelt/Sethares: два партиала в унисон (df=0) не дают
    диссонанса вовсе; разнесённые в пределах критической полосы (здесь
    ~40Гц вокруг 440Гц) — максимум диссонанса; разнесённые на октаву —
    почти ноль (консонанс). И sethares_dissonance, и vassilakis_roughness
    обязаны воспроизводить эту форму, не монотонный рост/спад."""
    amps = np.array([1.0, 1.0])
    unison = np.array([440.0, 440.0])
    close = np.array([440.0, 480.0])
    octave = np.array([440.0, 880.0])

    for fn in (sethares_dissonance, vassilakis_roughness):
        d_unison, d_close, d_octave = fn(unison, amps), fn(close, amps), fn(octave, amps)
        assert d_unison == 0.0, f"{fn.__name__}: унисон должен давать ровно 0, получили {d_unison}"
        assert d_close > d_octave * 100, (
            f"{fn.__name__}: интервал внутри критической полосы должен быть на порядки "
            f"диссонантнее октавы: close={d_close}, octave={d_octave}")


def test_goniometer_axis_ratio_zero_for_mono_one_for_wide_stereo():
    """ТЗ-05-предшествующий баг (см. docstring goniometer_stats): моно
    (L=R) и противофаза раньше давали concentration~0 через circular mean
    угла — ровно наоборот ожидаемому. Осевая мера (эллипс через собственные
    числа) обязана давать axis_ratio~0 для моно (вырожденная линия) и
    близко к 1 для широкого некоррелированного стерео (круг)."""
    n = SR
    mono = 0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
    g_mono = goniometer_stats(mono, mono)
    assert g_mono["goniometer_axis_ratio"] < 0.05, (
        f"моно (L=R) должно давать axis_ratio~0, получили {g_mono['goniometer_axis_ratio']}")

    rng = np.random.default_rng(0)
    wide_l = rng.standard_normal(n) * 0.2
    wide_r = rng.standard_normal(n) * 0.2  # независимый шум в каналах — максимально широкое стерео
    g_wide = goniometer_stats(wide_l, wide_r)
    assert g_wide["goniometer_axis_ratio"] > 0.9, (
        f"независимый шум в L/R должен давать axis_ratio~1 (круг), получили "
        f"{g_wide['goniometer_axis_ratio']}")


if __name__ == "__main__":
    test_reverb_rt60_recovered_from_known_exponential_decay()
    test_reverb_rt60_scales_with_decay_time()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_lufs_matches_bs1770_calibration_signal(Path(d))
        test_lufs_drops_10db_when_signal_gained_down_10db(Path(d))
    test_sharpness_increases_with_noise_band_center_frequency()
    test_dissonance_zero_at_unison_peaks_near_critical_bandwidth()
    test_goniometer_axis_ratio_zero_for_mono_one_for_wide_stereo()
    print("Все тесты Блока В (непокрытые модули) прошли.")
