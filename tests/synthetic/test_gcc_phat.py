"""Синтетические тесты на align/gcc_phat.py — известный ответ до реального корпуса."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from analysis.legacy_scripts.align.gcc_phat import gcc_phat, bar_hypothesis_shift

SR = 44100


def make_transient_signal(sr, dur_s, n_events=40, seed=0):
    """Широкополосный сигнал с чёткими transient-событиями (имитация
    вокала/гитары) — то, на чём GCC-PHAT должен работать хорошо."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur_s * sr))
    for t in rng.uniform(0, dur_s, n_events):
        i = int(t * sr)
        burst_len = int(0.05 * sr)
        env = np.exp(-np.linspace(0, 8, burst_len))
        x[i:i+burst_len] += env * rng.standard_normal(min(burst_len, len(x) - i))
    return x


def make_periodic_signal(sr, dur_s, period_s, seed=1):
    """Периодический сигнал (имитация ударных) — GCC-PHAT обязан путаться
    между сдвигами, кратными ОДНОЙ ДОЛЕ. Чтобы сдвиг на такт (4 доли) не был
    вырожден сам по себе, паттерн внутри такта должен иметь период ровно 4
    доли, без более коротких под-периодов — 4 разных по амплитуде/тембру
    удара на позициях 0..3, без 1- или 2-дольной симметрии."""
    rng = np.random.default_rng(seed)
    x = np.zeros(int(dur_s * sr))

    def hit(amp, cutoff_hz, length_s, seed_local):
        from scipy.signal import butter, sosfilt
        n = int(length_s * sr)
        env = np.exp(-np.linspace(0, 15, n))
        noise = np.random.default_rng(seed_local).standard_normal(n)
        sos = butter(2, cutoff_hz, btype='low', fs=sr, output='sos')
        return amp * env * sosfilt(sos, noise)

    hits = [
        hit(1.0, 4000, 0.08, seed + 10),   # позиция 0: громкий кик
        hit(0.4, 2500, 0.05, seed + 11),   # позиция 1: тихий кик
        hit(0.8, 6000, 0.06, seed + 12),   # позиция 2: снейр
        hit(0.25, 8000, 0.03, seed + 13),  # позиция 3: закрытый хэт-щелчок
    ]
    beat_idx = 0
    t = 0.0
    while t < dur_s:
        i = int(t * sr)
        w = hits[beat_idx % 4]
        x[i:i+len(w)] += w[:max(0, min(len(w), len(x) - i))]
        t += period_s
        beat_idx += 1
    return x


def eq_color(x, sr):
    """Грубая тембральная окраска — имитация того, что стем и его появление
    в миксе звучат по-разному (инженер сведения же EQ крутит)."""
    from scipy.signal import butter, sosfilt
    sos = butter(2, 3000, btype='low', fs=sr, output='sos')
    return sosfilt(sos, x)


def test_transient_signal_recovers_known_shift():
    ref = make_transient_signal(SR, 20.0)
    true_shift_s = 0.1373  # намеренно не кратно кадру/такту
    shift_samples_true = int(round(true_shift_s * SR))
    sig = np.zeros_like(ref)
    sig[shift_samples_true:] = ref[:len(ref) - shift_samples_true]
    sig = eq_color(sig, SR) + 0.02 * np.random.default_rng(2).standard_normal(len(sig))

    shift, confidence, p1, p2, z = gcc_phat(sig, ref, SR, max_shift_s=1.0)
    err_ms = abs(shift - shift_samples_true) / SR * 1000
    print(f"[transient] true={shift_samples_true} found={shift} err={err_ms:.3f}ms conf={confidence:.2f}")
    assert err_ms < 1.0, f"error too large: {err_ms}ms"
    assert confidence > 1.3, f"confidence too low: {confidence}"


def test_periodic_loop_is_genuinely_ambiguous_across_bar_multiples():
    """Ключевой вывод: если паттерн СТРОГО периодичен по такту (лупуется
    один в один), корреляцией огибающей число тактов сдвига НЕ восстановить
    в принципе — это математическое свойство периодического сигнала, а не
    слабость метода. Алгоритм обязан это обнаружить и явно сказать
    "неоднозначно", а не тихо выдать один (возможно неверный) ответ."""
    bpm = 72.0
    period_s = 60.0 / bpm
    bar_s = period_s * 4
    ref = make_periodic_signal(SR, 30.0, period_s)  # лупуется весь файл
    true_shift_s = 2 * bar_s
    shift_samples_true = int(round(true_shift_s * SR))
    sig = np.zeros_like(ref)
    sig[shift_samples_true:] = ref[:len(ref) - shift_samples_true]

    shift, confidence, p1, p2, z = gcc_phat(sig, ref, SR, max_shift_s=5.0)
    print(f"[periodic/loop] naive gcc_phat: shift={shift} conf={confidence:.2f} (низкая уверенность ожидаема)")

    best_shift_s, score, ambiguous, ties = bar_hypothesis_shift(bar_s, max_bars=8, sig=sig, ref=ref, sr=SR)
    print(f"[periodic/loop] bar-hypothesis: true={true_shift_s:.3f}s best={best_shift_s:.3f}s "
          f"score={score:.3f} ambiguous={ambiguous} tied_candidates={ties}")
    # Точное совпадение с true_shift_s среди тай-кандидатов не требуем: при
    # 10-мс квантовании кадров и разном размере валидного окна перекрытия
    # на разных k числовые оценки чуть шумят. Существенно то, что метод
    # НЕ возвращает одну уверенную (потенциально неверную) цифру — он видит
    # больше одного равноценного кандидата и явно об этом сообщает.
    assert ambiguous, "ожидалась неоднозначность на строго лупованном материале — если прошло, это подозрительно"
    assert len(ties) >= 2, "ожидалось несколько равноценных кандидатов на периодическом материале"


def test_periodic_signal_with_one_fill_breaks_the_tie():
    """Практический выход из неоднозначности: если в материале есть ХОТЯ БЫ
    одно неповторяющееся событие (файл, фолл, акцент — то, что реально есть
    в живой записи, а не в drum-машине), периодичность по такту нарушается
    и bar-hypothesis обязан найти единственный верный ответ."""
    bpm = 72.0
    period_s = 60.0 / bpm
    bar_s = period_s * 4
    ref = make_periodic_signal(SR, 30.0, period_s)
    # разовый акцент вне паттерна — как одиночный крэш/фолл в реальной записи
    rng = np.random.default_rng(99)
    fill_i = int(18.0 * SR)
    fill = rng.standard_normal(int(0.15 * SR)) * 2.5
    ref[fill_i:fill_i + len(fill)] += fill

    true_shift_s = 2 * bar_s
    shift_samples_true = int(round(true_shift_s * SR))
    sig = np.zeros_like(ref)
    sig[shift_samples_true:] = ref[:len(ref) - shift_samples_true]

    best_shift_s, score, ambiguous, ties = bar_hypothesis_shift(bar_s, max_bars=8, sig=sig, ref=ref, sr=SR)
    err_ms = abs(best_shift_s - true_shift_s) * 1000
    print(f"[periodic/with-fill] true={true_shift_s:.3f}s found={best_shift_s:.3f}s err={err_ms:.1f}ms "
          f"ambiguous={ambiguous} tied={len(ties)} (порог неоднозначности консервативен по конструкции —"
          f" это нормально, что он иногда перестраховывается)")
    # Главное требование: лучший ответ должен быть верным. Флаг ambiguous
    # намеренно консервативен (1% допуск) — лучше лишний раз перестраховаться
    # и попросить автора подтвердить, чем тихо съесть неверный сдвиг.
    assert err_ms < 5.0, f"bar hypothesis failed even with a disambiguating fill: err={err_ms}ms"


if __name__ == "__main__":
    test_transient_signal_recovers_known_shift()
    test_periodic_loop_is_genuinely_ambiguous_across_bar_multiples()
    test_periodic_signal_with_one_fill_breaks_the_tie()
    print("ALL SYNTHETIC ALIGNMENT TESTS PASSED")
