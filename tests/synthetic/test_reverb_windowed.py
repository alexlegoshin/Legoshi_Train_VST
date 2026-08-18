"""Блок 2 (среднее окно): группировка per-onset строк reverb.analyze_file
по позиции на треке — не новая метрика реверба, только локализация уже
посчитанных значений (см. analysis/metrics/reverb.py, windowed_summary)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from analysis.metrics.reverb import windowed_summary


def test_groups_onsets_by_position():
    df = pd.DataFrame([
        dict(onset_s=1.0, rt60_s=0.3, edt_s=0.2, c50_db=10.0, c80_db=12.0, drr_db=5.0),
        dict(onset_s=5.0, rt60_s=0.4, edt_s=0.25, c50_db=9.0, c80_db=11.0, drr_db=4.5),
        dict(onset_s=20.0, rt60_s=1.5, edt_s=1.0, c50_db=2.0, c80_db=3.0, drr_db=-2.0),
    ])
    windows = windowed_summary(df, win_s=15.0)
    assert len(windows) == 2
    w0, w1 = windows
    assert w0["t_start"] == 0.0 and w0["t_end"] == 15.0 and w0["n_onsets"] == 2
    assert abs(w0["rt60_s_median"] - 0.35) < 1e-9
    assert w1["t_start"] == 15.0 and w1["t_end"] == 30.0 and w1["n_onsets"] == 1
    assert w1["rt60_s_median"] == 1.5


def test_local_outlier_visible_only_in_its_window():
    """Смысл группировки: аномальный хвост (напр. бликид на одном участке)
    не размазывается в общую медиану трека, виден именно в своём окне."""
    normal_rows = [dict(onset_s=float(s), rt60_s=0.4, edt_s=0.3, c50_db=8.0,
                          c80_db=10.0, drr_db=4.0) for s in range(0, 15, 3)]
    outlier_row = dict(onset_s=20.0, rt60_s=3.0, edt_s=2.5, c50_db=-5.0, c80_db=-3.0, drr_db=-10.0)
    df = pd.DataFrame(normal_rows + [outlier_row])
    windows = windowed_summary(df, win_s=15.0)
    assert len(windows) == 2
    assert windows[0]["rt60_s_median"] == 0.4
    assert windows[1]["rt60_s_median"] == 3.0


def test_empty_input_returns_empty_list():
    assert windowed_summary(pd.DataFrame(), win_s=15.0) == []
    assert windowed_summary(None, win_s=15.0) == []


def test_last_window_t_end_clamped_to_track_duration():
    """Регрессия (найдена код-ревью): без track_duration_s последнее окно
    получало чисто номинальный t_end=(idx+1)*win_s, даже если реальный
    трек короче — на 42с треке с онсетом на 35с (окно idx=2, 30-45с)
    t_end указывал бы на 45с, хотя трек кончается на 42с. Тот же баг,
    которого noise.find_persistent_narrowband_windowed для точно такого
    же понятия "~15с окно" не имеет (там t_end честно обрезан по факту
    длины сигнала) — здесь нужна та же дисциплина."""
    df = pd.DataFrame([dict(onset_s=35.0, rt60_s=0.4, edt_s=0.3, c50_db=8.0, c80_db=10.0, drr_db=4.0)])
    windows = windowed_summary(df, win_s=15.0, track_duration_s=42.0)
    assert len(windows) == 1
    assert windows[0]["t_start"] == 30.0
    assert windows[0]["t_end"] == 42.0  # не 45.0 — трек реально кончается на 42с


def test_t_end_not_clamped_when_track_duration_not_given():
    """Обратная совместимость: без track_duration_s поведение как раньше
    (номинальная граница сетки) — не ломаем вызовы, где длина неизвестна."""
    df = pd.DataFrame([dict(onset_s=35.0, rt60_s=0.4, edt_s=0.3, c50_db=8.0, c80_db=10.0, drr_db=4.0)])
    windows = windowed_summary(df, win_s=15.0)
    assert windows[0]["t_end"] == 45.0


if __name__ == "__main__":
    test_groups_onsets_by_position()
    test_local_outlier_visible_only_in_its_window()
    test_empty_input_returns_empty_list()
    test_last_window_t_end_clamped_to_track_duration()
    test_t_end_not_clamped_when_track_duration_not_given()
    print("Все тесты оконной сводки реверба (Блок 2) прошли.")
