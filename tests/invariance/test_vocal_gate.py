"""ТЗ-05 А8: 4с тишины + 4с чистого инструментала (без вокала) -> все
F0-метрики окна должны быть NaN, не число (даже если pYIN что-то
"нашёл" на шуме/гармониках инструментала — гейт по voiced-доле обязан
это отсечь)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis import engine

SR = 44100


def test_no_vocal_window_gives_null_f0_metrics():
    # 8с: тишина, затем чистый инструментальный тон (220Гц) — ни то, ни
    # другое не должно пройти вокальный гейт
    silence = np.zeros(4 * SR)
    instrumental = 0.3 * np.sin(2 * np.pi * 220 * np.arange(4 * SR) / SR)
    mono = np.concatenate([silence, instrumental])

    # f0_df с voiced=False везде — как если бы pYIN честно не нашёл вокал
    t = np.arange(0, 8, 512 / SR)
    f0_df = pd.DataFrame({"t_s": t, "f0_hz": np.full(len(t), np.nan), "voiced": False})

    wdf = engine.window_metrics(mono, SR, "vocals", f0_df=f0_df, notes_df=None)
    assert len(wdf) > 0, "инструментальное окно должно пройти энергетический гейт (не тишина)"
    assert wdf["voiced_fraction"].isna().all(), "voiced_fraction обязан быть NaN без подтверждённого вокала"
    assert wdf["vibrato_depth_cents"].isna().all()
    assert wdf["intonation_dev_cents"].isna().all()


def test_vocal_window_with_high_voiced_frac_passes_gate():
    mono = 0.3 * np.sin(2 * np.pi * 220 * np.arange(8 * SR) / SR)
    t = np.arange(0, 8, 512 / SR)
    f0_df = pd.DataFrame({"t_s": t, "f0_hz": np.full(len(t), 220.0), "voiced": True})
    # >=2 ноты в окне — медиана вибрато не считается по одному наблюдению
    # (см. engine.window_metrics: `if len(nsub_vib) >= 2`)
    notes_df = pd.DataFrame({
        "t_start": [0.5, 2.5], "t_end": [2.0, 4.0],
        "vibrato_depth_cents": [40.0, 45.0], "intonation_deviation_cents": [12.0, -8.0],
    })

    wdf = engine.window_metrics(mono, SR, "vocals", f0_df=f0_df, notes_df=notes_df)
    assert (wdf["voiced_fraction"] > 0.9).all(), "полностью voiced-сигнал обязан пройти гейт"
    assert not wdf["vibrato_depth_cents"].isna().all(), "при подтверждённом вокале вибрато должно считаться"
