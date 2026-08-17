"""ТЗ-02 Задача 5: переклассификация vibrato_depth/voiced_fraction в группу
artifact_detectors + прямой показатель «доля выпрямленных нот».

vibrato_depth, посчитанная по F0 готового микса, — это в первую очередь
индикатор того, насколько питч был выпрямлен обработкой (тюном), не ось
вкуса. Прямее и честнее: сравнить каждый "аномально плоский" сегмент
F0 в МИКСЕ (уже находит §4.6, detect_tune_artifacts/flats) с тем же
временным окном в СЫРОМ СТЕМЕ ("вокал основной", необработанный) — если
в стеме там была настоящая нота (устойчивая, F0 тоже плоский) — это не
артефакт, честная деталь исполнения. Если в стеме было движение (тон,
глиссандо, вибрато), а в миксе плоско — это выпрямление тюном."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

from analysis.metrics.pitch_vocal import hz_to_cents

ROOT = Path(__file__).resolve().parents[4]
METRICS_DIR = ROOT / "_analysis" / "metrics"
OUT = ROOT / "_analysis" / "diff"

STEM_NAME = "основной трек__ТА__Track Out__вокал основной.wav"
STEM_OFFSET_SAMPLES = -94  # из alignment.parquet, align_stem: aligned[n]=raw[n+offset]
SR = 44100
FLAT_THRESH_CENTS = 5.0

MIX_FILES = {
    "v2": "основной трек__-__версия сведения 2.wav",
    "v3": "основной трек__-__версия сведения 3..wav",
    "v4": "основной трек__-__версия сведения 4 .wav",
    "v5": "основной трек__-__версия сведения 5.wav",
    "v6": "основной трек__-__версия сведения 6.wav",
    "v7": "основной трек__-__версия сведения 7.wav",
}


def main():
    stem_f0 = pd.read_parquet(METRICS_DIR / f"{STEM_NAME}.4_6_f0.parquet")
    # raw-время стема -> время в системе координат микса: t_mix = t_raw - offset/sr
    stem_f0 = stem_f0.copy()
    stem_f0["t_mix"] = stem_f0["t_s"] - STEM_OFFSET_SAMPLES / SR
    stem_cents = hz_to_cents(stem_f0["f0_hz"].to_numpy())

    rows = []
    for version, safe_name in MIX_FILES.items():
        flats_path = METRICS_DIR / f"{safe_name}.4_6_flats.parquet"
        notes_path = METRICS_DIR / f"{safe_name}.4_6_notes.parquet"
        if not flats_path.exists():
            continue
        flats = pd.read_parquet(flats_path)
        notes = pd.read_parquet(notes_path) if notes_path.exists() else pd.DataFrame()
        n_notes_total = len(notes[notes.type == "note"]) if len(notes) else np.nan

        n_flattened, n_genuine, examples = 0, 0, []
        for _, seg in flats.iterrows():
            mask = (stem_f0["t_mix"] >= seg.t_start) & (stem_f0["t_mix"] <= seg.t_end) & stem_f0["voiced"]
            window_cents = stem_cents[mask.to_numpy()]
            window_cents = window_cents[np.isfinite(window_cents)]
            if len(window_cents) < 3:
                continue
            stem_std = float(np.std(window_cents))
            if stem_std > FLAT_THRESH_CENTS:
                n_flattened += 1
                examples.append(dict(t_start=float(seg.t_start), t_end=float(seg.t_end), stem_std_cents=stem_std))
            else:
                n_genuine += 1

        n_checked = n_flattened + n_genuine
        frac = n_flattened / n_checked if n_checked else np.nan
        rows.append(dict(version=version, n_flat_segments_in_mix=len(flats),
                          n_checked_against_stem=n_checked, n_flattened_confirmed=n_flattened,
                          n_genuine_sustained=n_genuine, n_notes_total=n_notes_total,
                          доля_выпрямленных_нот=frac))
        if examples:
            pd.DataFrame(examples).to_parquet(OUT / f"flattened_examples_{version}.parquet", index=False)

    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "flattened_notes.parquet", index=False)
    table.to_csv(OUT / "flattened_notes.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
