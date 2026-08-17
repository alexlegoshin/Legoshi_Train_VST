"""Шаг 10 §10.3/§10.4: дубли для оценки внутренней согласованности + сборка
3 сессий. ~15% фрагментов встречаются дважды, по возможности в РАЗНЫХ
сессиях (проверка "оценка1 vs оценка2" — без неё корреляциям в Шаге 11
не с чем сравниваться, см. TZ-02-addendum §10.3).

102 уникальных клипа — немного больше плана (~90), поэтому сессии выходят
по ~39 слотов вместо строгих 30-35 (102 + ~15% дублей = 117, /3 = 39).
Не подрезаю искусственно ради точного числа — качество и покрытие
стратификации важнее точного попадания в диапазон."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
Q_DIR = ROOT / "_analysis" / "questionnaire"

RNG = np.random.default_rng(2026)
N_SESSIONS = 3
DUP_FRACTION = 0.15
N_WARMUP = 3


def main():
    manifest = pd.read_parquet(Q_DIR / "clips_manifest.parquet")
    n_unique = len(manifest)
    n_dup = round(n_unique * DUP_FRACTION)

    dup_ids = RNG.choice(manifest.fragment_id.to_numpy(), size=n_dup, replace=False)

    slots = []
    for _, row in manifest.iterrows():
        slots.append(dict(fragment_id=row.fragment_id, clip_path=row.clip_path,
                           version=row.version, group=row.group, is_duplicate_of=None))
    for fid in dup_ids:
        row = manifest[manifest.fragment_id == fid].iloc[0]
        slots.append(dict(fragment_id=fid, clip_path=row.clip_path,
                           version=row.version, group=row.group, is_duplicate_of=fid))

    slots_df = pd.DataFrame(slots)
    slots_df = slots_df.sample(frac=1, random_state=RNG.integers(1e6)).reset_index(drop=True)

    # раскладываем по сессиям с балансом источников: round-robin по версии
    per_session = [[] for _ in range(N_SESSIONS)]
    seen_in_session = [set() for _ in range(N_SESSIONS)]  # чтобы дубль не попал в ту же сессию
    session_sizes_target = [len(slots_df) // N_SESSIONS] * N_SESSIONS
    for i in range(len(slots_df) % N_SESSIONS):
        session_sizes_target[i] += 1

    remaining = slots_df.to_dict("records")
    RNG.shuffle(remaining)
    unplaced = []
    for rec in remaining:
        # предпочитаем сессию, где этот fragment_id ещё не встречался
        candidates = [i for i in range(N_SESSIONS)
                      if rec["fragment_id"] not in seen_in_session[i] and len(per_session[i]) < session_sizes_target[i]]
        if not candidates:
            candidates = [i for i in range(N_SESSIONS) if len(per_session[i]) < session_sizes_target[i]]
        if not candidates:
            unplaced.append(rec)
            continue
        # среди кандидатов — тот, где сейчас меньше всего этого источника (баланс)
        def source_count(i):
            return sum(1 for r in per_session[i] if r["version"] == rec["version"])
        chosen = min(candidates, key=source_count)
        per_session[chosen].append(rec)
        seen_in_session[chosen].add(rec["fragment_id"])
    for rec in unplaced:
        chosen = min(range(N_SESSIONS), key=lambda i: len(per_session[i]))
        per_session[chosen].append(rec)

    rows = []
    for s_idx, session in enumerate(per_session):
        order = RNG.permutation(len(session))
        for pos, orig_idx in enumerate(order):
            rec = session[orig_idx]
            rows.append(dict(
                session=s_idx + 1, position=pos + 1, is_warmup=pos < N_WARMUP,
                fragment_id=rec["fragment_id"], clip_path=rec["clip_path"],
                is_duplicate_of=rec["is_duplicate_of"],
            ))

    plan = pd.DataFrame(rows)
    plan.to_parquet(Q_DIR / "session_plan.parquet", index=False)
    plan.to_csv(Q_DIR / "session_plan.csv", index=False)

    print(f"Уникальных клипов: {n_unique}, дублей: {n_dup}, всего слотов: {len(plan)}")
    print(plan.groupby("session").size().to_string())
    print("\nДубли, оказавшиеся в разных сессиях (из числа дублированных):")
    dup_sessions = plan[plan.is_duplicate_of.notna()].groupby("fragment_id").session.apply(list)
    orig_sessions = plan[plan.is_duplicate_of.isna() & plan.fragment_id.isin(dup_ids)].set_index("fragment_id").session
    cross_session = 0
    for fid, sessions in dup_sessions.items():
        orig_s = orig_sessions.get(fid)
        if orig_s is not None and orig_s not in sessions:
            cross_session += 1
    print(f"{cross_session}/{len(dup_ids)}")


if __name__ == "__main__":
    main()
