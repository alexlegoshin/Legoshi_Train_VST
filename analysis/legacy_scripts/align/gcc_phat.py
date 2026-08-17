"""Переехало в ядро — ТЗ-05 А4: GCC-PHAT стал обязательной частью
оркестратора (проверка синхронности дорожек трек-аута), не только
исследовательским инструментом. Реализация теперь в `analysis.alignment`,
этот файл — совместимость для `run_alignment.py` и старых тестов."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from analysis.alignment import gcc_phat, activity_fraction, bar_hypothesis_shift  # noqa: F401
