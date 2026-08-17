"""ТЗ-05 А11. Эмпирически проверено вручную (см. комментарий в
orchestrate.run_demucs и отчёт): --shifts по умолчанию равен 1, даёт
max|diff|~0.05-0.08 между запусками на идентичном входе; --shifts 0 даёт
побайтово идентичный результат. Здесь — контрактный тест: сама команда
обязана содержать "--shifts","0", чтобы фикс не потерялся при рефакторинге
(гонять Demucs в юнит-тесте на каждый прогон — слишком медленно)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import orchestrate


def test_demucs_invocation_disables_random_shifts(tmp_path):
    fake_mix = tmp_path / "fake.wav"
    fake_mix.write_bytes(b"RIFF....")  # содержимое не важно — subprocess замокан

    with patch("orchestrate.engine.ensure_sr", return_value=fake_mix), \
         patch("orchestrate.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        try:
            orchestrate.run_demucs(fake_mix, tmp_path)
        except Exception:
            pass  # stem_dir не существует в тесте — не важно, проверяем только вызов

    assert mock_run.called, "subprocess.run не был вызван"
    cmd = mock_run.call_args[0][0]
    assert "--shifts" in cmd, f"--shifts не передан в команду: {cmd}"
    idx = cmd.index("--shifts")
    assert cmd[idx + 1] == "0", f"--shifts должен быть 0 для детерминированности, передан: {cmd[idx+1]}"
