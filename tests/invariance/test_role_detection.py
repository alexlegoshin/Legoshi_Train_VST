"""ТЗ-05 А10: русские ключевые слова наравне с английскими, устойчивость
к NFC/NFD (macOS APFS отдаёт имена файлов в NFD — кириллица разложена на
базовую букву + диакритику, байтово не совпадает с NFC-литералом в коде,
тот же баг, что уже ловили в исследовательском коде на «й»)."""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import orchestrate


def test_russian_keywords_recognized():
    assert orchestrate.classify_role("вокал основной.wav") == "vocals"
    assert orchestrate.classify_role("бас.wav") == "bass"
    assert orchestrate.classify_role("драмсы.wav") == "drums"
    assert orchestrate.classify_role("акустика микро.wav") == "other"  # не гитара по ключевым словам — ожидаемо


def test_english_keywords_still_recognized():
    assert orchestrate.classify_role("Lead Vocal Double.wav") == "vocals"
    assert orchestrate.classify_role("Bass DI.wav") == "bass"
    assert orchestrate.classify_role("Kick In.wav") == "drums"


def test_nfd_decomposed_filename_still_recognized():
    # реальный случай (уже ловили этот баг в исследовательском коде):
    # "й" в NFC — один кодпоинт (U+0439), в NFD — "и" (U+0438) + breve
    # (U+0306), байтово НЕ равно NFC-литералу в коде без нормализации.
    # Слова без "й" (просто "вокал", "бас") в NFD не меняются вовсе —
    # тест на них ничего бы не проверял, нужно именно слово с "й".
    name_nfc = "вокал дублирующий.wav"
    name_nfd = unicodedata.normalize("NFD", name_nfc)
    assert name_nfc != name_nfd, "тест не проверяет ничего, если платформа не дала настоящего NFD"
    assert orchestrate.classify_role(name_nfd) == "vocals"


def test_main_keyword_nfd():
    name_nfd = unicodedata.normalize("NFD", "общий микс.wav")
    assert orchestrate.is_main(name_nfd)


def test_unrecognized_falls_back_to_other():
    # d1.wav — из примера в самом ТЗ-05, ни один ключ не подходит
    assert orchestrate.classify_role("d1.wav") == "other"
