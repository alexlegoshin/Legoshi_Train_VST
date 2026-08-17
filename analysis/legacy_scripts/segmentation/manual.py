"""Ручная сегментация, §3.4 п.1 ТЗ-01. Парсер таймкодов переиспользуется и
для правок в §7 — там же явно перечислены форматы: "0:46", "1 мин 10 сек",
"2 мин 15 сек", "7 сек"."""
import re


def parse_timecode(s: str) -> float:
    """Секунды из разных форматов таймкодов, встречающихся в текстах автора."""
    s = s.strip()
    m = re.fullmatch(r"(\d+):(\d{1,2})(?:\.(\d+))?", s)
    if m:
        mins, secs, frac = m.groups()
        return int(mins) * 60 + int(secs) + (int(frac) / 10 ** len(frac) if frac else 0)
    m = re.fullmatch(r"(\d+)\s*мин\.?\s*(\d+)\s*сек\.?", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.fullmatch(r"(\d+)\s*сек\.?", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    raise ValueError(f"не распознан формат таймкода: {s!r}")


if __name__ == "__main__":
    tests = ["0:46", "1 мин 10 сек", "2 мин 15 сек", "7 сек", "0:54", "27", "3:06.5"]
    for t in tests:
        print(f"{t!r:20s} -> {parse_timecode(t)}с")
