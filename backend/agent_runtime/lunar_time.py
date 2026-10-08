"""Calendar-only expansion of user-stated Chinese lunar dates.

This module never reads answer labels or infers a lunar date from a vague
holiday mention. Unspecified years are supplied by the authorised album.
"""

from __future__ import annotations

import re
from datetime import date


_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_LUNAR_DAY = re.compile(
    r"(?:大年|正月|农历(?P<month>[一二三四五六七八九十\d]{1,2})月)"
    r"初(?P<day>[一二三四五六七八九十\d]{1,2})"
)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*年")
_SHORT_YEAR = re.compile(r"(?<!\d)(\d{2})\s*年")


def _number(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if len(value) == 1:
        return _DIGITS.get(value)
    if value.startswith("十"):
        return 10 + _DIGITS.get(value[1:], 0)
    return None


def lunar_solar_dates(query: str, album_years: set[int]) -> set[date]:
    """Map explicit lunar month/day wording to solar dates in album years."""
    match = _LUNAR_DAY.search(str(query or ""))
    if not match:
        return set()
    month = _number(match.group("month")) if match.group("month") else 1
    day = _number(match.group("day"))
    if not month or not day or not (1 <= month <= 12 and 1 <= day <= 10):
        return set()
    explicit_years = {int(y) for y in _YEAR.findall(query)}
    if not explicit_years:
        explicit_years = {2000 + int(y) for y in _SHORT_YEAR.findall(query)}
    years = explicit_years or album_years
    try:
        from lunar_python import Lunar
    except ImportError:
        return set()
    dates = set()
    for year in years:
        if not 1900 <= year <= 2099:
            continue
        try:
            solar = Lunar.fromYmd(year, month, day).getSolar()
            dates.add(date(solar.getYear(), solar.getMonth(), solar.getDay()))
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return dates
