"""Date and time utility helpers."""

from __future__ import annotations

from datetime import date, timedelta


def date_range(start: date, end: date, step_days: int = 1):
    """Yield dates from start to end (inclusive)."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=step_days)


def parse_date(s: str) -> date:
    """Parse a date string in YYYY-MM-DD or YYYYMMDD format."""
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    return date.fromisoformat(s)
