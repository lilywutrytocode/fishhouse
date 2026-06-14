"""交易日历:兜底周内日历 + 列表日历。"""

from __future__ import annotations

from datetime import date

from chanlun.data.calendars import ListCalendar, WeekdayCalendar, get_calendar


def test_weekday_calendar_excludes_weekend():
    cal = WeekdayCalendar()
    sess = cal.sessions(date(2024, 1, 1), date(2024, 1, 7))  # 周一..周日
    assert sess == [date(2024, 1, d) for d in range(1, 6)]  # 仅周一..周五


def test_list_calendar_filters_range():
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 10)]
    cal = ListCalendar(days)
    assert cal.sessions(date(2024, 1, 1), date(2024, 1, 5)) == days[:2]


def test_get_calendar_returns_something():
    # 真实环境有 exchange_calendars → ExchangeCalendar;否则 WeekdayCalendar。
    cal = get_calendar("A")
    sess = cal.sessions(date(2024, 1, 2), date(2024, 1, 5))
    assert len(sess) >= 1
