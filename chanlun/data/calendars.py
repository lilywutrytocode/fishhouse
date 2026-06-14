"""交易日历(§1.7 缺失判定、§1.8 半日市/session)。

提供统一接口 :class:`TradingCalendar`:给定 [start, end] 返回交易日列表
(假期不计)。真实日历用 ``exchange_calendars``(A=XSHG, 港=XHKG, 美=XNYS);
无该依赖时回落到 :class:`WeekdayCalendar`(周一至周五,不含假期,**仅兜底**)。
单元测试可用 :class:`ListCalendar` 注入确定的交易日集合。
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Protocol

from ..config import MARKET_CALENDAR


class TradingCalendar(Protocol):
    name: str

    def sessions(self, start: date, end: date) -> list[date]:
        """返回 [start, end] 闭区间内的交易日(升序),假期不计。"""
        ...


class ListCalendar:
    """由显式交易日集合构造的日历,主要供测试注入确定结果。"""

    def __init__(self, sessions: Iterable[date], name: str = "list"):
        self._sessions = sorted(set(sessions))
        self.name = name

    def sessions(self, start: date, end: date) -> list[date]:
        return [d for d in self._sessions if start <= d <= end]


class WeekdayCalendar:
    """周一至周五兜底日历(不含任何节假日)。

    仅在 ``exchange_calendars`` 不可用时兜底;真实缺失率会被节假日污染,
    因此真实运行应优先用 :class:`ExchangeCalendar`。
    """

    name = "weekday"

    def sessions(self, start: date, end: date) -> list[date]:
        out: list[date] = []
        d = start
        while d <= end:
            if d.weekday() < 5:  # 0=周一 .. 4=周五
                out.append(d)
            d += timedelta(days=1)
        return out


class ExchangeCalendar:
    """``exchange_calendars`` 适配器(惰性 import)。"""

    def __init__(self, code: str):
        import exchange_calendars as ec  # 惰性:仅真实运行时需要

        self._cal = ec.get_calendar(code)
        self.name = code

    def sessions(self, start: date, end: date) -> list[date]:
        idx = self._cal.sessions_in_range(
            self._cal.date_to_session(str(start), direction="next"),
            self._cal.date_to_session(str(end), direction="previous"),
        )
        return [ts.date() for ts in idx]


def get_calendar(market: str) -> TradingCalendar:
    """按市场返回交易日历;无 ``exchange_calendars`` 时回落 :class:`WeekdayCalendar`。"""
    code = MARKET_CALENDAR.get(market)
    if code is None:
        return WeekdayCalendar()
    try:
        return ExchangeCalendar(code)
    except Exception:
        return WeekdayCalendar()
