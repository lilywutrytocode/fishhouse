"""测试公共夹具与合成数据构造器(全部离线,不触网)。"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from chanlun.data.models import OHLCV_COLUMNS


def make_daily(
    dates,
    *,
    tz: str = "Asia/Shanghai",
    base: float = 100.0,
    step: float = 1.0,
    amount: float | None = 1.0e8,
) -> pd.DataFrame:
    """由自然日列表构造规范日线;每根 bar 给确定的小幅 OHLC,便于断言。"""
    rows = []
    idx = []
    for i, d in enumerate(dates):
        c = base + i * step
        rows.append({
            "open": c - 0.5, "high": c + 0.5, "low": c - 1.0, "close": c,
            "volume": 1000 + i, "amount": amount,
        })
        idx.append(pd.Timestamp(d))
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex(idx, name="date").tz_localize(tz)
    return df


def make_30min_for_day(
    day: date,
    *,
    tz: str = "America/New_York",
    open_=100.0,
    high=101.0,
    low=99.0,
    close=100.5,
    n_bars: int = 13,
) -> pd.DataFrame:
    """构造单交易日的 30min bar 序列(默认 13 根 ≈ 美股 regular session)。

    保证聚合后:open=首根 open、high=全程最高、low=全程最低、close=尾根 close。
    """
    start = datetime(day.year, day.month, day.day, 9, 30)
    rows, idx = [], []
    for i in range(n_bars):
        if i == 0:
            o, h, l, c = open_, open_, open_, open_
        elif i == n_bars - 1:
            o, h, l, c = close, close, close, close
        else:
            o = h = l = c = (open_ + close) / 2
        # 把极值塞进中间某根
        if i == 1:
            h = high
        if i == 2:
            l = low
        rows.append({"open": o, "high": h, "low": l, "close": c,
                     "volume": 100, "amount": pd.NA})
        idx.append(pd.Timestamp(start + timedelta(minutes=30 * i)))
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex(idx, name="date").tz_localize(tz)
    return df


def weekdays(start: date, n: int) -> list[date]:
    """从 start 起取 n 个工作日(周一至周五)。"""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


@pytest.fixture
def make_daily_fixture():
    return make_daily


@pytest.fixture
def weekdays_fixture():
    return weekdays
