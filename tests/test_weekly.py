"""§1.9 周线由日线合成。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from chanlun.data.models import HealthStatus, validate_canonical
from chanlun.data.weekly import synthesize_weekly
from tests.conftest import make_daily, weekdays


def test_weekly_aggregates_ohlc_correctly():
    # 两个完整自然周(周一..周五)
    days = weekdays(date(2024, 1, 1), 10)  # 1/1 周一 .. 1/12 周五
    df = make_daily(days, base=100.0, step=1.0)
    wk = synthesize_weekly(df)
    validate_canonical(wk)
    assert len(wk) == 2

    first_week = df.iloc[:5]
    w0 = wk.iloc[0]
    assert w0["open"] == first_week.iloc[0]["open"]   # 周一 open
    assert w0["close"] == first_week.iloc[-1]["close"]  # 周五 close
    assert w0["high"] == first_week["high"].max()
    assert w0["low"] == first_week["low"].min()
    assert w0["volume"] == first_week["volume"].sum()


def test_weekly_anchored_on_friday():
    days = weekdays(date(2024, 1, 1), 5)
    wk = synthesize_weekly(make_daily(days))
    # 锚定周五:索引落在 1/5(周五)
    assert wk.index[0].weekday() == 4


def test_weekly_rejects_when_daily_health_reject():
    df = make_daily(weekdays(date(2024, 1, 1), 5))
    with pytest.raises(ValueError, match="REJECT"):
        synthesize_weekly(df, daily_health_status=HealthStatus.REJECT.value)


def test_weekly_passes_when_daily_health_ok():
    df = make_daily(weekdays(date(2024, 1, 1), 5))
    wk = synthesize_weekly(df, daily_health_status=HealthStatus.OK.value)
    assert len(wk) == 1
