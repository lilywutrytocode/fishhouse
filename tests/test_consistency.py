"""§1.10 日内-日线一致性校验。"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from chanlun.data.consistency import aggregate_30min_to_daily, check_consistency
from chanlun.data.models import ConsistencyStatus, OHLCV_COLUMNS
from tests.conftest import make_30min_for_day


def _daily_from(day, *, open_, high, low, close, tz="America/New_York"):
    df = pd.DataFrame(
        [{"open": open_, "high": high, "low": low, "close": close,
          "volume": 1000, "amount": pd.NA}],
        columns=list(OHLCV_COLUMNS),
    )
    df.index = pd.DatetimeIndex([pd.Timestamp(day)], name="date").tz_localize(tz)
    return df


def test_aggregate_30min_matches_session_ohlc():
    day = date(2024, 3, 4)
    m30 = make_30min_for_day(day, open_=100, high=105, low=95, close=102)
    agg = aggregate_30min_to_daily(m30)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["open"] == 100
    assert row["high"] == 105
    assert row["low"] == 95
    assert row["close"] == 102


def test_consistency_ok_when_aligned():
    day = date(2024, 3, 4)
    m30 = make_30min_for_day(day, open_=100, high=105, low=95, close=102)
    daily = _daily_from(day, open_=100, high=105, low=95, close=102)
    rep = check_consistency(m30, daily, symbol="NVDA")
    assert rep.status == ConsistencyStatus.OK.value
    assert rep.compared_days == 1
    assert not rep.reject_daily_30min_lianli


def test_consistency_warn_on_small_close_dev():
    day = date(2024, 3, 4)
    m30 = make_30min_for_day(day, open_=100, high=105, low=95, close=102)
    # close 偏差 ~0.78% (>0.5%, <2%) → WARN
    daily = _daily_from(day, open_=100, high=105, low=95, close=102.8)
    rep = check_consistency(m30, daily, symbol="NVDA")
    assert rep.status == ConsistencyStatus.WARN.value
    assert day in rep.warn_days


def test_consistency_reject_lianli_on_large_dev():
    day = date(2024, 3, 4)
    m30 = make_30min_for_day(day, open_=100, high=105, low=95, close=102)
    # high 偏差 ~3% (>2%) → REJECT 联立(但不 REJECT 单级别日线)
    daily = _daily_from(day, open_=100, high=108.5, low=95, close=102)
    rep = check_consistency(m30, daily, symbol="NVDA")
    assert rep.status == ConsistencyStatus.REJECT_LIANLI.value
    assert rep.reject_daily_30min_lianli is True
    assert day in rep.reject_days


def test_consistency_only_compares_common_days():
    d1, d2 = date(2024, 3, 4), date(2024, 3, 5)
    m30 = pd.concat([
        make_30min_for_day(d1, open_=100, high=105, low=95, close=102),
        make_30min_for_day(d2, open_=102, high=106, low=100, close=104),
    ])
    daily = _daily_from(d1, open_=100, high=105, low=95, close=102)  # 仅 d1
    rep = check_consistency(m30, daily, symbol="NVDA")
    assert rep.compared_days == 1
