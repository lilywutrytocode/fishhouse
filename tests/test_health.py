"""§1.7 数据健康检查。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chanlun.data.calendars import ListCalendar
from chanlun.data.health import check_health
from chanlun.data.models import (
    FLAG_LISTED_DATE_UNKNOWN,
    FLAG_PRE_ANALYSIS_HISTORY_UNAVAILABLE,
    FLAG_SHORT_HISTORY,
    REASON_CONSEC_MISSING,
    REASON_MISSING_RATE,
    REASON_NO_DATA,
    HealthStatus,
)
from tests.conftest import make_daily, weekdays


def _cal_from(dates):
    return ListCalendar(dates)


def test_clean_data_is_ok():
    days = weekdays(date(2023, 1, 2), 400)
    df = make_daily(days)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
    )
    assert rep.status == HealthStatus.OK.value
    assert rep.missing_days == 0
    assert rep.short_history is False
    assert rep.source_start_date == days[0]
    assert rep.analysis_start_date == days[0]


def test_consecutive_missing_rejects():
    days = weekdays(date(2023, 1, 2), 400)
    # 抠掉连续 3 个交易日 → REJECT
    present = days[:50] + days[53:]
    df = make_daily(present)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
    )
    assert rep.status == HealthStatus.REJECT.value
    assert REASON_CONSEC_MISSING in rep.reasons
    assert rep.max_consecutive_missing == 3


def test_scattered_missing_below_threshold_is_warn():
    days = weekdays(date(2023, 1, 2), 400)
    # 抠掉 4 个不相邻交易日(<5%、连续<3)→ WARN
    drop = {days[10], days[40], days[80], days[120]}
    present = [d for d in days if d not in drop]
    df = make_daily(present)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
    )
    assert rep.status == HealthStatus.WARN.value
    assert rep.missing_days == 4
    assert rep.max_consecutive_missing == 1


def test_missing_rate_rejects():
    days = weekdays(date(2023, 1, 2), 100)
    # 隔位抠掉(交替缺失)→ 缺失率 ~50%、但连续<3 → 由缺失率触发 REJECT
    present = [d for i, d in enumerate(days) if i % 2 == 0]
    df = make_daily(present)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
    )
    assert rep.status == HealthStatus.REJECT.value
    assert REASON_MISSING_RATE in rep.reasons
    assert rep.max_consecutive_missing < 3  # 不是连续触发
    assert rep.missing_rate > 0.05


def test_suspended_not_counted_as_missing():
    days = weekdays(date(2023, 1, 2), 400)
    # 连续 5 天停牌:若按缺失会 REJECT;标 SUSPENDED 后应不 REJECT
    suspended = days[100:105]
    present = [d for d in days if d not in set(suspended)]
    df = make_daily(present)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
        suspended_dates=suspended,
    )
    assert rep.status == HealthStatus.OK.value
    assert rep.missing_days == 0
    assert rep.suspended_days == 5


def test_short_history_flag_does_not_reject():
    days = weekdays(date(2023, 1, 2), 300)  # < 1.5×250=375 → SHORT_HISTORY
    df = make_daily(days)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
    )
    assert rep.short_history is True
    assert rep.low_confidence is True
    assert FLAG_SHORT_HISTORY in rep.flags
    assert rep.status == HealthStatus.OK.value  # 短历史不拒算


def test_pre_analysis_history_unavailable_when_source_starts_after_listing():
    days = weekdays(date(2023, 1, 2), 400)
    df = make_daily(days)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days),
        listed_date=date(2015, 1, 1),  # 远早于源起始
    )
    assert rep.pre_analysis_history_unavailable is True
    assert FLAG_PRE_ANALYSIS_HISTORY_UNAVAILABLE in rep.flags


def test_listed_date_unknown_flagged():
    days = weekdays(date(2023, 1, 2), 400)
    df = make_daily(days)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=None,
    )
    assert rep.pre_analysis_history_unavailable is False
    assert FLAG_LISTED_DATE_UNKNOWN in rep.flags


def test_missing_rate_from_analysis_start_not_source_start():
    days = weekdays(date(2023, 1, 2), 400)
    # 源前 100 天有大量缺失,但 analysis 从第 100 天起且其后干净 → 应 OK
    early = [d for i, d in enumerate(days[:100]) if i % 2 == 0]  # 前段一半缺失
    present = early + days[100:]
    df = make_daily(present)
    rep = check_health(
        df, market="A", symbol="300502", level="daily",
        calendar=_cal_from(days), listed_date=days[0],
        analysis_start_date=days[100],
    )
    assert rep.analysis_start_date == days[100]
    assert rep.status == HealthStatus.OK.value
    assert rep.missing_days == 0


def test_empty_data_rejects():
    empty = make_daily(weekdays(date(2023, 1, 2), 1)).iloc[0:0]
    rep = check_health(
        empty, market="A", symbol="300502", level="daily",
        calendar=_cal_from([]),
    )
    assert rep.status == HealthStatus.REJECT.value
    assert REASON_NO_DATA in rep.reasons


def test_weekly_without_calendar_skips_missing():
    # 周线无逐日历概念 → calendar=None,仅长度/充足度
    days = weekdays(date(2020, 1, 2), 200)
    df = make_daily(days)
    rep = check_health(
        df, market="A", symbol="300502", level="weekly",
        calendar=None, listed_date=days[0],
    )
    assert rep.missing_days == 0
    assert rep.required_length == 150
    # 200 < 1.5×150=225 → 短历史
    assert rep.short_history is True
    assert rep.status == HealthStatus.OK.value
