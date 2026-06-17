"""data_health 日历误杀修复:weekday 兜底日历对 A 股节假日不 REJECT(只软化为 WARN)。

真实交易日历(XSHG 等)仍严格;min30 不套 daily session 缺失逻辑。
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from chanlun.cli import build_data_health
from chanlun.data.calendars import WeekdayCalendar
from chanlun.data.health import check_health
from chanlun.data.models import FLAG_APPROX_CALENDAR_WEEKDAY


def _daily_df(days, tz="Asia/Shanghai"):
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in days],
                           name="date").tz_localize(tz)
    n = len(days)
    return pd.DataFrame({"open": [10.0] * n, "high": [11.0] * n, "low": [9.0] * n,
                         "close": [10.0] * n, "volume": [1.0] * n,
                         "amount": [1.0] * n}, index=idx)


def _weekdays_with_holiday_gaps(start: date, n_weeks: int):
    """生成工作日序列,挖掉若干节假日缺口(含一段春节式连缺 7 个工作日)。"""
    days = []
    d = start
    while len(days) < n_weeks * 5:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    # 春节式连缺:删掉第 20~26 个工作日(连续 7 个)
    spring = set(days[20:27])
    # 国庆式:删掉第 60~64 个工作日(连续 5 个)
    national = set(days[60:65])
    return [x for x in days if x not in spring | national]


def test_a_daily_weekday_fallback_not_rejected():
    days = _weekdays_with_holiday_gaps(date(2023, 1, 2), 80)  # 充足历史,避免 SHORT
    df = _daily_df(days)
    hr = check_health(df, market="A", symbol="sz300308", level="daily",
                      calendar=WeekdayCalendar())
    assert hr.status != "REJECT"                  # 节假日不再误杀
    assert hr.status == "WARN"
    assert FLAG_APPROX_CALENDAR_WEEKDAY in hr.flags


def test_a_daily_weekday_extreme_gap_still_rejects():
    days = _weekdays_with_holiday_gaps(date(2023, 1, 2), 80)
    df = _daily_df(days[:25] + days[25 + 12:])    # 再挖一段连续 12 工作日 → 明显异常
    hr = check_health(df, market="A", symbol="sz300308", level="daily",
                      calendar=WeekdayCalendar())
    assert hr.status == "REJECT"
    assert "CONSEC_MISSING" in hr.reasons


def test_min30_not_rejected_by_holidays():
    # min30:即使带节假日缺口,build_data_health 不套 daily session 逻辑 → 不 REJECT
    base = pd.Timestamp("2025-01-02 10:00", tz="Asia/Shanghai")
    idx = pd.date_range(base, periods=2000, freq="30min", tz="Asia/Shanghai")
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0,
                       "volume": 1.0, "amount": 1.0}, index=idx)
    h = build_data_health(df, symbol="sz300308", market="A", level="min30")
    assert h["status"] != "REJECT"


def test_report_reject_banner():
    from chanlun.cli import format_report
    from chanlun.output import build_output
    out = build_output(symbol="sz300308", level="daily",
                       data_health={"status": "REJECT",
                                    "reasons": ["CONSEC_MISSING", "MISSING_RATE"]})
    report = format_report(out)
    assert "数据门禁: REJECT" in report
    assert "CONSEC_MISSING, MISSING_RATE" in report
    assert "不会输出" in report
