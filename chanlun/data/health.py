"""§1.7 数据健康检查【确定性】★

区分三个日期(listed / source / analysis),缺失率**自 analysis_start_date 算**,
不从源起始简单算。停牌 ≠ 缺失。判定:
- 连续缺 ≥3 交易日 → REJECT
- 总缺失率(自 analysis_start_date)> 5% → REJECT
- 否则有缺 → WARN
- 历史不足 1.5×长度 → 不拒算,标 SHORT_HISTORY + 低置信(不影响 REJECT/WARN)
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

from ..config import DEFAULT_CONFIG, Config
from .calendars import TradingCalendar
from .models import (
    FLAG_LISTED_DATE_UNKNOWN,
    FLAG_PRE_ANALYSIS_HISTORY_UNAVAILABLE,
    FLAG_SHORT_HISTORY,
    REASON_CONSEC_MISSING,
    REASON_HAS_MISSING,
    REASON_MISSING_RATE,
    REASON_NO_DATA,
    HealthReport,
    HealthStatus,
)


def _index_dates(df: pd.DataFrame) -> list[date]:
    """规范 df 索引 → 去时区后的自然日列表(升序)。"""
    return [ts.date() for ts in df.index]


def check_health(
    df: pd.DataFrame,
    *,
    market: str,
    symbol: str,
    level: str,
    calendar: TradingCalendar | None = None,
    listed_date: date | None = None,
    analysis_start_date: date | None = None,
    suspended_dates: Iterable[date] | None = None,
    config: Config = DEFAULT_CONFIG,
) -> HealthReport:
    """对单标的×级别做数据健康检查。

    参数
    ----
    df: 规范 OHLCV(tz-aware DatetimeIndex)。
    calendar: 交易日历;``None`` 表示跳过缺失判定(如周线,仅做长度/充足度)。
    listed_date: 上市/重新上市日;``None`` 则无法判定 pre-analysis 历史是否缺。
    analysis_start_date: 结构计算起点;``None`` 默认取 source_start_date(用全量)。
    suspended_dates: 停牌日集合(A/港),计为 SUSPENDED,不计缺失、不触发 REJECT。
    """
    reasons: list[str] = []
    flags: list[str] = []
    required_length = config.min_length_for(level)

    # ── 空数据直接 REJECT ────────────────────────────────────────────────
    if df.empty:
        return HealthReport(
            symbol=symbol, market=market, level=level,
            status=HealthStatus.REJECT.value,
            listed_date=listed_date, source_start_date=None,
            analysis_start_date=None, pre_analysis_history_unavailable=False,
            expected_sessions=0, present_bars=0, suspended_days=0,
            missing_days=0, missing_rate=0.0, max_consecutive_missing=0,
            bars_available=0, required_length=required_length,
            short_history=True, low_confidence=True,
            reasons=[REASON_NO_DATA], flags=list(flags),
        )

    present = _index_dates(df)
    source_start_date = present[0]
    end_date = present[-1]
    if analysis_start_date is None:
        analysis_start_date = source_start_date

    # ── pre-analysis 历史是否缺(诚实标注)────────────────────────────────
    if listed_date is None:
        flags.append(FLAG_LISTED_DATE_UNKNOWN)
        pre_unavailable = False
    else:
        pre_unavailable = source_start_date > listed_date
        if pre_unavailable:
            flags.append(FLAG_PRE_ANALYSIS_HISTORY_UNAVAILABLE)

    # 自 analysis_start_date 起的存在 bar
    present_in_range = [d for d in present if d >= analysis_start_date]
    present_set = set(present_in_range)
    bars_available = len(present_in_range)

    suspended_set = {d for d in (suspended_dates or []) if d >= analysis_start_date}

    # ── 缺失判定(需日历)────────────────────────────────────────────────
    if calendar is not None:
        sessions = [
            d for d in calendar.sessions(analysis_start_date, end_date)
            if d not in suspended_set  # 停牌日剔出期望序列(≠ 缺失)
        ]
        expected_sessions = len(sessions)
        missing = [d for d in sessions if d not in present_set]
        missing_days = len(missing)
        missing_rate = (missing_days / expected_sessions) if expected_sessions else 0.0

        max_consec = 0
        run = 0
        for d in sessions:  # 已剔停牌,连续性按剩余期望序列算
            if d in present_set:
                run = 0
            else:
                run += 1
                max_consec = max(max_consec, run)

        if max_consec >= config.missing_consecutive_reject:
            reasons.append(REASON_CONSEC_MISSING)
        if missing_rate > config.missing_rate_reject:
            reasons.append(REASON_MISSING_RATE)
        if missing_days and not reasons:
            reasons.append(REASON_HAS_MISSING)
    else:
        expected_sessions = bars_available
        missing_days = 0
        missing_rate = 0.0
        max_consec = 0

    suspended_in_range = len([d for d in suspended_set if d <= end_date])

    # ── 历史充足度(不影响 REJECT/WARN,仅标志 + 低置信)──────────────────
    if required_length is not None:
        threshold = config.short_history_factor * required_length
        short_history = bars_available < threshold
    else:
        short_history = False
    if short_history:
        flags.append(FLAG_SHORT_HISTORY)

    # ── 汇总状态 ─────────────────────────────────────────────────────────
    if REASON_CONSEC_MISSING in reasons or REASON_MISSING_RATE in reasons:
        status = HealthStatus.REJECT.value
    elif REASON_HAS_MISSING in reasons:
        status = HealthStatus.WARN.value
    else:
        status = HealthStatus.OK.value

    return HealthReport(
        symbol=symbol, market=market, level=level, status=status,
        listed_date=listed_date, source_start_date=source_start_date,
        analysis_start_date=analysis_start_date,
        pre_analysis_history_unavailable=pre_unavailable,
        expected_sessions=expected_sessions, present_bars=bars_available,
        suspended_days=suspended_in_range, missing_days=missing_days,
        missing_rate=missing_rate, max_consecutive_missing=max_consec,
        bars_available=bars_available, required_length=required_length,
        short_history=short_history, low_confidence=short_history,
        reasons=reasons, flags=flags,
    )
