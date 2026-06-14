"""§1.10 ★ 日内-日线一致性校验【确定性·可配置】

将 30min(regular session)聚合成日线 OHLCV,与日线源比较:
- close 偏差 > 0.5% → WARN
- high/low 偏差 > 1% → WARN
- 任一核心价偏差 > 2% → REJECT 日-30min 联立

★ 此 REJECT **仅作用于日-30min 联立**,不 REJECT 单级别日线分析。
"""

from __future__ import annotations

import pandas as pd

from ..config import DEFAULT_CONFIG, Config
from .models import (
    PRICE_COLUMNS,
    ConsistencyReport,
    ConsistencyStatus,
    validate_canonical,
)

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def aggregate_30min_to_daily(min30: pd.DataFrame) -> pd.DataFrame:
    """把 30min bar 按自然日聚合为日线 OHLC(open=首/high=高/low=低/close=尾)。"""
    grouped = min30.groupby(min30.index.normalize()).agg(_AGG)
    grouped.index.name = "date"
    return grouped


def _rel_dev(a: float, b: float) -> float:
    """相对偏差 |a-b|/|b|;基准为 0 时回退到绝对差以避免除零。"""
    if b == 0:
        return abs(a - b)
    return abs(a - b) / abs(b)


def check_consistency(
    min30: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    symbol: str,
    config: Config = DEFAULT_CONFIG,
) -> ConsistencyReport:
    """比较 30min 聚合日线 与 日线源,产出一致性报告。

    仅比较两侧共有的交易日;任一侧缺该日则跳过(缺失由 §1.7 健康检查负责)。
    """
    validate_canonical(min30)
    validate_canonical(daily)

    agg = aggregate_30min_to_daily(min30)
    daily_by_day = daily.copy()
    daily_by_day.index = daily_by_day.index.normalize()

    common = agg.index.intersection(daily_by_day.index)

    max_close = max_high = max_low = 0.0
    warn_days: list = []
    reject_days: list = []

    for ts in common:
        a = agg.loc[ts]
        d = daily_by_day.loc[ts]
        devs = {c: _rel_dev(float(a[c]), float(d[c])) for c in PRICE_COLUMNS}
        max_close = max(max_close, devs["close"])
        max_high = max(max_high, devs["high"])
        max_low = max(max_low, devs["low"])

        day = ts.date()
        if any(v > config.consistency_reject_pct for v in devs.values()):
            reject_days.append(day)
        elif (
            devs["close"] > config.consistency_close_pct
            or devs["high"] > config.consistency_highlow_pct
            or devs["low"] > config.consistency_highlow_pct
        ):
            warn_days.append(day)

    if reject_days:
        status = ConsistencyStatus.REJECT_LIANLI.value
    elif warn_days:
        status = ConsistencyStatus.WARN.value
    else:
        status = ConsistencyStatus.OK.value

    return ConsistencyReport(
        symbol=symbol, status=status, compared_days=len(common),
        max_close_dev=max_close, max_high_dev=max_high, max_low_dev=max_low,
        warn_days=warn_days, reject_days=reject_days,
    )
