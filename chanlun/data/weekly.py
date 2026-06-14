"""§1.9 ★ 周线由日线合成。

周线**一律由已过健康检查的日线快照合成**(锚定周五 resample),
不直接读数据源周线——保证复权口径与日线一致、可复现。
"""

from __future__ import annotations

import pandas as pd

from .models import OHLCV_COLUMNS, HealthStatus, validate_canonical

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
}


def synthesize_weekly(
    daily: pd.DataFrame,
    *,
    daily_health_status: str | None = None,
) -> pd.DataFrame:
    """把规范日线合成规范周线(锚定周五)。

    参数
    ----
    daily: 规范日线 OHLCV(tz-aware)。**应为已过健康检查的快照**。
    daily_health_status: 传入日线健康状态;若为 REJECT 则拒绝合成(数据门禁)。

    返回:规范周线 OHLCV,索引为各周周五(tz 同日线)。
    """
    validate_canonical(daily)
    if daily_health_status == HealthStatus.REJECT.value:
        raise ValueError("日线健康状态为 REJECT,拒绝合成周线(§1.9 数据门禁)")
    if daily.empty:
        return daily.copy()

    weekly = daily.resample("W-FRI").agg(_AGG)
    # resample 在无交易的周也会产生全 NaN 行,剔除(close 为空即该周无数据)
    weekly = weekly.dropna(subset=["open", "high", "low", "close"])
    weekly = weekly[list(OHLCV_COLUMNS)]
    weekly.index.name = "date"
    return weekly
