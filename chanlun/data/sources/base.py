"""源适配器协议与公共类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd


class FetchError(RuntimeError):
    """源拉取失败(网络/限频/无数据/非前复权等),触发兜底回落。"""


@dataclass
class SourceResult:
    """单次源拉取结果。"""

    df: pd.DataFrame   # 规范 OHLCV(tz-aware)
    source: str        # 源名(akshare / eastmoney / yfinance)
    adjust: str        # 复权口径(qfq)


@runtime_checkable
class DataSource(Protocol):
    name: str

    def fetch(self, symbol: str, market: str, level: str) -> SourceResult:
        """拉取规范化的前复权 OHLC;失败抛 :class:`FetchError`。"""
        ...
