"""东方财富源适配器(A 股 / 港股兜底,§1.1)。

东财行情通过 akshare 的东财接口取(``*_em`` 系列),与主源同样前复权全四价;
独立成类以表达「主源失败→兜底回落」的语义。惰性 import,缺依赖时抛
:class:`FetchError`。
"""

from __future__ import annotations

import pandas as pd

from ...config import MARKET_TZ
from ..models import Adjust, to_canonical
from .base import FetchError, SourceResult


class EastmoneySource:
    name = "eastmoney"

    def fetch(self, symbol: str, market: str, level: str) -> SourceResult:
        try:
            import akshare as ak  # 东财接口经 akshare 暴露
        except Exception as e:  # pragma: no cover
            raise FetchError(f"东财(akshare em 接口)不可用:{e}") from e

        tz = MARKET_TZ[market]
        try:
            if market == "A" and level == "daily":  # pragma: no cover
                raw = ak.stock_zh_a_hist(
                    symbol=symbol, period="daily", adjust="qfq",
                )
            elif market == "HK" and level == "daily":  # pragma: no cover
                raw = ak.stock_hk_hist_em(symbol=symbol, period="daily", adjust="qfq")
            else:
                raise FetchError(f"东财兜底不支持 {market}/{level}")
        except FetchError:
            raise
        except Exception as e:  # pragma: no cover
            raise FetchError(f"东财拉取失败:{e}") from e

        if raw is None or raw.empty:  # pragma: no cover
            raise FetchError("东财返回空数据")
        mapping = {
            "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
            "收盘": "close", "成交量": "volume", "成交额": "amount",
        }
        df = to_canonical(raw.rename(columns=mapping), tz=tz)  # pragma: no cover
        return SourceResult(df=df, source=self.name, adjust=Adjust.QFQ.value)
