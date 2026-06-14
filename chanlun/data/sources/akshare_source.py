"""akshare 源适配器(A 股 / 港股主源)。前复权全四价(§1.2)。

惰性 import akshare。本环境未安装 akshare 时,实例化不报错,
仅在 :meth:`fetch` 调用时抛 :class:`FetchError`,从而触发兜底回落。
"""

from __future__ import annotations

import pandas as pd

from ...config import MARKET_TZ
from ..models import Adjust, to_canonical
from .base import FetchError, SourceResult

# akshare 周期参数
_PERIOD = {"daily": "daily", "min30": "30"}


class AkshareSource:
    name = "akshare"

    def fetch(self, symbol: str, market: str, level: str) -> SourceResult:
        try:
            import akshare as ak
        except Exception as e:  # pragma: no cover - 依赖缺失路径
            raise FetchError(f"akshare 不可用:{e}") from e

        tz = MARKET_TZ[market]
        try:
            if market == "A":
                raw = self._fetch_a(ak, symbol, level)
            elif market == "HK":
                raw = self._fetch_hk(ak, symbol, level)
            else:
                raise FetchError(f"akshare 不支持市场 {market}")
        except FetchError:
            raise
        except Exception as e:  # pragma: no cover - 网络/解析错误
            raise FetchError(f"akshare 拉取失败:{e}") from e

        if raw is None or raw.empty:
            raise FetchError("akshare 返回空数据")
        df = to_canonical(raw, tz=tz)
        return SourceResult(df=df, source=self.name, adjust=Adjust.QFQ.value)

    def _fetch_a(self, ak, symbol: str, level: str) -> pd.DataFrame:  # pragma: no cover
        if level in ("daily",):
            df = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
        elif level == "min30":
            df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="30", adjust="qfq")
        else:
            raise FetchError(f"不支持级别 {level}")
        return self._rename(df)

    def _fetch_hk(self, ak, symbol: str, level: str) -> pd.DataFrame:  # pragma: no cover
        if level == "daily":
            df = ak.stock_hk_hist(symbol=symbol, period="daily", adjust="qfq")
        elif level == "min30":
            df = ak.stock_hk_hist_min_em(symbol=symbol, period="30", adjust="qfq")
        else:
            raise FetchError(f"不支持级别 {level}")
        return self._rename(df)

    @staticmethod
    def _rename(df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        # akshare 中文列 → 规范英文列
        mapping = {
            "日期": "date", "时间": "date",
            "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
            "成交量": "volume", "成交额": "amount",
        }
        return df.rename(columns=mapping)
