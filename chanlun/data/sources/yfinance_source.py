"""yfinance 源适配器(美股,§1.1)。

``auto_adjust=True`` 取复权后 OHLC(全四价)。美股只用 regular session
(§1.8,剔 pre/post);30min 历史 yfinance 仅近 ~60 天(§1.3)。
惰性 import,缺依赖时抛 :class:`FetchError`。
"""

from __future__ import annotations

import pandas as pd

from ...config import MARKET_TZ
from ..models import Adjust, to_canonical
from .base import FetchError, SourceResult

_INTERVAL = {"daily": "1d", "min30": "30m"}


class YFinanceSource:
    name = "yfinance"

    def fetch(self, symbol: str, market: str, level: str) -> SourceResult:
        if market != "US":
            raise FetchError("yfinance 适配器仅用于美股")
        try:
            import yfinance as yf
        except Exception as e:  # pragma: no cover
            raise FetchError(f"yfinance 不可用:{e}") from e

        interval = _INTERVAL.get(level)
        if interval is None:
            raise FetchError(f"不支持级别 {level}")

        try:  # pragma: no cover - 网络
            period = "60d" if level == "min30" else "max"
            raw = yf.download(
                symbol, period=period, interval=interval,
                auto_adjust=True, prepost=False, progress=False,
            )
        except Exception as e:  # pragma: no cover
            raise FetchError(f"yfinance 拉取失败:{e}") from e

        if raw is None or raw.empty:  # pragma: no cover
            raise FetchError("yfinance 返回空数据")
        raw = self._normalize(raw)  # pragma: no cover
        df = to_canonical(raw, tz=MARKET_TZ[market])  # pragma: no cover
        return SourceResult(df=df, source=self.name, adjust=Adjust.QFQ.value)

    @staticmethod
    def _normalize(raw: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df["amount"] = pd.NA  # yfinance 不提供成交额
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume", "amount"]]
