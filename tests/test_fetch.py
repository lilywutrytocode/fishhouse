"""§1.1 拉取编排:主源→兜底回落 + 前复权校验(用桩源,不触网)。"""

from __future__ import annotations

from datetime import date

import pytest

from chanlun.data.fetch import fetch
from chanlun.data.models import Adjust
from chanlun.data.sources.base import FetchError, SourceResult
from tests.conftest import make_daily, weekdays


class StubSource:
    """可配置的桩源:可成功、可抛 FetchError、可返回非前复权。"""

    def __init__(self, name, *, fail=False, adjust=Adjust.QFQ.value, df=None):
        self.name = name
        self._fail = fail
        self._adjust = adjust
        self._df = df
        self.calls = 0

    def fetch(self, symbol, market, level):
        self.calls += 1
        if self._fail:
            raise FetchError(f"{self.name} 故意失败")
        df = self._df if self._df is not None else make_daily(
            weekdays(date(2024, 1, 1), 10)
        )
        return SourceResult(df=df, source=self.name, adjust=self._adjust)


def test_uses_primary_when_it_succeeds():
    primary = StubSource("akshare")
    fallback = StubSource("eastmoney")
    res = fetch("300502", market="A", level="daily", sources=[primary, fallback])
    assert res.source == "akshare"
    assert primary.calls == 1
    assert fallback.calls == 0  # 主源成功 → 不回落


def test_falls_back_when_primary_fails():
    primary = StubSource("akshare", fail=True)
    fallback = StubSource("eastmoney")
    res = fetch("300502", market="A", level="daily", sources=[primary, fallback])
    assert res.source == "eastmoney"
    assert primary.calls == 1
    assert fallback.calls == 1


def test_falls_back_when_primary_not_qfq():
    # 主源返回非前复权 → 视为不合格,回落兜底(§1.2 前复权钉死)
    primary = StubSource("akshare", adjust=Adjust.NONE.value)
    fallback = StubSource("eastmoney")
    res = fetch("300502", market="A", level="daily", sources=[primary, fallback])
    assert res.source == "eastmoney"
    assert res.adjust == Adjust.QFQ.value


def test_raises_when_all_sources_fail():
    primary = StubSource("akshare", fail=True)
    fallback = StubSource("eastmoney", fail=True)
    with pytest.raises(FetchError, match="所有源均失败"):
        fetch("300502", market="A", level="daily", sources=[primary, fallback])


def test_market_inferred_from_symbol():
    primary = StubSource("akshare")
    res = fetch("300502", level="daily", sources=[primary])  # 不传 market
    assert res.source == "akshare"


def test_us_has_no_fallback_single_source():
    # 美股仅 yfinance;桩源失败应直接抛错(无兜底)
    primary = StubSource("yfinance", fail=True)
    with pytest.raises(FetchError):
        fetch("NVDA", market="US", level="daily", sources=[primary])
