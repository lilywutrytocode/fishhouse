"""模型层:规范 schema 校验与整理。"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from chanlun.config import market_of
from chanlun.data.models import (
    OHLCV_COLUMNS,
    to_canonical,
    validate_canonical,
)
from tests.conftest import make_daily, weekdays


def test_market_of():
    assert market_of("300502") == "A"
    assert market_of("09992") == "HK"
    assert market_of("NVDA") == "US"
    with pytest.raises(KeyError):
        market_of("999999")


def test_validate_canonical_ok():
    df = make_daily(weekdays(date(2024, 1, 1), 10))
    validate_canonical(df)  # 不抛即通过


def test_validate_rejects_naive_index():
    df = make_daily(weekdays(date(2024, 1, 1), 5))
    df.index = df.index.tz_localize(None)
    with pytest.raises(ValueError, match="时区"):
        validate_canonical(df)


def test_validate_rejects_non_monotonic():
    df = make_daily(weekdays(date(2024, 1, 1), 5))
    df = df.iloc[::-1]  # 逆序
    with pytest.raises(ValueError, match="递增"):
        validate_canonical(df)


def test_validate_rejects_high_below_low():
    df = make_daily(weekdays(date(2024, 1, 1), 5))
    df.iloc[2, df.columns.get_loc("high")] = df.iloc[2]["low"] - 1
    with pytest.raises(ValueError, match="high < low"):
        validate_canonical(df)


def test_to_canonical_fills_amount_and_orders_columns():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"]),
        "open": [1, 2, 9], "high": [2, 3, 9], "low": [0.5, 1.5, 9],
        "close": [1.5, 2.5, 9], "volume": [10, 20, 99],
        # 故意缺 amount,且含重复日期 + 乱序
    })
    out = to_canonical(raw, tz="Asia/Shanghai")
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert "amount" in out.columns
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique  # 重复日期去重(保留 last)
    assert out.index.tz is not None
