"""本地 CSV 加载器(§1.6):中文表头 → 规范 schema + 自带 MACD 进 cross-check 列。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from chanlun.data.loaders import load_local_csv
from chanlun.data.models import OHLCV_COLUMNS, validate_canonical
from chanlun.structure.beichi import compute_macd

RAW_DAILY = "chanlun/data/xinyisheng_300502_daily_20250701_20260609_macd.csv"
RAW_30MIN = "chanlun/data/xinyisheng_300502_30min_20260408_20260610_macd.csv"


def test_load_daily_chinese_header_to_canonical():
    res = load_local_csv(RAW_DAILY, level="daily")
    assert res.df.shape == (228, 6)
    assert list(res.df.columns) == list(OHLCV_COLUMNS)
    validate_canonical(res.df)                          # tz-aware、递增唯一、high≥low
    assert res.df.index.tz is not None
    assert str(res.source_start_date) == "2025-07-01"


def test_file_macd_goes_to_crosscheck_only_and_converges():
    res = load_local_csv(RAW_DAILY, level="daily")
    assert res.cross_check is not None
    assert list(res.cross_check.columns) == ["cc_dif", "cc_dea", "cc_macd"]
    assert len(res.cross_check) == 228
    # ★ 引擎自算 MACD,与文件自带在尾部(暖机后)收敛 → 证明自算正确
    m = compute_macd(res.df["close"])
    tail = np.abs(m["dif"].values[-50:] - res.cross_check["cc_dif"].values[-50:])
    assert tail.max() < 0.01


def test_load_intraday_30min():
    res = load_local_csv(RAW_30MIN, level="min30")
    assert res.df.shape == (344, 6)
    assert res.df.index.tz is not None
    assert res.df.index[0].hour == 10                   # 日内时间戳保留
    assert res.cross_check is not None


def test_ohlcv_only_has_no_crosscheck(tmp_path):
    # 真实 akshare/yfinance 路径只返回 OHLCV、无 MACD → cross_check 为 None
    p = tmp_path / "ohlcv.csv"
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [10, 11, 12], "high": [11, 12, 13], "low": [9, 10, 11],
        "close": [10.5, 11.5, 12.5], "volume": [100, 110, 120],
        "amount": [1e6, 1.1e6, 1.2e6],
    }).to_csv(p, index=False)
    res = load_local_csv(str(p), level="daily")
    assert res.cross_check is None
    assert list(res.df.columns) == list(OHLCV_COLUMNS)
