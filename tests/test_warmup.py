"""§7.1 MACD 暖机守卫:前 ~130 根 EMA 暖机区不发背驰/买卖点。"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from chanlun.cli import analyze
from chanlun.config import Config
from chanlun.data.loaders import load_local_csv
from chanlun.data.models import OHLCV_COLUMNS

NO_WARMUP = Config(macd_warmup_factor=0)
RAW_DAILY = "chanlun/data/raw/300502/300502_daily_short.csv"


def _div_df(tz="Asia/Shanghai"):
    pts = [100, 70, 85, 68, 82, 66, 78, 67, 76]
    closes = [pts[0]]
    for k in range(1, len(pts)):
        closes += list(np.linspace(pts[k - 1], pts[k], 7))[1:]
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 100, "amount": 1.0} for c in closes]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex(
        [pd.Timestamp(date(2024, 1, 1)) + pd.Timedelta(days=i)
         for i in range(len(closes))], name="date").tz_localize(tz)
    return df


def test_default_warmup_bars_is_130():
    assert Config().macd_warmup_bars() == 130           # 5 × 26


def test_short_data_fully_in_warmup_suppresses_all_signals():
    # 49 根 < 130 → 全程暖机区:默认配置不发任何背驰/买卖点
    df = _div_df()
    out = analyze(df, symbol="TEST")
    assert out["macd_warmup"]["fully_in_warmup"] is True
    assert out["beichi"] == []
    assert out["mai_mai_dian"] == []
    # 关掉暖机守卫 → 信号回来(证明是守卫抑制,不是无结构)
    nw = analyze(df, symbol="TEST", config=NO_WARMUP)
    assert len(nw["beichi"]) >= 1 and len(nw["mai_mai_dian"]) >= 1
    # 笔/线段不受暖机影响(不依赖 MACD)
    assert len(out["bi"]) >= 1


def test_300502_daily_warmup_only_recent_signals():
    df = load_local_csv(RAW_DAILY, level="daily").df
    out = analyze(df, symbol="300502")                  # 默认暖机 130
    mw = out["macd_warmup"]
    assert mw["bars"] == 130
    assert mw["cutoff_date"] is not None
    cutoff = pd.Timestamp(mw["cutoff_date"])

    # 默认:所有已确认背驰/买卖点 confirm 都 ≥ 暖机截止(只在最近 ~100 根)
    for b in out["beichi"]:
        if b["confirm_date"]:
            assert pd.Timestamp(b["confirm_date"]) >= cutoff
    for m in out["mai_mai_dian"]:
        if m["confirm_date"]:
            assert pd.Timestamp(m["confirm_date"]) >= cutoff

    # 关掉暖机守卫 → 信号更多,且存在落在暖机区(早段)的背驰被守卫剔除了
    nw = analyze(df, symbol="300502", config=NO_WARMUP)
    assert len(nw["beichi"]) >= len(out["beichi"])
    early = [b for b in nw["beichi"]
             if b["confirm_date"] and pd.Timestamp(b["confirm_date"]) < cutoff]
    assert len(early) >= 1                               # 暖机区确有被抑制的背驰
    # 笔不受影响
    assert len(out["bi"]) == len(nw["bi"])
