"""本地 CSV 加载器(§1.6):中文表头 → 规范 schema + 自带 MACD 进 cross-check 列。"""

from __future__ import annotations

import pandas as pd

from chanlun.data.loaders import load_local_csv
from chanlun.data.models import OHLCV_COLUMNS, validate_canonical


def _write(p, header, rows):
    p.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(p)


def test_load_daily_chinese_header_to_canonical(tmp_path):
    path = _write(tmp_path / "d.csv",
                  "日期,开盘,最高,最低,收盘,成交量,成交额",
                  ["2024-01-02,10.0,10.5,9.8,10.2,1000,1e7",
                   "2024-01-03,10.2,10.9,10.1,10.7,1100,1.1e7",
                   "2024-01-04,10.7,11.0,10.5,10.6,1200,1.2e7"])
    res = load_local_csv(path, level="daily")
    assert list(res.df.columns) == list(OHLCV_COLUMNS)   # 中文表头已映射
    assert res.df.shape == (3, 6)
    validate_canonical(res.df)
    assert res.df.index.tz is not None
    assert str(res.source_start_date) == "2024-01-02"
    assert res.cross_check is None                        # 无 MACD 列


def test_file_macd_goes_to_crosscheck_only(tmp_path):
    # ★ 文件自带 DIF/DEA/MACD柱 只进 cross-check 列,不作信号源
    path = _write(tmp_path / "dm.csv",
                  "日期,开盘,最高,最低,收盘,成交量,成交额,DIF,DEA,MACD柱",
                  ["2024-01-02,10,10.5,9.8,10.2,1000,1e7,0.11,0.05,0.12",
                   "2024-01-03,10.2,10.9,10.1,10.7,1100,1.1e7,0.18,0.08,0.20"])
    res = load_local_csv(path, level="daily")
    assert list(res.df.columns) == list(OHLCV_COLUMNS)   # 规范 df 不含 MACD
    assert "DIF" not in res.df.columns
    assert res.cross_check is not None
    assert list(res.cross_check.columns) == ["cc_dif", "cc_dea", "cc_macd"]
    assert len(res.cross_check) == 2
    assert res.cross_check.index.equals(res.df.index)


def test_load_intraday_30min(tmp_path):
    path = _write(tmp_path / "m30.csv",
                  "日期时间,开盘,最高,最低,收盘,成交量,成交额",
                  ["2024-01-02 10:00:00,10,10.5,9.8,10.2,500,5e6",
                   "2024-01-02 10:30:00,10.2,10.6,10.1,10.4,400,4e6"])
    res = load_local_csv(path, level="min30")
    assert res.df.shape == (2, 6)
    assert res.df.index.tz is not None
    assert res.df.index[0].hour == 10                     # 日内时间戳保留
    assert res.cross_check is None


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


def test_real_300502_long_loads(tmp_path):
    # 真实文件 smoke:长日线规范加载(无 MACD 列)
    res = load_local_csv("chanlun/data/300502_daily_long.csv", level="daily")
    assert res.df.shape[1] == 6
    validate_canonical(res.df)
    assert res.cross_check is None
