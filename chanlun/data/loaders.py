"""本地 CSV 加载器(§1.6):中文表头 → 规范 schema + 校验。

支持东财导出的中文表头(``日期/日期时间, 开盘, 最高, 最低, 收盘, 成交量, 成交额``),
以及纯 OHLCV(真实 akshare/yfinance 路径)。

★ 引擎**永远自己按 §1.4 算 MACD**;文件自带的 ``DIF/DEA/MACD柱`` **只读进单独的
cross-check 列**(用于核对),**绝不作为信号源**——否则真实数据(无 MACD)就跑不了。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .models import OHLCV_COLUMNS, to_canonical, validate_canonical

_RENAME = {
    "日期": "date", "日期时间": "date", "时间": "date",
    "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
    "成交量": "volume", "成交额": "amount", "成交金额": "amount",
}
_CC_COLS = ("DIF", "DEA", "MACD柱")


@dataclass
class LoadResult:
    df: pd.DataFrame                 # §1.6 规范 OHLCV(信号唯一来源)
    cross_check: pd.DataFrame | None  # 文件自带 cc_dif/cc_dea/cc_macd(仅核对,非信号源)
    level: str
    source_start_date: date


def load_local_csv(path, *, level: str = "daily",
                   tz: str = "Asia/Shanghai") -> LoadResult:
    """读本地 CSV → 规范 §1.6 OHLCV;文件自带 MACD(若有)进 cross-check 列。"""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    renamed = raw.rename(columns=_RENAME)
    cols = ["date"] + [c for c in OHLCV_COLUMNS if c in renamed.columns]
    canon = to_canonical(renamed[cols], tz=tz)
    validate_canonical(canon)

    cross_check = None
    if all(c in raw.columns for c in _CC_COLS):
        cc = renamed[["date"]].copy()
        cc["cc_dif"] = raw["DIF"].to_numpy()
        cc["cc_dea"] = raw["DEA"].to_numpy()
        cc["cc_macd"] = raw["MACD柱"].to_numpy()
        cc["date"] = pd.to_datetime(cc["date"])
        cc = cc.set_index("date")
        cc.index = (cc.index.tz_localize(tz) if cc.index.tz is None
                    else cc.index.tz_convert(tz))
        cc.index.name = "date"
        cross_check = cc.reindex(canon.index)   # 对齐规范索引(只读核对)

    return LoadResult(df=canon, cross_check=cross_check, level=level,
                      source_start_date=canon.index[0].date())
