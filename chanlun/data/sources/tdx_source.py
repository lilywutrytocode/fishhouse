"""通达信 TDX 源适配器(A 股短周期主源,§免费数据源)。

惰性 import ``pytdx``。本环境未安装 / 连不上服务器时,实例化不报错,
仅在 :meth:`fetch` 调用时抛 :class:`FetchError`。

★ TDX 日线/分钟线为**不复权(raw)**:``adjust="raw"``;长周期前复权结构请用 qfq 源。
本轮支持 ``market="A"`` + ``level in {daily, min30}``;个股走 get_security_bars,
指数(sh000xxx / sz399xxx)走 get_index_bars。
"""

from __future__ import annotations

import pandas as pd

from ..models import to_canonical
from ..symbols import normalize_cn_symbol
from .base import FetchError, SourceResult

# TDX K 线类别:9=日K线,2=30分钟K线
_CATEGORY = {"daily": 9, "min30": 2}

# 免费行情服务器候选(首个可连即用)
_SERVERS = [
    ("119.147.212.81", 7709),
    ("60.12.136.250", 7709),
    ("218.108.98.244", 7709),
    ("123.125.108.14", 7709),
]
_PAGE = 800        # TDX 单次最多 800 根
_MAX_PAGES = 60    # 安全上限(≈ 48000 根)


class TdxSource:
    name = "tdx"

    def fetch(self, symbol: str, market: str, level: str, *,
              start: str | None = None, end: str | None = None) -> SourceResult:
        if market != "A":
            raise FetchError(f"tdx 仅支持 A 股(market=A),收到 {market}")
        if level not in _CATEGORY:
            raise FetchError(f"tdx 仅支持 daily/min30,收到 {level}")
        try:
            from pytdx.hq import TdxHq_API
        except Exception as e:  # pragma: no cover - 依赖缺失
            raise FetchError(f"pytdx 不可用:{e}") from e

        sym = normalize_cn_symbol(symbol)
        cat = _CATEGORY[level]
        api = TdxHq_API(raise_exception=True)

        srv = self._connect(api)
        try:
            rows = self._page_all(api, cat, sym, level)
        except Exception as e:  # pragma: no cover - 网络/解析
            raise FetchError(f"tdx 拉取失败({srv}):{e}") from e
        finally:
            try:
                api.disconnect()
            except Exception:  # pragma: no cover
                pass

        if not rows:
            raise FetchError(f"tdx 返回空数据:{sym.exch_symbol} {level}")

        df = self._to_frame(rows, level)
        df = self._crop(df, start, end, level)
        if df.empty:
            raise FetchError(
                f"tdx 区间内无数据:{sym.exch_symbol} {level} {start}~{end}")
        canon = to_canonical(df, tz="Asia/Shanghai")
        return SourceResult(df=canon, source=self.name, adjust="raw")

    # ── 内部 ────────────────────────────────────────────────────────────────
    def _connect(self, api):  # pragma: no cover - 联网
        last = None
        for ip, port in _SERVERS:
            try:
                if api.connect(ip, port):
                    return f"{ip}:{port}"
            except Exception as e:
                last = e
        raise FetchError(f"tdx 无法连接任一行情服务器:{last}")

    def _page_all(self, api, cat: int, sym, level: str) -> list[dict]:  # pragma: no cover
        getter = api.get_index_bars if sym.is_index else api.get_security_bars
        out: list[dict] = []
        for p in range(_MAX_PAGES):
            chunk = getter(cat, sym.tdx_market, sym.code, p * _PAGE, _PAGE)
            if not chunk:
                break
            out = list(chunk) + out          # chunk 越早页越旧 → 前置拼接
            if len(chunk) < _PAGE:
                break
        return out

    def _to_frame(self, rows: list[dict], level: str) -> pd.DataFrame:
        recs = []
        for r in rows:
            dt = str(r.get("datetime") or "")
            recs.append({
                "date": dt,
                "open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r.get("vol", 0) or 0),
                "amount": float(r.get("amount", 0) or 0),   # 缺失填 0
            })
        df = pd.DataFrame(recs)
        df["date"] = pd.to_datetime(df["date"])
        if level == "daily":
            df["date"] = df["date"].dt.normalize()           # 日线去掉时分
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _crop(df: pd.DataFrame, start: str | None, end: str | None,
              level: str) -> pd.DataFrame:
        if start:
            df = df[df["date"] >= pd.to_datetime(start)]
        if end:
            # end 当日含全天(min30 也算到 23:59)
            df = df[df["date"] <= pd.to_datetime(end) + pd.Timedelta(days=1)
                    - pd.Timedelta(minutes=1)]
        return df.reset_index(drop=True)
