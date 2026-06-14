"""§1.2 行情快照:落盘 / 读取 + 内容派生的 ``data_snapshot_id``。

快照(带拉取日)只读,保证可复现。``data_snapshot_id`` 由元信息 + 规范化后的
内容哈希派生 —— 同一标的不同时间跑出不同结果时,可区分"行情变了"还是"快照变了"。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pandas as pd

from .models import OHLCV_COLUMNS, SnapshotMeta, validate_canonical


def compute_snapshot_id(
    df: pd.DataFrame, *, symbol: str, market: str, level: str,
    source: str, adjust: str,
) -> str:
    """对规范 df + 关键元信息求 sha256,取前 16 位作为快照 id。

    内容哈希用规范化 CSV(固定列序、UTC、ISO 时间、float 统一格式),保证
    同样的数据 → 同样的 id,与读写顺序/平台无关。
    """
    norm = df.copy()
    norm = norm[list(OHLCV_COLUMNS)]
    norm.index = norm.index.tz_convert("UTC")
    payload = norm.to_csv(date_format="%Y-%m-%dT%H:%M:%S%z", float_format="%.10g")
    h = hashlib.sha256()
    h.update(f"{symbol}|{market}|{level}|{source}|{adjust}|".encode())
    h.update(payload.encode())
    return h.hexdigest()[:16]


def save_snapshot(
    df: pd.DataFrame, *, symbol: str, market: str, level: str, source: str,
    adjust: str, tz: str, fetch_date: date, root: str | Path,
) -> SnapshotMeta:
    """把规范 df 落盘为快照(parquet 数据 + json 元信息),返回 :class:`SnapshotMeta`。"""
    validate_canonical(df)
    snapshot_id = compute_snapshot_id(
        df, symbol=symbol, market=market, level=level, source=source, adjust=adjust,
    )
    meta = SnapshotMeta(
        symbol=symbol, market=market, level=level, source=source, adjust=adjust,
        fetch_date=fetch_date.isoformat(), tz=tz, row_count=len(df),
        first_date=df.index[0].isoformat() if len(df) else None,
        last_date=df.index[-1].isoformat() if len(df) else None,
        data_snapshot_id=snapshot_id,
    )
    base = Path(root) / f"{market}_{symbol}_{level}_{fetch_date.isoformat()}_{snapshot_id}"
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base.with_suffix(".parquet"))
    base.with_suffix(".meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return meta


def load_snapshot(path: str | Path) -> tuple[pd.DataFrame, SnapshotMeta]:
    """读取快照(传 parquet 路径或不带后缀的 base 路径)。返回 (只读 df, meta)。"""
    p = Path(path)
    base = p.with_suffix("") if p.suffix in (".parquet", ".json") else p
    if base.name.endswith(".meta"):
        base = base.with_suffix("")
    df = pd.read_parquet(base.with_suffix(".parquet"))
    validate_canonical(df)
    meta_dict = json.loads(base.with_suffix(".meta.json").read_text(encoding="utf-8"))
    df.flags.allows_duplicate_labels = False  # 只读语义提示
    return df, SnapshotMeta(**meta_dict)
