"""§1.2 快照落盘/读取 + data_snapshot_id 复现性。"""

from __future__ import annotations

from datetime import date

from chanlun.data.snapshot import (
    compute_snapshot_id,
    load_snapshot,
    save_snapshot,
)
from tests.conftest import make_daily, weekdays


def test_snapshot_id_is_content_deterministic():
    days = weekdays(date(2024, 1, 1), 20)
    df = make_daily(days)
    kw = dict(symbol="300502", market="A", level="daily",
              source="akshare", adjust="qfq")
    id1 = compute_snapshot_id(df, **kw)
    id2 = compute_snapshot_id(df.copy(), **kw)
    assert id1 == id2  # 同内容 → 同 id


def test_snapshot_id_changes_with_data():
    days = weekdays(date(2024, 1, 1), 20)
    df = make_daily(days)
    kw = dict(symbol="300502", market="A", level="daily",
              source="akshare", adjust="qfq")
    id1 = compute_snapshot_id(df, **kw)
    df2 = df.copy()
    df2.iloc[5, df2.columns.get_loc("close")] += 0.01  # 行情变了
    id2 = compute_snapshot_id(df2, **kw)
    assert id1 != id2


def test_snapshot_id_changes_with_source():
    days = weekdays(date(2024, 1, 1), 20)
    df = make_daily(days)
    base = dict(symbol="300502", market="A", level="daily", adjust="qfq")
    assert (
        compute_snapshot_id(df, source="akshare", **base)
        != compute_snapshot_id(df, source="eastmoney", **base)
    )


def test_save_and_load_roundtrip(tmp_path):
    days = weekdays(date(2024, 1, 1), 20)
    df = make_daily(days)
    meta = save_snapshot(
        df, symbol="300502", market="A", level="daily", source="akshare",
        adjust="qfq", tz="Asia/Shanghai", fetch_date=date(2024, 2, 1),
        root=tmp_path,
    )
    assert meta.row_count == 20
    assert meta.data_snapshot_id

    base = tmp_path / f"A_300502_daily_2024-02-01_{meta.data_snapshot_id}"
    loaded, loaded_meta = load_snapshot(base.with_suffix(".parquet"))
    assert loaded_meta.data_snapshot_id == meta.data_snapshot_id
    assert len(loaded) == len(df)
    # 复权口径与拉取日随快照固定,保证可复现
    assert loaded_meta.adjust == "qfq"
    assert loaded_meta.fetch_date == "2024-02-01"
