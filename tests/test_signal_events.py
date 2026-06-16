"""#5 事件流最小闭环:confirmed-only + 无 pivot 触发 + §6.7 去重 + §8.6 二三买重合。"""

from __future__ import annotations

import pandas as pd

from chanlun.cli import analyze, assemble_signal_events, run_pipeline
from chanlun.config import DEFAULT_CONFIG
from chanlun.data.loaders import load_local_csv
from chanlun.output import REQUIRED_EVENT_FIELDS, output_schema_complete
from chanlun.probability import to_backtest_triggers
from chanlun.structure.maimaidian import MaiMaiDian

_000001 = "chanlun/data/raw/000001/000001_sh_daily_20170601_20190630_ohlcv.csv"
_TZ = "Asia/Shanghai"


def _mmd(kind, side, *, pivot, pday, cday, cprice, exe, gid, grade=None, subkind=None):
    return MaiMaiDian(
        kind=kind, side=side, level="daily", status="confirmed", subkind=subkind,
        pivot_date=pd.Timestamp(pday, tz=_TZ), pivot_price=pivot,
        confirm_date=pd.Timestamp(cday, tz=_TZ), confirm_price=cprice,
        executable_price=exe, beichi_grade=grade, id=gid)


def _df(dates):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=_TZ) for d in dates], name="date")
    return pd.DataFrame({"close": [1.0] * len(dates)}, index=idx)


# ── 事件流只来自 confirmed,触发无 pivot,末根 live_pending 不进 ─────────────
def test_event_stream_confirmed_only_no_pivot_trigger():
    df = load_local_csv(_000001, level="daily").df
    r = run_pipeline(df, symbol="000001")
    ev = r["signal_events"]
    assert ev
    for e in ev:
        assert e.confirm_date is not None and e.executable_price is not None
    triggers = to_backtest_triggers(ev)
    assert len(triggers) == len(ev)                 # confirmed 事件均可触发
    for t in triggers:
        assert "confirm_date" in t and "executable_price" in t
        assert not any("pivot" in k for k in t)     # ★ 触发不含 pivot_*
    # 待确认/无 executable 的买卖点不进事件流
    pending = [m for m in r["maimaidians"]
               if m.confirm_date is None or m.executable_price is None]
    ev_ids = {e.id for e in ev}
    assert all(m.id not in ev_ids for m in pending)


# ── §6.7 同级别同向同类 → 去重(其余进 supporting_signals)──────────────────
def test_event_stream_dedupe_same_cluster():
    df = _df(["2024-03-01", "2024-03-04", "2024-03-05"])
    b1 = _mmd("一买", "buy", pivot=99, pday="2024-02-28", cday="2024-03-01",
              cprice=100.0, exe=100.0, gid="signal_d_001", grade="DIF背驰")
    b2 = _mmd("一买", "buy", pivot=99, pday="2024-02-28", cday="2024-03-05",
              cprice=100.5, exe=100.5, gid="signal_d_002", grade="标准背驰")
    ev = assemble_signal_events([b1, b2], symbol="X", df=df, config=DEFAULT_CONFIG)
    assert len(ev) == 1                              # 3 根内、1% 内、同类 → 合并
    assert ev[0].id == "signal_d_002"                # 标准背驰优先为触发
    assert "signal_d_001" in ev[0].supporting_signals


# ── §8.6 二三买重合 → overlap + 互入 supporting_signals ────────────────────
def test_event_stream_overlap_2_3():
    df = _df(["2024-03-03"])
    s2 = _mmd("二买", "buy", pivot=10.0, pday="2024-03-01", cday="2024-03-03",
              cprice=10.5, exe=10.6, gid="signal_d_001")
    s3 = _mmd("三买", "buy", pivot=10.0, pday="2024-03-01", cday="2024-03-03",
              cprice=10.5, exe=10.6, gid="signal_d_002")
    ev = assemble_signal_events([s2, s3], symbol="X", df=df, config=DEFAULT_CONFIG)
    assert {e.id for e in ev} == {"signal_d_001", "signal_d_002"}   # 不同类不合并
    assert s2.overlap_2_3 is True and s3.overlap_2_3 is True
    by = {e.id: e for e in ev}
    assert "signal_d_002" in by["signal_d_001"].supporting_signals
    assert "signal_d_001" in by["signal_d_002"].supporting_signals


# ── #2 输出 schema 完整性:signal_events / label / min30_consistency / macd_warmup ─
def test_output_schema_includes_event_stream_and_new_keys():
    df = load_local_csv(_000001, level="daily").df
    out = analyze(df, symbol="000001")
    assert output_schema_complete(out)
    for k in ("signal_events", "min30_consistency", "macd_warmup"):
        assert k in out
    assert out["signal_events"]                       # 非空
    for e in out["signal_events"]:
        assert all(f in e for f in REQUIRED_EVENT_FIELDS)
    for m in out["mai_mai_dian"]:
        assert "label" in m
    # 事件流里 2019-01-04 标准趋势一买在
    assert any(e["confirm_date"][:7] == "2019-01" and e["kind"] == "一买"
               for e in out["signal_events"])
