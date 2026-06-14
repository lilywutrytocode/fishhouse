"""模块 11 · 输出 schema 与版本化 + 监控位 + CLI(§11.1–11.4)。"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from chanlun.cli import analyze, format_report
from chanlun.config import DEFAULT_CONFIG
from chanlun.data.models import OHLCV_COLUMNS
from chanlun.monitor import CAUTION, REASSESSMENT, TARGET, derive_monitor_levels
from chanlun.output import (
    REQUIRED_TOP_KEYS,
    algorithm_config_hash,
    build_output,
    output_schema_complete,
    to_json,
)
from chanlun.structure.maimaidian import ABOVE, BELOW, INSIDE
from chanlun.structure.zhongshu import ZUnit, build_zhongshu
from tests.conftest import weekdays


def wave(cs, tz="Asia/Shanghai") -> pd.DataFrame:
    days = weekdays(date(2024, 1, 1), len(cs))
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 100, "amount": 1.0} for c in cs]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in days],
                                name="date").tz_localize(tz)
    return df


# ── §11.1 版本化 + schema ─────────────────────────────────────────────────
def test_output_top_level_schema_and_versioning():
    out = build_output(symbol="300502")
    assert output_schema_complete(out)
    assert all(k in out for k in REQUIRED_TOP_KEYS)
    assert out["spec_version"] == "v1.2"
    assert out["engine_version"]
    assert len(out["algorithm_config_hash"]) == 16


def test_config_hash_changes_with_config():
    from dataclasses import replace
    h1 = algorithm_config_hash(DEFAULT_CONFIG)
    h2 = algorithm_config_hash(replace(DEFAULT_CONFIG, beichi_k=0.8))
    assert h1 != h2                       # 规格/代码阈值变 → hash 变


def test_output_is_json_serializable():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    out = analyze(df, symbol="300502", level="daily")
    s = to_json(out)
    parsed = json.loads(s)               # Timestamp 等已转 ISO,可解析
    assert parsed["symbol"] == "300502"


# ── §11.2 监控位上下文提示语 ──────────────────────────────────────────────
def _zs(zd=10.0, zg=14.0, gg=18.0, id="zs1"):
    from types import SimpleNamespace
    return SimpleNamespace(ZD=zd, ZG=zg, GG=gg, id=id)


def test_monitor_above_zhongshu_hint():
    levels = derive_monitor_levels(current_price=16.0, zhongshu=_zs())
    caution = [m for m in levels if m.tier == CAUTION][0]
    assert caution.price == 14.0         # ZG
    assert "三买" in caution.hint and "不等于三卖" in caution.hint


def test_monitor_below_zhongshu_hint():
    levels = derive_monitor_levels(current_price=8.0, zhongshu=_zs())
    caution = [m for m in levels if m.tier == CAUTION][0]
    assert caution.price == 10.0         # ZD
    assert "三卖" in caution.hint


def test_monitor_inside_no_3buy_3sell():
    levels = derive_monitor_levels(current_price=12.0, zhongshu=_zs())
    for m in levels:
        if m.tier == CAUTION:
            assert "仅震荡边界" in m.hint and "不出三买/三卖" in m.hint


def test_monitor_reassessment_and_target():
    levels = derive_monitor_levels(
        current_price=16.0, zhongshu=_zs(), recent_first_buy_low=6.0, prev_high=20.0)
    tiers = {m.tier for m in levels}
    assert REASSESSMENT in tiers and TARGET in tiers
    rea = [m for m in levels if m.tier == REASSESSMENT][0]
    assert "一买失效" in rea.hint


# ── §11.3/11.4 CLI analyze + 右端未确认 ───────────────────────────────────
def test_analyze_full_pipeline_keys():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    out = analyze(df, symbol="300502", level="daily")
    assert output_schema_complete(out)
    assert len(out["bi"]) >= 1
    # §11.4 右端未确认笔显式标 forming
    assert any(b["status"] == "forming" for b in out["bi"])
    # 所有笔带 §0.6 通用字段
    for b in out["bi"]:
        for f in ("id", "parent_id", "source_unit_ids", "level", "direction",
                  "status", "pivot_date", "pivot_price"):
            assert f in b


def test_format_report_mentions_pending():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    out = analyze(df, symbol="300502")
    report = format_report(out)
    assert "标的: 300502" in report
    assert "未确认" in report or "笔:" in report


def test_zhongshu_serialized_with_zg_zd_gg_dd():
    units = [ZUnit(high=12, low=8, start_date=pd.Timestamp("2024-01-01", tz="UTC"),
                   start_price=10, confirm_date=pd.Timestamp("2024-01-02", tz="UTC"),
                   confirm_price=10, id=f"u{i}") for i in range(3)]
    units[1].high, units[1].low = 11, 7
    units[2].high, units[2].low = 13, 9
    zs = build_zhongshu(units, kind="bi")
    out = build_output(symbol="x", zhongshu=zs)
    z = out["zhongshu"][0]
    for f in ("ZG", "ZD", "GG", "DD", "n_segments", "source_unit_ids", "parent_id"):
        assert f in z
