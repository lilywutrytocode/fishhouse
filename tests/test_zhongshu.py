"""模块 6 · 中枢(§6.1–6.7)。"""

from __future__ import annotations

import pandas as pd

from chanlun.config import Config
from chanlun.structure.zhongshu import (
    BI,
    XIANDUAN,
    SignalEvent,
    ZUnit,
    Zhongshu,
    build_zhongshu,
    classify_relation,
    dedupe_event_cluster,
)

_BASE = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")


def mkunits(specs) -> list[ZUnit]:
    """specs: [(high, low), ...];自动赋递增 start/confirm 日期与 bar。"""
    units = []
    for i, (hi, lo) in enumerate(specs):
        units.append(ZUnit(
            high=float(hi), low=float(lo),
            start_date=_BASE + pd.Timedelta(days=i),
            start_price=(hi + lo) / 2,
            confirm_date=_BASE + pd.Timedelta(days=i + 1),
            confirm_price=(hi + lo) / 2,
            confirm_bar=i,
            id=f"u{i}",
        ))
    return units


# ── §6.1/6.2 成立 ─────────────────────────────────────────────────────────
def test_zhongshu_forms_zg_min_high_zd_max_low():
    units = mkunits([(12, 8), (11, 7), (13, 9)])
    zs = build_zhongshu(units, kind=BI)
    assert len(zs) == 1
    z = zs[0]
    assert z.ZG == 11          # min(12,11,13)
    assert z.ZD == 9           # max(8,7,9)
    assert z.ZD <= z.ZG
    assert z.n_segments == 3
    # §0.5:pivot=首段起点,confirm=第三段成立
    assert z.pivot_date == units[0].start_date
    assert z.confirm_date == units[2].confirm_date
    assert z.confirm_date > z.pivot_date


def test_no_zhongshu_when_three_units_dont_overlap():
    # ZD > ZG → 不成立
    units = mkunits([(12, 11), (10, 9), (8, 7)])
    assert build_zhongshu(units, kind=BI) == []


# ── §6.3 延伸 >9 段:ZG/ZD 固定、GG/DD 刷新 ────────────────────────────────
def test_extension_beyond_9_segments_fixes_zgzd_refreshes_ggdd():
    specs = [
        (12, 8), (11, 7), (13, 9),   # 形成 → ZG=11, ZD=9
        (15, 9.5),                   # GG 抬到 15
        (10.5, 6),                   # DD 压到 6
        (11, 9), (10, 9.2), (11, 9), (10.5, 9.1),
        (16, 9.3),                   # GG 抬到 16
        (10.2, 5),                   # DD 压到 5
        (11, 9), (10, 9.5),          # 共 13 段
    ]
    units = mkunits(specs)
    zs = build_zhongshu(units, kind=BI)
    assert len(zs) == 1
    z = zs[0]
    assert z.n_segments == 13 and z.n_segments > 9
    # ZG/ZD 由前三段固定,不随延伸变
    assert z.ZG == 11 and z.ZD == 9
    # GG/DD 随延伸刷新为全体极值
    assert z.GG == max(h for h, _ in specs)   # 16
    assert z.DD == min(l for _, l in specs)   # 5
    assert z.extending is True                # 右端仍在延伸(未破坏)


# ── §6.5 破坏(定理三)────────────────────────────────────────────────────
def test_theorem3_break_when_leave_and_pullback_fails():
    # 前三成立 [9,11];u3 整段在上(low12>11)离开,u4 仍在上不回 → 破坏
    units = mkunits([(12, 8), (11, 7), (13, 9), (14, 12), (15, 13)])
    zs = build_zhongshu(units, kind=BI)
    assert zs[0].n_segments == 3            # 中枢只含前三段
    assert zs[0].broken is True
    assert zs[0].extending is False


def test_single_leave_with_pullback_returns_extends():
    # u3 离开(low12>11)但 u4 回到 [9,11] → 单次离开计入延伸,中枢继续
    units = mkunits([(12, 8), (11, 7), (13, 9), (14, 12), (11, 9.5)])
    zs = build_zhongshu(units, kind=BI)
    assert zs[0].n_segments == 5
    assert zs[0].GG == 14                   # 离开段的高计入 GG
    assert zs[0].ZG == 11 and zs[0].ZD == 9 # 核心不变


# ── §6.4 扩展(定理二)────────────────────────────────────────────────────
def test_theorem2_classification():
    z1 = Zhongshu(kind=BI, level="daily", ZG=11, ZD=9, GG=14, DD=6,
                  start_unit=0, end_unit=2, n_segments=3,
                  pivot_date=_BASE, pivot_price=10,
                  confirm_date=_BASE + pd.Timedelta(days=3), confirm_price=10,
                  executable_price=None)
    z_up = Zhongshu(kind=BI, level="daily", ZG=20, ZD=16, GG=22, DD=15,
                    start_unit=3, end_unit=5, n_segments=3,
                    pivot_date=_BASE + pd.Timedelta(days=4), pivot_price=18,
                    confirm_date=_BASE + pd.Timedelta(days=7), confirm_price=18,
                    executable_price=None)
    assert classify_relation(z1, z_up) == "上涨延续"   # 后 DD15 > 前 GG14
    z_dn = Zhongshu(kind=BI, level="daily", ZG=5, ZD=2, GG=5, DD=1,
                    start_unit=3, end_unit=5, n_segments=3,
                    pivot_date=_BASE + pd.Timedelta(days=4), pivot_price=3,
                    confirm_date=_BASE + pd.Timedelta(days=7), confirm_price=3,
                    executable_price=None)
    assert classify_relation(z1, z_dn) == "下跌延续"   # 后 GG5 < 前 DD6


# ── 线段中枢与笔中枢都做 ─────────────────────────────────────────────────
def test_xianduan_zhongshu_same_logic():
    units = mkunits([(12, 8), (11, 7), (13, 9)])
    zs = build_zhongshu(units, level="daily", kind=XIANDUAN)
    assert len(zs) == 1 and zs[0].kind == XIANDUAN
    assert zs[0].id.startswith("zhongshu_xi_d_")


# ── §6.7 去重事件簇 ───────────────────────────────────────────────────────
def _sig(id, *, bar, price=100.0, level="daily", direction="buy", kind="一买",
         source_rank=0, beichi_rank=0, level30_rank=0):
    return SignalEvent(
        id=id, symbol="300502", level=level, direction=direction, kind=kind,
        confirm_date=_BASE + pd.Timedelta(days=bar), confirm_price=price,
        confirm_bar=bar, source_rank=source_rank, beichi_rank=beichi_rank,
        level30_rank=level30_rank,
    )


def test_dedupe_same_cluster_keeps_segment_zhongshu_trigger():
    # 同级别同向同类、≤3根K、≤1%:线段中枢(source_rank2)优先于笔中枢(1)
    bi = _sig("bi1", bar=10, price=100.0, source_rank=1)
    xd = _sig("xd1", bar=12, price=100.5, source_rank=2)
    out = dedupe_event_cluster([bi, xd])
    assert len(out) == 1
    assert out[0].id == "xd1"                         # 线段中枢触发
    assert out[0].supporting_signals == ["bi1"]       # 笔中枢进 supporting


def test_dedupe_cross_level_not_merged():
    # 同类同向但不同级别 → 不同事件,都保留(★ 跨级别绝不合并)
    d = _sig("d1", bar=10, level="daily")
    w = _sig("w1", bar=11, level="weekly")
    out = dedupe_event_cluster([d, w])
    assert len(out) == 2
    assert all(not s.supporting_signals for s in out)


def test_dedupe_earliest_confirm_triggers_when_same_priority():
    a = _sig("a", bar=10, price=100.0, source_rank=1)
    b = _sig("b", bar=12, price=100.2, source_rank=1)
    out = dedupe_event_cluster([a, b])
    assert len(out) == 1
    assert out[0].id == "a"                           # 同优先级 → confirm 早者触发
    assert out[0].supporting_signals == ["b"]


def test_dedupe_outside_thresholds_kept_separate():
    # 间隔 >3 根K → 不同簇
    a = _sig("a", bar=10)
    b = _sig("b", bar=20)
    out = dedupe_event_cluster([a, b])
    assert len(out) == 2
    # 价格相差 >1% → 不同簇
    c = _sig("c", bar=10, price=100.0)
    d = _sig("d", bar=11, price=105.0)
    out2 = dedupe_event_cluster([c, d])
    assert len(out2) == 2


def test_dedupe_standard_beichi_beats_panbei_when_same_source():
    std = _sig("std", bar=11, source_rank=1, beichi_rank=2)
    pan = _sig("pan", bar=10, source_rank=1, beichi_rank=1)
    out = dedupe_event_cluster([std, pan])
    assert len(out) == 1
    assert out[0].id == "std"                         # 标准背驰优先于盘背(即便确认稍晚)
    assert out[0].supporting_signals == ["pan"]
