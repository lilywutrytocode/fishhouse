"""模块 5 · 线段 §5.7 七态状态机(特征序列法)。

用合成笔序列直接驱动状态机,精确覆盖:第一种(无缺口)、第二种(带缺口)、
第三笔落在第一笔内的待定、右端 WAIT_SECOND_FEATURE 不得当已完成线段。
"""

from __future__ import annotations

import pandas as pd

from chanlun.structure.inclusion import DOWN, UP
from chanlun.structure.xianduan import (
    Pen,
    SegmentMachine,
    XDState,
    build_segments,
)

_BASE = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")


def mkpens(specs) -> list[Pen]:
    """specs: [(dir, low, high), ...];自动赋递增端点日期(笔终点)。"""
    pens = []
    for i, (d, lo, hi) in enumerate(specs):
        pens.append(Pen(
            direction=d, low=float(lo), high=float(hi), idx=i,
            start_date=_BASE + pd.Timedelta(days=i),
            end_date=_BASE + pd.Timedelta(days=i + 1),
        ))
    return pens


# 上升线段 p0..p4(顶=20 在 p4);特征序列=下降笔
_UP_SEG_HEAD = [
    (UP, 0, 10),     # 0
    (DOWN, 4, 10),   # 1  feature
    (UP, 4, 15),     # 2
    (DOWN, 9, 15),   # 3  feature  (e₁ 区)
    (UP, 9, 20),     # 4  顶=20
]


# ── §5.3 第一种(无缺口)→ CONFIRMED_END ──────────────────────────────────
def test_case1_no_gap_confirms_end():
    # 转折后下降笔回到 13(< 前一回调顶 15)→ e₁–e₂ 无重叠? 实为重叠 → 无缺口
    pens = mkpens(_UP_SEG_HEAD + [
        (DOWN, 13, 20),  # 5  e₂(转折后第一笔),与 e₁[9,15] 重叠 → 无缺口
        (UP, 13, 18),    # 6
        (DOWN, 11, 18),  # 7  C:完成特征序列顶分型
    ])
    m = build_segments(pens)
    assert len(m.confirmed) == 1
    seg = m.confirmed[0]
    assert seg.state == XDState.CONFIRMED_END.value
    assert seg.direction == UP
    assert seg.feeds_zhongshu is True
    assert seg.pivot_price == 20
    assert seg.pivot_date == pens[4].end_date          # 端点极值=p4 顶
    assert seg.confirm_date == pens[7].end_date         # 特征序列分型出现日(C=p7)
    assert seg.confirm_date > seg.pivot_date            # §0.5 右侧确认


# ── §5.3 第二种(有缺口)→ WAIT_SECOND_FEATURE,右端不得当已完成线段 ───────
def test_case2_gap_enters_wait_second_feature_and_not_completed():
    pens = mkpens(_UP_SEG_HEAD + [
        (DOWN, 16, 20),  # 5  e₂ 回到 16(> e₁ 顶 15)→ 有缺口
        (UP, 16, 18),    # 6
        (DOWN, 13, 18),  # 7  完成特征序列顶分型,但有缺口 → 等第二特征序列
    ])
    m = build_segments(pens)
    # 右端处于 WAIT_SECOND_FEATURE:线段尚未完成
    assert m.state == XDState.WAIT_SECOND_FEATURE
    assert len(m.confirmed) == 0                        # ★ 不得当已完成线段
    cur = m.current_segment()
    assert cur.feeds_zhongshu is False                  # 右端非 CONFIRMED_END,不喂中枢
    assert cur.status == "未确认"


def test_case2_gap_confirms_on_second_feature_fractal():
    # 在 WAIT_SECOND_FEATURE 后,第二特征序列(新下降段的上升笔)出底分型 → 确认
    pens = mkpens(_UP_SEG_HEAD + [
        (DOWN, 16, 20),  # 5  缺口
        (UP, 16, 18),    # 6  第二特征序列元素 g1
        (DOWN, 10, 18),  # 7
        (UP, 10, 14),    # 8  g2
        (DOWN, 8, 14),   # 9
        (UP, 8, 12),     # 10 g3(底,最低)
        (DOWN, 9, 12),   # 11
        (UP, 9, 13),     # 12 g4 → 完成第二特征序列底分型(g3 为底)
    ])
    m = build_segments(pens)
    up_seg = next(s for s in m.confirmed if s.direction == UP)
    assert up_seg.state == XDState.CONFIRMED_END.value
    assert up_seg.pivot_price == 20
    assert up_seg.pivot_date == pens[4].end_date
    # 第二种:confirm = 第二特征序列分型出现日(晚于第一种会更晚)
    assert up_seg.confirm_date == pens[12].end_date
    assert up_seg.confirm_date > up_seg.pivot_date
    assert up_seg.feeds_zhongshu is True


# ── §5.7 WAIT_SECOND_FEATURE → INVALIDATED(原方向创新高)──────────────────
def test_wait_second_feature_invalidated_on_new_high():
    pens = mkpens(_UP_SEG_HEAD + [
        (DOWN, 16, 20),  # 5  缺口 → WAIT_SECOND_FEATURE
        (UP, 16, 18),    # 6
        (DOWN, 13, 18),  # 7  进入 WAIT_SECOND_FEATURE
        (UP, 13, 25),    # 8  原方向创新高 25 > 20 → 候选作废,回 EXTENDING
    ])
    m = build_segments(pens)
    assert m.state == XDState.EXTENDING                 # 候选作废后回延伸
    assert len(m.confirmed) == 0                        # 线段未终结,延伸中


# ── §5.6③ 第三笔落在第一笔内 → PENDING_DIRECTION(待定)────────────────────
def test_third_pen_inside_first_pending_direction():
    pens = mkpens([
        (UP, 0, 20),     # 0  第一笔
        (DOWN, 5, 20),   # 1
        (UP, 5, 15),     # 2  第三笔 [5,15] 完全落在第一笔 [0,20] 内
    ])
    m = build_segments(pens)
    assert m.state == XDState.PENDING_DIRECTION
    assert len(m.confirmed) == 0
    cur = m.current_segment()
    assert cur.status == "待定"
    assert cur.feeds_zhongshu is False


def test_pending_direction_resolves_to_extending_on_continuation():
    pens = mkpens([
        (UP, 0, 20),
        (DOWN, 5, 20),
        (UP, 5, 15),     # PENDING_DIRECTION
        (DOWN, 14, 22),  # 先破延续侧:创新高 22 > 20 → EXTENDING(向上延伸)
    ])
    m = build_segments(pens)
    assert m.state == XDState.EXTENDING
    assert m.seg_dir == UP


# ── 累计不足 / 状态枚举完整 ───────────────────────────────────────────────
def test_forming_when_fewer_than_three_pens():
    m = SegmentMachine()
    m.feed(mkpens([(UP, 0, 10)])[0])
    assert m.state == XDState.FORMING
    m.feed(Pen(direction=DOWN, low=4, high=10, start_date=_BASE, end_date=_BASE))
    assert m.state == XDState.FORMING                   # 仍 <3 笔
    assert len(m.confirmed) == 0


def test_only_confirmed_end_feeds_zhongshu():
    # 第一种确认后:历史线段喂中枢,右端在建线段不喂
    pens = mkpens(_UP_SEG_HEAD + [
        (DOWN, 13, 20), (UP, 13, 18), (DOWN, 11, 18),
    ])
    m = build_segments(pens)
    for seg in m.confirmed:
        assert seg.feeds_zhongshu == (seg.state == XDState.CONFIRMED_END.value)
    cur = m.current_segment()
    if cur is not None:
        assert cur.feeds_zhongshu is False
