"""模块 4 · 笔(§4.1–4.4 + §0.5 防未来函数)。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chanlun.data.models import OHLCV_COLUMNS
from chanlun.structure.bi import (
    CONFIRMED,
    FORMING,
    MIN_MID_K_GAP,
    build_bi,
    build_bi_from_df,
)
from chanlun.structure.fractal import detect_fractals
from chanlun.structure.inclusion import DOWN, UP, process_inclusion
from tests.conftest import weekdays


def wave(cs, tz="Asia/Shanghai") -> pd.DataFrame:
    """由中心值序列 cs 构造规范日线:high=c+1, low=c-1。

    cs 严格无相邻相等 → 标准 K == 原始 K(无包含),分型落在 cs 的局部极值,
    标准 K 索引 == 原始索引,便于精确断言 pivot/confirm/gap。
    """
    days = weekdays(date(2024, 1, 1), len(cs))
    rows = [
        {"open": c, "high": c + 1, "low": c - 1, "close": c,
         "volume": 100, "amount": 1.0}
        for c in cs
    ]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in days], name="date"
    ).tz_localize(tz)
    return df


# ── §4.1/4.2 基础:交替成笔,方向/端点正确 ────────────────────────────────
def test_basic_alternating_bis():
    # 顶@4 底@8 顶@12,间距均为 4(≥5 根标准 K);末根回落 → 右端在形成一笔下降笔
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    bis = build_bi_from_df(df)
    # 两笔历史(顶4→底8、底8→顶12)+ 顶12→右端的在途下降笔
    assert len(bis) == 3

    b0 = bis[0]
    assert b0.direction == DOWN          # 顶@4 → 底@8
    assert b0.start_k == 4 and b0.end_k == 8
    assert b0.pivot_date == df.index[8]  # 端点极值=末端底
    assert b0.pivot_price == -1          # low@8 = 0-1
    assert b0.status == CONFIRMED

    b1 = bis[1]
    assert b1.direction == UP            # 底@8 → 顶@12
    assert b1.status == CONFIRMED        # 被后续反向运动锁定

    b2 = bis[2]
    assert b2.direction == DOWN          # 顶@12 → 右端待定底
    assert b2.status == FORMING          # 右端未确认/延伸


def test_gap_constant_matches_spec():
    # ②③ 合并为中间 K 间距 ≥ 4(= ≥5 根标准 K)
    assert MIN_MID_K_GAP == 4


# ── §0.5 confirm_date > pivot_date,且 confirm 来自末端分型第三根 K ─────────
def test_confirmed_bi_confirm_after_pivot():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    bis = build_bi_from_df(df)
    confirmed = [b for b in bis if b.status == CONFIRMED]
    assert confirmed
    for b in confirmed:
        assert b.confirm_date is not None
        assert b.confirm_date > b.pivot_date          # 右侧确认,严禁用 pivot 触发
        assert b.executable_price is not None
    # 底@8 的 confirm = 第三根标准 K@9 收盘
    assert confirmed[0].confirm_date == df.index[9]


def test_forming_bi_has_no_confirm_or_executable():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0])
    bis = build_bi_from_df(df)
    last = bis[-1]
    assert last.status == FORMING
    assert last.confirm_date is None
    assert last.confirm_price is None
    assert last.executable_price is None


# ── §4.3 步骤③ 顺延 + 连续同类去重(取价格极值)──────────────────────────
def test_suspend_and_dedupe_reselects_lower_bottom():
    # 底@1(c=2)→ 顶@3 太近(间距2<4)被顺延 → 底@5(c=0 更低)去重重选为起点
    df = wave([6, 2, 4, 7, 4, 0, 3, 6, 9, 12, 9, 6, 3, 0, 3, 6, 9, 12, 9])
    merged = process_inclusion(df)
    fxs = detect_fractals(merged, df)
    bis = build_bi(fxs, merged)

    # 第一笔起点应是更低的底@5,而非底@1;太近的顶@3 不作端点
    assert bis[0].start_k == 5
    all_ks = {b.start_k for b in bis} | {b.end_k for b in bis}
    assert 3 not in all_ks               # 被顺延掉的顶@3 不出现
    assert 1 not in all_ks               # 被更低底@5 去重替代

    # 确认端点序列:底5 → 顶9 → 底13 → 顶17(三笔历史) + 顶17→右端在途下降笔
    confirmed = [(b.start_k, b.end_k) for b in bis if b.status == CONFIRMED]
    assert confirmed == [(5, 9), (9, 13), (13, 17)]
    assert bis[-1].status == FORMING
    assert bis[-1].start_k == 17         # 在途笔自末端顶@17 起


def test_tie_break_keeps_earliest_bottom_pivot_not_advancing_confirm():
    # 底@1 与 底@5 同价(c=0)→ 同价取最先(底@1),只影响 pivot 选择
    df = wave([6, 0, 3, 6, 3, 0, 3, 6, 9, 12, 9, 6, 3, 6, 9])
    merged = process_inclusion(df)
    fxs = detect_fractals(merged, df)
    bis = build_bi(fxs, merged)

    # 顶@3 太近被顺延;两个同价底去重取最先 → 起点为底@1
    assert bis[0].start_k == 1
    assert bis[0].start_date == df.index[1]
    # 起端底@1 自身的分型 confirm 未被提前(仍是其第三根 K@2)
    b1_fx = next(f for f in fxs if f.mid_k == 1)
    assert b1_fx.confirm_date == df.index[2]


# ── §4.4 右端未确认:在途笔抵达右端待定极值 ───────────────────────────────
def test_right_end_forming_bi_reaches_tip():
    # 顶@4 → 底@8(确认下降笔)→ 右端持续走高未出确认顶 → 在途上升笔抵达右端待定顶
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5])
    bis = build_bi_from_df(df)
    assert bis[0].status == CONFIRMED and bis[0].direction == DOWN  # 顶4→底8 已确认
    last = bis[-1]
    assert last.direction == UP
    assert last.status == FORMING
    # 在途上升笔的 pivot 落在右端待定顶(c=5 → high=6),而非中途的 c=4
    assert last.pivot_price == 6
    assert last.pivot_date == df.index[-1]


def test_too_few_fractals_only_forming_or_empty():
    # 单调上行 → 无确认笔;若有反向在途则一笔 forming,否则空
    df = wave([0, 1, 2, 3, 4, 5, 6])
    bis = build_bi_from_df(df)
    assert all(b.status == FORMING for b in bis)
    assert len(bis) <= 1


# ── 不变量:确认笔不靠 pivot 触发(executable 来自 confirm 之后)───────────
def test_no_confirmed_bi_triggers_on_pivot():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0,
               1, 2, 3, 4, 3, 2, 1, 0])
    bis = build_bi_from_df(df)
    for b in bis:
        if b.status == CONFIRMED:
            assert b.confirm_date is not None and b.confirm_date > b.pivot_date
            assert b.executable_price is not None
        else:
            assert b.executable_price is None
