"""模块 2 · 包含关系(§2.1–2.5)。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chanlun.data.models import OHLCV_COLUMNS
from chanlun.structure.inclusion import (
    DOWN,
    UP,
    initial_direction,
    is_contained,
    merged_to_frame,
    process_inclusion,
)
from tests.conftest import weekdays


def bars(hl, tz="Asia/Shanghai") -> pd.DataFrame:
    """由 (high, low) 列表构造规范日线(open/close 取中点,保证 high≥low)。"""
    days = weekdays(date(2024, 1, 1), len(hl))
    rows = [
        {"open": (h + l) / 2, "high": h, "low": l, "close": (h + l) / 2,
         "volume": 100, "amount": 1.0}
        for h, l in hl
    ]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in days], name="date"
    ).tz_localize(tz)
    return df


def assert_no_adjacent_containment(merged):
    for a, b in zip(merged, merged[1:]):
        assert not is_contained(a.high, a.low, b.high, b.low), (
            f"标准 K 间不应有包含:{(a.high, a.low)} vs {(b.high, b.low)}"
        )


# ── 基础 ────────────────────────────────────────────────────────────────
def test_is_contained_basic():
    assert is_contained(12, 9, 11, 9.5)      # 前含后
    assert is_contained(11, 9.5, 12, 9)      # 后含前(对称)
    assert not is_contained(12, 9, 11, 8)    # 互不含(后者更低)
    # §2.5 单边等高仍算包含
    assert is_contained(10, 8, 10, 7)


def test_non_containment_passthrough():
    # 全程非包含 → 标准 K 数量等于原始,值不变
    df = bars([(10, 8), (12, 9), (14, 11), (16, 13)])
    merged = process_inclusion(df)
    assert len(merged) == 4
    assert [m.high for m in merged] == [10, 12, 14, 16]
    assert all(m.direction == UP for m in merged[1:])
    assert_no_adjacent_containment(merged)


# ── §2.1 取值 ─────────────────────────────────────────────────────────────
def test_up_merge_takes_max_low_max_high():
    # b0,b1 上行定方向 up;b2 被 b1 包含 → [max低, max高]
    df = bars([(10, 8), (12, 9), (11, 9.5)])
    merged = process_inclusion(df)
    assert len(merged) == 2
    tip = merged[1]
    assert tip.high == 12 and tip.low == 9.5
    assert tip.direction == UP
    assert tip.raw_indices == [1, 2]
    assert tip.high_idx == 1 and tip.low_idx == 2


def test_down_merge_takes_min_low_min_high():
    # b0,b1 下行定方向 down;b2 被 b1 包含 → [min低, min高]
    df = bars([(12, 10), (10, 7), (9, 8)])
    merged = process_inclusion(df)
    assert len(merged) == 2
    tip = merged[1]
    assert tip.high == 9 and tip.low == 7
    assert tip.direction == DOWN
    assert tip.raw_indices == [1, 2]
    assert tip.high_idx == 2 and tip.low_idx == 1


# ── §2.2 方向由非包含标准 K 决定 ──────────────────────────────────────────
def test_direction_recomputed_on_each_non_containment():
    df = bars([(10, 5), (12, 6), (9, 4)])  # 上, 然后向下非包含
    merged = process_inclusion(df)
    assert len(merged) == 3
    assert merged[1].direction == UP    # 12>10
    assert merged[2].direction == DOWN  # 9<12


# ── §2.3 顺序原则(先合 1、2,再比 3;不传递)──────────────────────────────
def test_sequential_merge_then_compare_with_third():
    # b1,b2 合并成 M(12,7);b3 再与 M 比并入 → M(12,8)
    df = bars([(10, 5), (12, 6), (11, 7), (11.5, 8)])
    merged = process_inclusion(df)
    assert len(merged) == 2
    tip = merged[1]
    assert tip.high == 12 and tip.low == 8
    assert tip.raw_indices == [1, 2, 3]
    assert_no_adjacent_containment(merged)


# ── §2.4 初始方向:回溯前缀 ───────────────────────────────────────────────
def test_initial_direction_backfills_prefix_down():
    # 前三根递减且互含,第一对非包含是 (b2,b3) → down;用 down 回溯合并前缀
    df = bars([(11, 5), (10, 6), (9, 7), (8, 4)])
    assert initial_direction(df["high"].to_numpy(), df["low"].to_numpy()) == DOWN
    merged = process_inclusion(df)
    assert len(merged) == 2
    head = merged[0]
    assert head.high == 9 and head.low == 5  # min高=9(b2), min低=5(b0)
    assert head.raw_indices == [0, 1, 2]
    assert head.direction == DOWN
    assert merged[1].high == 8 and merged[1].low == 4


def test_all_mutually_contained_collapse_to_one_bar():
    # 全程互含(嵌套收缩)→ 合并为单根,默认方向 up
    df = bars([(20, 1), (18, 3), (16, 5), (14, 7)])
    merged = process_inclusion(df)
    assert len(merged) == 1
    assert merged[0].raw_indices == [0, 1, 2, 3]
    # up 合并:max高=20(b0)、max低=7(b3)
    assert merged[0].high == 20 and merged[0].low == 7


# ── §2.5 等高 ─────────────────────────────────────────────────────────────
def test_fully_equal_bars_collapse_to_one():
    df = bars([(10, 8), (10, 8), (12, 9)])
    merged = process_inclusion(df)
    assert len(merged) == 2
    assert merged[0].raw_indices == [0, 1]
    assert merged[0].high == 10 and merged[0].low == 8


def test_single_side_equal_absorbed_by_maxmin():
    # 仅高相等:b0,b1 等高、b1 更低 → 包含,up 方向吸收为 (10, max(8,7)=8)
    df = bars([(10, 8), (10, 7), (12, 9)])
    merged = process_inclusion(df)
    assert len(merged) == 2
    assert merged[0].high == 10 and merged[0].low == 8
    assert merged[0].raw_indices == [0, 1]


# ── §11.5 极端 1:连续嵌套包含 + 方向回溯 ─────────────────────────────────
def test_extreme_continuous_nested_with_flip():
    # 前 4 根嵌套收缩(互含),第一对非包含是 (b3,b4) 向上突破 → 初始方向 up;
    # 用 up 回溯合并前缀。验证顺序处理 + 不出现相邻包含。
    df = bars([
        (20, 4),    # 0
        (18, 6),    # 1 被 0 含
        (16, 8),    # 2 被前含
        (15, 9),    # 3 被前含
        (22, 10),   # 4 向上非包含(突破)
        (21, 12),   # 5 被 4 含
        (24, 13),   # 6 向上非包含
    ])
    merged = process_inclusion(df)
    assert_no_adjacent_containment(merged)
    assert len(merged) == 3
    # 前 4 根互含 → 合一(up 回溯:max高=20(b0)、max低=9(b3))
    assert merged[0].raw_indices == [0, 1, 2, 3]
    assert merged[0].high == 20 and merged[0].low == 9
    assert merged[0].direction == UP
    assert merged[0].high_idx == 0 and merged[0].low_idx == 3
    # 4、5 合(up:max高=22、max低=12)
    assert merged[1].high == 22 and merged[1].low == 12
    assert merged[1].raw_indices == [4, 5]
    assert merged[1].direction == UP
    # 6 独立
    assert merged[2].high == 24 and merged[2].low == 13


def test_merged_to_frame_dates_and_provenance():
    df = bars([(10, 8), (12, 9), (11, 9.5)])
    merged = process_inclusion(df)
    frame = merged_to_frame(merged, df)
    assert list(frame.columns) == [
        "high", "low", "direction", "n_raw",
        "start_date", "end_date", "high_date", "low_date",
    ]
    assert frame.index.is_monotonic_increasing
    # 第二根标准 K:high 来自 b1、low 来自 b2
    row = frame.iloc[1]
    assert row["high_date"] == df.index[1]
    assert row["low_date"] == df.index[2]
    assert row["n_raw"] == 2
