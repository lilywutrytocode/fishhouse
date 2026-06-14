"""模块 3 · 分型(§3.1–3.2 + §0.5 防未来函数)。"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chanlun.data.models import OHLCV_COLUMNS
from chanlun.structure.fractal import (
    BOTTOM,
    CONFIRMED,
    LIVE_PENDING,
    PENDING,
    TOP,
    Fractal,
    dedup_consecutive_same_type,
    detect_fractals,
)
from chanlun.structure.inclusion import process_inclusion
from tests.conftest import weekdays


def bars(hl, tz="Asia/Shanghai") -> pd.DataFrame:
    """由 (high, low) 列表构造规范日线;open=close=(h+l)/2,便于断言 confirm/executable。"""
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


def _mid(price):
    return price  # 仅用于可读断言


# ── §3.1 检测 + pivot/confirm 锚 ─────────────────────────────────────────
def test_detect_bottom_top_with_pivot_and_confirm():
    # b0↓b1(底)↑b2(顶)↓b3(底)↑b4 ;全程非包含 → 标准 K == 原始 K
    df = bars([(10, 8), (9, 5), (13, 9), (11, 7), (12, 10)])
    merged = process_inclusion(df)
    assert len(merged) == 5

    fx = detect_fractals(merged, df, level="daily")
    confirmed = [f for f in fx if f.status in (CONFIRMED, LIVE_PENDING)]
    kinds = [f.kind for f in confirmed]
    assert kinds == [BOTTOM, TOP, BOTTOM]

    bottom = confirmed[0]
    # pivot = 中间 K(b1) 的极值;confirm = 第三根 K(b2)收盘
    assert bottom.kind == BOTTOM
    assert bottom.pivot_date == df.index[1]
    assert bottom.pivot_price == 5
    assert bottom.confirm_date == df.index[2]            # 第三根 K,不是中间 K
    assert bottom.confirm_date != bottom.pivot_date
    assert bottom.confirm_price == df.iloc[2]["close"]    # =(13+9)/2=11
    assert bottom.status == CONFIRMED
    assert bottom.executable_price == df.iloc[3]["open"]  # 下一根 open

    top = confirmed[1]
    assert top.kind == TOP
    assert top.pivot_date == df.index[2]
    assert top.pivot_price == 13
    assert top.confirm_date == df.index[3]
    assert top.status == CONFIRMED


def test_confirm_date_is_third_k_not_middle():
    df = bars([(10, 8), (9, 5), (13, 9), (11, 7), (12, 10)])
    merged = process_inclusion(df)
    for f in detect_fractals(merged, df):
        if f.confirm_date is not None:
            # confirm 必须严格晚于 pivot,且 ≠ 中间 K 日期
            assert f.confirm_date > f.pivot_date
            assert f.confirm_date != f.pivot_date


# ── §0.5 右端:live_pending / pending ─────────────────────────────────────
def test_last_fractal_is_live_pending_and_right_end_pending_marked():
    df = bars([(10, 8), (9, 5), (13, 9), (11, 7), (12, 10)])
    merged = process_inclusion(df)
    fx = detect_fractals(merged, df)

    # 第三个分型(底@b3)confirm 落在末根 b4 → live_pending、无 executable
    last_confirmed = [f for f in fx if f.status in (CONFIRMED, LIVE_PENDING)][-1]
    assert last_confirmed.kind == BOTTOM
    assert last_confirmed.confirm_date == df.index[4]    # 末根
    assert last_confirmed.status == LIVE_PENDING
    assert last_confirmed.executable_price is None

    # 右端待定候选:b4 相对 b3 上行 → 潜在顶,pending,无 confirm
    pend = [f for f in fx if f.status == PENDING]
    assert len(pend) == 1
    assert pend[0].kind == TOP
    assert pend[0].confirm_date is None
    assert pend[0].executable_price is None
    assert pend[0].mid_k == 4


def test_too_few_bars_only_pending_no_confirmed():
    # 只有上行两根 → 无三根窗口,只出右端待定顶
    df = bars([(10, 8), (12, 9)])
    merged = process_inclusion(df)
    fx = detect_fractals(merged, df)
    assert all(f.status == PENDING for f in fx)
    assert len(fx) == 1 and fx[0].kind == TOP


# ── §3:confirm 取"第三根标准 K"的收盘(跨包含合并时取末根原始 K)──────────
def test_confirm_uses_third_standard_k_end_when_merged():
    # 第三根标准 K = merge(b2,b3);confirm 应落在 b3(末根原始 K),非 b2
    df = bars([(13, 9), (9, 4), (14, 10), (13, 11), (12, 8)])
    merged = process_inclusion(df)
    # 标准 K:K0=b0, K1=b1, K2=merge(b2,b3), K3=b4
    assert len(merged) == 4
    assert merged[2].raw_indices == [2, 3]

    fx = detect_fractals(merged, df)
    bottom = [f for f in fx if f.kind == BOTTOM and f.status == CONFIRMED][0]
    assert bottom.pivot_date == df.index[1]              # 中间 K=b1
    assert bottom.pivot_price == 4
    assert bottom.confirm_date == df.index[3]            # 第三根标准 K 末根=b3
    assert bottom.confirm_price == df.iloc[3]["close"]
    assert bottom.executable_price == df.iloc[4]["open"]  # 下一根 b4 open


# ── §3.2 连续同类去重(顶顶顶后出底取最高顶,同价取最先)──────────────────
def _fx(kind, mid_k, price, *, confirm_offset=2):
    """构造用于去重测试的分型对象(pivot 在 mid_k,confirm 在其后第 offset 根)。"""
    base = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    pivot_date = base + pd.Timedelta(days=mid_k)
    confirm_date = base + pd.Timedelta(days=mid_k + confirm_offset)
    return Fractal(
        kind=kind, level="daily", status=CONFIRMED, mid_k=mid_k,
        pivot_date=pivot_date, pivot_price=float(price),
        confirm_date=confirm_date, confirm_price=float(price),
        executable_price=float(price), source_unit_ids=[mid_k],
    )


def test_dedup_three_tops_then_bottom_keeps_highest_top():
    # 顶顶顶(12,15,13)后出底 → 保留最高顶(15),并保留其自身 confirm
    tops = [_fx(TOP, 1, 12), _fx(TOP, 3, 15), _fx(TOP, 5, 13)]
    bottom = _fx(BOTTOM, 7, 4)
    out = dedup_consecutive_same_type(tops + [bottom])
    assert [f.kind for f in out] == [TOP, BOTTOM]
    survivor = out[0]
    assert survivor.pivot_price == 15
    assert survivor.mid_k == 3
    # tie-break/选择不提前 confirm:survivor 仍用自身 confirm(mid_k=3 → confirm=5)
    assert survivor.confirm_date == pd.Timestamp("2024-01-06", tz="Asia/Shanghai")


def test_dedup_tie_keeps_earliest():
    # 同价两顶 → 取最先(mid_k 较小者),只影响 pivot 选择
    out = dedup_consecutive_same_type([_fx(TOP, 1, 15), _fx(TOP, 3, 15)])
    assert len(out) == 1
    assert out[0].mid_k == 1


def test_dedup_three_bottoms_keeps_lowest():
    bottoms = [_fx(BOTTOM, 1, 6), _fx(BOTTOM, 3, 4), _fx(BOTTOM, 5, 5)]
    top = _fx(TOP, 7, 20)
    out = dedup_consecutive_same_type(bottoms + [top])
    assert [f.kind for f in out] == [BOTTOM, TOP]
    assert out[0].pivot_price == 4 and out[0].mid_k == 3


# ── 不变量:确认分型严禁用 pivot 触发(executable 来自 confirm 之后)────────
def test_executable_never_equals_pivot_trigger():
    df = bars([(10, 8), (9, 5), (13, 9), (11, 7), (12, 10), (8, 4), (14, 11)])
    merged = process_inclusion(df)
    for f in detect_fractals(merged, df):
        if f.status == CONFIRMED:
            # 可执行价来自 confirm 之后的下一根,confirm 严格晚于 pivot
            assert f.executable_price is not None
            assert f.confirm_date > f.pivot_date
