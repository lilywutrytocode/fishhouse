"""§11.5 六类极端单元测试 + 未来函数回归断言。

1 包含连续嵌套(方向回溯) / 2 连续同类分型(顶顶顶取最高·同价取最先) /
3 右端未确认(末段疑似但确认不足→pending) / 4 30min 降级(无 30min 不得最高强度共振) /
5 中枢延伸 >9 段(ZG/ZD 固定、GG/DD 刷新) / 6 未来函数回归断言。
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from chanlun.data.models import OHLCV_COLUMNS
from chanlun.probability import to_backtest_triggers, to_signal_event
from chanlun.structure.bi import CONFIRMED as BI_CONFIRMED, FORMING, build_bi_from_df
from chanlun.structure.fractal import (
    BOTTOM,
    CONFIRMED as FX_CONFIRMED,
    PENDING,
    TOP,
    Fractal,
    dedupe_same_type_fractals,
    detect_fractals,
)
from chanlun.structure.inclusion import is_contained, process_inclusion
from chanlun.structure.lianli import StructureSignal, classify_lianli
from chanlun.structure.maimaidian import BUY, MaiMaiDian, detect_first
from chanlun.structure.xianduan import Pen, XDState, build_segments
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


def hl_bars(hl, tz="Asia/Shanghai") -> pd.DataFrame:
    days = weekdays(date(2024, 1, 1), len(hl))
    rows = [{"open": (h + l) / 2, "high": h, "low": l, "close": (h + l) / 2,
             "volume": 100, "amount": 1.0} for h, l in hl]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in days],
                                name="date").tz_localize(tz)
    return df


# ── 1) 包含连续嵌套(方向回溯)───────────────────────────────────────────
def test_extreme1_continuous_nested_containment():
    df = hl_bars([(20, 4), (18, 6), (16, 8), (15, 9), (22, 10), (21, 12), (24, 13)])
    merged = process_inclusion(df)
    # 标准 K 间不得有相邻包含
    for a, b in zip(merged, merged[1:]):
        assert not is_contained(a.high, a.low, b.high, b.low)
    # 前 4 根互含 → 合一(首个非包含对向上 → 方向回溯为 up)
    assert merged[0].raw_indices == [0, 1, 2, 3]
    assert merged[0].high == 20 and merged[0].low == 9


# ── 2) 连续同类分型:顶顶顶取最高,同价取最先 ─────────────────────────────
def _fx(kind, mid_k, price):
    base = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    return Fractal(kind=kind, level="daily", status=FX_CONFIRMED, mid_k=mid_k,
                   pivot_date=base + pd.Timedelta(days=mid_k), pivot_price=float(price),
                   confirm_date=base + pd.Timedelta(days=mid_k + 2),
                   confirm_price=float(price), executable_price=float(price),
                   source_unit_ids=[mid_k])


def test_extreme2_consecutive_tops_pick_highest_earliest():
    out = dedupe_same_type_fractals(
        [_fx(TOP, 1, 12), _fx(TOP, 3, 15), _fx(TOP, 5, 15), _fx(BOTTOM, 7, 4)])
    assert [f.kind for f in out] == [TOP, BOTTOM]
    assert out[0].pivot_price == 15 and out[0].mid_k == 3   # 取最高;同价(15)取最先


# ── 3) 右端未确认:末段必须 pending/forming,不得 confirmed ────────────────
def test_extreme3_right_end_pending():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4])
    fx = detect_fractals(process_inclusion(df), df)
    assert any(f.status == PENDING for f in fx)            # 右端待定分型
    bis = build_bi_from_df(df)
    assert bis[-1].status == FORMING                       # 右端未确认笔
    # 线段右端非 CONFIRMED_END
    pens = [Pen(direction=b.direction,
                high=max(b.start_price, b.pivot_price),
                low=min(b.start_price, b.pivot_price),
                start_date=b.start_date, end_date=b.pivot_date)
            for b in bis]
    machine = build_segments(pens)
    cur = machine.current_segment()
    if cur is not None:
        assert cur.state != XDState.CONFIRMED_END.value
        assert cur.feeds_zhongshu is False


# ── 4) 30min 降级:无真 30min → 不得最高强度共振 ──────────────────────────
def test_extreme4_30min_downgrade_no_top_resonance():
    sig = classify_lianli(weekly_standard=True, daily_standard=True,
                          min30_standard=True, min30_is_approx=True)
    assert sig == StructureSignal.DOWNGRADED
    assert sig != StructureSignal.RESONANCE


# ── 5) 中枢延伸 >9 段:ZG/ZD 固定、GG/DD 刷新 ────────────────────────────
def test_extreme5_zhongshu_extension_over_9():
    base = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    specs = [(12, 8), (11, 7), (13, 9), (15, 9.5), (10.5, 6), (11, 9),
             (10, 9.2), (11, 9), (16, 9.3), (10.2, 5), (11, 9), (10, 9.5)]
    units = [ZUnit(high=h, low=l, start_date=base + pd.Timedelta(days=i),
                   start_price=(h + l) / 2,
                   confirm_date=base + pd.Timedelta(days=i + 1),
                   confirm_price=(h + l) / 2, confirm_bar=i, id=f"u{i}")
             for i, (h, l) in enumerate(specs)]
    zs = build_zhongshu(units, kind="bi")[0]
    assert zs.n_segments == 12 and zs.n_segments > 9
    assert zs.ZG == 11 and zs.ZD == 9                      # 固定
    assert zs.GG == 16 and zs.DD == 5                      # 刷新


# ── 6) 未来函数回归断言 ───────────────────────────────────────────────────
def _collect_confirmed_structures():
    df = wave([0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0,
               1, 2, 3, 4, 3, 2, 1, 0])
    merged = process_inclusion(df)
    fractals = detect_fractals(merged, df)
    bis = build_bi_from_df(df)
    pens = [Pen(direction=b.direction,
                high=max(b.start_price, b.pivot_price),
                low=min(b.start_price, b.pivot_price),
                start_date=b.start_date, end_date=b.pivot_date)
            for b in bis]
    machine = build_segments(pens)
    return fractals, bis, machine.confirmed


def test_extreme6_no_future_function_confirm_ge_pivot():
    fractals, bis, segments = _collect_confirmed_structures()
    structs = (
        [f for f in fractals if f.confirm_date is not None]
        + [b for b in bis if b.confirm_date is not None]
        + [s for s in segments if s.confirm_date is not None]
    )
    assert structs, "应至少有一个已确认结构"
    for s in structs:
        # confirm_date >= pivot_date;需右侧确认者(分型/笔/线段)严格 >
        assert s.confirm_date >= s.pivot_date
        assert s.confirm_date > s.pivot_date          # 均为需右侧确认结构


def test_extreme6_confirmed_fractal_never_triggers_on_pivot():
    fractals, _, _ = _collect_confirmed_structures()
    for f in fractals:
        if f.status == FX_CONFIRMED:
            # confirm 必为第三根 K(晚于 pivot),signal_date==pivot_date 即未来函数
            assert f.confirm_date != f.pivot_date


def test_extreme6_backtest_uses_confirm_executable_not_pivot():
    # 一买:confirmed → 进回测触发,且触发只含 confirm_date + executable_price
    from types import SimpleNamespace
    from chanlun.structure.beichi import BeichiStatus, BeichiType
    base = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    zs = SimpleNamespace(ZD=10.0, ZG=14.0, id="zs1")
    bc = SimpleNamespace(type=BeichiType.TREND.value, pivot_date=base,
                         pivot_price=8.0, beichi_status=BeichiStatus.CONFIRMED.value,
                         confirm_date=base + pd.Timedelta(days=3), confirm_price=12.0,
                         executable_price=12.1, id="bc1")
    mmd = detect_first(bc, zs, side=BUY)
    mmd.id = "signal_d_001"
    triggers = to_backtest_triggers([to_signal_event(mmd)])
    assert len(triggers) == 1
    t = triggers[0]
    assert "confirm_date" in t and "executable_price" in t
    assert not any("pivot" in k for k in t)               # ★ 触发不得用 pivot


def test_extreme6_pending_signal_excluded_from_backtest():
    # 待确认买卖点(无 confirm/executable)→ 不进回测
    base = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")
    pending = MaiMaiDian(
        kind="一买", side="buy", level="daily", status="待确认", subkind="标准",
        pivot_date=base, pivot_price=8.0, confirm_date=None, confirm_price=None,
        executable_price=None, id="signal_d_002")
    assert to_backtest_triggers([to_signal_event(pending)]) == []
