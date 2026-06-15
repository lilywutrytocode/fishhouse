"""模块 8 · 三类买卖点(§8.1–8.6 + §0.5)。"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from chanlun.structure.beichi import BeichiStatus, BeichiType
from chanlun.structure.inclusion import DOWN, UP
from chanlun.structure.maimaidian import (
    ABOVE,
    BELOW,
    BUY,
    INSIDE,
    ST_BEICHI,
    ST_CONFIRMED,
    ST_PENDING,
    ST_SUBLEVEL,
    SUB_PANBEI,
    SUB_STANDARD,
    Unit,
    detect_first,
    detect_second,
    detect_third,
    mark_overlap_2_3,
    relation_to_zhongshu,
)

_BASE = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")


def _d(n):
    return _BASE + pd.Timedelta(days=n)


def mk_zhongshu(zd=10.0, zg=14.0, id="zs1"):
    return SimpleNamespace(ZD=zd, ZG=zg, id=id)


def mk_beichi(*, btype=BeichiType.TREND.value, pivot=8.0, confirm=12.0,
              status=BeichiStatus.CONFIRMED.value, grade="标准背驰",
              is_main_signal=True, id="bc1"):
    return SimpleNamespace(
        type=btype, pivot_date=_d(5), pivot_price=pivot,
        beichi_status=status, confirm_date=_d(7) if confirm is not None else None,
        confirm_price=confirm, executable_price=confirm,
        grade=grade, is_main_signal=is_main_signal, id=id,
    )


def mk_unit(direction, *, low, high, pivot_day, confirm_day, confirmed=True, n_bars=5, id="u"):
    pivot_price = high if direction == UP else low
    return Unit(
        direction=direction, high=high, low=low,
        pivot_date=_d(pivot_day), pivot_price=pivot_price,
        confirm_date=_d(confirm_day), confirm_price=pivot_price,
        executable_price=pivot_price, confirmed=confirmed, n_bars=n_bars, id=id,
    )


# ── 相对中枢位置 ──────────────────────────────────────────────────────────
def test_relation_to_zhongshu():
    assert relation_to_zhongshu(8, 10, 14) == BELOW
    assert relation_to_zhongshu(12, 10, 14) == INSIDE
    assert relation_to_zhongshu(16, 10, 14) == ABOVE


# ── §8.1 一买:标准/盘背 + pivot/confirm 相对中枢 ─────────────────────────
def test_first_buy_standard_vs_panbei():
    zs = mk_zhongshu()
    std = detect_first(mk_beichi(btype=BeichiType.TREND.value), zs)
    assert std.kind == "一买" and std.subkind == SUB_STANDARD
    pan = detect_first(mk_beichi(btype=BeichiType.CONSOLIDATION.value), zs)
    assert pan.subkind == SUB_PANBEI


def test_first_buy_pivot_below_but_confirm_inside():
    # ★ pivot 在中枢下方(结构成立),confirm 已回到中枢内(操作风险不同)
    zs = mk_zhongshu(zd=10, zg=14)
    mmd = detect_first(mk_beichi(pivot=8.0, confirm=12.0), zs)
    assert mmd.pivot_relation_to_zhongshu == BELOW
    assert mmd.confirm_relation_to_zhongshu == INSIDE
    assert mmd.status == ST_BEICHI
    assert mmd.confirm_date > mmd.pivot_date


# ── §8.1 强度档闸:标准档→主信号;面积/DIF 弱档→一买·弱、不进主信号 ──────────
def test_first_buy_standard_grade_is_main():
    zs = mk_zhongshu()
    mmd = detect_first(mk_beichi(grade="标准背驰", is_main_signal=True), zs)
    assert mmd.strength == "标准" and mmd.is_main is True
    assert mmd.beichi_grade == "标准背驰"
    assert mmd.label == "一买·标准"          # 趋势 + 标准档


def test_first_buy_weak_grade_marks_weak_not_main():
    zs = mk_zhongshu()
    # 趋势背驰但只落 DIF 档(弱)→ 一买·弱,不进主信号;子类(趋势→标准)正交保留
    bc = mk_beichi(btype=BeichiType.TREND.value, grade="DIF背驰", is_main_signal=False)
    mmd = detect_first(bc, zs)
    assert mmd.subkind == SUB_STANDARD       # 趋势子类仍记录(与强度正交)
    assert mmd.strength == "弱"
    assert mmd.is_main is False
    assert mmd.label == "一买·弱"             # 标 一买·弱,而非 一买·标准
    assert mmd.label != "一买·标准"


def test_first_buy_area_grade_also_weak():
    zs = mk_zhongshu()
    bc = mk_beichi(btype=BeichiType.CONSOLIDATION.value, grade="面积背驰",
                   is_main_signal=False)
    mmd = detect_first(bc, zs)
    assert mmd.strength == "弱" and mmd.is_main is False
    assert mmd.label == "一买·弱"             # 盘整弱档也标 弱


def test_first_buy_pending_when_beichi_not_confirmed():
    zs = mk_zhongshu()
    bc = mk_beichi(status=BeichiStatus.EARLY.value, confirm=None)
    mmd = detect_first(bc, zs)
    assert mmd.status == ST_PENDING
    assert mmd.confirm_date is None
    assert mmd.confirm_relation_to_zhongshu is None


def test_first_buy_sublevel_confirm_state():
    zs = mk_zhongshu()
    sub = mk_unit(UP, low=9, high=13, pivot_day=8, confirm_day=9, id="subup")
    mmd = detect_first(mk_beichi(), zs, sublevel_confirm=sub)
    assert mmd.status == ST_SUBLEVEL
    assert mmd.confirm_date == _d(9)


# ── §8.2 二买五步 ─────────────────────────────────────────────────────────
def test_second_buy_five_steps():
    zs = mk_zhongshu()
    first = detect_first(mk_beichi(pivot=8.0), zs)        # 一买 low=8,confirmed
    sub_units = [
        mk_unit(UP, low=8, high=15, pivot_day=8, confirm_day=9, id="up1"),     # ① 次级别向上
        mk_unit(DOWN, low=11, high=15, pivot_day=10, confirm_day=11, id="dn1"),  # ② 第一个向下完成
    ]
    second = detect_second(first, sub_units)
    assert second is not None and second.kind == "二买"
    assert second.pivot_price == 11                       # ③④ 低点 11 > 一买 8
    assert second.pivot_date == _d(10)
    assert second.confirm_date == _d(11)                  # ⑤ 完成日=confirm
    assert second.confirm_date > second.pivot_date


def test_second_buy_rejected_when_low_breaks_first_pivot():
    zs = mk_zhongshu()
    first = detect_first(mk_beichi(pivot=8.0), zs)
    sub_units = [
        mk_unit(UP, low=8, high=15, pivot_day=8, confirm_day=9, id="up1"),
        mk_unit(DOWN, low=7, high=15, pivot_day=10, confirm_day=11, id="dn1"),  # 低点 7 < 8
    ]
    assert detect_second(first, sub_units) is None        # 跌破一买 → 非二买


def test_second_buy_none_when_first_not_confirmed():
    zs = mk_zhongshu()
    first = detect_first(mk_beichi(status=BeichiStatus.EARLY.value, confirm=None), zs)
    assert first.confirm_date is None
    assert detect_second(first, []) is None


def test_second_buy_skips_until_first_subdown_completes():
    # 一买后先有段内小波动(未确认向下)不算,等第一个"确认"的次级别向下
    zs = mk_zhongshu()
    first = detect_first(mk_beichi(pivot=8.0), zs)
    sub_units = [
        mk_unit(UP, low=8, high=15, pivot_day=8, confirm_day=9, id="up1"),
        mk_unit(DOWN, low=12, high=15, pivot_day=10, confirm_day=11,
                confirmed=False, id="dn_extending"),                 # 未确认 → 跳过
        mk_unit(DOWN, low=11, high=15, pivot_day=12, confirm_day=13, id="dn_done"),
    ]
    second = detect_second(first, sub_units)
    assert second.pivot_date == _d(12) and second.confirm_date == _d(13)


# ── §8.3 三买:离开须确认次级别单位明确脱离,单根刺破不算 ──────────────────
def test_third_buy_valid():
    zs = mk_zhongshu(zd=10, zg=14)
    leave = mk_unit(UP, low=12, high=18, pivot_day=8, confirm_day=9, n_bars=5, id="lv")
    retest = mk_unit(DOWN, low=15, high=18, pivot_day=10, confirm_day=11, id="rt")
    t = detect_third(zs, leave, retest)
    assert t is not None and t.kind == "三买"
    assert t.pivot_price == 15 and t.pivot_price > zs.ZG    # 回试低点 > ZG
    assert t.related_leave_unit_id == "lv"
    assert t.related_retest_unit_id == "rt"
    assert t.status == ST_CONFIRMED and t.confirm_date > t.pivot_date


def test_third_buy_single_bar_poke_not_counted():
    # ★ 单根 K 刺破 ZG(n_bars==1)不算离开
    zs = mk_zhongshu(zd=10, zg=14)
    poke = mk_unit(UP, low=12, high=18, pivot_day=8, confirm_day=9, n_bars=1, id="poke")
    retest = mk_unit(DOWN, low=15, high=18, pivot_day=10, confirm_day=11, id="rt")
    assert detect_third(zs, poke, retest) is None


def test_third_buy_rejected_when_retest_returns_into_zhongshu():
    zs = mk_zhongshu(zd=10, zg=14)
    leave = mk_unit(UP, low=12, high=18, pivot_day=8, confirm_day=9, n_bars=5, id="lv")
    retest = mk_unit(DOWN, low=13, high=18, pivot_day=10, confirm_day=11, id="rt")  # 13 <= ZG14
    assert detect_third(zs, leave, retest) is None


def test_third_buy_pending_when_retest_extending():
    zs = mk_zhongshu(zd=10, zg=14)
    leave = mk_unit(UP, low=12, high=18, pivot_day=8, confirm_day=9, n_bars=5, id="lv")
    retest = mk_unit(DOWN, low=15, high=18, pivot_day=10, confirm_day=11,
                     confirmed=False, id="rt")
    t = detect_third(zs, leave, retest)
    assert t.status == ST_PENDING and t.confirm_date is None   # extending 不得 confirmed


# ── §8.6 二三买重合 ───────────────────────────────────────────────────────
def test_overlap_2_3_marked():
    zs = mk_zhongshu(zd=10, zg=14)
    first = detect_first(mk_beichi(pivot=8.0), zs)
    second = detect_second(first, [
        mk_unit(UP, low=8, high=18, pivot_day=8, confirm_day=9, id="up1"),
        mk_unit(DOWN, low=15, high=18, pivot_day=10, confirm_day=11, id="dn1"),
    ])
    leave = mk_unit(UP, low=12, high=18, pivot_day=6, confirm_day=7, n_bars=5, id="lv")
    third = detect_third(zs, leave, mk_unit(
        DOWN, low=15, high=18, pivot_day=10, confirm_day=11, id="rt"))
    # 二买与三买同 pivot(低点 15 @ day10)→ 重合
    mark_overlap_2_3(second, third)
    assert second.overlap_2_3 is True and third.overlap_2_3 is True
