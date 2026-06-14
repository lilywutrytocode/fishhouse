"""模块 7 · 背驰(§7.1–7.5 + §0.5)。"""

from __future__ import annotations

import pandas as pd

from chanlun.structure.beichi import (
    Beichi,
    BeichiStatus,
    BeichiType,
    Grade,
    MacdReset,
    SegEnergy,
    ToleranceState,
    classify_grade,
    compute_macd,
    evaluate_divergence,
    macd_reset_status,
    segment_area,
    segment_dif_peak,
    tolerance_state,
)
from chanlun.structure.inclusion import DOWN, UP

_BASE = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")


def _segC(area, dif, *, confirmed=True, new_extreme=True, direction=UP):
    return SegEnergy(
        area=area, dif_peak=dif, direction=direction,
        confirmed=confirmed, makes_new_extreme=new_extreme,
        pivot_date=_BASE + pd.Timedelta(days=20), pivot_price=30.0,
        confirm_date=_BASE + pd.Timedelta(days=22), confirm_price=29.5,
        executable_price=29.6, id="C",
    )


def _segA(area, dif, direction=UP):
    return SegEnergy(area=area, dif_peak=dif, direction=direction, id="A")


# ── §7.1 MACD 与动能 ──────────────────────────────────────────────────────
def test_compute_macd_columns():
    close = pd.Series(range(1, 60), dtype=float)
    macd = compute_macd(close)
    assert list(macd.columns) == ["dif", "dea", "hist"]
    assert len(macd) == len(close)
    # 单调上行 → DIF 应为正
    assert macd["dif"].iloc[-1] > 0


def test_segment_area_and_dif_peak():
    hist = pd.Series([1.0, 2.0, -0.5, 3.0, -1.0])
    assert segment_area(hist, UP) == 6.0       # 正柱累计 1+2+3
    assert segment_area(hist, DOWN) == 1.5     # 负柱绝对值累计 0.5+1.0
    dif = pd.Series([0.5, 1.2, -0.3, 0.8])
    assert segment_dif_peak(dif, UP) == 1.2
    assert segment_dif_peak(dif, DOWN) == -0.3


# ── §7.3 容差三态边界 [0.9,1.0] 落疑似 ────────────────────────────────────
def test_tolerance_state_boundaries():
    assert tolerance_state(0.89) == ToleranceState.MET.value
    assert tolerance_state(0.90) == ToleranceState.SUSPECT.value   # 边界入疑似
    assert tolerance_state(0.95) == ToleranceState.SUSPECT.value
    assert tolerance_state(1.00) == ToleranceState.SUSPECT.value   # 边界入疑似
    assert tolerance_state(1.01) == ToleranceState.UNMET.value


# ── §7.2 三档分类 ─────────────────────────────────────────────────────────
def test_classify_three_grades():
    M, S, U = (ToleranceState.MET.value, ToleranceState.SUSPECT.value,
               ToleranceState.UNMET.value)
    assert classify_grade(M, M) == Grade.STANDARD.value
    assert classify_grade(M, U) == Grade.AREA.value
    assert classify_grade(U, M) == Grade.DIF.value
    assert classify_grade(U, U) == Grade.NONE.value
    assert classify_grade(M, S) == Grade.AREA.value     # 仅面积满足
    assert classify_grade(S, M) == Grade.DIF.value


def test_three_grades_via_evaluate():
    A = _segA(area=100.0, dif=2.0)
    # 标准:面积↓且 DIF↓
    std = evaluate_divergence(A, _segC(area=50.0, dif=1.0))
    assert std.grade == Grade.STANDARD.value and std.is_main_signal is True
    # 面积背驰:仅面积↓(DIF 反而更大)
    area_only = evaluate_divergence(A, _segC(area=50.0, dif=3.0))
    assert area_only.grade == Grade.AREA.value and area_only.is_main_signal is False
    # DIF 背驰:仅 DIF↓(面积反而更大)
    dif_only = evaluate_divergence(A, _segC(area=120.0, dif=1.0))
    assert dif_only.grade == Grade.DIF.value and dif_only.is_main_signal is False


# ── §7.2 ★ C 段未完成绝不 confirmed ───────────────────────────────────────
def test_c_extending_never_confirmed():
    A = _segA(area=100.0, dif=2.0)
    bc = evaluate_divergence(A, _segC(area=40.0, dif=0.8, confirmed=False))
    assert bc.grade == Grade.STANDARD.value
    assert bc.beichi_status == BeichiStatus.EARLY.value       # 提前判
    assert bc.beichi_status != BeichiStatus.CONFIRMED.value
    assert bc.is_main_signal is False                         # 非 confirmed → 非主信号
    assert bc.confirm_date is None and bc.executable_price is None


def test_c_confirmed_standard_is_main_signal():
    A = _segA(area=100.0, dif=2.0)
    bc = evaluate_divergence(A, _segC(area=40.0, dif=0.8, confirmed=True))
    assert bc.beichi_status == BeichiStatus.CONFIRMED.value
    assert bc.is_main_signal is True
    assert bc.confirm_date is not None and bc.confirm_date > bc.pivot_date


# ── §7.3→7.2 容差疑似落到 beichi_status=疑似 ───────────────────────────────
def test_suspect_tolerance_yields_suspect_status():
    A = _segA(area=100.0, dif=2.0)
    # 比值 0.95/0.95 → 两条件皆疑似 → grade=无、status=疑似
    bc = evaluate_divergence(A, _segC(area=95.0, dif=1.9, confirmed=True))
    assert bc.area_state == ToleranceState.SUSPECT.value
    assert bc.dif_state == ToleranceState.SUSPECT.value
    assert bc.grade == Grade.NONE.value
    assert bc.beichi_status == BeichiStatus.SUSPECT.value
    assert bc.is_main_signal is False


# ── §7.2 前提:无新高/新低 → 无背驰 ───────────────────────────────────────
def test_no_divergence_without_new_extreme():
    A = _segA(area=100.0, dif=2.0)
    assert evaluate_divergence(A, _segC(area=40.0, dif=0.8, new_extreme=False)) is None


def test_no_divergence_when_c_stronger():
    A = _segA(area=100.0, dif=2.0)
    # 面积、DIF 均 >1.0 不满足且无疑似 → None
    assert evaluate_divergence(A, _segC(area=130.0, dif=3.0)) is None


# ── §7.4 趋势:macd_reset_status 字段(不作硬门槛)─────────────────────────
def test_macd_reset_status():
    assert macd_reset_status([1.0, 0.5, -0.2, 0.3]) == MacdReset.RESET.value  # 穿0
    assert macd_reset_status([1.0, 0.5, 0.1]) == MacdReset.NEAR.value         # 接近
    assert macd_reset_status([1.0, 0.9, 0.8]) == MacdReset.NOT.value          # 未回拉


def test_trend_outputs_reset_consolidation_does_not():
    A = _segA(area=100.0, dif=2.0)
    trend = evaluate_divergence(
        A, _segC(area=40.0, dif=0.8), btype=BeichiType.TREND.value,
        reset_dif_values=[2.0, 0.1, 1.5],
    )
    assert trend.type == BeichiType.TREND.value
    assert trend.macd_reset_status == MacdReset.NEAR.value
    cons = evaluate_divergence(A, _segC(area=40.0, dif=0.8))
    assert cons.type == BeichiType.CONSOLIDATION.value
    assert cons.macd_reset_status is None
