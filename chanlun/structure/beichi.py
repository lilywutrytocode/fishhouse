"""模块 7 · 背驰【确定性 + 约定】

出处:中泰 p58/p6 / 缠师 24、27 课。

判据(严格按 §7 + §0.5):
- §7.1 动能 = 段内 MACD 柱面积(上升取红柱、下降取绿柱累计绝对值)。
- §7.2 三档(前提:价格创新高/新低):
  - **标准背驰** = 面积↓ 且 DIF 峰值↓ → 主信号(仓位/共振只用此档)。
  - **面积背驰(弱)** = 仅面积↓;**DIF 背驰(弱)** = 仅 DIF↓ → 进报告/回测,非主信号。
  - ★ C 段完成才可测 confirmed:笔级=C 笔 confirmed、线段级=C 线段 CONFIRMED_END。
    C 段仍 extending/pending → ``beichi_status ∈ {提前判, 疑似, 待确认}``,**绝不 confirmed**。
- §7.3 容差三态 k=0.9:比值 C/A ``<0.9 满足`` / ``[0.9,1.0] 疑似`` / ``>1.0 不满足``。
- §7.4 趋势(≥2 中枢,A 进入 / C 离开,同向)只输出 ``macd_reset_status``(0 轴回拉不作硬门槛);
  盘整按"最近一组同向 A/C"取段,延伸 >3 段不固定前三段。
- §7.5 比较单元跟随中枢单位(线段中枢→线段级、笔中枢→笔级),都算、各标级别。
- §0.5:``pivot`` = 价格新高/新低点(C 段端点);``confirm`` = C 段确认完成(晚于 pivot)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..config import DEFAULT_CONFIG, Config
from .inclusion import DOWN, UP


class BeichiType(str, Enum):
    TREND = "趋势"
    CONSOLIDATION = "盘整"


class Grade(str, Enum):
    STANDARD = "标准背驰"
    AREA = "面积背驰"
    DIF = "DIF背驰"
    NONE = "无"


class ToleranceState(str, Enum):
    MET = "满足"
    SUSPECT = "疑似"
    UNMET = "不满足"


class BeichiStatus(str, Enum):
    CONFIRMED = "confirmed"
    EARLY = "提前判"
    SUSPECT = "疑似"
    PENDING = "待确认"


class MacdReset(str, Enum):
    RESET = "已回拉0轴"
    NEAR = "接近0轴"
    NOT = "未回拉"


@dataclass
class SegEnergy:
    """一个比较段的动能度量 + 端点信息(C 段含 confirm/executable)。"""

    area: float
    dif_peak: float
    direction: str
    confirmed: bool = True
    makes_new_extreme: bool = True
    pivot_date: pd.Timestamp | None = None
    pivot_price: float | None = None
    confirm_date: pd.Timestamp | None = None
    confirm_price: float | None = None
    executable_price: float | None = None
    id: str | None = None


@dataclass
class Beichi:
    """一个背驰判定(带 §0.6 通用纪律字段)。"""

    type: str               # BeichiType
    compare_unit: str       # bi / xianduan(比较单元)
    level: str
    grade: str              # Grade
    area_state: str         # ToleranceState
    dif_state: str
    area_ratio: float
    dif_ratio: float
    beichi_status: str      # BeichiStatus(C 段未完成绝不 confirmed)
    is_main_signal: bool    # 仅 标准背驰 + confirmed
    macd_reset_status: str | None   # 仅趋势背驰输出
    pivot_date: pd.Timestamp | None
    pivot_price: float | None
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    related_zhongshu_id: str | None = None
    a_unit_id: str | None = None
    c_unit_id: str | None = None
    seg_start_date: pd.Timestamp | None = None   # 背驰段起点(A 段起点),供 §9.2 区间套嵌套
    id: str | None = None

    def __post_init__(self):
        if self.confirm_date is not None and self.pivot_date is not None:
            assert self.confirm_date > self.pivot_date, (
                "confirm_date 必须晚于 pivot_date(§0.5 背驰 confirm=C 段确认完成)"
            )
        if self.beichi_status != BeichiStatus.CONFIRMED.value:
            # C 段未完成/疑似 → 不得带 confirmed 触发信息
            assert self.confirm_date is None, "非 confirmed 背驰不得带 confirm_date"


# ── §7.1 动能度量 ─────────────────────────────────────────────────────────
def compute_macd(close: pd.Series, *, config: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """MACD(12/26/9,收盘价):返回 DIF/DEA/HIST(柱=DIF-DEA)。"""
    ema_fast = close.ewm(span=config.macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=config.macd_slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=config.macd_signal, adjust=False).mean()
    hist = dif - dea
    return pd.DataFrame({"dif": dif, "dea": dea, "hist": hist}, index=close.index)


def segment_area(hist: pd.Series, direction: str) -> float:
    """段内 MACD 柱面积:上升取红柱(正)累计,下降取绿柱(负)累计绝对值。"""
    if direction == UP:
        return float(hist[hist > 0].sum())
    return float(-hist[hist < 0].sum())


def segment_dif_peak(dif: pd.Series, direction: str) -> float:
    """段内 DIF 峰值:上升取最大,下降取最小(最负)。"""
    return float(dif.max()) if direction == UP else float(dif.min())


# ── §7.3 容差三态 ─────────────────────────────────────────────────────────
def tolerance_state(ratio: float, *, k: float = 0.9) -> str:
    """C/A 比值三态:<k 满足;[k,1.0] 疑似;>1.0 不满足。"""
    if ratio < k:
        return ToleranceState.MET.value
    if ratio <= 1.0:
        return ToleranceState.SUSPECT.value
    return ToleranceState.UNMET.value


# ── §7.2 分档 ─────────────────────────────────────────────────────────────
def classify_grade(area_state: str, dif_state: str) -> str:
    """按"满足哪些"分三档(疑似/不满足均不计入满足)。"""
    a = area_state == ToleranceState.MET.value
    d = dif_state == ToleranceState.MET.value
    if a and d:
        return Grade.STANDARD.value
    if a:
        return Grade.AREA.value
    if d:
        return Grade.DIF.value
    return Grade.NONE.value


# ── §7.4 0 轴回拉(只输出字段,不作硬门槛)──────────────────────────────────
def macd_reset_status(dif_values, *, near_ratio: float = 0.2) -> str:
    """趋势背驰的黄白线回拉 0 轴状态(v1 仅字段)。"""
    arr = [float(v) for v in dif_values if v == v]  # 去 NaN
    if not arr:
        return MacdReset.NOT.value
    crossed = any(a * b < 0 for a, b in zip(arr, arr[1:]))
    min_abs = min(abs(v) for v in arr)
    if crossed or min_abs == 0:
        return MacdReset.RESET.value
    peak = max(abs(v) for v in arr) or 1.0
    if min_abs <= near_ratio * peak:
        return MacdReset.NEAR.value
    return MacdReset.NOT.value


def _beichi_status(grade: str, suspect: bool, c_confirmed: bool) -> str:
    """C 段未完成绝不 confirmed(§7.2 ★)。"""
    if not c_confirmed:
        if grade != Grade.NONE.value:
            return BeichiStatus.EARLY.value      # 提前判:满足但 C 段未完成
        if suspect:
            return BeichiStatus.SUSPECT.value     # 疑似
        return BeichiStatus.PENDING.value         # 待确认
    if grade != Grade.NONE.value:
        return BeichiStatus.CONFIRMED.value
    return BeichiStatus.SUSPECT.value             # C 完成但仅容差疑似


def evaluate_divergence(
    a: SegEnergy,
    c: SegEnergy,
    *,
    btype: str = BeichiType.CONSOLIDATION.value,
    compare_unit: str = "bi",
    level: str = "daily",
    config: Config = DEFAULT_CONFIG,
    related_zhongshu_id: str | None = None,
    reset_dif_values=None,
    seg_start_date=None,
) -> Beichi | None:
    """比较进入段 A 与离开段 C,产出背驰判定(无背驰且非疑似 → None)。"""
    if not c.makes_new_extreme:           # §7.2 前提:价格须创新高/新低
        return None
    if a.area == 0:
        return None
    area_ratio = c.area / a.area
    dif_ratio = abs(c.dif_peak) / (abs(a.dif_peak) or 1e-12)
    area_state = tolerance_state(area_ratio, k=config.beichi_k)
    dif_state = tolerance_state(dif_ratio, k=config.beichi_k)
    grade = classify_grade(area_state, dif_state)
    suspect = ToleranceState.SUSPECT.value in (area_state, dif_state)
    if grade == Grade.NONE.value and not suspect:
        return None                        # C 比 A 更强:无背驰

    status = _beichi_status(grade, suspect, c.confirmed)
    confirmed = status == BeichiStatus.CONFIRMED.value
    reset = None
    if btype == BeichiType.TREND.value:
        reset = (macd_reset_status(reset_dif_values)
                 if reset_dif_values is not None else MacdReset.NOT.value)

    return Beichi(
        type=btype, compare_unit=compare_unit, level=level, grade=grade,
        area_state=area_state, dif_state=dif_state,
        area_ratio=area_ratio, dif_ratio=dif_ratio,
        beichi_status=status,
        is_main_signal=(grade == Grade.STANDARD.value and confirmed),
        macd_reset_status=reset,
        pivot_date=c.pivot_date, pivot_price=c.pivot_price,
        confirm_date=c.confirm_date if confirmed else None,
        confirm_price=c.confirm_price if confirmed else None,
        executable_price=c.executable_price if confirmed else None,
        related_zhongshu_id=related_zhongshu_id,
        a_unit_id=a.id, c_unit_id=c.id,
        seg_start_date=seg_start_date,         # A 段起点(供区间套嵌套)
    )


# ── §7.4 选段(结构集成层)────────────────────────────────────────────────
def select_consolidation_ac(units: list, zhongshu) -> tuple[int, int] | None:
    """盘整背驰:取"最近一组同向 A/C"——C=离开中枢的同向段(end+1),
    A=其前一个同向段(C-2,通常为中枢内最近一次同向摆动)。

    ★ 按最近一组同向段配对,延伸 >3 段也不固定取前三段、不靠 start_unit,避免错位(§7.4)。
    """
    c_idx = zhongshu.end_unit + 1
    a_idx = c_idx - 2
    if a_idx < 0 or c_idx >= len(units):
        return None
    if units[a_idx].direction != units[c_idx].direction:
        return None
    return a_idx, c_idx
