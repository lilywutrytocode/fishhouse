"""模块 9 · 区间套 / 级别联立【确定性 + 主观分层】

出处:缠师 REF p67/p38 / 中泰 p78。

判据(严格按 §9):
- §9.1 级别链 周⊃日⊃30min;美股历史段 30min 降级为日线内部近似(标注)。
- §9.2 区间套:嵌套背驰段 [周]⊃[日]⊃[30min] 逐级收缩,精确点落最低可得级别背驰段内。
- §9.3 联立结构信号【确定性】★ **仅标准背驰参与共振**;★ 30min 用日线内部近似的
  **一律不进"共振背驰"最高强度,只进"降级共振"**。
- §9.4 操作映射【主观·可配置】引擎只出 {结构信号 + 级别 + 共振状态(含是否降级) + 区间套区间},
  **不出算出来的仓位数字**,仓位只给强度档;降级共振降一档(顶减仓不清仓/底试仓不重仓)。
- §9.5 盲复核旁路【主观·非确定·非复现】LLM 不见 policy,出同一动作词汇,引擎比对标
  ``policy_divergence``;★ LLM 输出只进 ``review_notes``,不写结构确定字段、不把未确认改 confirmed。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .beichi import BeichiStatus, Grade


class StructureSignal(str, Enum):
    RESONANCE = "共振背驰"              # 三级标准齐全(真 30min)→ 最高强度
    DOWNGRADED = "降级共振"            # 30min 用日线内部近似 → 降一档
    PENDING_30MIN = "日周共振·待30min"  # 日+周标准但 30min 缺失 → 降一档(防把没取到的当已确认共振)
    LEVEL_TURN = "本级别转折成立"
    SUBLEVEL_PROTECT = "次级别回调·保护级"
    SMALL_TO_BIG = "小转大/信号失效"
    NONE = "无"


class Stance(str, Enum):
    STRONG_ADD = "strong_add"
    ADD = "add"
    HOLD = "hold"
    REDUCE = "reduce"
    STRONG_REDUCE = "strong_reduce"


# ── §9.3 标准背驰判定(仅标准背驰参与共振)─────────────────────────────────
def is_standard_resonance_grade(beichi) -> bool:
    """该级别是否为可参与共振的标准背驰(标准档 + confirmed)。"""
    return (beichi is not None
            and beichi.grade == Grade.STANDARD.value
            and beichi.beichi_status == BeichiStatus.CONFIRMED.value)


def is_any_beichi(beichi) -> bool:
    """该级别是否有任意背驰(标准/面积/DIF,任意 confirmed/疑似)。"""
    return beichi is not None and beichi.grade != Grade.NONE.value


# ── §9.3 联立结构信号表 ───────────────────────────────────────────────────
def classify_lianli(
    *,
    weekly_standard: bool,
    daily_standard: bool,
    min30_standard: bool,
    daily_any: bool = False,
    min30_any: bool = False,
    min30_confirm: bool = False,
    min30_is_approx: bool = False,
    daily_continuation_failed: bool = False,
    min30_available: bool = True,
) -> StructureSignal:
    """按 §9.3 表判联立结构信号。

    ★ ``min30_is_approx``(30min 用日线内部近似)时,三标准齐备也只出 ``降级共振``,
    绝不进 ``共振背驰`` 最高强度。
    ★ ``min30_available=False``(30min 留空接口,日-周两级):**仅标准档参与**——
    日+周标准背驰 → 共振背驰;仅日标准背驰 → 本级别转折成立;否则无(弱信号不进主动作)。
    """
    if daily_continuation_failed:                     # 背驰后未反向、盘整顺原向
        return StructureSignal.SMALL_TO_BIG
    if not min30_available:
        # 日-周两级联立(30min 缺失/未取):★ 降级隔离——不得把没取到的 30min 当已确认共振。
        # 日+周标准 → 日周共振·待30min(降一档);仅日标准 → 本级别转折;弱信号 → 无。
        if weekly_standard and daily_standard:
            return StructureSignal.PENDING_30MIN
        if daily_standard:
            return StructureSignal.LEVEL_TURN
        return StructureSignal.NONE
    if weekly_standard and daily_standard and min30_standard:
        # ★ 30min 近似 → 降级共振;真 30min → 共振背驰
        return (StructureSignal.DOWNGRADED if min30_is_approx
                else StructureSignal.RESONANCE)
    if (not weekly_standard) and daily_any and (min30_any or min30_confirm):
        return StructureSignal.LEVEL_TURN
    if (not weekly_standard) and (not daily_any) and min30_any:
        return StructureSignal.SUBLEVEL_PROTECT
    return StructureSignal.NONE


def is_downgraded(signal: StructureSignal) -> bool:
    """降一档共振:30min 近似(降级共振)或 30min 缺失(待30min)均算降级。"""
    return signal in (StructureSignal.DOWNGRADED, StructureSignal.PENDING_30MIN)


# ── §9.2 区间套定位 ───────────────────────────────────────────────────────
@dataclass
class IntervalNest:
    """嵌套背驰区间(周⊃日⊃30min)与精确点。"""

    weekly_range: tuple | None
    daily_range: tuple | None
    min30_range: tuple | None
    precise_point: float | None     # 落在最低可得级别背驰段内
    lowest_level: str | None
    nested_ok: bool                 # 逐级是否真嵌套
    note: str = ""


def _within(inner, outer) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def interval_nesting(
    weekly_range: tuple | None,
    daily_range: tuple | None,
    min30_range: tuple | None,
    *,
    min30_is_approx: bool = False,
) -> IntervalNest:
    """逐级收缩定位:校验 [周]⊃[日]⊃[30min],精确点取最低可得级别区间中点。"""
    nested = True
    if weekly_range is not None and daily_range is not None:
        nested = nested and _within(daily_range, weekly_range)
    if daily_range is not None and min30_range is not None:
        nested = nested and _within(min30_range, daily_range)

    if min30_range is not None:
        lowest, rng = ("min30", min30_range)
    elif daily_range is not None:
        lowest, rng = ("daily", daily_range)
    elif weekly_range is not None:
        lowest, rng = ("weekly", weekly_range)
    else:
        lowest, rng = (None, None)
    point = (rng[0] + rng[1]) / 2 if rng is not None else None

    note = "30min 日线内部近似" if min30_is_approx else ""
    return IntervalNest(
        weekly_range=weekly_range, daily_range=daily_range, min30_range=min30_range,
        precise_point=point, lowest_level=lowest, nested_ok=nested, note=note,
    )


# ── §9.4 操作映射(主观·可配置·只给强度档)────────────────────────────────
@dataclass
class Policy:
    """主观操作建议:只给强度档与动作词,**不出仓位百分比**。"""

    signal: str
    side: str               # top(顶)/ bottom(底)
    tier: str               # 强度档
    action: str             # 动作描述词
    stance: str             # Stance(供盲复核比对)
    downgraded: bool = False
    upgradable: bool = False  # 降级共振补齐真 30min 可升级
    note: str = "主观·可配置·非仓位数字"


def map_policy(signal: StructureSignal, *, side: str) -> Policy:
    """§9.4 默认操作映射(主观层,可配置)。side ∈ {top, bottom}。"""
    is_top = side == "top"
    if signal == StructureSignal.RESONANCE:
        return Policy(signal.value, side, "最高强度",
                      "清仓/大幅减" if is_top else "重仓建仓",
                      Stance.STRONG_REDUCE.value if is_top else Stance.STRONG_ADD.value)
    if signal in (StructureSignal.DOWNGRADED, StructureSignal.PENDING_30MIN):
        # 降一档:30min 近似(降级共振)或缺失(待30min)同口径——顶减仓不清仓/底分批不重仓
        return Policy(signal.value, side, "降一档",
                      "减仓/降风险(不清仓)" if is_top else "试仓/分批建仓(不重仓)",
                      Stance.REDUCE.value if is_top else Stance.ADD.value,
                      downgraded=True, upgradable=True)
    if signal == StructureSignal.LEVEL_TURN:
        return Policy(signal.value, side, "本级别",
                      "减仓" if is_top else "建仓",
                      Stance.REDUCE.value if is_top else Stance.ADD.value)
    if signal == StructureSignal.SUBLEVEL_PROTECT:
        return Policy(signal.value, side, "保护级",
                      "减仓/对冲(非清仓)", Stance.REDUCE.value)
    if signal == StructureSignal.SMALL_TO_BIG:
        return Policy(signal.value, side, "失效",
                      "撤销减仓信号,持有/顺势", Stance.HOLD.value)
    return Policy(signal.value, side, "无", "观望", Stance.HOLD.value)


# ── §9.5 盲复核旁路(LLM 只进 review_notes)────────────────────────────────
@dataclass
class ReviewNote:
    """LLM 盲复核留痕(非确定·非复现);只进 review_notes,不写结构字段。"""

    action: str             # 加/持/减/清(+ 强度)
    strength: str
    reason: str
    deterministic: bool = False
    reproducible: bool = False
    source: str = "LLM·盲复核"


_ACTION_STANCE = {
    "加": Stance.ADD, "建": Stance.ADD, "试": Stance.ADD,
    "持": Stance.HOLD, "减": Stance.REDUCE, "清": Stance.STRONG_REDUCE,
}


def _stance_family(stance: str) -> str:
    if stance in (Stance.ADD.value, Stance.STRONG_ADD.value):
        return "add"
    if stance in (Stance.REDUCE.value, Stance.STRONG_REDUCE.value):
        return "reduce"
    return "hold"


def compare_review(policy: Policy, review: ReviewNote) -> bool:
    """比对引擎 policy 与 LLM 盲复核动作,方向族不一致 → ``policy_divergence=True``。"""
    llm_stance = _ACTION_STANCE.get(review.action[:1], Stance.HOLD)
    return _stance_family(policy.stance) != _stance_family(llm_stance.value)


# ── §11.1 lianli 聚合 ─────────────────────────────────────────────────────
@dataclass
class Lianli:
    """级别联立输出(§11.1)。"""

    level_beichi: dict             # {'weekly':..,'daily':..,'min30':..} 三级背驰状态
    structure_signal: str
    downgraded: bool
    interval_nest: IntervalNest | None
    policy: Policy                 # 主观·强度档
    min30_status: str = "缺失"      # 真30min / 近似 / 缺失(降级隔离标记)
    policy_divergence: bool | None = None
    review_notes: list = field(default_factory=list)


def build_lianli(
    *,
    weekly_beichi=None,
    daily_beichi=None,
    min30_beichi=None,
    min30_is_approx: bool = False,
    side: str = "bottom",
    daily_continuation_failed: bool = False,
    interval_nest: IntervalNest | None = None,
    review: ReviewNote | None = None,
) -> Lianli:
    """从三级背驰构建联立输出(确定层),并挂主观 policy 与可选盲复核留痕。"""
    w_std = is_standard_resonance_grade(weekly_beichi)
    d_std = is_standard_resonance_grade(daily_beichi)
    m_std = is_standard_resonance_grade(min30_beichi)
    signal = classify_lianli(
        weekly_standard=w_std, daily_standard=d_std, min30_standard=m_std,
        daily_any=is_any_beichi(daily_beichi),
        min30_any=is_any_beichi(min30_beichi),
        min30_is_approx=min30_is_approx,
        daily_continuation_failed=daily_continuation_failed,
        min30_available=(min30_beichi is not None),
    )
    policy = map_policy(signal, side=side)
    divergence = compare_review(policy, review) if review is not None else None
    if min30_beichi is None:
        min30_status = "缺失"
    elif min30_is_approx:
        min30_status = "近似"
    else:
        min30_status = "真30min"
    return Lianli(
        level_beichi={
            "weekly": getattr(weekly_beichi, "grade", None),
            "daily": getattr(daily_beichi, "grade", None),
            "min30": getattr(min30_beichi, "grade", None),
        },
        structure_signal=signal.value,
        downgraded=is_downgraded(signal),
        interval_nest=interval_nest,
        policy=policy,
        min30_status=min30_status,
        policy_divergence=divergence,
        review_notes=[review] if review is not None else [],
    )
