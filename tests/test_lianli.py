"""模块 9 · 区间套 / 级别联立(§9.1–9.5)。"""

from __future__ import annotations

from types import SimpleNamespace

from chanlun.structure.beichi import BeichiStatus, Grade
from chanlun.structure.lianli import (
    IntervalNest,
    Policy,
    ReviewNote,
    Stance,
    StructureSignal,
    build_lianli,
    classify_lianli,
    compare_review,
    interval_nesting,
    is_standard_resonance_grade,
    map_policy,
)


def mk_beichi(grade=Grade.STANDARD.value, status=BeichiStatus.CONFIRMED.value):
    return SimpleNamespace(grade=grade, beichi_status=status, id="bc")


# ── §9.3 标准背驰判定 ─────────────────────────────────────────────────────
def test_only_standard_confirmed_participates():
    assert is_standard_resonance_grade(mk_beichi()) is True
    assert is_standard_resonance_grade(mk_beichi(grade=Grade.AREA.value)) is False
    assert is_standard_resonance_grade(
        mk_beichi(status=BeichiStatus.EARLY.value)) is False
    assert is_standard_resonance_grade(None) is False


# ── §9.3 ★ 日线内部近似不得产生最高强度共振背驰 ───────────────────────────
def test_daily_internal_approx_never_top_resonance():
    # 三级都标准背驰,但 30min 为日线内部近似 → 只能 降级共振
    sig = classify_lianli(
        weekly_standard=True, daily_standard=True, min30_standard=True,
        min30_is_approx=True,
    )
    assert sig == StructureSignal.DOWNGRADED
    assert sig != StructureSignal.RESONANCE


def test_real_30min_gives_top_resonance():
    sig = classify_lianli(
        weekly_standard=True, daily_standard=True, min30_standard=True,
        min30_is_approx=False,
    )
    assert sig == StructureSignal.RESONANCE


def test_other_lianli_rows():
    # 周无,日背驰,30min背驰/确认 → 本级别转折
    assert classify_lianli(
        weekly_standard=False, daily_standard=False, min30_standard=False,
        daily_any=True, min30_any=True,
    ) == StructureSignal.LEVEL_TURN
    # 周无,日无,30min背驰 → 次级别回调·保护级
    assert classify_lianli(
        weekly_standard=False, daily_standard=False, min30_standard=False,
        daily_any=False, min30_any=True,
    ) == StructureSignal.SUBLEVEL_PROTECT
    # 背驰后未反向、盘整顺原向 → 小转大/失效
    assert classify_lianli(
        weekly_standard=True, daily_standard=True, min30_standard=True,
        daily_continuation_failed=True,
    ) == StructureSignal.SMALL_TO_BIG
    # 全无
    assert classify_lianli(
        weekly_standard=False, daily_standard=False, min30_standard=False,
    ) == StructureSignal.NONE


# ── §9.2 区间套定位 ───────────────────────────────────────────────────────
def test_interval_nesting_shrinks_to_lowest():
    nest = interval_nesting((10, 30), (15, 25), (18, 22))
    assert nest.nested_ok is True
    assert nest.lowest_level == "min30"
    assert nest.precise_point == 20            # 30min 区间中点

def test_interval_nesting_falls_back_to_daily_when_no_30min():
    nest = interval_nesting((10, 30), (15, 25), None, min30_is_approx=True)
    assert nest.lowest_level == "daily"
    assert nest.precise_point == 20
    assert "近似" in nest.note

def test_interval_nesting_detects_non_nested():
    nest = interval_nesting((10, 30), (15, 35), (18, 22))   # 日 [15,35] 不在周 [10,30] 内
    assert nest.nested_ok is False


# ── §9.4 操作映射:降级共振降一档(顶不清仓/底不重仓)──────────────────────
def test_policy_resonance_top_and_bottom():
    top = map_policy(StructureSignal.RESONANCE, side="top")
    assert top.tier == "最高强度" and "清仓" in top.action
    assert top.stance == Stance.STRONG_REDUCE.value
    bot = map_policy(StructureSignal.RESONANCE, side="bottom")
    assert "重仓" in bot.action and bot.stance == Stance.STRONG_ADD.value


def test_policy_downgraded_steps_down_one_tier():
    top = map_policy(StructureSignal.DOWNGRADED, side="top")
    assert top.tier == "降一档" and "不清仓" in top.action
    assert top.stance == Stance.REDUCE.value and top.downgraded and top.upgradable
    bot = map_policy(StructureSignal.DOWNGRADED, side="bottom")
    assert "不重仓" in bot.action and bot.stance == Stance.ADD.value


def test_policy_outputs_no_position_number():
    # 引擎只给强度档/动作词,不出"算出来的"仓位百分比
    pol = map_policy(StructureSignal.RESONANCE, side="bottom")
    assert "%" not in pol.action and "%" not in pol.tier


# ── §9.5 盲复核旁路:policy_divergence ─────────────────────────────────────
def test_review_divergence_flag():
    pol = map_policy(StructureSignal.RESONANCE, side="bottom")   # stance=strong_add
    agree = ReviewNote(action="加仓", strength="强", reason="底背驰")
    assert compare_review(pol, agree) is False                   # 同为 add 族
    disagree = ReviewNote(action="减仓", strength="中", reason="谨慎")
    assert compare_review(pol, disagree) is True                 # add vs reduce → 分歧


def test_review_note_is_non_deterministic_only_in_notes():
    note = ReviewNote(action="持", strength="中", reason="观望")
    assert note.deterministic is False and note.reproducible is False


# ── 聚合 build_lianli ─────────────────────────────────────────────────────
def test_build_lianli_downgraded_path():
    li = build_lianli(
        weekly_beichi=mk_beichi(), daily_beichi=mk_beichi(), min30_beichi=mk_beichi(),
        min30_is_approx=True, side="bottom",
        interval_nest=interval_nesting((10, 30), (15, 25), None, min30_is_approx=True),
        review=ReviewNote(action="试仓", strength="轻", reason="降级共振谨慎"),
    )
    assert li.structure_signal == StructureSignal.DOWNGRADED.value
    assert li.downgraded is True
    assert li.policy.tier == "降一档"
    assert li.policy_divergence is False         # LLM 试仓(add)与底·降级(add)一致
    assert len(li.review_notes) == 1


def test_build_lianli_real_resonance_path():
    li = build_lianli(
        weekly_beichi=mk_beichi(), daily_beichi=mk_beichi(), min30_beichi=mk_beichi(),
        min30_is_approx=False, side="bottom",
    )
    assert li.structure_signal == StructureSignal.RESONANCE.value
    assert li.downgraded is False
    assert li.policy.tier == "最高强度"
