"""端到端集成:一段构造数据跑完整链路,真正产出 beichi/mai_mai_dian/lianli。

断言:买卖点非空;每个已确认信号 executable_price 已填(下一 bar open);
回测触发只用 confirm_date + executable_price,不含 pivot;executable 在线段/背驰/买卖点层落地。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from chanlun.cli import analyze, run_pipeline
from chanlun.config import Config
from chanlun.data.models import OHLCV_COLUMNS, validate_canonical
from chanlun.data.weekly import synthesize_weekly
from chanlun.output import output_schema_complete
from chanlun.probability import to_backtest_triggers, to_signal_event
from chanlun.structure.beichi import BeichiType, Grade
from chanlun.structure.lianli import StructureSignal
from tests.conftest import weekdays


def _seg(a, b, n):
    return list(np.linspace(a, b, n + 1))[1:]


# 信号逻辑测试用:关闭 MACD 暖机守卫(暖机守卫另有专测)
NO_WARMUP = Config(macd_warmup_factor=0)


def make_divergence_df(tz="Asia/Shanghai") -> pd.DataFrame:
    """构造下跌盘整背驰:逐次更低低点但跌势减弱(MACD 面积/DIF 收缩)+ 回抽确认。"""
    pts = [100, 70, 85, 68, 82, 66, 78, 67, 76]   # 低点 70/68/66 递低且跌幅减弱
    closes = [pts[0]]
    for k in range(1, len(pts)):
        closes += _seg(pts[k - 1], pts[k], 6)
    d = date(2024, 1, 1)
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 100, "amount": 1.0} for c in closes]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex([pd.Timestamp(d + timedelta(days=i))
                                 for i in range(len(closes))],
                                name="date").tz_localize(tz)
    validate_canonical(df)
    return df


def test_end_to_end_produces_signals():
    df = make_divergence_df()
    out = analyze(df, symbol="TEST", level="daily", config=NO_WARMUP)
    assert output_schema_complete(out)
    # 不再占位空:背驰 / 买卖点 / 联立 均非空
    assert len(out["beichi"]) >= 1
    assert len(out["mai_mai_dian"]) >= 1
    assert out["lianli"] is not None


def test_signals_have_executable_filled():
    df = make_divergence_df()
    out = analyze(df, symbol="TEST", config=NO_WARMUP)
    confirmed = [m for m in out["mai_mai_dian"] if m["confirm_date"] is not None]
    assert confirmed, "应产出至少一个已确认买卖点"
    for m in confirmed:
        assert m["executable_price"] is not None        # 下一 bar open 已填
        assert m["confirm_date"] > m["pivot_date"]      # 右侧确认


def test_beichi_is_standard_consolidation_with_executable():
    df = make_divergence_df()
    r = run_pipeline(df, config=NO_WARMUP)
    bc = r["beichis"][0]
    assert bc.type == BeichiType.CONSOLIDATION.value     # 盘整背驰
    assert bc.grade == Grade.STANDARD.value              # 面积↓且 DIF↓ → 标准档
    assert bc.is_main_signal is True
    assert bc.executable_price is not None                # 背驰层 executable 已填


def test_backtest_triggers_use_confirm_executable_not_pivot():
    df = make_divergence_df()
    r = run_pipeline(df, config=NO_WARMUP)
    events = [to_signal_event(m) for m in r["maimaidians"]]
    triggers = to_backtest_triggers(events)
    assert len(triggers) >= 1
    for t in triggers:
        assert "confirm_date" in t and "executable_price" in t
        assert t["executable_price"] is not None
        assert not any("pivot" in k for k in t)          # ★ 触发不含 pivot


def test_segment_executable_filled_from_df():
    df = make_divergence_df()
    r = run_pipeline(df, config=NO_WARMUP)
    confirmed_segs = [s for s in r["segments"]
                      if s.state == "CONFIRMED_END" and s.confirm_date is not None]
    for s in confirmed_segs:
        # 非末根确认的线段应已填 executable(下一 bar open)
        if s.confirm_date != df.index[-1]:
            assert s.executable_price is not None


def make_df_from_points(pts, leg=6, tz="Asia/Shanghai") -> pd.DataFrame:
    closes = [pts[0]]
    for k in range(1, len(pts)):
        closes += _seg(pts[k - 1], pts[k], leg)
    days = weekdays(date(2020, 1, 6), len(closes))
    rows = [{"open": c, "high": c + 1, "low": c - 1, "close": c,
             "volume": 100, "amount": 1.0} for c in closes]
    df = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
    df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in days],
                                name="date").tz_localize(tz)
    validate_canonical(df)
    return df


def _signal_objs(df):
    return run_pipeline(df, config=NO_WARMUP)["maimaidians"]


# ── 趋势背驰一买(≥2 中枢)──────────────────────────────────────────────────
def test_trend_first_buy_fires():
    df = make_df_from_points(
        [100, 80, 92, 82, 90, 81, 88, 62, 72, 64, 70, 63, 71, 58, 64])
    r = run_pipeline(df, config=NO_WARMUP)
    assert any(b.type == BeichiType.TREND.value for b in r["beichis"])  # ≥2 中枢 → 趋势背驰
    firsts = [m for m in r["maimaidians"] if m.kind == "一买" and m.subkind == "标准"]
    assert firsts, "应 fire 趋势子类一买"
    m = firsts[0]
    assert m.executable_price is not None
    assert m.confirm_date is not None and m.confirm_date > m.pivot_date
    assert m.related_zhongshu_id and m.related_beichi_id        # 引用 id 齐全


# ── §8.1 强度档闸:弱档(面积/DIF)趋势背驰 → 一买·弱、不进主信号 ──────────────
def test_weak_trend_beichi_marks_first_buy_weak():
    df = make_df_from_points(
        [100, 80, 92, 82, 90, 81, 88, 62, 72, 64, 70, 63, 71, 58, 64])
    r = run_pipeline(df, config=NO_WARMUP)
    trend_bcs = [b for b in r["beichis"] if b.type == BeichiType.TREND.value]
    assert trend_bcs
    tb = trend_bcs[0]
    assert tb.grade in (Grade.AREA.value, Grade.DIF.value)      # 仅面积/DIF 档(弱)
    assert tb.is_main_signal is False                          # 弱档背驰非主信号
    weak = [m for m in r["maimaidians"]
            if m.kind == "一买" and m.related_beichi_id == tb.id]
    assert weak, "弱档趋势背驰应仍产出一买(标弱)"
    m = weak[0]
    assert m.strength == "弱"                                   # 标 弱
    assert m.is_main is False                                   # ★ 不进主信号
    assert m.label == "一买·弱" and m.label != "一买·标准"
    assert m.subkind == "标准"                                  # 趋势子类正交保留


def test_consolidation_first_buy_is_main_when_standard_grade():
    # 对照:盘整标准背驰 → 一买·盘背、主信号
    df = make_divergence_df()
    r = run_pipeline(df, config=NO_WARMUP)
    firsts = [m for m in r["maimaidians"] if m.kind == "一买"]
    assert firsts
    m = firsts[0]
    assert m.beichi_grade == Grade.STANDARD.value
    assert m.strength == "标准" and m.is_main is True
    assert m.label == "一买·盘背"


# ── 二买(§8.2 五步)──────────────────────────────────────────────────────
def test_second_buy_fires():
    df = make_df_from_points([100, 70, 85, 68, 82, 66, 78, 67, 76])
    seconds = [m for m in _signal_objs(df) if m.kind == "二买"]
    assert seconds, "应 fire 二买"
    m = seconds[0]
    assert m.status == "confirmed"
    assert m.executable_price is not None
    assert m.confirm_date > m.pivot_date
    assert m.related_zhongshu_id and m.related_beichi_id        # 承一买引用


# ── 三买(§8.3 离开/回试 + leave/retest id)───────────────────────────────
def test_third_buy_fires():
    df = make_df_from_points([70, 80, 72, 82, 74, 84, 76, 95, 88, 96])
    thirds = [m for m in _signal_objs(df) if m.kind == "三买"]
    assert thirds, "应 fire 三买"
    m = thirds[0]
    assert m.status == "confirmed"
    assert m.executable_price is not None
    assert m.confirm_date > m.pivot_date
    assert m.related_zhongshu_id                                # 关联中枢
    assert m.related_leave_unit_id and m.related_retest_unit_id  # leave/retest id 齐全
    assert m.pivot_price > 0


# ── ① 日+周标准背驰 + 30min 缺失 → 日周共振·待30min,降一档(降级隔离)─────────
def test_daily_weekly_resonance_pending_30min():
    # 长日线下跌(低点递低、跌势减弱),合成周线后日/周同向标准背驰;30min 缺失 → 不顶格
    df = make_df_from_points([100, 68, 84, 66, 80, 64, 76, 63, 72], leg=20)
    wdf = synthesize_weekly(df)
    assert len(wdf) >= 20                               # 周线由日线合成且足量
    r = run_pipeline(df, config=NO_WARMUP)
    assert any(b.is_main_signal for b in r["beichis"])          # 日线标准背驰
    assert any(b.is_main_signal for b in r["weekly_beichis"])   # 周线标准背驰
    li = r["lianli"]
    # ★ 30min 缺失 → 不得判共振·最高强度;降为 日周共振·待30min、降一档
    assert li.structure_signal == StructureSignal.PENDING_30MIN.value
    assert li.structure_signal != StructureSignal.RESONANCE.value
    assert li.policy.tier == "降一档"
    assert li.policy.stance == "add"                   # 底 → 分批不重仓(非 strong_add)
    assert "不重仓" in li.policy.action
    assert li.downgraded is True
    assert li.min30_status == "缺失"
    assert li.level_beichi["daily"] == "标准背驰"
    assert li.level_beichi["min30"] is None


def test_three_level_real_30min_gives_top_resonance():
    # 顶格只在三级齐全(真 30min)时出现:周+日+真 30min 标准背驰 → 共振·最高强度
    from types import SimpleNamespace

    from chanlun.structure.lianli import build_lianli
    bc = SimpleNamespace(grade="标准背驰", beichi_status="confirmed",
                         is_main_signal=True, id="bc")
    li = build_lianli(weekly_beichi=bc, daily_beichi=bc, min30_beichi=bc,
                      min30_is_approx=False, side="bottom")
    assert li.structure_signal == StructureSignal.RESONANCE.value
    assert li.policy.tier == "最高强度"
    assert li.policy.stance == "strong_add"
    assert li.min30_status == "真30min"
    assert li.downgraded is False

    # 30min 用日线内部近似 → 降级共振(降一档),不顶格
    li2 = build_lianli(weekly_beichi=bc, daily_beichi=bc, min30_beichi=bc,
                       min30_is_approx=True, side="bottom")
    assert li2.structure_signal == StructureSignal.DOWNGRADED.value
    assert li2.policy.tier == "降一档"
    assert li2.min30_status == "近似"


def test_weekly_synthesized_from_daily_anchored_friday():
    df = make_df_from_points([100, 68, 84, 66, 80, 64, 76, 63, 72], leg=20)
    wdf = synthesize_weekly(df)
    assert wdf.index[0].weekday() == 4                  # §1.9 锚定周五
    validate_canonical(wdf)


def test_daily_only_beichi_is_level_turn_not_resonance():
    # ② 只有日线标准背驰、周线无背驰 → 本级别转折成立(而非共振)
    df = make_divergence_df()                          # 短日线 → 合成周线不足以成背驰
    r = run_pipeline(df, config=NO_WARMUP)
    assert not any(b.is_main_signal for b in r["weekly_beichis"])  # 周线无主背驰
    li = r["lianli"]
    assert li is not None
    assert li.structure_signal == StructureSignal.LEVEL_TURN.value  # 本级别转折成立
    assert li.structure_signal != StructureSignal.RESONANCE.value
    assert li.policy.tier != "最高强度"               # 非最高强度


def test_policy_filters_weak_signal_no_main_action():
    # policy 层统一按 is_main 过滤:仅弱背驰背景(非主信号)→ 不进任何主信号动作
    from types import SimpleNamespace

    from chanlun.cli import build_lianli_two_level
    from chanlun.structure.inclusion import DOWN

    weak = SimpleNamespace(grade="面积背驰", beichi_status="confirmed",
                           is_main_signal=False, id="bc")
    li = build_lianli_two_level([(weak, None, DOWN)], [], level="daily")
    assert li.structure_signal == StructureSignal.NONE.value
    assert li.policy.tier != "最高强度"
    assert li.policy.stance == "hold"                  # 弱信号 → 观望,无主动作
