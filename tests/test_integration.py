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


def test_monitor_uses_latest_zhongshu_and_first_buy():
    # 确定层 bug 回归:监控位取时间最近的中枢与最近的一买(非拼接顺序最后/最早)
    from chanlun.data.loaders import load_local_csv

    df = load_local_csv("chanlun/data/300502_daily_long.csv", level="daily").df
    r = run_pipeline(df)
    zss = r["zhongshus"]
    firsts = [m for m in r["maimaidians"] if m.kind == "一买"]
    assert zss and firsts
    latest_zs = max(zss, key=lambda z: z.confirm_date)
    latest_buy = max(firsts, key=lambda m: m.pivot_date)
    caution = [m for m in r["monitor"] if m.tier == "caution"]
    rea = [m for m in r["monitor"] if m.tier == "reassessment"]
    assert caution and caution[0].price in (latest_zs.ZG, latest_zs.ZD)
    assert rea and rea[0].price == latest_buy.pivot_price


def test_policy_filters_weak_signal_no_main_action():
    # policy 层统一按 is_main 过滤:仅弱背驰背景(非主信号)→ 不进任何主信号动作
    from types import SimpleNamespace

    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import DOWN

    weak = SimpleNamespace(grade="面积背驰", beichi_status="confirmed",
                           is_main_signal=False, id="bc")
    li = build_lianli_nested([(weak, None, DOWN)], [], level="daily")
    assert li.structure_signal == StructureSignal.NONE.value
    assert li.policy.tier != "最高强度"
    assert li.policy.stance == "hold"                  # 弱信号 → 观望,无主动作


# ── (B) §1.10 一致性门禁:基准不一致的日-30min 不进联立 ─────────────────────
def test_consistency_gate_rejects_mixed_adjustment_basis():
    from chanlun.data.loaders import load_local_csv
    ld = load_local_csv("chanlun/data/300502_daily_long.csv", level="daily").df
    sd = load_local_csv(
        "chanlun/data/xinyisheng_300502_daily_20250701_20260609_macd.csv",
        level="daily").df
    m30 = load_local_csv(
        "chanlun/data/xinyisheng_300502_30min_20260408_20260610_macd.csv",
        level="min30").df
    # 长日线与 30min 前复权基准不一致 → REJECT_LIANLI(30min 不进联立)
    o_bad = analyze(ld, symbol="300502", min30_df=m30)
    assert o_bad["min30_consistency"] == "REJECT_LIANLI"
    assert o_bad["lianli"]["min30_status"] == "缺失"        # 30min 未参与
    # 短日线与 30min 一致 → OK
    o_ok = analyze(sd, symbol="300502", min30_df=m30)
    assert o_ok["min30_consistency"] == "OK"


# ── (A) §9.2 区间套嵌套:旧周线背驰不嵌套当前日线 → 右端无共振 ───────────────
def _mkbc(pivot, confirm, start, grade="标准背驰"):
    from types import SimpleNamespace
    return SimpleNamespace(
        grade=grade, beichi_status="confirmed",
        is_main_signal=(grade == "标准背驰"),
        pivot_date=pd.Timestamp(pivot, tz="Asia/Shanghai"),
        confirm_date=pd.Timestamp(confirm, tz="Asia/Shanghai"),
        seg_start_date=pd.Timestamp(start, tz="Asia/Shanghai"), id="bc")


def test_stale_weekly_beichi_does_not_resonate():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import DOWN
    daily = _mkbc("2026-02-02", "2026-02-03", "2026-01-10")   # 当前日线底背驰
    weekly_stale = _mkbc("2023-02-10", "2023-02-24", "2022-10-14")  # 旧周线底背驰
    li = build_lianli_nested([(daily, None, DOWN)], [(weekly_stale, None, DOWN)],
                             level="daily")
    # 旧周线段不含当前日线背驰点 → 不嵌套 → 本级别转折(非共振/待30min)
    assert li.structure_signal == StructureSignal.LEVEL_TURN.value


def test_current_weekly_nests_daily_pending_30min():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import DOWN
    daily = _mkbc("2026-02-02", "2026-02-10", "2026-01-10")
    weekly_now = _mkbc("2026-01-15", "2026-03-01", "2025-11-01")  # 段含日线背驰点
    li = build_lianli_nested([(daily, None, DOWN)], [(weekly_now, None, DOWN)],
                             level="daily")
    # 周线段嵌套当前日线 + 30min 缺失 → 日周共振·待30min(降一档)
    assert li.structure_signal == StructureSignal.PENDING_30MIN.value
    assert li.policy.tier == "降一档"


def test_three_level_nesting_real_30min_resonance():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import DOWN
    daily = _mkbc("2026-02-02", "2026-02-10", "2026-01-10")
    weekly_now = _mkbc("2026-01-15", "2026-03-01", "2025-11-01")
    m30 = _mkbc("2026-02-03", "2026-02-05", "2026-01-20")      # 嵌于日线段内
    li = build_lianli_nested([(daily, None, DOWN)], [(weekly_now, None, DOWN)],
                             level="daily", min30_tuples=[(m30, None, DOWN)])
    # 三级时间嵌套齐 + 真30min → 共振·最高强度
    assert li.structure_signal == StructureSignal.RESONANCE.value
    assert li.policy.tier == "最高强度"
    assert li.min30_status == "真30min"


# ── §9.3 锚点背驰失效(小转大):确认后价格顺原向越过 pivot → 信号失效 ─────────
def _price_df(dates_closes, tz="Asia/Shanghai"):
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz=tz) for d, _ in dates_closes], name="date")
    return pd.DataFrame({"close": [c for _, c in dates_closes]}, index=idx)


def test_anchor_top_beichi_invalidated_when_price_exceeds_pivot():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import UP
    top = _mkbc("2026-02-02", "2026-02-03", "2025-11-24")     # 顶背驰 pivot 默认...
    top.pivot_price = 336.0
    px = _price_df([("2026-03-02", 400.0), ("2026-06-12", 506.0)])  # 后续超越 336
    li = build_lianli_nested([(top, None, UP)], [], level="daily", price_df=px)
    assert li.structure_signal == StructureSignal.SMALL_TO_BIG.value   # 信号失效
    assert li.policy.stance == "hold"                                  # 持有/顺势
    assert "持有" in li.policy.action and "减仓" not in li.policy.action.replace("撤销减仓", "")


def test_anchor_bottom_beichi_invalidated_when_price_breaks_pivot():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import DOWN
    bot = _mkbc("2024-01-10", "2024-01-12", "2023-11-01")
    bot.pivot_price = 50.0
    px = _price_df([("2024-02-01", 45.0), ("2024-03-01", 40.0)])       # 后续跌破 50
    li = build_lianli_nested([(bot, None, DOWN)], [], level="daily", price_df=px)
    assert li.structure_signal == StructureSignal.SMALL_TO_BIG.value


def test_anchor_not_invalidated_when_price_reverses():
    from chanlun.cli import build_lianli_nested
    from chanlun.structure.inclusion import UP
    top = _mkbc("2026-02-02", "2026-02-03", "2025-11-24")
    top.pivot_price = 336.0
    px = _price_df([("2026-03-02", 300.0), ("2026-04-01", 280.0)])     # 反向(未越 pivot)
    li = build_lianli_nested([(top, None, UP)], [], level="daily", price_df=px)
    assert li.structure_signal != StructureSignal.SMALL_TO_BIG.value   # 未失效 → 本级别转折
    assert li.structure_signal == StructureSignal.LEVEL_TURN.value


def test_300502_right_end_anchor_beichi_invalidated():
    from chanlun.data.loaders import load_local_csv
    daily = load_local_csv("chanlun/data/raw/300502/300502_daily.csv", level="daily").df
    m30 = load_local_csv("chanlun/data/raw/300502/300502_30min.csv", level="min30").df
    o = analyze(daily, symbol="300502", min30_df=m30)
    # 日线顶背驰 336@2026-02 被随后涨至 506 超越 → 信号失效/顺势,而非减仓
    assert o["lianli"]["structure_signal"] == StructureSignal.SMALL_TO_BIG.value
    assert o["lianli"]["policy"]["stance"] == "hold"


# ── 回归①:同一 A/C 被多个相邻中枢对命中 → 只输出一个趋势背驰(不重复)──────────
def test_trend_beichi_deduplicated_300750():
    from chanlun.data.loaders import load_local_csv
    df = load_local_csv(
        "chanlun/data/raw/300750/300750_qfq_daily_20210101_20230731.csv",
        level="daily").df
    r = run_pipeline(df)
    trend = [b for b in r["beichis"] if b.type == BeichiType.TREND.value]
    # 同 (a_unit_id,c_unit_id,pivot,confirm) 不得重复
    keys = [(b.a_unit_id, b.c_unit_id, b.pivot_date, b.confirm_date) for b in trend]
    assert len(keys) == len(set(keys)), "趋势背驰存在重复"
    # 由该趋势背驰派生的买卖点也不重复(同 pivot/confirm/kind)
    sig_keys = [(m.kind, m.pivot_date, m.confirm_date, m.pivot_price)
                for m in r["maimaidians"] if m.subkind == "标准"]
    assert len(sig_keys) == len(set(sig_keys)), "趋势派生买卖点存在重复"
    # 该弱档趋势背驰派生的卖点应标 ·弱,不得标 ·标准
    for m in r["maimaidians"]:
        if m.subkind == "标准" and m.strength == "弱":
            assert m.label.endswith("·弱")
            assert not m.label.endswith("·标准")
