"""端到端集成:一段构造数据跑完整链路,真正产出 beichi/mai_mai_dian/lianli。

断言:买卖点非空;每个已确认信号 executable_price 已填(下一 bar open);
回测触发只用 confirm_date + executable_price,不含 pivot;executable 在线段/背驰/买卖点层落地。
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from chanlun.cli import analyze, run_pipeline
from chanlun.data.models import OHLCV_COLUMNS, validate_canonical
from chanlun.output import output_schema_complete
from chanlun.probability import to_backtest_triggers, to_signal_event
from chanlun.structure.beichi import BeichiType, Grade


def _seg(a, b, n):
    return list(np.linspace(a, b, n + 1))[1:]


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
    out = analyze(df, symbol="TEST", level="daily")
    assert output_schema_complete(out)
    # 不再占位空:背驰 / 买卖点 / 联立 均非空
    assert len(out["beichi"]) >= 1
    assert len(out["mai_mai_dian"]) >= 1
    assert out["lianli"] is not None


def test_signals_have_executable_filled():
    df = make_divergence_df()
    out = analyze(df, symbol="TEST")
    confirmed = [m for m in out["mai_mai_dian"] if m["confirm_date"] is not None]
    assert confirmed, "应产出至少一个已确认买卖点"
    for m in confirmed:
        assert m["executable_price"] is not None        # 下一 bar open 已填
        assert m["confirm_date"] > m["pivot_date"]      # 右侧确认


def test_beichi_is_standard_consolidation_with_executable():
    df = make_divergence_df()
    r = run_pipeline(df)
    bc = r["beichis"][0]
    assert bc.type == BeichiType.CONSOLIDATION.value     # 盘整背驰
    assert bc.grade == Grade.STANDARD.value              # 面积↓且 DIF↓ → 标准档
    assert bc.is_main_signal is True
    assert bc.executable_price is not None                # 背驰层 executable 已填


def test_backtest_triggers_use_confirm_executable_not_pivot():
    df = make_divergence_df()
    r = run_pipeline(df)
    events = [to_signal_event(m) for m in r["maimaidians"]]
    triggers = to_backtest_triggers(events)
    assert len(triggers) >= 1
    for t in triggers:
        assert "confirm_date" in t and "executable_price" in t
        assert t["executable_price"] is not None
        assert not any("pivot" in k for k in t)          # ★ 触发不含 pivot


def test_segment_executable_filled_from_df():
    df = make_divergence_df()
    r = run_pipeline(df)
    confirmed_segs = [s for s in r["segments"]
                      if s.state == "CONFIRMED_END" and s.confirm_date is not None]
    for s in confirmed_segs:
        # 非末根确认的线段应已填 executable(下一 bar open)
        if s.confirm_date != df.index[-1]:
            assert s.executable_price is not None


def test_lianli_non_empty_records_daily_beichi():
    # 单级别 CSV(无 30min)→ 跨级别共振信号合规为"无"(§9.3 需 30min 参与);
    # 但 lianli 非空:记录日线背驰档 + 主观 policy。
    df = make_divergence_df()
    out = analyze(df, symbol="TEST")
    li = out["lianli"]
    assert li is not None
    assert li["level_beichi"]["daily"] == "标准背驰"     # 日线档已记录
    assert li["structure_signal"] == "无"               # 缺 30min → 无最高强度共振
    assert "policy" in li and "tier" in li["policy"]
