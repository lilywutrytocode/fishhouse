"""模块 10 · 概率 / LLM 层(§10.1–10.3)。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from chanlun.probability import (
    BacktestSpec,
    LLMLayer,
    LLMMode,
    LLMPermissionError,
    SignalEventRecord,
    assert_unmutated,
    event_schema_complete,
    run_backtest,
    snapshot_deterministic,
    to_backtest_triggers,
    to_signal_event,
)
from chanlun.structure.lianli import ReviewNote
from chanlun.structure.maimaidian import MaiMaiDian

_BASE = pd.Timestamp("2024-01-01", tz="Asia/Shanghai")


def mk_mmd(*, confirmed=True):
    return MaiMaiDian(
        kind="一买", side="buy", level="daily",
        status="confirmed" if confirmed else "待确认", subkind="标准",
        pivot_date=_BASE, pivot_price=8.0,
        confirm_date=_BASE + pd.Timedelta(days=3) if confirmed else None,
        confirm_price=12.0 if confirmed else None,
        executable_price=12.1 if confirmed else None,
        related_zhongshu_id="zs1", related_beichi_id="bc1", id="signal_d_001",
    )


# ── §10.2 事件流 schema 完整 ──────────────────────────────────────────────
def test_signal_event_fields_complete():
    ev = to_signal_event(mk_mmd(), downgraded=True, beichi_grade="标准背驰")
    assert event_schema_complete(ev) is True
    # §0.6 通用 + 扩展字段齐全
    assert ev.executable_price == 12.1
    assert ev.downgraded is True and ev.beichi_grade == "标准背驰"
    assert ev.related_zhongshu_id == "zs1" and ev.related_beichi_id == "bc1"
    assert ev.confirm_date is not None and ev.pivot_date is not None


def test_backtest_uses_only_confirm_and_executable():
    confirmed = to_signal_event(mk_mmd(confirmed=True))
    pending = to_signal_event(mk_mmd(confirmed=False))   # 无 confirm/executable
    triggers = to_backtest_triggers([confirmed, pending])
    assert len(triggers) == 1                            # 待确认事件被剔除,不进回测
    t = triggers[0]
    assert set(t.keys()) == {"id", "confirm_date", "executable_price",
                             "direction", "downgraded"}
    assert "pivot" not in t and "pivot_price" not in t   # ★ 不含 pivot


def test_run_backtest_reserved():
    with pytest.raises(NotImplementedError):
        run_backtest([], BacktestSpec())


def test_backtest_spec_placeholder():
    spec = BacktestSpec()
    assert spec.after_n_days == (5, 10, 20)
    assert "占位" in spec.win_rate_definition


# ── §10.3 LLM 边界 ────────────────────────────────────────────────────────
@dataclass
class _FakeStructure:
    """模拟 bi/xianduan/zhongshu/beichi 的确定字段 + review_notes。"""

    status: str = "未确认"
    confirm_date: object = None
    pivot_price: float = 8.0
    grade: str = "标准背驰"
    review_notes: list = field(default_factory=list)


def test_llm_default_zero_token():
    layer = LLMLayer(client=None)
    assert layer.is_zero_token() is True
    out = layer.review(structure_outputs={}, numeric_snapshot={},
                       mode=LLMMode.NARRATIVE)
    assert out == []                                     # 平时 0 token,不产出


def test_llm_output_only_into_review_notes():
    struct = _FakeStructure()
    before = snapshot_deterministic(struct)
    layer = LLMLayer()
    note = ReviewNote(action="试仓", strength="轻", reason="底背驰盲复核")
    layer.write_structure(struct, "review_notes", note)  # 唯一允许的写
    layer.add_review(note)
    # 结构确定字段未被篡改
    assert_unmutated(struct, before)
    assert struct.review_notes == [note]
    assert layer.review_notes == [note]


def test_llm_cannot_write_deterministic_fields():
    struct = _FakeStructure()
    layer = LLMLayer()
    for f in ("status", "confirm_date", "pivot_price", "grade"):
        with pytest.raises(LLMPermissionError):
            layer.write_structure(struct, f, "篡改")


def test_llm_cannot_flip_unconfirmed_to_confirmed():
    struct = _FakeStructure(status="未确认")
    before = snapshot_deterministic(struct)
    layer = LLMLayer()
    with pytest.raises(LLMPermissionError):
        layer.write_structure(struct, "status", "confirmed")   # 不得把未确认改 confirmed
    assert_unmutated(struct, before)
    assert struct.status == "未确认"


def test_llm_review_note_must_be_non_deterministic():
    layer = LLMLayer()
    bad = ReviewNote(action="加", strength="强", reason="x",
                     deterministic=True, reproducible=True)
    with pytest.raises(LLMPermissionError):
        layer.add_review(bad)


def test_llm_review_with_client_appends_notes_only():
    struct = _FakeStructure()
    before = snapshot_deterministic(struct)

    def fake_client(outputs, snapshot, mode):
        return [ReviewNote(action="持", strength="中", reason="LLM 叙事")]

    layer = LLMLayer(client=fake_client)
    assert layer.is_zero_token() is False
    notes = layer.review(structure_outputs={"bi": []}, numeric_snapshot={},
                         mode=LLMMode.BLIND_REVIEW)
    assert len(notes) == 1 and notes[0].action == "持"
    assert_unmutated(struct, before)                    # 结构未被触碰
