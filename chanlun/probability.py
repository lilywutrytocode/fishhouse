"""模块 10 · 概率 / LLM 层【约定】

§10.1 先跑 **A+C**:A=结构=确定层(模块 1–9);C=LLM 盲复核/叙事=主观层。**B 回测留接口**后加。

- §10.2 回测接口(预留·接口锁)::class:`SignalEventRecord` 信号事件流 schema =
  §0.6 通用字段 + ``executable_price`` + 是否降级 + 背驰档 + ``related_*``。
  ★ **B 统计只用 ``confirm_date`` + ``executable_price``**(严禁 ``pivot_*``);
  统计口径占位(胜率定义 / 后 N 日 N=5/10/20 / 样本范围)。``run_backtest`` 暂 ``NotImplementedError``。
- §10.3 LLM 边界:平时 **0 token**;仅 叙事 / 盲复核 / 对 ``[需判断]`` 分支给建议 三情形;
  输出**只进 ``review_notes``**,不写 ``bi/xianduan/zhongshu/beichi/mai_mai_dian`` 确定字段、
  不把 ``未确认/待定`` 改 ``confirmed``;每次留痕标 ``非确定·非复现``(同 §9.5)。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import pandas as pd

from .structure.lianli import ReviewNote


# ── §10.2 信号事件流 schema(回测接口锁)──────────────────────────────────
@dataclass
class SignalEventRecord:
    """回测信号事件(§0.6 通用字段 + §10.2 扩展)。"""

    # §0.6 通用字段
    id: str
    parent_id: str | None
    source_unit_ids: list
    level: str
    direction: str
    status: str
    pivot_date: pd.Timestamp | None
    pivot_price: float | None
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    # §10.2 扩展
    kind: str | None = None
    subkind: str | None = None
    downgraded: bool = False
    beichi_grade: str | None = None
    related_zhongshu_id: str | None = None
    related_beichi_id: str | None = None
    related_leave_unit_id: str | None = None
    related_retest_unit_id: str | None = None
    supporting_signals: list = field(default_factory=list)


# 事件流必备字段(供 schema 完整性校验)
REQUIRED_EVENT_FIELDS = (
    "id", "parent_id", "source_unit_ids", "level", "direction", "status",
    "pivot_date", "pivot_price", "confirm_date", "confirm_price", "executable_price",
    "executable_price", "downgraded", "beichi_grade",
    "related_zhongshu_id", "related_beichi_id",
    "related_leave_unit_id", "related_retest_unit_id", "supporting_signals",
)


def to_signal_event(
    mmd, *, downgraded: bool = False, beichi_grade: str | None = None,
    parent_id: str | None = None, source_unit_ids: list | None = None,
) -> SignalEventRecord:
    """买卖点(MaiMaiDian)→ 回测信号事件。"""
    return SignalEventRecord(
        id=mmd.id, parent_id=parent_id, source_unit_ids=source_unit_ids or [],
        level=mmd.level, direction=mmd.side, status=mmd.status,
        pivot_date=mmd.pivot_date, pivot_price=mmd.pivot_price,
        confirm_date=mmd.confirm_date, confirm_price=mmd.confirm_price,
        executable_price=mmd.executable_price,
        kind=mmd.kind, subkind=mmd.subkind, downgraded=downgraded,
        beichi_grade=beichi_grade,
        related_zhongshu_id=mmd.related_zhongshu_id,
        related_beichi_id=mmd.related_beichi_id,
        related_leave_unit_id=mmd.related_leave_unit_id,
        related_retest_unit_id=mmd.related_retest_unit_id,
        supporting_signals=list(mmd.supporting_signals),
    )


def event_schema_complete(record: SignalEventRecord) -> bool:
    """校验事件流字段完整(§10.2)。"""
    present = {f.name for f in fields(record)}
    return all(name in present for name in REQUIRED_EVENT_FIELDS)


# ── §10.2 回测接口(预留;B 后加)──────────────────────────────────────────
@dataclass
class BacktestSpec:
    """B 统计口径占位(接口锁)。"""

    after_n_days: tuple = (5, 10, 20)
    win_rate_definition: str = "占位:后 N 日 executable_price 相对收益 > 0 为胜(待定稿)"
    sample_range: tuple | None = None


def to_backtest_triggers(events: list[SignalEventRecord]) -> list[dict]:
    """★ 只用 ``confirm_date`` + ``executable_price`` 抽取可回测触发(严禁 pivot)。

    无 confirm_date 或无 executable_price(右端 live_pending/pending/未确认)的事件被剔除,
    不进回测。
    """
    triggers = []
    for e in events:
        if e.confirm_date is None or e.executable_price is None:
            continue
        triggers.append({
            "id": e.id, "confirm_date": e.confirm_date,
            "executable_price": e.executable_price, "direction": e.direction,
            "downgraded": e.downgraded,
        })
    return triggers


def run_backtest(events, spec: BacktestSpec | None = None):
    """B 回测(预留)。接口已锁:只用 confirm_date + executable_price。"""
    raise NotImplementedError(
        "B 回测后加(§10.1);接口已锁:统计只用 confirm_date + executable_price"
    )


# ── §10.3 LLM 边界 ────────────────────────────────────────────────────────
class LLMMode(str, Enum):
    NARRATIVE = "叙事"
    BLIND_REVIEW = "盲复核"
    ADVICE = "需判断建议"


# LLM 绝不可写的结构确定字段(§9.5 / §10.3)
FORBIDDEN_STRUCTURE_FIELDS = frozenset({
    "direction", "status", "state", "grade", "beichi_status", "kind", "subkind",
    "pivot_date", "pivot_price", "confirm_date", "confirm_price", "executable_price",
    "ZG", "ZD", "GG", "DD", "feeds_zhongshu",
})


class LLMPermissionError(PermissionError):
    """LLM 试图写结构确定字段或确认未确认结构时抛出。"""


class LLMLayer:
    """C 层:LLM 盲复核 / 叙事 / 建议。平时 0 token;输出只进 ``review_notes``。"""

    def __init__(self, client=None):
        # client 为 None → 不调用、0 token
        self._client = client
        self.review_notes: list[ReviewNote] = []

    def is_zero_token(self) -> bool:
        return self._client is None

    def add_review(self, note: ReviewNote) -> None:
        """把 LLM 留痕加入 review_notes(必须 非确定·非复现)。"""
        if note.deterministic or note.reproducible:
            raise LLMPermissionError("LLM 留痕必须标 非确定·非复现(§10.3)")
        self.review_notes.append(note)

    def write_structure(self, obj, field_name: str, value) -> None:
        """LLM 唯一允许的"写"= review_notes;写任何结构确定字段一律拒绝。"""
        if field_name in FORBIDDEN_STRUCTURE_FIELDS:
            raise LLMPermissionError(
                f"LLM 不得写结构确定字段 {field_name!r}(§9.5/§10.3)"
            )
        if field_name != "review_notes":
            raise LLMPermissionError("LLM 只能写 review_notes")
        getattr(obj, "review_notes").append(value)

    def review(self, *, structure_outputs, numeric_snapshot, mode: LLMMode) -> list:
        """三情形复核入口。无 client → 返回空(0 token),不触碰结构。"""
        if self._client is None:
            return []
        notes = self._client(structure_outputs, numeric_snapshot, mode)  # 主观调用
        for n in notes:
            self.add_review(n)
        return list(self.review_notes)


def snapshot_deterministic(obj) -> dict:
    """对结构对象拍快照,用于校验 LLM 未篡改确定字段。"""
    return {f: getattr(obj, f) for f in FORBIDDEN_STRUCTURE_FIELDS if hasattr(obj, f)}


def assert_unmutated(obj, before: dict) -> None:
    for f, v in before.items():
        assert getattr(obj, f) == v, f"结构确定字段 {f!r} 被篡改(违反 §10.3)"
