"""模块 5 · 线段【确定性】特征序列法 + §5.7 七态状态机。

出处:中泰 p30 / 缠师 67、71、78 课。**乙 · 特征序列法**。

核心是 §5.7 的七态状态机(显式转移表,非 if/else 堆),逐笔喂入驱动:
``FORMING / EXTENDING / BREAK_CANDIDATE / WAIT_SECOND_FEATURE /
PENDING_DIRECTION / CONFIRMED_END / INVALIDATED``。

判据:
- §5.1 特征序列元素 = 反向笔,区间 = ``[笔底点, 笔顶点]``;缺口 = 相邻特征元素无重叠。
- §5.2 标准特征序列:仅同序列内包含处理、**绝不跨转折点**;向上线段看顶分型、向下看底分型。
  实现上检测在"创新高/低的上升段"内做包含,遇到回落即定峰(C 不与 B 合并)→ 不跨转折点。
- §5.3 终结:e₁=转折前最后特征元素,e₂=转折点后第一笔。
  第一种(e₁–e₂ **无缺口**)出分型即终结;第二种(**有缺口**)需第二特征序列出反向分型才确认。
- §5.4 成段前提:≥3 笔、前三笔有重叠、顶>底;"含笔数单数"仅作 assert(不筛选)。
- §5.6 右端三类待定显式标注:第二特征序列未出→未确认(WAIT_SECOND_FEATURE);
  第三笔落在第一笔内/方向未定→待定(PENDING_DIRECTION)。

★ §0.5 / §5.7:``pivot`` = 端点极值;``confirm`` = 第一种特征序列分型出现日 /
第二种第二特征序列分型出现日(均晚于 pivot)。**只有 CONFIRMED_END 产出的线段才喂中枢**;
右端非 CONFIRMED_END 的线段标未确认/待定,不入确定性中枢。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .inclusion import DOWN, UP


class XDState(str, Enum):
    FORMING = "FORMING"
    EXTENDING = "EXTENDING"
    BREAK_CANDIDATE = "BREAK_CANDIDATE"
    WAIT_SECOND_FEATURE = "WAIT_SECOND_FEATURE"
    PENDING_DIRECTION = "PENDING_DIRECTION"
    CONFIRMED_END = "CONFIRMED_END"
    INVALIDATED = "INVALIDATED"


# §5.7 status 输出映射
STATUS_BY_STATE: dict[XDState, str] = {
    XDState.FORMING: "未确认",
    XDState.EXTENDING: "未确认·延伸",
    XDState.BREAK_CANDIDATE: "未确认",
    XDState.WAIT_SECOND_FEATURE: "未确认",
    XDState.PENDING_DIRECTION: "待定",
    XDState.CONFIRMED_END: "已确认",
    XDState.INVALIDATED: "未确认",
}

# 同步状态(输入事件=「判定」,无需新笔即可解析)
_SYNC_STATES = {XDState.BREAK_CANDIDATE, XDState.CONFIRMED_END, XDState.INVALIDATED}

_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}


@dataclass
class Pen:
    """喂给线段状态机的笔(区间 [low, high] + 方向 + 端点日期)。"""

    direction: str          # up / down
    high: float
    low: float
    idx: int = -1
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    bi_id: str | None = None

    @property
    def end_price(self) -> float:
        """笔终点价(上升笔=high、下降笔=low)。"""
        return self.high if self.direction == UP else self.low


@dataclass
class _FE:
    """标准特征序列元素(区间 + 贡献极值的笔索引)。"""

    low: float
    high: float
    pen_idx: int


@dataclass
class _Cand:
    """终结候选(转折信息)。"""

    turn_pen_idx: int       # 终结于此笔(线段最后一笔,端点极值在其终点)
    turn_price: float
    gap: bool
    confirm_pen_idx: int | None = None


@dataclass
class XianDuan:
    """一条线段(带 §0.6 通用纪律字段)。"""

    direction: str
    level: str
    state: str              # XDState
    status: str             # 已确认/未确认/待定 …
    feeds_zhongshu: bool    # ★ 仅 CONFIRMED_END 为 True
    start_pen_idx: int
    end_pen_idx: int
    pivot_date: pd.Timestamp | None
    pivot_price: float | None
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    source_unit_ids: list[int] = field(default_factory=list)
    id: str | None = None

    def __post_init__(self):
        if self.confirm_date is not None and self.pivot_date is not None:
            assert self.confirm_date > self.pivot_date, (
                f"confirm_date({self.confirm_date}) 必须严格晚于 "
                f"pivot_date({self.pivot_date})(§0.5 右侧确认)"
            )


# ── 特征序列工具 ──────────────────────────────────────────────────────────
def _fe_contains(a: _FE, b: _FE) -> bool:
    return a.high >= b.high and a.low <= b.low


def _merge(a: _FE, b: _FE, kind: str) -> _FE:
    """同序列内包含合并:顶(向上序列)取 [max低,max高];底(向下序列)取 [min低,min高]。"""
    if kind == "top":
        high, pen = (a.high, a.pen_idx) if a.high >= b.high else (b.high, b.pen_idx)
        return _FE(low=max(a.low, b.low), high=high, pen_idx=pen)
    low, pen = (a.low, a.pen_idx) if a.low <= b.low else (b.low, b.pen_idx)
    return _FE(low=low, high=min(a.high, b.high), pen_idx=pen)


def _scan_fractal(feats: list[_FE], kind: str) -> _Cand | None:
    """在特征序列上找终结分型(向上线段找顶 kind='top',向下找底 'bottom')。

    ★ §5.2 绝不跨转折点:在创新高/低的上升段内做包含合并;一旦回落即定峰(B),
    当前回落元素作 C,**不与 B 合并**。
    """
    std: list[_FE] = []
    for f in feats:
        if std and (_fe_contains(std[-1], f) or _fe_contains(f, std[-1])):
            std[-1] = _merge(std[-1], f, kind)
            continue
        if len(std) >= 2:
            A, B = std[-2], std[-1]
            if kind == "top":
                is_fractal = (
                    f.high < B.high and B.high > A.high
                    and B.low > A.low and B.low > f.low
                )
                if is_fractal:
                    return _Cand(turn_pen_idx=B.pen_idx - 1, turn_price=B.high,
                                 gap=(A.high < B.low))
            else:
                is_fractal = (
                    f.low > B.low and B.low < A.low
                    and B.high < A.high and B.high < f.high
                )
                if is_fractal:
                    return _Cand(turn_pen_idx=B.pen_idx - 1, turn_price=B.low,
                                 gap=(A.low > B.high))
        std.append(f)
    return None


def _features(pens: list[Pen], start: int, end: int, seg_dir: str) -> list[_FE]:
    """取 [start, end) 内反向于 seg_dir 的笔,作为特征序列元素。"""
    return [
        _FE(low=p.low, high=p.high, pen_idx=p.idx)
        for p in pens[start:end]
        if p.direction != seg_dir
    ]


# ── §5.7 状态机 ───────────────────────────────────────────────────────────
class SegmentMachine:
    """逐笔驱动的线段状态机(§5.7)。"""

    def __init__(self, level: str = "daily"):
        self.level = level
        self.pens: list[Pen] = []
        self.seg_start = 0
        self.seg_dir: str | None = None
        self.state = XDState.FORMING
        self.cand: _Cand | None = None
        self.confirmed: list[XianDuan] = []
        self._reeval = False

    # 驱动:喂一笔
    def feed(self, pen: Pen) -> None:
        pen.idx = len(self.pens)
        self.pens.append(pen)
        self.state = _HANDLERS[self.state](self, pen)
        guard = 0
        while True:
            guard += 1
            assert guard < 10000, "状态机未收敛(疑似环路)"
            if self.state in _SYNC_STATES:
                self.state = _HANDLERS[self.state](self, None)
                continue
            if self._reeval and self.state in (XDState.FORMING, XDState.EXTENDING):
                self._reeval = False
                prev = self.state
                self.state = _HANDLERS[self.state](self, pen)
                if self.state != prev or self.state in _SYNC_STATES:
                    self._reeval = True
                    continue
            break

    def feed_all(self, pens: list[Pen]) -> "SegmentMachine":
        for p in pens:
            self.feed(p)
        return self

    # 右端在建线段(始终非 CONFIRMED_END → 不喂中枢)
    def current_segment(self) -> XianDuan | None:
        if self.seg_start >= len(self.pens):
            return None
        end = len(self.pens) - 1
        pivot_date = pivot_price = None
        if self.seg_dir is not None:
            # 在建线段端点暂取当前方向上的极值笔
            seg = self.pens[self.seg_start:]
            if self.seg_dir == UP:
                ext = max(seg, key=lambda p: p.high)
                pivot_price, pivot_date = ext.high, ext.end_date
            else:
                ext = min(seg, key=lambda p: p.low)
                pivot_price, pivot_date = ext.low, ext.end_date
        return XianDuan(
            direction=self.seg_dir or "?", level=self.level,
            state=self.state.value, status=STATUS_BY_STATE[self.state],
            feeds_zhongshu=False, start_pen_idx=self.seg_start, end_pen_idx=end,
            pivot_date=pivot_date, pivot_price=pivot_price,
            confirm_date=None, confirm_price=None, executable_price=None,
            source_unit_ids=list(range(self.seg_start, end + 1)),
        )

    def all_segments(self) -> list[XianDuan]:
        cur = self.current_segment()
        return self.confirmed + ([cur] if cur is not None else [])


# 各状态处理器(显式转移表)──────────────────────────────────────────────────
def _h_forming(m: SegmentMachine, pen: Pen) -> XDState:
    seg = m.pens[m.seg_start:]
    if len(seg) < 3:                                  # 累计 <3 笔
        return XDState.FORMING
    p0, p1, p2 = seg[0], seg[1], seg[2]
    p2_in_p0 = p0.low <= p2.low and p2.high <= p0.high
    overlap = max(p0.low, p2.low) <= min(p0.high, p2.high)
    if p2_in_p0:                                      # §5.6③ 第三笔落在第一笔内 → 待定
        m.seg_dir = p0.direction
        return XDState.PENDING_DIRECTION
    if overlap:                                       # 前三笔有重叠 → 成段方向
        m.seg_dir = p0.direction
        return XDState.EXTENDING
    m.seg_start += 1                                  # 前三笔无重叠 → 顺延起点
    return XDState.FORMING


def _h_extending(m: SegmentMachine, pen: Pen) -> XDState:
    assert m.seg_dir is not None
    feats = _features(m.pens, m.seg_start, len(m.pens), m.seg_dir)
    kind = "top" if m.seg_dir == UP else "bottom"
    cand = _scan_fractal(feats, kind)
    if cand is not None and cand.turn_pen_idx >= m.seg_start:
        m.cand = cand
        return XDState.BREAK_CANDIDATE
    return XDState.EXTENDING


def _h_break_candidate(m: SegmentMachine, _pen) -> XDState:
    assert m.cand is not None
    if not m.cand.gap:                                # 第一种:无缺口 → 即时终结
        m.cand.confirm_pen_idx = len(m.pens) - 1
        return XDState.CONFIRMED_END
    return XDState.WAIT_SECOND_FEATURE                 # 第二种:有缺口 → 等第二特征序列


def _h_wait_second_feature(m: SegmentMachine, pen: Pen) -> XDState:
    assert m.cand is not None and m.seg_dir is not None
    latest = m.pens[-1]
    # 原方向重新创新高/低 → 候选作废
    if m.seg_dir == UP and latest.high > m.cand.turn_price:
        return XDState.INVALIDATED
    if m.seg_dir == DOWN and latest.low < m.cand.turn_price:
        return XDState.INVALIDATED
    # 第二特征序列(新线段方向)出反向分型 → 确认
    new_dir = DOWN if m.seg_dir == UP else UP
    new_start = m.cand.turn_pen_idx + 1
    feats2 = _features(m.pens, new_start, len(m.pens), new_dir)
    kind2 = "bottom" if new_dir == DOWN else "top"
    c2 = _scan_fractal(feats2, kind2)
    if c2 is not None:
        m.cand.confirm_pen_idx = len(m.pens) - 1
        return XDState.CONFIRMED_END
    return XDState.WAIT_SECOND_FEATURE


def _h_pending_direction(m: SegmentMachine, pen: Pen) -> XDState:
    seg = m.pens[m.seg_start:]
    p0 = seg[0]
    latest = m.pens[-1]
    # 先破延续侧(沿 p0 方向创新极值)→ 成段延伸
    if p0.direction == UP and latest.high > p0.high:
        m.seg_dir = UP
        return XDState.EXTENDING
    if p0.direction == DOWN and latest.low < p0.low:
        m.seg_dir = DOWN
        return XDState.EXTENDING
    # 先破终结侧(反向越过 p0 起点端)→ 段方向翻转,自新方向延伸
    if p0.direction == UP and latest.low < seg[1].low:
        m.seg_dir = DOWN
        return XDState.EXTENDING
    if p0.direction == DOWN and latest.high > seg[1].high:
        m.seg_dir = UP
        return XDState.EXTENDING
    return XDState.PENDING_DIRECTION


def _h_confirmed_end(m: SegmentMachine, _pen) -> XDState:
    assert m.cand is not None and m.seg_dir is not None
    turn = m.cand.turn_pen_idx
    confirm_idx = m.cand.confirm_pen_idx
    n_pens = turn - m.seg_start + 1
    assert n_pens % 2 == 1, f"线段含笔数应为单数(§5.4 assert),实得 {n_pens}"  # 仅 assert

    pivot_pen = m.pens[turn]
    confirm_pen = m.pens[confirm_idx]
    seg = XianDuan(
        direction=m.seg_dir, level=m.level,
        state=XDState.CONFIRMED_END.value, status="已确认",
        feeds_zhongshu=True,                          # ★ 仅 CONFIRMED_END 喂中枢
        start_pen_idx=m.seg_start, end_pen_idx=turn,
        pivot_date=pivot_pen.end_date, pivot_price=m.cand.turn_price,
        confirm_date=confirm_pen.end_date, confirm_price=confirm_pen.end_price,
        executable_price=None,                        # 由 df 集成层补 next-open
        source_unit_ids=list(range(m.seg_start, turn + 1)),
    )
    seg.id = f"xianduan_{_LEVEL_CODE.get(m.level, m.level)}_{len(m.confirmed) + 1:03d}"
    m.confirmed.append(seg)

    # 新线段自终结点起(下一笔)
    m.seg_start = turn + 1
    m.seg_dir = None
    m.cand = None
    m._reeval = True
    return XDState.FORMING


def _h_invalidated(m: SegmentMachine, _pen) -> XDState:
    m.cand = None
    return XDState.EXTENDING


_HANDLERS = {
    XDState.FORMING: _h_forming,
    XDState.EXTENDING: _h_extending,
    XDState.BREAK_CANDIDATE: _h_break_candidate,
    XDState.WAIT_SECOND_FEATURE: _h_wait_second_feature,
    XDState.PENDING_DIRECTION: _h_pending_direction,
    XDState.CONFIRMED_END: _h_confirmed_end,
    XDState.INVALIDATED: _h_invalidated,
}


def build_segments(pens: list[Pen], *, level: str = "daily") -> SegmentMachine:
    """便捷入口:喂入笔序列,返回驱动后的状态机(含 confirmed 线段与右端状态)。"""
    return SegmentMachine(level=level).feed_all(pens)
