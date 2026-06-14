"""模块 3 · 分型【确定性】

出处:中泰 p22 / 缠师 65 课。在**包含处理后的标准 K**(模块 2)上识别顶/底分型。

判据(严格按 §3 + §0.5 防未来函数):
- §3.1 定义:
  - 顶分型:中间标准 K 的**高点最高且低点最高**(严格高于左右两根)。
  - 底分型:中间标准 K 的**低点最低且高点最低**(严格低于左右两根)。
  - ``pivot`` = 中间 K 的极值(顶取 high、底取 low);``pivot_date`` 取贡献该极值的
    **原始 K 日期**(模块 2 的 ``high_idx/low_idx``,同价取最先)。
  - ``confirm`` = **第三根标准 K 收盘**:``confirm_date`` = 第三根标准 K 的末根原始 K 日期,
    ``confirm_price`` = 该 bar 的 close。confirm 严格晚于 pivot(右侧确认)。
- §3.2 相邻同类取舍【约定】:连续顶取价格最高、连续底取价格最低,**同价取最先**(详见 §4.3)。
  ★ tie-break 同价取最先**只影响 pivot 选择,不提前 confirm_date**(survivor 保留自身 confirm)。

★ 防未来函数(§0.5):
- ``executable_price`` = confirm bar **下一根原始 K 的 open**(日线=下一交易日 open)。
- confirm bar 即末根 bar(无下一根)→ ``executable_price=None``、status=``live_pending``,不入回测。
- 第三根标准 K 尚未形成的右端 → status=``pending``(待定),``confirm_*``/executable 均 None。
- 回测/实盘只准用 ``confirm_date`` + ``executable_price``,严禁用 ``pivot_*`` 触发。

注:在合法的标准 K 序列上,严格顶/底定义使分型**天然交替**(两顶之间必有一底),
故 §3.2 的"连续同类去重"在纯分型层通常为空操作,主要供模块 4 笔在过滤掉中间分型后调用;
此处实现为可独立调用的规则(:func:`dedup_consecutive_same_type`)并单测之。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .inclusion import MergedK


class FractalKind(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"


TOP = FractalKind.TOP.value
BOTTOM = FractalKind.BOTTOM.value


class FractalStatus(str, Enum):
    CONFIRMED = "confirmed"        # 三根标准 K 齐备、confirm bar 非末根 → 可入回测
    LIVE_PENDING = "live_pending"  # 结构齐备但 confirm bar 是末根 → 不入回测(§0.5)
    PENDING = "pending"            # 右端第三根标准 K 未形成 → 待定


CONFIRMED = FractalStatus.CONFIRMED.value
LIVE_PENDING = FractalStatus.LIVE_PENDING.value
PENDING = FractalStatus.PENDING.value

_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}


@dataclass
class Fractal:
    """一个分型结构(带 §0.6 通用纪律字段)。"""

    kind: str                    # FractalKind
    level: str
    status: str                  # FractalStatus
    mid_k: int                   # 中间标准 K 位置(分型定位)
    pivot_date: pd.Timestamp
    pivot_price: float
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    source_unit_ids: list[int]   # 覆盖的原始 K 位置(左/中/右三根标准 K)
    left_k: int | None = None
    right_k: int | None = None
    id: str | None = None

    def __post_init__(self):
        # 防未来函数:确认必须严格晚于极值(分型需右侧第三根 K 确认)
        if self.confirm_date is not None:
            assert self.confirm_date > self.pivot_date, (
                f"confirm_date({self.confirm_date}) 必须严格晚于 "
                f"pivot_date({self.pivot_date})(§0.5 右侧确认)"
            )


def _close_at(df: pd.DataFrame, ts: pd.Timestamp) -> float:
    return float(df.loc[ts, "close"])


def _open_after(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    pos = df.index.get_loc(ts)
    if pos + 1 < len(df):
        return float(df.iloc[pos + 1]["open"])
    return None


def _make_fractal(kind: str, i: int, merged: list[MergedK], df: pd.DataFrame,
                  level: str) -> Fractal:
    L, M, R = merged[i - 1], merged[i], merged[i + 1]
    if kind == TOP:
        pivot_date = df.index[M.high_idx]
        pivot_price = M.high
    else:
        pivot_date = df.index[M.low_idx]
        pivot_price = M.low

    # confirm = 第三根标准 K 收盘(末根原始 K 的 close)
    confirm_date = df.index[R.raw_indices[-1]]
    confirm_price = _close_at(df, confirm_date)
    executable_price = _open_after(df, confirm_date)
    status = CONFIRMED if executable_price is not None else LIVE_PENDING

    source_unit_ids = sorted(L.raw_indices + M.raw_indices + R.raw_indices)
    return Fractal(
        kind=kind, level=level, status=status, mid_k=i,
        pivot_date=pivot_date, pivot_price=float(pivot_price),
        confirm_date=confirm_date, confirm_price=confirm_price,
        executable_price=executable_price, source_unit_ids=source_unit_ids,
        left_k=i - 1, right_k=i + 1,
    )


def _classify(L: MergedK, M: MergedK, R: MergedK) -> str | None:
    """§3.1:中间 K 是顶/底/都不是。"""
    if M.high > L.high and M.high > R.high and M.low > L.low and M.low > R.low:
        return TOP
    if M.low < L.low and M.low < R.low and M.high < L.high and M.high < R.high:
        return BOTTOM
    return None


def dedup_consecutive_same_type(fractals: list[Fractal]) -> list[Fractal]:
    """§3.2:把相邻同类分型压成极值代表(顶取最高/底取最低,**同价取最先**)。

    survivor 保留自身的 ``pivot_*`` 与 ``confirm_*``;tie-break 只影响保留谁,
    绝不提前 confirm_date。
    """
    out: list[Fractal] = []
    for f in fractals:
        if out and out[-1].kind == f.kind:
            prev = out[-1]
            if f.kind == TOP:
                if f.pivot_price > prev.pivot_price:  # 严格更高才替换;同价保留更早
                    out[-1] = f
            else:  # BOTTOM
                if f.pivot_price < prev.pivot_price:   # 严格更低才替换;同价保留更早
                    out[-1] = f
        else:
            out.append(f)
    return out


def _right_end_pending(merged: list[MergedK], df: pd.DataFrame,
                       level: str) -> Fractal | None:
    """右端待定候选:最后一根标准 K 相对前一根的单边极值(第三根 K 未形成)。"""
    n = len(merged)
    if n < 2:
        return None
    L, M = merged[n - 2], merged[n - 1]
    if M.high > L.high and M.low > L.low:        # 上行 → 潜在顶
        kind, pivot_date, pivot_price = TOP, df.index[M.high_idx], M.high
    elif M.high < L.high and M.low < L.low:      # 下行 → 潜在底
        kind, pivot_date, pivot_price = BOTTOM, df.index[M.low_idx], M.low
    else:
        return None
    return Fractal(
        kind=kind, level=level, status=PENDING, mid_k=n - 1,
        pivot_date=pivot_date, pivot_price=float(pivot_price),
        confirm_date=None, confirm_price=None, executable_price=None,
        source_unit_ids=sorted(L.raw_indices + M.raw_indices),
        left_k=n - 2, right_k=None,
    )


def _assign_ids(fractals: list[Fractal], level: str) -> None:
    code = _LEVEL_CODE.get(level, level)
    for seq, f in enumerate(fractals, start=1):
        f.id = f"fx_{code}_{seq:03d}"


def detect_fractals(
    merged: list[MergedK],
    df: pd.DataFrame,
    *,
    level: str = "daily",
    include_pending: bool = True,
) -> list[Fractal]:
    """在标准 K 序列上识别分型,返回(去重后的确认/待确认 + 右端待定)列表。

    参数
    ----
    merged: 模块 2 输出的标准 K 列表。
    df: 对应的规范 OHLCV(用于取 confirm 收盘 / executable 开盘 / 极值日期)。
    include_pending: 是否附加右端待定候选(默认 True,右端结构显式标 ``pending``)。
    """
    raw: list[Fractal] = []
    for i in range(1, len(merged) - 1):
        kind = _classify(merged[i - 1], merged[i], merged[i + 1])
        if kind is not None:
            raw.append(_make_fractal(kind, i, merged, df, level))

    result = dedup_consecutive_same_type(raw)

    if include_pending:
        pending = _right_end_pending(merged, df, level)
        if pending is not None:
            result = result + [pending]

    _assign_ids(result, level)
    return result


def fractals_to_frame(fractals: list[Fractal]) -> pd.DataFrame:
    """分型列表 → DataFrame(便于检视/输出)。"""
    rows = [{
        "id": f.id, "kind": f.kind, "status": f.status, "mid_k": f.mid_k,
        "pivot_date": f.pivot_date, "pivot_price": f.pivot_price,
        "confirm_date": f.confirm_date, "confirm_price": f.confirm_price,
        "executable_price": f.executable_price,
    } for f in fractals]
    return pd.DataFrame(rows, columns=[
        "id", "kind", "status", "mid_k", "pivot_date", "pivot_price",
        "confirm_date", "confirm_price", "executable_price",
    ])
