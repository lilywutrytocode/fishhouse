"""模块 4 · 笔【确定性】

出处:中泰 p25–26 / 缠师 65、66 课。**纯新笔(无缺口豁免)**。
在模块 3 的分型序列上,按缠师 66 课三步划分笔。

判据(严格按 §4 + §0.5 防未来函数):
- §4.1 新笔判定:① 顶底分型相邻;② 各自独立、不共用 K;③ 之间 ≥1 独立 K;④ 顶点>底点。
  在标准 K 索引上,②③ 合并为**两分型中间 K 间距 ≥ 4**(两个 3-K 窗口不相交且之间 ≥1 独立 K),
  等价于「经包含处理 ≥5 根 K」;④ 为端点价格比较(上升笔 顶>底、下降笔同)。**无缺口豁免**。
- §4.2 方向有效性:上升笔 底点<顶点;下降笔反向。
- §4.3 三步:① 定分型(模块 3);② 连续同类去重(顶取最高/底取最低,**同价取最先**,
  复用 :func:`dedupe_same_type_fractals`);③ 连接成笔,不成笔则**顺延**到下一个满足 4.1 的反向分型,
  顺延暴露出的连续同类再次去重(★ 同价 tie-break 只影响 pivot 选择,不提前 confirm_date)。
- §4.4 延伸/破坏:笔只被笔破坏;新笔未成立算原笔延伸;**右端未确认 → 笔·未确认/延伸**。

★ pivot/confirm/executable(§0.5):
- ``pivot`` = 端点(反向)分型极值 K:上升笔取末端顶的 high、下降笔取末端底的 low。
- ``confirm`` = 反向分型满足 4.1 成笔那刻 = 末端分型的 ``confirm``(晚于端点 pivot)。
- ``executable_price`` = confirm bar 下一根 open(承自末端分型)。
- **最右侧笔恒标 ``forming``(未确认/延伸)**:其端点可延伸,confirm/executable 置空、不入回测;
  历史笔(非最右)才 ``confirmed``。回测只用 confirm_date + executable_price,严禁用 pivot 触发。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .fractal import (
    BOTTOM,
    CONFIRMED as FX_CONFIRMED,
    LIVE_PENDING as FX_LIVE_PENDING,
    PENDING as FX_PENDING,
    TOP,
    Fractal,
    dedupe_same_type_fractals,
    detect_fractals,
)
from .inclusion import DOWN, UP, MergedK, process_inclusion


class BiStatus(str, Enum):
    CONFIRMED = "confirmed"   # 历史笔:末端分型已确认且非最右 → 入回测
    FORMING = "forming"       # 右端未确认/延伸:端点可延伸,不入回测


CONFIRMED = BiStatus.CONFIRMED.value
FORMING = BiStatus.FORMING.value

_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}

# §4.1 ②③:两分型中间 K 间距下限(= 不共用 K + 之间 ≥1 独立 K = ≥5 根标准 K)
MIN_MID_K_GAP = 4


@dataclass
class Bi:
    """一笔(带 §0.6 通用纪律字段)。"""

    direction: str               # up / down
    level: str
    status: str                  # BiStatus
    start_k: int                 # 起端分型中间标准 K 位置
    end_k: int                   # 末端分型中间标准 K 位置
    start_date: pd.Timestamp     # 起端极值(上一笔 pivot)
    start_price: float
    pivot_date: pd.Timestamp     # 末端分型极值(§0.5 端点极值)
    pivot_price: float
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    source_unit_ids: list[int]   # 覆盖的原始 K 位置
    start_fx_id: str | None = None
    end_fx_id: str | None = None
    id: str | None = None

    def __post_init__(self):
        if self.confirm_date is not None:
            assert self.confirm_date > self.pivot_date, (
                f"confirm_date({self.confirm_date}) 必须严格晚于 "
                f"pivot_date({self.pivot_date})(§0.5 右侧确认)"
            )


def _valid_bi(a: Fractal, b: Fractal) -> bool:
    """§4.1:相邻反向分型 a→b 能否成纯新笔。"""
    if a.kind == b.kind:
        return False
    if (b.mid_k - a.mid_k) < MIN_MID_K_GAP:   # ②③ 不共用K + ≥1独立K(无缺口豁免)
        return False
    if a.kind == BOTTOM:   # 上升笔:顶(b) > 底(a)
        return b.pivot_price > a.pivot_price
    return a.pivot_price > b.pivot_price       # 下降笔:顶(a) > 底(b)


def _make_bi(start_fx: Fractal, end_fx: Fractal, merged: list[MergedK],
             level: str, status: str) -> Bi:
    direction = UP if start_fx.kind == BOTTOM else DOWN
    if status == CONFIRMED:
        confirm_date = end_fx.confirm_date
        confirm_price = end_fx.confirm_price
        executable_price = end_fx.executable_price
    else:  # FORMING:右端未确认,不给可执行触发
        confirm_date = confirm_price = executable_price = None

    sk, ek = min(start_fx.mid_k, end_fx.mid_k), max(start_fx.mid_k, end_fx.mid_k)
    source = sorted({i for k in range(sk, ek + 1) for i in merged[k].raw_indices})
    return Bi(
        direction=direction, level=level, status=status,
        start_k=start_fx.mid_k, end_k=end_fx.mid_k,
        start_date=start_fx.pivot_date, start_price=start_fx.pivot_price,
        pivot_date=end_fx.pivot_date, pivot_price=end_fx.pivot_price,
        confirm_date=confirm_date, confirm_price=confirm_price,
        executable_price=executable_price, source_unit_ids=source,
        start_fx_id=start_fx.id, end_fx_id=end_fx.id,
    )


def _connect_endpoints(fxs: list[Fractal]) -> list[Fractal]:
    """§4.3 步骤③:连接成笔 + 顺延 + 暴露的连续同类去重。返回交替的端点序列。"""
    if not fxs:
        return []
    endpoints = [fxs[0]]
    for cand in fxs[1:]:
        last = endpoints[-1]
        if cand.kind == last.kind:
            # 顺延跳过反向分型后暴露的连续同类 → 去重重选极值端点(同价取最先)
            endpoints[-1] = dedupe_same_type_fractals([last, cand])[0]
        elif _valid_bi(last, cand):
            endpoints.append(cand)
        # else: 不成笔 → 顺延(跳过 cand,下一根同类会触发重选)
    return endpoints


def build_bi(
    fractals: list[Fractal],
    merged: list[MergedK],
    *,
    level: str = "daily",
) -> list[Bi]:
    """由分型列表(模块 3 输出)构建笔序列。

    参数
    ----
    fractals: :func:`detect_fractals` 的输出(含右端 ``pending`` 候选)。
    merged: 对应标准 K 列表(取 source_unit_ids)。
    """
    # 步骤②:连续同类去重(防御性;detect 已去重,此处满足「build_bi 必须调用去重」)
    settled = dedupe_same_type_fractals(
        [f for f in fractals if f.status in (FX_CONFIRMED, FX_LIVE_PENDING)]
    )
    pending_fx = next((f for f in fractals if f.status == FX_PENDING), None)

    endpoints = _connect_endpoints(settled)

    # 历史笔:每笔末端分型已确认 → confirmed(§0.5 confirm = 末端分型成笔)
    bis: list[Bi] = [
        _make_bi(endpoints[k - 1], endpoints[k], merged, level, CONFIRMED)
        for k in range(1, len(endpoints))
    ]

    # §4.4 右端未确认/延伸:依右端待定分型与末端端点的相对方向处理
    if endpoints and pending_fx is not None:
        last_ep = endpoints[-1]
        if pending_fx.kind != last_ep.kind:
            # 反向在途:从末端端点起一笔正在形成(新笔未成立 → 未确认/延伸)
            bis.append(_make_bi(last_ep, pending_fx, merged, level, FORMING))
        elif bis:
            # 同向延伸:末端笔端点延伸到右端更极值处(同价取最先),转为未确认/延伸
            a = endpoints[-2]
            end_fx = dedupe_same_type_fractals([last_ep, pending_fx])[0]
            bis[-1] = _make_bi(a, end_fx, merged, level, FORMING)

    for seq, bi in enumerate(bis, start=1):
        bi.id = f"bi_{_LEVEL_CODE.get(level, level)}_{seq:03d}"
    return bis


def build_bi_from_df(df: pd.DataFrame, *, level: str = "daily") -> list[Bi]:
    """便捷流水线:规范 OHLCV → 包含 → 分型 → 笔。"""
    merged = process_inclusion(df)
    fractals = detect_fractals(merged, df, level=level)
    return build_bi(fractals, merged, level=level)


def bis_to_frame(bis: list[Bi]) -> pd.DataFrame:
    """笔列表 → DataFrame(便于检视/输出)。"""
    rows = [{
        "id": b.id, "direction": b.direction, "status": b.status,
        "start_k": b.start_k, "end_k": b.end_k,
        "start_date": b.start_date, "start_price": b.start_price,
        "pivot_date": b.pivot_date, "pivot_price": b.pivot_price,
        "confirm_date": b.confirm_date, "confirm_price": b.confirm_price,
        "executable_price": b.executable_price,
    } for b in bis]
    return pd.DataFrame(rows, columns=[
        "id", "direction", "status", "start_k", "end_k",
        "start_date", "start_price", "pivot_date", "pivot_price",
        "confirm_date", "confirm_price", "executable_price",
    ])
