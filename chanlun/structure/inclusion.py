"""模块 2 · 包含关系【确定性】

出处:中泰 p23 / 缠师 65 课。把原始 K 序列处理为**标准 K 序列**(相邻无包含)。

判据(严格按 §2):
- §2.1 取值:方向向上 → 合并为 ``[max低, max高]``;向下 → ``[min低, min高]``。
- §2.2 方向:在**非包含**的标准 K 之间比较 —— ``gₙ ≥ gₙ₋₁`` → 上;``dₙ ≤ dₙ₋₁`` → 下。
  (两根非包含 K 的高点必严格不等,故方向唯一。)
- §2.3 顺序原则:逐根处理 —— 先合 1、2,再用合并结果与第 3 根比;**不满足传递律**,
  绝不回头比较非相邻 K(只与当前标准 K 尾比较)。
- §2.4 初始方向【约定·A】:序列开头若互相包含、方向未定,则**向后找第一对非包含 K**
  定向,用该方向回溯合并前缀。全程互含(无非包含对)→ 方向不可定,默认向上并入单根。
- §2.5 等高【约定】:完全相等(高低都等)→ 并为一根;单边相等(仅高等或仅低等)
  天然属于包含,由 max/min 吸收(``is_contained`` 用 ``≥/≤`` 已覆盖)。

★ 不引入未来函数:逐根顺序处理,每根标准 K 仅由其覆盖的原始 K 决定;
``high_idx/low_idx`` 记录贡献极值的原始 K 位置(同价取最先),供模块 3 分型定位 pivot。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..data.models import validate_canonical


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


UP = Direction.UP.value
DOWN = Direction.DOWN.value


@dataclass
class MergedK:
    """一根标准 K(包含处理后的合并 K)。

    - ``high/low``:合并后的高低点。
    - ``direction``:形成该标准 K 时的趋势方向(领先趋势,§2.2);仅信息用途。
    - ``raw_indices``:被并入的原始 K 位置(升序、连续区间)。
    - ``high_idx/low_idx``:贡献 ``high``/``low`` 的原始 K 位置(同价取最先)。
    """

    high: float
    low: float
    direction: str
    raw_indices: list[int]
    high_idx: int
    low_idx: int


def is_contained(hi_a: float, lo_a: float, hi_b: float, lo_b: float) -> bool:
    """两根 K(各自高低)是否构成包含关系(任一方含另一方)。

    用 ``≥/≤`` 判定,故单边等高/等低也算包含(§2.5 由 max/min 吸收)。
    """
    a_contains_b = hi_a >= hi_b and lo_a <= lo_b
    b_contains_a = hi_b >= hi_a and lo_b <= lo_a
    return a_contains_b or b_contains_a


def initial_direction(highs, lows) -> str:
    """§2.4:向后找第一对**非包含**相邻原始 K 定方向;全互含则默认向上。"""
    for i in range(len(highs) - 1):
        if not is_contained(highs[i], lows[i], highs[i + 1], lows[i + 1]):
            return UP if highs[i + 1] > highs[i] else DOWN
    return UP


def _merge(tip: MergedK, i: int, hi: float, lo: float, direction: str) -> MergedK:
    """把原始 K ``i`` 并入标准 K ``tip``,按方向取值(§2.1),同价取最先。"""
    if direction == UP:
        # 向上:[max低, max高]
        if tip.high >= hi:
            new_high, new_high_idx = tip.high, tip.high_idx
        else:
            new_high, new_high_idx = hi, i
        if tip.low >= lo:
            new_low, new_low_idx = tip.low, tip.low_idx
        else:
            new_low, new_low_idx = lo, i
    else:
        # 向下:[min低, min高]
        if tip.high <= hi:
            new_high, new_high_idx = tip.high, tip.high_idx
        else:
            new_high, new_high_idx = hi, i
        if tip.low <= lo:
            new_low, new_low_idx = tip.low, tip.low_idx
        else:
            new_low, new_low_idx = lo, i
    return MergedK(
        high=new_high, low=new_low, direction=tip.direction,
        raw_indices=tip.raw_indices + [i],
        high_idx=new_high_idx, low_idx=new_low_idx,
    )


def process_inclusion(df: pd.DataFrame) -> list[MergedK]:
    """对规范 OHLCV 做包含处理,返回标准 K 列表(相邻无包含)。"""
    validate_canonical(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(highs)
    if n == 0:
        return []

    direction = initial_direction(highs, lows)
    merged = [MergedK(
        high=float(highs[0]), low=float(lows[0]), direction=direction,
        raw_indices=[0], high_idx=0, low_idx=0,
    )]

    for i in range(1, n):
        tip = merged[-1]
        hi, lo = float(highs[i]), float(lows[i])
        if is_contained(tip.high, tip.low, hi, lo):
            merged[-1] = _merge(tip, i, hi, lo, direction)
        else:
            # §2.2:非包含 → 由这对标准 K 重定方向(高点必严格不等)
            direction = UP if hi > tip.high else DOWN
            merged.append(MergedK(
                high=hi, low=lo, direction=direction,
                raw_indices=[i], high_idx=i, low_idx=i,
            ))
    return merged


def merged_to_frame(merged: list[MergedK], df: pd.DataFrame) -> pd.DataFrame:
    """把标准 K 列表转为 DataFrame,索引=各标准 K 末根原始 K 日期(严格递增)。

    列:``high, low, direction, n_raw, start_date, end_date, high_date, low_date``。
    其中 ``high_date/low_date`` 为贡献极值的原始 K 日期,供分型 pivot 定位。
    """
    idx = df.index
    rows = []
    end_dates = []
    for m in merged:
        rows.append({
            "high": m.high,
            "low": m.low,
            "direction": m.direction,
            "n_raw": len(m.raw_indices),
            "start_date": idx[m.raw_indices[0]],
            "end_date": idx[m.raw_indices[-1]],
            "high_date": idx[m.high_idx],
            "low_date": idx[m.low_idx],
        })
        end_dates.append(idx[m.raw_indices[-1]])
    out = pd.DataFrame(rows, columns=[
        "high", "low", "direction", "n_raw",
        "start_date", "end_date", "high_date", "low_date",
    ])
    out.index = pd.DatetimeIndex(end_dates, name="date")
    return out
