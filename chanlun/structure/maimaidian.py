"""模块 8 · 三类买卖点【确定性】

出处:中泰 p11/p59/p60/p62 / 缠师 14、21、67 课 + 第三类定理。

判据(严格按 §8 + §0.5):
- §8.1 一买:趋势背驰→``一买·标准``;盘整背驰→``一买·盘背``(类一买)。输出
  ``pivot_relation_to_zhongshu`` 与 ``confirm_relation_to_zhongshu`` ∈ {below, inside, above}
  (相对最近同级别中枢):结构成立看 pivot(应 below),操作风险看 confirm(可能已回 inside)。
- §8.4 一买两态:日线背驰→``背驰确认``;次级别向上→``次级别确认``;右端未出→``待确认``。
- §8.2 二买五步:一买 confirmed 后 ① 等次级别向上确认 → ② 等其后第一个次级别向下完成 →
  ③ 该向下低点 > 一买 pivot_price → ④ 低点=二买 pivot → ⑤ 完成日=二买 confirm。
- §8.3 三买:离开必须有**已确认次级别走势单位**明确脱离 ZG(**不能单根 K 刺破**);
  回试走势必须**确认结束**;记 ``leave_unit_id`` / ``retest_unit_id``;回试低点 > ZG。
- §8.5 级别 = 背驰/中枢级别;§8.6 二买与三买可重合→``二三买重合``;一买/二买、一买/三买不重合。
- §8.7 中泰 p63 均线/量能默认**不纳入**(仅可选叠加层贴标,本模块不实现)。
- §0.5:一买 pivot=下跌极值低点;二买 pivot=回调低点;三买 pivot=回试低点;confirm 均晚于 pivot。
- ★ 卖点对称(顶/ZD)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from .beichi import BeichiStatus, BeichiType, Grade
from .inclusion import DOWN, UP


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


BUY = Side.BUY.value
SELL = Side.SELL.value

# 相对中枢位置
BELOW, INSIDE, ABOVE = "below", "inside", "above"

# 一买子类 / 确认状态
SUB_STANDARD = "标准"
SUB_PANBEI = "盘背"
ST_BEICHI = "背驰确认"
ST_SUBLEVEL = "次级别确认"
ST_PENDING = "待确认"
ST_CONFIRMED = "confirmed"

_KIND = {
    (BUY, 1): "一买", (BUY, 2): "二买", (BUY, 3): "三买",
    (SELL, 1): "一卖", (SELL, 2): "二卖", (SELL, 3): "三卖",
}
_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}


@dataclass
class Unit:
    """次级别走势单位(笔/线段),买卖点识别用。"""

    direction: str          # up / down
    high: float
    low: float
    pivot_date: pd.Timestamp
    pivot_price: float       # 终端极值(up=high / down=low)
    confirm_date: pd.Timestamp
    confirm_price: float
    executable_price: float | None = None
    confirmed: bool = True
    n_bars: int = 5          # 单根 K 刺破 → n_bars==1(非走势单位)
    id: str | None = None


@dataclass
class MaiMaiDian:
    """一个买卖点(带 §0.6 通用纪律 + 关联 id)。"""

    kind: str                # 一买/二买/三买/一卖/二卖/三卖
    side: str                # buy / sell
    level: str
    status: str              # confirmed/背驰确认/次级别确认/待确认
    subkind: str | None      # 标准(趋势)/盘背(盘整)——子类,与强度档正交(仅一买/一卖)
    pivot_date: pd.Timestamp
    pivot_price: float
    confirm_date: pd.Timestamp | None
    confirm_price: float | None
    executable_price: float | None
    pivot_relation_to_zhongshu: str | None = None
    confirm_relation_to_zhongshu: str | None = None
    related_zhongshu_id: str | None = None
    related_beichi_id: str | None = None
    related_leave_unit_id: str | None = None
    related_retest_unit_id: str | None = None
    overlap_2_3: bool = False
    supporting_signals: list = field(default_factory=list)
    # §8.1 强度档闸(对齐 §7.2):背驰档透传 + 主信号判定(与 subkind 正交)
    beichi_grade: str | None = None     # 标准背驰/面积背驰/DIF背驰
    strength: str | None = None         # 标准(建立在标准档背驰)/ 弱(仅面积/DIF档)
    is_main: bool = False               # 仅标准档背驰之上的一买为主买点
    invalidated: bool = False           # §9.3 小转大/失效:confirm 后顺原向越过 pivot(结果字段,不删样本)
    label: str | None = None            # 显示标签(__post_init__ 计算)
    id: str | None = None

    def __post_init__(self):
        if self.confirm_date is not None:
            assert self.confirm_date > self.pivot_date, (
                "confirm_date 必须晚于 pivot_date(§0.5 买卖点右侧确认)"
            )
        self.label = self._compute_label()

    def _compute_label(self) -> str:
        """标签区分『趋势子类』与『背驰强弱档』,不混:
        - 趋势标准背驰 → 一买/一卖·标准
        - 趋势面积/DIF 弱背驰 → 一买/一卖·弱
        - 盘整背驰 → 一买/一卖·盘背(强弱另见 is_main/strength)
        """
        if self.kind in ("一买", "一卖"):
            if self.subkind == SUB_PANBEI:          # 盘整背驰
                return f"{self.kind}·盘背"
            if self.strength == "弱":               # 趋势但弱档(面积/DIF)
                return f"{self.kind}·弱"
            return f"{self.kind}·标准"               # 趋势标准档
        if self.subkind:
            return f"{self.kind}·{self.subkind}"
        return self.kind


def relation_to_zhongshu(price: float, zd: float, zg: float) -> str:
    """价格相对中枢 [ZD,ZG] 的位置。"""
    if price < zd:
        return BELOW
    if price > zg:
        return ABOVE
    return INSIDE


# ── §8.1 / §8.4 一买(对称一卖)────────────────────────────────────────────
def detect_first(
    beichi, zhongshu, *, side: str = BUY, level: str = "daily",
    sublevel_confirm: Unit | None = None,
) -> MaiMaiDian | None:
    """由背驰 + 最近同级别中枢识别一买/一卖(两态确认)。"""
    subkind = SUB_STANDARD if beichi.type == BeichiType.TREND.value else SUB_PANBEI
    pivot_date, pivot_price = beichi.pivot_date, beichi.pivot_price
    if pivot_date is None:
        return None

    # §8.4 两态:次级别向上(buy)/向下(sell)确认优先,其次背驰确认,否则待确认
    if sublevel_confirm is not None and sublevel_confirm.confirmed:
        status = ST_SUBLEVEL
        confirm_date = sublevel_confirm.confirm_date
        confirm_price = sublevel_confirm.confirm_price
        executable = sublevel_confirm.executable_price
    elif beichi.beichi_status == BeichiStatus.CONFIRMED.value:
        status = ST_BEICHI
        confirm_date = beichi.confirm_date
        confirm_price = beichi.confirm_price
        executable = beichi.executable_price
    else:
        status = ST_PENDING
        confirm_date = confirm_price = executable = None

    # §8.1 强度档闸:主一买须建立在标准档背驰之上;面积/DIF 弱档 → 一买·弱(不进主信号)
    beichi_grade = getattr(beichi, "grade", None)
    is_main = bool(getattr(beichi, "is_main_signal", False))   # 标准档 + 背驰 confirmed
    strength = "标准" if beichi_grade == Grade.STANDARD.value else "弱"

    confirm_rel = (relation_to_zhongshu(confirm_price, zhongshu.ZD, zhongshu.ZG)
                   if confirm_price is not None else None)
    mmd = MaiMaiDian(
        kind=_KIND[(side, 1)], side=side, level=level, status=status, subkind=subkind,
        pivot_date=pivot_date, pivot_price=pivot_price,
        confirm_date=confirm_date, confirm_price=confirm_price, executable_price=executable,
        pivot_relation_to_zhongshu=relation_to_zhongshu(pivot_price, zhongshu.ZD, zhongshu.ZG),
        confirm_relation_to_zhongshu=confirm_rel,
        related_zhongshu_id=zhongshu.id, related_beichi_id=beichi.id,
        beichi_grade=beichi_grade, strength=strength, is_main=is_main,
    )
    return mmd


# ── §8.2 二买五步(对称二卖)──────────────────────────────────────────────
def detect_second(
    first: MaiMaiDian, sub_units: list[Unit], *, side: str = BUY, level: str = "daily",
) -> MaiMaiDian | None:
    """一买/一卖 confirmed 后,按五步识别第一个次级别回调 → 二买/二卖。"""
    if first.confirm_date is None:
        return None  # 一买未确认,不进二买
    first_dir = UP if side == BUY else DOWN
    second_dir = DOWN if side == BUY else UP

    # ① 第一个次级别(向上/向下)确认
    i_first = next((i for i, u in enumerate(sub_units)
                    if u.direction == first_dir and u.confirmed), None)
    if i_first is None:
        return None
    # ② 其后第一个次级别反向走势完成
    i_rev = next((j for j in range(i_first + 1, len(sub_units))
                  if sub_units[j].direction == second_dir and sub_units[j].confirmed), None)
    if i_rev is None:
        return None
    rev = sub_units[i_rev]
    # ③ 回调极值未越过一买 pivot(buy:低点 > 一买低;sell:高点 < 一卖高)
    if side == BUY and not (rev.low > first.pivot_price):
        return None
    if side == SELL and not (rev.high < first.pivot_price):
        return None

    pivot_price = rev.low if side == BUY else rev.high       # ④ 低/高点=二买 pivot
    mmd = MaiMaiDian(
        kind=_KIND[(side, 2)], side=side, level=level, status=ST_CONFIRMED, subkind=None,
        pivot_date=rev.pivot_date, pivot_price=pivot_price,
        confirm_date=rev.confirm_date, confirm_price=rev.confirm_price,  # ⑤ 完成日=confirm
        executable_price=rev.executable_price,
        related_zhongshu_id=first.related_zhongshu_id,
        related_beichi_id=first.related_beichi_id,           # 二买承一买的背驰引用
    )
    return mmd


# ── §8.3 三买(对称三卖)──────────────────────────────────────────────────
def detect_third(
    zhongshu, leave_unit: Unit, retest_unit: Unit, *,
    side: str = BUY, level: str = "daily",
) -> MaiMaiDian | None:
    """次级别离开中枢 + 回试 → 三买/三卖。离开须确认次级别单位明确脱离,非单根刺破。"""
    leave_dir = UP if side == BUY else DOWN
    retest_dir = DOWN if side == BUY else UP

    # 离开:已确认次级别走势单位(非单根 K)且明确脱离 ZG(buy)/ZD(sell)
    if leave_unit.direction != leave_dir or not leave_unit.confirmed:
        return None
    if leave_unit.n_bars < 2:                      # ★ 单根 K 刺破不算
        return None
    if side == BUY and not (leave_unit.high > zhongshu.ZG):
        return None
    if side == SELL and not (leave_unit.low < zhongshu.ZD):
        return None

    # 回试:方向相反,回试极值不回中枢(buy:低点 > ZG;sell:高点 < ZD)
    if retest_unit.direction != retest_dir:
        return None
    if side == BUY and not (retest_unit.low > zhongshu.ZG):
        return None
    if side == SELL and not (retest_unit.high < zhongshu.ZD):
        return None

    pivot_price = retest_unit.low if side == BUY else retest_unit.high
    if retest_unit.confirmed:                       # 回试确认结束 → confirmed
        status = ST_CONFIRMED
        confirm_date = retest_unit.confirm_date
        confirm_price = retest_unit.confirm_price
        executable = retest_unit.executable_price
    else:                                           # extending → 不得 confirmed
        status = ST_PENDING
        confirm_date = confirm_price = executable = None

    return MaiMaiDian(
        kind=_KIND[(side, 3)], side=side, level=level, status=status, subkind=None,
        pivot_date=retest_unit.pivot_date, pivot_price=pivot_price,
        confirm_date=confirm_date, confirm_price=confirm_price, executable_price=executable,
        related_zhongshu_id=zhongshu.id,
        related_leave_unit_id=leave_unit.id, related_retest_unit_id=retest_unit.id,
    )


# ── §8.6 二三买重合 ───────────────────────────────────────────────────────
def mark_overlap_2_3(second: MaiMaiDian | None, third: MaiMaiDian | None) -> None:
    """二买与三买可重合 → 双方标 ``二三买重合``(同 pivot)。一买不与二/三重合。"""
    if second is None or third is None:
        return
    if second.pivot_date == third.pivot_date and second.pivot_price == third.pivot_price:
        second.overlap_2_3 = True
        third.overlap_2_3 = True


def assign_ids(mmds: list[MaiMaiDian], *, level: str = "daily") -> None:
    code = _LEVEL_CODE.get(level, level)
    for seq, m in enumerate([x for x in mmds if x is not None], start=1):
        m.id = f"signal_{code}_{seq:03d}"
