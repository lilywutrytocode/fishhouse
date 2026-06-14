"""模块 6 · 中枢【确定性】笔中枢 + 线段中枢。

出处:中泰 p39/p44/p46 / 缠师 17 课 + 中心定理一二三。

判据(严格按 §6 + §0.5):
- §6.1 成立:连续三个次级别走势单位 U1/U2/U3 →
  ``ZG = min(三高)``、``ZD = max(三低)``;``ZD ≤ ZG`` 成立(§6.2)。
  成立后 **ZG/ZD 固定**;``GG = max(各延伸段高)``、``DD = min(各延伸段低)`` **随延伸刷新**。
- §6.3 延伸(定理一):后续单位与 [ZD,ZG] 有重叠则延伸,不设段数硬上限;
  整段在上(low>ZG)或在下(high<ZD)即离开。
- §6.5 破坏(定理三):离开后回抽不回 [ZD,ZG] 内 → 旧中枢破坏(下一结构起);
  若回抽回到 [ZD,ZG],单次离开计入延伸(GG/DD 扩张),中枢继续。
- §6.4 扩展(定理二)::func:`classify_relation` 判两相邻中枢关系。
- §6.6 起点:标准起点(上一中枢之后顺序取),不做重新组合优化。
- §0.5:``pivot`` = 首段起点;``confirm`` = 第三段成立、ZD≤ZG 那刻(晚于 pivot)。
- ★ 只有已确认单位喂中枢(线段须 CONFIRMED_END,见 :func:`units_from_segments`)。

§6.7 信号去重/事件簇见 :func:`dedupe_event_cluster`(笔+线段中枢都用的必备规则):
**只在同级别同向同类内合并,跨级别都保留**;线段中枢>笔中枢>…,confirm 早者触发,
其余进 ``supporting_signals``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from ..config import DEFAULT_CONFIG, Config
from .inclusion import DOWN, UP

_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}


class ZhongshuKind(str, Enum):
    BI = "bi"             # 笔中枢
    XIANDUAN = "xianduan"  # 线段中枢


BI = ZhongshuKind.BI.value
XIANDUAN = ZhongshuKind.XIANDUAN.value


@dataclass
class ZUnit:
    """中枢的次级别走势单位(笔或线段),只取区间与确认信息。"""

    high: float
    low: float
    start_date: pd.Timestamp
    start_price: float
    confirm_date: pd.Timestamp
    confirm_price: float
    confirm_bar: int = -1
    direction: str | None = None
    id: str | None = None


@dataclass
class Zhongshu:
    """一个中枢(带 §0.6 通用纪律字段)。"""

    kind: str               # bi / xianduan
    level: str
    ZG: float
    ZD: float
    GG: float
    DD: float
    start_unit: int         # 起始单位下标(首段)
    end_unit: int           # 末单位下标
    n_segments: int         # 含段数(不设硬上限,§6.3)
    pivot_date: pd.Timestamp
    pivot_price: float      # 首段起点(§0.5)
    confirm_date: pd.Timestamp
    confirm_price: float
    executable_price: float | None
    member_unit_ids: list = field(default_factory=list)
    extending: bool = False  # 右端仍在延伸(未见破坏/离开)
    broken: bool = False     # 已被破坏(定理三)
    id: str | None = None

    def __post_init__(self):
        assert self.confirm_date > self.pivot_date, (
            f"confirm_date({self.confirm_date}) 必须晚于 pivot_date"
            f"({self.pivot_date})(§0.5 中枢 confirm=第三段成立)"
        )


def _overlaps_core(u: ZUnit, zd: float, zg: float) -> bool:
    """单位区间与中枢核心 [ZD,ZG] 是否有重叠(定理一)。"""
    return u.low <= zg and u.high >= zd


def build_zhongshu(
    units: list[ZUnit], *, level: str = "daily", kind: str = BI,
) -> list[Zhongshu]:
    """从次级别走势单位序列构建中枢列表(§6.1–6.6)。

    顺序扫描(§6.6 标准起点,不重组):连续三单位 ZD≤ZG 成立中枢,后续重叠延伸、
    GG/DD 刷新;整段离开且回抽不回 → 破坏并从离开单位起继续找下一中枢。
    """
    out: list[Zhongshu] = []
    n = len(units)
    i = 0
    while i + 2 < n + 1 and i + 2 < n:
        u1, u2, u3 = units[i], units[i + 1], units[i + 2]
        zg = min(u1.high, u2.high, u3.high)
        zd = max(u1.low, u2.low, u3.low)
        if zd > zg:                       # §6.2 不成立 → 顺延一个单位(标准起点)
            i += 1
            continue

        members = [i, i + 1, i + 2]
        gg = max(u1.high, u2.high, u3.high)
        dd = min(u1.low, u2.low, u3.low)
        broken = False
        j = i + 3
        while j < n:
            u = units[j]
            if _overlaps_core(u, zd, zg):           # 定理一:重叠 → 延伸
                members.append(j)
                gg = max(gg, u.high)
                dd = min(dd, u.low)
                j += 1
            elif j + 1 < n and _overlaps_core(units[j + 1], zd, zg):
                # 定理三:离开后回抽回到 [ZD,ZG] → 单次离开计入延伸,中枢继续
                members.append(j)
                gg = max(gg, u.high)
                dd = min(dd, u.low)
                j += 1
            else:                                    # 定理三:离开且回抽不回 → 破坏
                broken = True
                break

        u1 = units[members[0]]
        u3 = units[members[2]]
        zs = Zhongshu(
            kind=kind, level=level, ZG=zg, ZD=zd, GG=gg, DD=dd,
            start_unit=members[0], end_unit=members[-1], n_segments=len(members),
            pivot_date=u1.start_date, pivot_price=u1.start_price,
            confirm_date=u3.confirm_date, confirm_price=u3.confirm_price,
            executable_price=None,
            member_unit_ids=[units[k].id for k in members],
            extending=(not broken),    # 没被破坏 = 右端仍在延伸
            broken=broken,
        )
        zs.id = f"zhongshu_{kind[:2]}_{_LEVEL_CODE.get(level, level)}_{len(out) + 1:03d}"
        out.append(zs)
        i = members[-1] + 1            # §6.6 标准起点:从中枢之后顺序继续
    return out


def classify_relation(prev: Zhongshu, nxt: Zhongshu) -> str:
    """§6.4 定理二:判两相邻同级别中枢关系。"""
    if nxt.GG < prev.DD:
        return "下跌延续"
    if nxt.DD > prev.GG:
        return "上涨延续"
    if nxt.ZG < prev.ZD and nxt.GG >= prev.DD:
        return "高级别中枢"
    if nxt.ZD > prev.ZG and nxt.DD <= prev.GG:
        return "高级别中枢"
    return "无关"


def units_from_segments(segments) -> list[ZUnit]:
    """把线段(仅 CONFIRMED_END / feeds_zhongshu=True)转为中枢单位。"""
    units = []
    for s in segments:
        if not getattr(s, "feeds_zhongshu", False):
            continue
        hi = max(s.pivot_price, s.confirm_price or s.pivot_price)
        lo = min(s.pivot_price, s.confirm_price or s.pivot_price)
        units.append(ZUnit(
            high=hi, low=lo, start_date=s.pivot_date, start_price=s.pivot_price,
            confirm_date=s.confirm_date, confirm_price=s.confirm_price or s.pivot_price,
            direction=s.direction, id=s.id,
        ))
    return units


# ── §6.7 信号去重 / 事件簇【约定·可配置】──────────────────────────────────
@dataclass
class SignalEvent:
    """买卖点/中枢信号事件(供 §6.7 去重)。"""

    id: str
    symbol: str
    level: str
    direction: str          # buy/sell 或 up/down
    kind: str               # 同类买卖点(一买/二买/三买…)
    confirm_date: pd.Timestamp
    confirm_price: float
    confirm_bar: int        # 以 K 线计的确认位(用于 ≤3 根K 判定)
    source_rank: int = 0    # 线段中枢=2 > 笔中枢=1
    beichi_rank: int = 0    # 标准背驰=2 > 盘背=1
    level30_rank: int = 0   # 真 30min=1 > 日线内部近似=0
    supporting_signals: list = field(default_factory=list)


def _cluster_key(s: SignalEvent) -> tuple:
    """同一事件簇的前置分组键:同标的、同级别、同方向、同类(★ 跨级别绝不合并)。"""
    return (s.symbol, s.level, s.direction, s.kind)


def _trigger_priority(s: SignalEvent) -> tuple:
    """簇内触发优先级(越小越优先):线段中枢>笔中枢>…,标准背驰>盘背,
    真30min>近似,最后 confirm 早者。"""
    return (-s.source_rank, -s.beichi_rank, -s.level30_rank, s.confirm_bar)


def dedupe_event_cluster(
    signals: list[SignalEvent], *, config: Config = DEFAULT_CONFIG,
) -> list[SignalEvent]:
    """§6.7:把同一事件簇压成一个触发信号,其余进 ``supporting_signals``。

    同簇条件(全满足):同标的/级别/方向/类;``|Δconfirm_bar| ≤ N``;
    ``|Δconfirm_price|/price ≤ x``(N=dedup_confirm_bars、x=dedup_confirm_price_pct)。
    ★ 只在同级别内合并;跨级别为不同事件,都保留。
    """
    n_bar = config.dedup_confirm_bars
    x_pct = config.dedup_confirm_price_pct

    # 先按分组键聚类(跨级别天然分开)
    groups: dict[tuple, list[SignalEvent]] = {}
    for s in signals:
        groups.setdefault(_cluster_key(s), []).append(s)

    triggers: list[SignalEvent] = []
    for _, group in groups.items():
        group = sorted(group, key=lambda s: s.confirm_bar)
        clusters: list[list[SignalEvent]] = []
        for s in group:
            placed = False
            for cl in clusters:
                anchor = cl[0]
                near_bar = abs(s.confirm_bar - anchor.confirm_bar) <= n_bar
                base = abs(anchor.confirm_price) or 1.0
                near_price = abs(s.confirm_price - anchor.confirm_price) / base <= x_pct
                if near_bar and near_price:
                    cl.append(s)
                    placed = True
                    break
            if not placed:
                clusters.append([s])

        for cl in clusters:
            trigger = min(cl, key=_trigger_priority)
            trigger.supporting_signals = [s.id for s in cl if s is not trigger]
            triggers.append(trigger)

    triggers.sort(key=lambda s: s.confirm_bar)
    return triggers
