"""模块 11 · CLI(§11.3/11.4):代号/名称 → JSON + 可读报告。

``analyze`` 把规范 OHLCV 串起完整链路 包含→分型→笔→线段→中枢→背驰→买卖点→联立,
端到端产出 ``beichi/mai_mai_dian/lianli``;``executable_price`` 在 线段/背驰/买卖点 层
统一由 df 自动算"下一 bar open",末根 bar 标 ``live_pending``(executable=None)。
右端未完成结构显式标注(笔 forming / 线段非 CONFIRMED_END / 分型 pending)。
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import DEFAULT_CONFIG, Config, market_of
from .data.calendars import get_calendar
from .data.consistency import check_consistency
from .data.health import check_health
from .data.models import HealthStatus
from .data.snapshot import compute_snapshot_id
from .data.weekly import synthesize_weekly
from .monitor import derive_monitor_levels
from .output import build_output, to_json
from .structure.beichi import (
    BeichiType,
    SegEnergy,
    compute_macd,
    evaluate_divergence,
    segment_area,
    segment_dif_peak,
)
from .structure.bi import build_bi
from .structure.fractal import detect_fractals
from .structure.inclusion import DOWN, UP, process_inclusion
from .structure.lianli import (
    build_lianli,
    classify_lianli,
    is_any_beichi,
    is_standard_resonance_grade,
)
from .structure.maimaidian import (
    BUY,
    SELL,
    Unit,
    assign_ids,
    detect_first,
    detect_second,
    detect_third,
    mark_overlap_2_3,
)
from .structure.zhongshu import SignalEvent, dedupe_event_cluster
from .probability import to_signal_event
from .structure.xianduan import Pen, build_segments
from .structure.zhongshu import BI, XIANDUAN, ZUnit, build_zhongshu

_LEVEL_CODE = {"daily": "d", "weekly": "w", "min30": "30m"}


# ── executable_price:下一 bar open(末根 → None=live_pending)──────────────
def executable_after(df: pd.DataFrame, confirm_date) -> tuple[float | None, bool]:
    """返回 (下一 bar open, is_live_pending)。confirm_date 为末根 bar → (None, True)。"""
    if confirm_date is None:
        return None, False
    try:
        pos = df.index.get_loc(confirm_date)
    except KeyError:
        pos = int(df.index.searchsorted(confirm_date, side="right")) - 1
    if 0 <= pos and pos + 1 < len(df):
        return float(df.iloc[pos + 1]["open"]), False
    return None, True


# ── 结构 → 中枢单位 ───────────────────────────────────────────────────────
def _zunit_from_bi(b) -> ZUnit:
    return ZUnit(
        high=max(b.start_price, b.pivot_price), low=min(b.start_price, b.pivot_price),
        start_date=b.start_date, start_price=b.start_price,
        confirm_date=b.confirm_date, confirm_price=b.confirm_price,
        direction=b.direction, id=b.id,
    )


def _bi_to_pen(b, idx: int) -> Pen:
    return Pen(direction=b.direction, high=max(b.start_price, b.pivot_price),
               low=min(b.start_price, b.pivot_price), idx=idx,
               start_date=b.start_date, end_date=b.pivot_date, bi_id=b.id)


def _units_from_segments(segments) -> list[ZUnit]:
    units = []
    for s in segments:
        if not getattr(s, "feeds_zhongshu", False):
            continue
        cp = s.confirm_price if s.confirm_price is not None else s.pivot_price
        units.append(ZUnit(
            high=max(s.pivot_price, cp), low=min(s.pivot_price, cp),
            start_date=s.pivot_date, start_price=s.pivot_price,
            confirm_date=s.confirm_date, confirm_price=cp,
            direction=s.direction, id=s.id))
    return units


# ── 背驰:笔级动能比较(进入 A vs 离开 C,绕中枢)───────────────────────────
def _bi_energy(bi, macd: pd.DataFrame):
    sl = macd.loc[bi.start_date:bi.pivot_date]
    if len(sl) == 0:
        return None
    return segment_area(sl["hist"], bi.direction), segment_dif_peak(sl["dif"], bi.direction)


def _zhongshu_covering(bi_zhongshu, i: int):
    """找一个中枢,其成员区间覆盖 A/C 之间的摆动(中间笔 i+1 为中枢成员)。"""
    for zs in bi_zhongshu:
        if zs.start_unit <= i + 1 <= zs.end_unit:
            return zs
    return None


def detect_beichis(bi_zhongshu, confirmed_bis, macd, df, *, level, config):
    """§7.4 盘整背驰:扫相邻同向笔对(A=bis[i],C=bis[i+2],C 创新高/低),
    在覆盖其摆动的笔中枢内比较 MACD 面积/DIF 峰值。

    返回 [(beichi, zhongshu, side)],side=DOWN(底→一买)/UP(顶→一卖)。
    """
    out = []
    if not bi_zhongshu:
        return out
    for i in range(len(confirmed_bis) - 2):
        A, C = confirmed_bis[i], confirmed_bis[i + 2]
        if A.direction != C.direction:
            continue
        new_ext = (C.pivot_price < A.pivot_price if C.direction == DOWN
                   else C.pivot_price > A.pivot_price)
        if not new_ext:                        # 前提:C 须创新高/新低
            continue
        zs = _zhongshu_covering(bi_zhongshu, i)
        if zs is None:
            continue
        ea, ec = _bi_energy(A, macd), _bi_energy(C, macd)
        if ea is None or ec is None:
            continue
        exe, _live = executable_after(df, C.confirm_date)
        bc = evaluate_divergence(
            SegEnergy(area=ea[0], dif_peak=ea[1], direction=A.direction, id=A.id),
            SegEnergy(area=ec[0], dif_peak=ec[1], direction=C.direction, confirmed=True,
                      makes_new_extreme=new_ext, pivot_date=C.pivot_date,
                      pivot_price=C.pivot_price, confirm_date=C.confirm_date,
                      confirm_price=C.confirm_price, executable_price=exe, id=C.id),
            btype=BeichiType.CONSOLIDATION.value, compare_unit="bi",
            level=level, config=config, related_zhongshu_id=zs.id,
            seg_start_date=A.start_date)
        # 只收已确认且成档的背驰(C 已 confirmed)
        if bc is not None and bc.confirm_date is not None:
            bc.id = f"beichi_{_LEVEL_CODE.get(level, level)}_{len(out) + 1:03d}"
            out.append((bc, zs, C.direction))
    return out


def detect_trend_beichis(bi_zhongshu, confirmed_bis, macd, df, *, level, config):
    """§7.4 趋势背驰:对每一对同向趋势中枢 (zs1, zs2),取**局部 A/C**比较——
    A = 进入 zs1 的同向段(zs1 之前最近的同向笔);
    C = 离开/试图离开 zs2 的同向段(zs2 段内[含 end+1]达趋势极值 DD/GG 的同向笔)。

    ★ 不再从全序列头尾取首/末同向笔(那会跨年错配、漏判局部趋势背驰,并致重复输出)。
    """
    out = []
    cb = confirmed_bis
    for k in range(len(bi_zhongshu) - 1):
        zs1, zs2 = bi_zhongshu[k], bi_zhongshu[k + 1]
        if zs2.ZG < zs1.ZD:
            trend = DOWN
        elif zs2.ZD > zs1.ZG:
            trend = UP
        else:
            continue
        # A = 进入 zs1 的同向段:zs1 起点之前最近的一根同向笔
        a_idx = next((i for i in range(zs1.start_unit - 1, -1, -1)
                      if cb[i].direction == trend), None)
        # C = 离开/试图离开 zs2 的同向段:zs2 段内[含 end+1]达趋势极值(下→最低/上→最高)的同向笔
        lo, hi = zs2.start_unit, min(len(cb) - 1, zs2.end_unit + 1)
        cands = [i for i in range(lo, hi + 1) if cb[i].direction == trend]
        if not cands:
            continue
        c_idx = (min(cands, key=lambda i: cb[i].pivot_price) if trend == DOWN
                 else max(cands, key=lambda i: cb[i].pivot_price))
        if a_idx is None or c_idx <= a_idx:
            continue
        A, C = cb[a_idx], cb[c_idx]
        new_ext = (C.pivot_price < A.pivot_price if trend == DOWN
                   else C.pivot_price > A.pivot_price)
        if not new_ext:
            continue
        ea, ec = _bi_energy(A, macd), _bi_energy(C, macd)
        if ea is None or ec is None:
            continue
        exe, _live = executable_after(df, C.confirm_date)
        reset = macd["dif"].loc[A.pivot_date:C.pivot_date].tolist()
        bc = evaluate_divergence(
            SegEnergy(area=ea[0], dif_peak=ea[1], direction=trend, id=A.id),
            SegEnergy(area=ec[0], dif_peak=ec[1], direction=trend, confirmed=True,
                      makes_new_extreme=new_ext, pivot_date=C.pivot_date,
                      pivot_price=C.pivot_price, confirm_date=C.confirm_date,
                      confirm_price=C.confirm_price, executable_price=exe, id=C.id),
            btype=BeichiType.TREND.value, compare_unit="bi", level=level, config=config,
            related_zhongshu_id=zs2.id, reset_dif_values=reset,
            seg_start_date=A.start_date)
        if bc is not None and bc.confirm_date is not None:
            out.append((bc, zs2, trend))
    # ★ 去重:不同中枢对若命中同一 A段/C段 → 同一趋势背驰,只保留一个
    #   (同 a_unit_id/c_unit_id/pivot/confirm),其余不作为独立背驰进入事件流。
    seen, deduped = set(), []
    for bc, zs, tr in out:
        key = (bc.a_unit_id, bc.c_unit_id, bc.pivot_date, bc.pivot_price,
               bc.confirm_date, bc.confirm_price)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((bc, zs, tr))
    for i, (bc, zs, tr) in enumerate(deduped, 1):
        bc.id = f"beichi_{_LEVEL_CODE.get(level, level)}_t{i:03d}"
    return deduped


def detect_maimaidians(beichi_tuples, *, level):
    """由背驰 + 中枢识别一买/一卖(趋势→标准、盘整→盘背)。"""
    mmds = []
    for bc, zs, side_dir in beichi_tuples:
        side = BUY if side_dir == DOWN else SELL
        mmd = detect_first(bc, zs, side=side, level=level)
        if mmd is not None:
            mmds.append(mmd)
    return mmds


def bi_to_unit(b) -> Unit:
    """已确认笔 → 次级别走势单位(供二买/三买识别)。"""
    return Unit(
        direction=b.direction, high=max(b.start_price, b.pivot_price),
        low=min(b.start_price, b.pivot_price), pivot_date=b.pivot_date,
        pivot_price=b.pivot_price, confirm_date=b.confirm_date,
        confirm_price=b.confirm_price, executable_price=b.executable_price,
        confirmed=b.confirm_date is not None,
        n_bars=len(b.source_unit_ids) or 5, id=b.id)


def _inherit_strength(child, parent) -> None:
    """二/三买强度继承:从来源/锚点一买继承 强度档 + 主信号判定(§8.1 强度档闸沿用)。"""
    if parent is None:
        return
    child.strength = parent.strength
    child.is_main = parent.is_main
    child.beichi_grade = parent.beichi_grade


def detect_second_buys(first_buys, confirmed_bis, *, level):
    """§8.2 五步:每个已确认一买/一卖后,取其后笔为次级别单位识别二买/二卖。

    ★ 二买继承其来源一买的强度档/is_main(标准一买→主二买;弱一买→弱二买)。
    """
    out = []
    for fb in first_buys:
        if fb.confirm_date is None:
            continue
        subs = [bi_to_unit(b) for b in confirmed_bis if b.start_date >= fb.pivot_date]
        sb = detect_second(fb, subs, side=fb.side, level=level)
        if sb is not None:
            _inherit_strength(sb, fb)               # 继承来源一买
            out.append(sb)
    return out


def detect_third_buys(bi_zhongshu, confirmed_bis, *, level, first_buys=None):
    """§8.3:笔中枢内向上离开 ZG 的已确认单位(leave)+ 其后反向回试不回(retest)→ 三买。

    离开段常作为中枢末成员把 GG 抬过 ZG;回试段为中枢后第一根反向笔(low > ZG)。
    ★ 三买继承"锚点一买"(同向、pivot 早于三买的最近一买)的强度档/is_main。
    """
    first_buys = first_buys or []
    out = []
    for zs in bi_zhongshu:
        ri = zs.end_unit + 1
        if ri >= len(confirmed_bis):
            continue
        retest_b = confirmed_bis[ri]
        if retest_b.direction != DOWN:
            continue
        leave_b = None
        for i in range(zs.end_unit, zs.start_unit - 1, -1):     # 末清离 ZG 的向上单位
            b = confirmed_bis[i]
            if b.direction == UP and max(b.start_price, b.pivot_price) > zs.ZG:
                leave_b = b
                break
        if leave_b is None:
            continue
        t = detect_third(zs, bi_to_unit(leave_b), bi_to_unit(retest_b),
                         side=BUY, level=level)
        if t is not None:
            anchors = [m for m in first_buys if m.kind == "一买"
                       and m.pivot_date is not None and m.pivot_date <= t.pivot_date]
            anchor = max(anchors, key=lambda m: m.pivot_date) if anchors else None
            _inherit_strength(t, anchor)            # 继承锚点一买
            out.append(t)
    return out


def _level_structures(df: pd.DataFrame, *, level: str, config: Config) -> dict:
    """单级别结构链路:包含→分型→笔→线段→中枢→背驰(趋势+盘整)。"""
    merged = process_inclusion(df)
    fractals = detect_fractals(merged, df, level=level)
    bis = build_bi(fractals, merged, level=level)

    pens = [_bi_to_pen(b, i) for i, b in enumerate(bis)]
    machine = build_segments(pens, level=level)
    for s in machine.confirmed:                       # 线段层补 executable(末根→None)
        s.executable_price, _ = executable_after(df, s.confirm_date)
    segments = machine.all_segments()

    confirmed_bis = [b for b in bis if b.confirm_date is not None]
    bi_zhongshu = build_zhongshu([_zunit_from_bi(b) for b in confirmed_bis],
                                 level=level, kind=BI)
    xd_zhongshu = build_zhongshu(_units_from_segments(machine.confirmed),
                                 level=level, kind=XIANDUAN)
    zhongshus = bi_zhongshu + xd_zhongshu

    macd = compute_macd(df["close"], config=config)   # ★ 引擎自算 MACD(§1.4),不用文件自带
    cons = detect_beichis(bi_zhongshu, confirmed_bis, macd, df, level=level, config=config)
    trend = detect_trend_beichis(bi_zhongshu, confirmed_bis, macd, df,
                                 level=level, config=config)

    # §7.1 MACD 暖机守卫:前 warmup 根为 EMA 暖机区,不发背驰(C 段落在暖机区者剔除)
    warmup = config.macd_warmup_bars()

    def _pos(dt):
        try:
            return df.index.get_loc(dt)
        except KeyError:
            return len(df)

    beichi_tuples = [t for t in (trend + cons) if _pos(t[0].confirm_date) >= warmup]
    return {
        "bis": bis, "segments": segments, "zhongshus": zhongshus,
        "bi_zhongshu": bi_zhongshu, "confirmed_bis": confirmed_bis,
        "beichi_tuples": beichi_tuples,                # 趋势优先(标准一买)
    }


def _beichi_span(bc):
    """背驰段时间跨度 [A段起点, C段确认]。"""
    start = bc.seg_start_date or bc.pivot_date
    return start, bc.confirm_date


def _time_contains(outer, inner) -> bool:
    """§9.2 区间套:小级别背驰的**精确点**(pivot→confirm)落在大级别背驰段时间范围内。

    (用小级别的极值/确认点而非其 A 段起点,避免跨级别重采样导致的起点错位;
    但大级别背驰段必须『当前在进行』——若其 confirm 早于小级别背驰点则不嵌套 → 无共振。)
    """
    os_, oe = _beichi_span(outer)
    ip, ic = inner.pivot_date, inner.confirm_date
    if None in (os_, oe, ip, ic):
        return False
    return os_ <= ip and ic <= oe


def _anchor_invalidated(bc, side, price_df) -> bool:
    """§9.3 小转大/信号失效:锚点背驰 confirm 后,价格顺原趋势越过 pivot(未反向)→ 失效。

    顶背驰(side=UP):confirm 后收盘超越 pivot 高点 → 失效;
    底背驰(side=DOWN):confirm 后收盘跌破 pivot 低点 → 失效。
    """
    if price_df is None or bc.confirm_date is None or bc.pivot_price is None:
        return False
    after = price_df.loc[price_df.index > bc.confirm_date, "close"]
    if after.empty:
        return False
    if side == UP:
        return float(after.max()) > bc.pivot_price
    return float(after.min()) < bc.pivot_price


def _signal_invalidated(m, price_df) -> bool:
    """§9.3 每信号失效:confirm 后价格顺原趋势越过该买卖点 pivot(反转未成)。

    买点(buy):后续收盘**跌破** pivot 低点 → 失效;卖点(sell):**升破** pivot 高点 → 失效。
    ★ 仅作结果字段,不从事件流/回测触发剔除(实盘右端可隐藏,回测全计入)。
    """
    if m.confirm_date is None or m.pivot_price is None:
        return False
    after = price_df.loc[price_df.index > m.confirm_date, "close"]
    if after.empty:
        return False
    if m.side == "buy":
        return float(after.min()) < m.pivot_price
    return float(after.max()) > m.pivot_price


def build_lianli_nested(daily_tuples, weekly_tuples, *, level,
                        min30_tuples=None, min30_consistent=True, price_df=None):
    """§9.2 区间套联立:以右端当前日线主背驰为锚,要求小级别背驰**时间嵌套**于大级别同向背驰段。

    - 周线同向主背驰的段【时间范围包含】日线背驰段 → 日嵌于周(共振候选);
      大级别当前无在进行的同向背驰(如周线最近标准背驰是 2023 旧底)→ 不嵌套 → 右端无共振。
    - 真 30min(且与日线同前复权基准)同向主背驰【嵌于日线段内】→ 三级齐 → 共振·最高强度;
      否则(30min 缺失/不一致)→ 日周共振·待30min 降一档。
    - ★ §9.3 锚点背驰失效:confirm 后价格顺原向越过 pivot(未反向)→ 信号失效(小转大),
      policy 撤销转折动作(持有/顺势)。
    - 无日线主背驰(仅弱信号)→ structure_signal=无,不进任何主信号动作。
    """
    min30_tuples = min30_tuples or []
    if not daily_tuples:
        return None
    d_mains = [(bc, s) for bc, _z, s in daily_tuples if bc.is_main_signal]
    if not d_mains:
        bc, _z, s = daily_tuples[0]                     # 仅弱背驰背景 → 无主信号
        return build_lianli(daily_beichi=bc, side="bottom" if s == DOWN else "top")

    d_bc, d_side = max(d_mains, key=lambda t: t[0].confirm_date)  # 右端当前日线主背驰
    side = "bottom" if d_side == DOWN else "top"
    # ★ 锚点背驰失效优先判定:确认后顺原向越过 pivot → 信号失效(小转大)
    if _anchor_invalidated(d_bc, d_side, price_df):
        return build_lianli(daily_beichi=d_bc, side=side, daily_continuation_failed=True)
    # 周线:同向标准主背驰,其段时间范围包含日线背驰段 → 嵌套
    w_bc = next((bc for bc, _z, s in weekly_tuples
                 if bc.is_main_signal and s == d_side and _time_contains(bc, d_bc)), None)
    # 30min:一致基准 + 同向标准主背驰,且嵌于日线段内
    m_bc = None
    if min30_consistent:
        m_bc = next((bc for bc, _z, s in min30_tuples
                     if bc.is_main_signal and s == d_side and _time_contains(d_bc, bc)), None)
    return build_lianli(weekly_beichi=w_bc, daily_beichi=d_bc, min30_beichi=m_bc, side=side)


def assemble_signal_events(maimaidians, *, symbol, df, config):
    """§10.2 事件流最小闭环出口:
    - 只取 **confirmed** 买卖点(有 confirm_date 且有 executable_price;末根 live_pending 自动排除);
    - 统一 SignalEventRecord 结构,触发只用 confirm_date + executable_price(无 pivot_*);
    - §8.6 二三买重合 → overlap_2_3 + 互入 supporting_signals;
    - §6.7 同级别同向同类去重(dedupe_event_cluster),其余进 supporting_signals,不重复计入。
    """
    confirmed = [m for m in maimaidians
                 if m.confirm_date is not None and m.executable_price is not None]
    # §8.6 二三买重合(同向同 pivot)
    seconds = [m for m in confirmed if m.kind in ("二买", "二卖")]
    thirds = [m for m in confirmed if m.kind in ("三买", "三卖")]
    for s in seconds:
        for t in thirds:
            if (s.side == t.side and s.pivot_date == t.pivot_date
                    and s.pivot_price == t.pivot_price):
                mark_overlap_2_3(s, t)
                if t.id not in s.supporting_signals:
                    s.supporting_signals.append(t.id)
                if s.id not in t.supporting_signals:
                    t.supporting_signals.append(s.id)
    records = [to_signal_event(m, beichi_grade=m.beichi_grade) for m in confirmed]

    # §6.7 事件簇去重(适配 SignalEvent → dedupe → 映回 SignalEventRecord)
    adapters = []
    for r in records:
        try:
            bar = int(df.index.get_loc(r.confirm_date))
        except (KeyError, TypeError):
            bar = -1
        adapters.append(SignalEvent(
            id=r.id, symbol=symbol, level=r.level, direction=r.direction, kind=r.kind,
            confirm_date=r.confirm_date, confirm_price=r.confirm_price or 0.0,
            confirm_bar=bar, source_rank=1,
            beichi_rank=(2 if r.beichi_grade == "标准背驰"
                         else (1 if r.beichi_grade else 0)),
            level30_rank=0))
    triggers = dedupe_event_cluster(adapters, config=config)
    by_id = {r.id: r for r in records}
    out = []
    for t in triggers:
        rec = by_id[t.id]
        rec.supporting_signals = list(dict.fromkeys(
            list(rec.supporting_signals) + list(t.supporting_signals)))
        out.append(rec)
    out.sort(key=lambda r: r.confirm_date)
    return out


def run_pipeline(
    df: pd.DataFrame, *, level: str = "daily", config: Config = DEFAULT_CONFIG,
    weekly_df: pd.DataFrame | None = None, min30_df: pd.DataFrame | None = None,
    symbol: str = "",
) -> dict:
    """跑完整结构链路 + 区间套联立,返回**原始结构对象**(供输出层与测试使用)。

    ``min30_df`` 给定时:§1.10 自动校验其与日线是否同前复权基准,不一致(REJECT_LIANLI)
    则该 30min **不进日-30min 联立**(只用一致基准),仅日线分析照常。
    """
    d = _level_structures(df, level=level, config=config)
    beichi_tuples = d["beichi_tuples"]
    beichis = [bc for bc, _z, _s in beichi_tuples]

    first_buys = detect_maimaidians(beichi_tuples, level=level)
    second_buys = detect_second_buys(first_buys, d["confirmed_bis"], level=level)
    third_buys = detect_third_buys(d["bi_zhongshu"], d["confirmed_bis"], level=level,
                                   first_buys=first_buys)
    maimaidians = first_buys + second_buys + third_buys

    # §7.1 暖机守卫:confirm 落在暖机区的买卖点不发(右端未确认 confirm_date=None 保留)
    warmup = config.macd_warmup_bars()
    cutoff_date = df.index[warmup].isoformat() if len(df) > warmup else None

    def _confirm_in_warmup(m):
        if m.confirm_date is None:
            return False
        try:
            return df.index.get_loc(m.confirm_date) < warmup
        except KeyError:
            return False

    maimaidians = [m for m in maimaidians if not _confirm_in_warmup(m)]
    assign_ids(maimaidians, level=level)
    for m in maimaidians:                          # §9.3 每信号失效结果字段(不删样本)
        m.invalidated = _signal_invalidated(m, df)

    # §1.9 周线由日线合成 + §1.10 30min 一致性门禁 + §9.2 区间套联立
    weekly_tuples = []
    weekly_beichis = []
    min30_tuples = []
    min30_beichis = []
    min30_consistency = None       # None=未提供 / OK / WARN / REJECT_LIANLI
    if level == "daily":
        wdf = weekly_df if weekly_df is not None else synthesize_weekly(df)
        if len(wdf) >= 5:
            w = _level_structures(wdf, level="weekly", config=config)
            weekly_tuples = w["beichi_tuples"]
            weekly_beichis = [bc for bc, _z, _s in weekly_tuples]
        if min30_df is not None:
            cons = check_consistency(min30_df, df, symbol="", config=config)
            min30_consistency = cons.status
            if not cons.reject_daily_30min_lianli:     # 一致基准才进联立
                m = _level_structures(min30_df, level="min30", config=config)
                min30_tuples = m["beichi_tuples"]
                min30_beichis = [bc for bc, _z, _s in min30_tuples]
    lianli = build_lianli_nested(
        beichi_tuples, weekly_tuples, level=level, min30_tuples=min30_tuples,
        min30_consistent=(min30_consistency != "REJECT_LIANLI"), price_df=df)

    monitor = []
    if len(df):
        current = float(df["close"].iloc[-1])
        # 按 confirm_date 取时间最近的中枢与最近的一买(防拿到旧中枢/最早一买)
        zss = d["zhongshus"]
        latest_zs = (max(zss, key=lambda z: z.confirm_date) if zss else None)
        firsts = [m for m in maimaidians if m.kind == "一买" and m.pivot_date is not None]
        first_buy_low = (max(firsts, key=lambda m: m.pivot_date).pivot_price
                         if firsts else None)
        monitor = derive_monitor_levels(current_price=current, zhongshu=latest_zs,
                                        recent_first_buy_low=first_buy_low)

    macd_warmup = {
        "bars": warmup,
        "cutoff_date": cutoff_date,                    # analysis_start_date ≥ 此日
        "analysis_start_date": cutoff_date,
        "fully_in_warmup": len(df) <= warmup,
        "note": (f"前 {warmup} 根为 EMA 暖机区(MACD_WARMUP·低置信);"
                 "该区间不发背驰/买卖点"),
    }
    signal_events = assemble_signal_events(maimaidians, symbol=symbol, df=df, config=config)
    return {
        "bis": d["bis"], "segments": d["segments"], "zhongshus": d["zhongshus"],
        "beichis": beichis, "maimaidians": maimaidians, "lianli": lianli,
        "monitor": monitor, "weekly_beichis": weekly_beichis,
        "min30_beichis": min30_beichis, "min30_consistency": min30_consistency,
        "macd_warmup": macd_warmup, "signal_events": signal_events,
    }


def _resolve_market(symbol: str, market: str | None) -> str | None:
    """显式 market 优先;否则按 SYMBOLS 反查;未知则 None(只能做长度判定)。"""
    if market is not None:
        return market
    try:
        return market_of(symbol)
    except KeyError:
        return None


def _health_to_dict(hr) -> dict:
    """§1.7 HealthReport → 顶层 data_health(诚实暴露门禁状态;不假装正常)。

    ★ status 取门禁判定 OK/WARN/REJECT;无缺失但历史不足时升格暴露为 SHORT_HISTORY
    (与 REJECT/WARN 正交:仅在本会判 OK 时才显示 SHORT_HISTORY,绝不掩盖 REJECT/WARN)。
    """
    base = hr.status
    if base == HealthStatus.OK.value and hr.short_history:
        status = "SHORT_HISTORY"
    else:
        status = base
    notes: list[str] = []
    if hr.short_history:
        notes.append("SHORT_HISTORY·历史不足,趋势背驰/多中枢/区间套等长历史判定标低置信")
    if hr.pre_analysis_history_unavailable:
        notes.append("分析起点前历史不可得(pre_analysis_history_unavailable)")
    if hr.reasons:
        notes.append("reasons=" + ",".join(hr.reasons))
    if hr.flags:
        notes.append("flags=" + ",".join(hr.flags))

    def _iso(d):
        return d.isoformat() if d is not None else None

    return {
        "status": status,
        "missing_rate": hr.missing_rate,
        "missing_count": hr.missing_days,
        "max_consecutive_missing": hr.max_consecutive_missing,
        "expected_sessions": hr.expected_sessions,
        "present_bars": hr.present_bars,
        "bars_available": hr.bars_available,
        "required_length": hr.required_length,
        "analysis_start_date": _iso(hr.analysis_start_date),
        "source_start_date": _iso(hr.source_start_date),
        "listed_date": _iso(hr.listed_date),
        "pre_analysis_history_unavailable": hr.pre_analysis_history_unavailable,
        "short_history": hr.short_history,
        "low_confidence": hr.low_confidence,
        "rejected": hr.rejected,
        "reasons": list(hr.reasons),
        "flags": list(hr.flags),
        "warnings": notes,
        "notes": "; ".join(notes) if notes else None,
    }


def build_data_health(
    df: pd.DataFrame, *, symbol: str, market: str | None, level: str = "daily",
    config: Config = DEFAULT_CONFIG, listed_date=None, analysis_start_date=None,
    suspended_dates=None,
) -> dict:
    """对输入 df 跑 §1.7 健康检查 → 顶层 data_health dict(不再为 null)。

    market 可解析 → 用交易所日历判缺失;不可解析 → 仅长度/充足度判定(calendar=None)。
    """
    cal = get_calendar(market) if market is not None else None
    hr = check_health(
        df, market=market or "?", symbol=symbol, level=level, calendar=cal,
        listed_date=listed_date, analysis_start_date=analysis_start_date,
        suspended_dates=suspended_dates, config=config,
    )
    return _health_to_dict(hr)


def build_data_snapshot(
    df: pd.DataFrame, *, symbol: str, market: str | None, level: str = "daily",
    source: str = "local_csv", adjust: str = "qfq",
) -> dict:
    """本地最小可复现快照(不联网):内容派生 data_snapshot_id + 轻量摘要(不 dump 全量)。

    id 由 symbol|market|level|source(含 CSV 文件名)|adjust + 规范化内容哈希派生:
    同份数据重跑稳定;内容/文件名/源任一变化 → id 变化;不含当前时间(可复现)。
    """
    snapshot_id = compute_snapshot_id(
        df, symbol=symbol, market=market or "?", level=level,
        source=source, adjust=adjust,
    )
    return {
        "data_snapshot_id": snapshot_id,
        "symbol": symbol,
        "market": market,
        "level": level,
        "source": source,
        "adjust": adjust,
        "row_count": int(len(df)),
        "first_date": df.index[0].isoformat() if len(df) else None,
        "last_date": df.index[-1].isoformat() if len(df) else None,
    }


def analyze(
    df: pd.DataFrame, *, symbol: str, market: str | None = None,
    level: str = "daily", data_health=None, snapshot_meta=None,
    config: Config = DEFAULT_CONFIG, min30_df: pd.DataFrame | None = None,
    source: str = "local_csv", adjust: str = "qfq",
    listed_date=None, analysis_start_date=None, suspended_dates=None,
) -> dict:
    """规范 OHLCV → §11.1 输出 dict(完整链路 + executable_price)。

    数据门禁 + 可复现快照(§1.2/§1.7)自动接入:
    - 未显式传 ``data_health`` → 自动跑 :func:`build_data_health`,顶层 data_health 不再为 null;
    - 自动派生内容哈希 ``data_snapshot_id`` + 轻量 ``data_snapshot`` 摘要;
    - 健康 ``REJECT`` → 不输出 confirmed 结构(结构列表清空),只诚实给出拒算的 data_health;
      OK/WARN/SHORT_HISTORY 正常输出(SHORT_HISTORY 标低置信,不伪装完整历史)。
    ★ 30min 一致性 REJECT_LIANLI 仍只拒日-30min 联立,不拒单级别日线(在 run_pipeline 内处理)。

    ``min30_df`` 给定 → §1.10 一致基准校验后接入日-30min 区间套联立。
    """
    mk = _resolve_market(symbol, market)
    if data_health is None:
        data_health = build_data_health(
            df, symbol=symbol, market=mk, level=level, config=config,
            listed_date=listed_date, analysis_start_date=analysis_start_date,
            suspended_dates=suspended_dates,
        )
    snapshot = build_data_snapshot(
        df, symbol=symbol, market=mk, level=level, source=source, adjust=adjust)

    rejected = isinstance(data_health, dict) and data_health.get("status") == "REJECT"
    if rejected:
        # 数据门禁 REJECT:不输出已确认结构,只诚实暴露拒算状态(不假装正常)。
        return build_output(
            symbol=symbol, level=level, data_health=data_health,
            data_snapshot=snapshot, data_snapshot_id=snapshot["data_snapshot_id"],
            bi=[], xianduan=[], zhongshu=[], beichi=[], mai_mai_dian=[],
            lianli=None, monitor_levels=[], signal_events=[],
            min30_consistency=None, macd_warmup=None, config=config)

    r = run_pipeline(df, level=level, config=config, min30_df=min30_df, symbol=symbol)
    return build_output(
        symbol=symbol, level=level, data_health=data_health, snapshot_meta=snapshot_meta,
        data_snapshot=snapshot, data_snapshot_id=snapshot["data_snapshot_id"],
        bi=r["bis"], xianduan=r["segments"], zhongshu=r["zhongshus"],
        beichi=r["beichis"], mai_mai_dian=r["maimaidians"], lianli=r["lianli"],
        monitor_levels=r["monitor"], signal_events=r["signal_events"],
        min30_consistency=r["min30_consistency"], macd_warmup=r["macd_warmup"],
        config=config)


def format_report(output: dict) -> str:
    """可读报告(§11.3)。"""
    lines = [
        f"标的: {output['symbol']}  级别: {output['level']}",
        f"spec={output['spec_version']} engine={output['engine_version']} "
        f"config_hash={output['algorithm_config_hash']}",
        f"笔: {len(output['bi'])}  线段: {len(output['xianduan'])}  "
        f"中枢: {len(output['zhongshu'])}  背驰: {len(output['beichi'])}  "
        f"买卖点: {len(output['mai_mai_dian'])}",
    ]
    mw = output.get("macd_warmup")
    if mw:
        lines.append(f"MACD 暖机区: 前 {mw['bars']} 根(截至 {mw['cutoff_date']})"
                     f" MACD_WARMUP·低置信,不发背驰/买卖点"
                     + ("  ⚠ 全程在暖机区" if mw["fully_in_warmup"] else ""))
    pending_bi = [b for b in output["bi"] if b.get("status") == "forming"]
    pending_xd = [s for s in output["xianduan"] if s.get("state") != "CONFIRMED_END"]
    if pending_bi:
        lines.append(f"右端未确认笔: {[b['id'] for b in pending_bi]}")
    if pending_xd:
        lines.append(f"右端未确认线段: state={[s['state'] for s in pending_xd]}")
    for m in output["mai_mai_dian"]:
        lines.append(f"买卖点 {m.get('label') or m['kind']} "
                     f"pivot={m['pivot_price']}({m.get('pivot_relation_to_zhongshu')}) "
                     f"confirm={m.get('confirm_relation_to_zhongshu')} "
                     f"executable={m['executable_price']} is_main={m.get('is_main')} "
                     f"status={m['status']}")
    if output["zhongshu"]:
        z = output["zhongshu"][-1]
        lines.append(f"最近中枢 [{z['ZD']}, {z['ZG']}] GG={z['GG']} DD={z['DD']} "
                     f"段数={z['n_segments']} extending={z['extending']}")
    for m in output["monitor_levels"]:
        lines.append(f"监控位 [{m['tier']}] {m['price']} — {m['hint']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="缠论引擎 CLI:代号/名称 → JSON + 报告")
    p.add_argument("symbol", help="标的代号(或名称)")
    p.add_argument("--level", default="daily", choices=["daily", "weekly", "min30"])
    p.add_argument("--csv", help="规范 OHLCV CSV 路径(离线);省略则尝试在线拉取")
    p.add_argument("--json-only", action="store_true", help="只打印 JSON")
    args = p.parse_args(argv)

    try:
        market = market_of(args.symbol)
    except KeyError:
        market = None

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["date"]).set_index("date")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = "date"
    else:
        from .data.fetch import fetch
        df = fetch(args.symbol, market=market, level=args.level).df

    output = analyze(df, symbol=args.symbol, market=market, level=args.level)
    print(to_json(output))
    if not args.json_only:
        print("\n" + format_report(output), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
