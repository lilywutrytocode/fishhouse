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
)
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
            level=level, config=config, related_zhongshu_id=zs.id)
        # 只收已确认且成档的背驰(C 已 confirmed)
        if bc is not None and bc.confirm_date is not None:
            bc.id = f"beichi_{_LEVEL_CODE.get(level, level)}_{len(out) + 1:03d}"
            out.append((bc, zs, C.direction))
    return out


def detect_trend_beichis(bi_zhongshu, confirmed_bis, macd, df, *, level, config):
    """§7.4 趋势背驰:≥2 同级别中枢且 zs2 在 zs1 之外(同向趋势),比较趋势首/末同向笔。"""
    out = []
    for k in range(len(bi_zhongshu) - 1):
        zs1, zs2 = bi_zhongshu[k], bi_zhongshu[k + 1]
        if zs2.ZG < zs1.ZD:
            trend = DOWN
        elif zs2.ZD > zs1.ZG:
            trend = UP
        else:
            continue
        # A = 趋势初始同向推动(从头第一个同向笔);C = 趋势末段同向推动(最后一个)
        a_idx = next((i for i in range(0, zs1.end_unit + 1)
                      if confirmed_bis[i].direction == trend), None)
        c_idx = next((i for i in range(len(confirmed_bis) - 1, zs2.start_unit - 1, -1)
                      if confirmed_bis[i].direction == trend), None)
        if a_idx is None or c_idx is None or c_idx <= a_idx:
            continue
        A, C = confirmed_bis[a_idx], confirmed_bis[c_idx]
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
            related_zhongshu_id=zs2.id, reset_dif_values=reset)
        if bc is not None and bc.confirm_date is not None:
            bc.id = f"beichi_{_LEVEL_CODE.get(level, level)}_t{len(out) + 1:03d}"
            out.append((bc, zs2, trend))
    return out


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


def detect_second_buys(first_buys, confirmed_bis, *, level):
    """§8.2 五步:每个已确认一买/一卖后,取其后笔为次级别单位识别二买/二卖。"""
    out = []
    for fb in first_buys:
        if fb.confirm_date is None:
            continue
        subs = [bi_to_unit(b) for b in confirmed_bis if b.start_date >= fb.pivot_date]
        side = fb.side
        sb = detect_second(fb, subs, side=side, level=level)
        if sb is not None:
            out.append(sb)
    return out


def detect_third_buys(bi_zhongshu, confirmed_bis, *, level):
    """§8.3:笔中枢内向上离开 ZG 的已确认单位(leave)+ 其后反向回试不回(retest)→ 三买。

    离开段常作为中枢末成员把 GG 抬过 ZG;回试段为中枢后第一根反向笔(low > ZG)。
    """
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

    macd = compute_macd(df["close"], config=config)
    cons = detect_beichis(bi_zhongshu, confirmed_bis, macd, df, level=level, config=config)
    trend = detect_trend_beichis(bi_zhongshu, confirmed_bis, macd, df,
                                 level=level, config=config)
    return {
        "bis": bis, "segments": segments, "zhongshus": zhongshus,
        "bi_zhongshu": bi_zhongshu, "confirmed_bis": confirmed_bis,
        "beichi_tuples": trend + cons,                 # 趋势优先(标准一买)
    }


def build_lianli_two_level(daily_tuples, weekly_tuples, *, level):
    """§9.3 日-周两级联立:取日线主背驰(标准 is_main)+ 同向周线主背驰 → 共振/转折。

    ★ policy 按 is_main 过滤:无日线主背驰(仅弱信号)→ 不进任何主信号动作(structure_signal=无)。
    """
    daily_main = next(((bc, side) for bc, _z, side in daily_tuples
                       if bc.is_main_signal), None)
    if daily_main is None:
        if not daily_tuples:
            return None
        bc, _z, side = daily_tuples[0]                  # 仅弱背驰背景 → 无主信号
        return build_lianli(daily_beichi=bc,
                            side="bottom" if side == DOWN else "top")
    d_bc, d_side = daily_main
    w_bc = next((bc for bc, _z, side in weekly_tuples
                 if bc.is_main_signal and side == d_side), None)
    return build_lianli(daily_beichi=d_bc, weekly_beichi=w_bc,
                        side="bottom" if d_side == DOWN else "top")


def run_pipeline(
    df: pd.DataFrame, *, level: str = "daily", config: Config = DEFAULT_CONFIG,
    weekly_df: pd.DataFrame | None = None,
) -> dict:
    """跑完整结构链路 + 日-周联立,返回**原始结构对象**(供输出层与测试使用)。"""
    d = _level_structures(df, level=level, config=config)
    beichi_tuples = d["beichi_tuples"]
    beichis = [bc for bc, _z, _s in beichi_tuples]

    first_buys = detect_maimaidians(beichi_tuples, level=level)
    second_buys = detect_second_buys(first_buys, d["confirmed_bis"], level=level)
    third_buys = detect_third_buys(d["bi_zhongshu"], d["confirmed_bis"], level=level)
    maimaidians = first_buys + second_buys + third_buys
    assign_ids(maimaidians, level=level)

    # §1.9 周线由日线合成 + §9.3 日-周联立(30min 先留空接口)
    weekly_tuples = []
    weekly_beichis = []
    if level == "daily":
        wdf = weekly_df if weekly_df is not None else synthesize_weekly(df)
        if len(wdf) >= 5:
            w = _level_structures(wdf, level="weekly", config=config)
            weekly_tuples = w["beichi_tuples"]
            weekly_beichis = [bc for bc, _z, _s in weekly_tuples]
    lianli = build_lianli_two_level(beichi_tuples, weekly_tuples, level=level)

    monitor = []
    if len(df):
        current = float(df["close"].iloc[-1])
        latest_zs = d["zhongshus"][-1] if d["zhongshus"] else None
        first_buy_low = next((m.pivot_price for m in maimaidians
                              if m.kind == "一买"), None)
        monitor = derive_monitor_levels(current_price=current, zhongshu=latest_zs,
                                        recent_first_buy_low=first_buy_low)

    return {
        "bis": d["bis"], "segments": d["segments"], "zhongshus": d["zhongshus"],
        "beichis": beichis, "maimaidians": maimaidians, "lianli": lianli,
        "monitor": monitor, "weekly_beichis": weekly_beichis,
    }


def analyze(
    df: pd.DataFrame, *, symbol: str, market: str | None = None,
    level: str = "daily", data_health=None, snapshot_meta=None,
    config: Config = DEFAULT_CONFIG,
) -> dict:
    """规范 OHLCV → §11.1 输出 dict(完整链路 + executable_price)。"""
    r = run_pipeline(df, level=level, config=config)
    return build_output(
        symbol=symbol, level=level, data_health=data_health, snapshot_meta=snapshot_meta,
        bi=r["bis"], xianduan=r["segments"], zhongshu=r["zhongshus"],
        beichi=r["beichis"], mai_mai_dian=r["maimaidians"], lianli=r["lianli"],
        monitor_levels=r["monitor"], config=config)


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
    pending_bi = [b for b in output["bi"] if b.get("status") == "forming"]
    pending_xd = [s for s in output["xianduan"] if s.get("state") != "CONFIRMED_END"]
    if pending_bi:
        lines.append(f"右端未确认笔: {[b['id'] for b in pending_bi]}")
    if pending_xd:
        lines.append(f"右端未确认线段: state={[s['state'] for s in pending_xd]}")
    for m in output["mai_mai_dian"]:
        lines.append(f"买卖点 {m['kind']}·{m.get('subkind') or ''} "
                     f"pivot={m['pivot_price']}({m.get('pivot_relation_to_zhongshu')}) "
                     f"confirm={m.get('confirm_relation_to_zhongshu')} "
                     f"executable={m['executable_price']} status={m['status']}")
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
