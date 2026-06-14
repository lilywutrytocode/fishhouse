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
from .structure.maimaidian import BUY, SELL, assign_ids, detect_first
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


def detect_maimaidians(beichi_tuples, *, level):
    """由背驰 + 中枢识别一买/一卖(确定层)。"""
    mmds = []
    for bc, zs, side_dir in beichi_tuples:
        side = BUY if side_dir == DOWN else SELL
        mmd = detect_first(bc, zs, side=side, level=level)
        if mmd is not None:
            mmds.append(mmd)
    assign_ids(mmds, level=level)
    return mmds


def build_lianli_single_level(beichi_tuples, *, level):
    """单级别(本 CSV)联立:仅日线档位有数据 → 本级别转折/无。"""
    if not beichi_tuples:
        return None
    bc, _zs, side_dir = beichi_tuples[0]
    return build_lianli(
        daily_beichi=bc, min30_is_approx=False,
        side="bottom" if side_dir == DOWN else "top")


def run_pipeline(
    df: pd.DataFrame, *, level: str = "daily", config: Config = DEFAULT_CONFIG,
) -> dict:
    """跑完整结构链路,返回**原始结构对象**(供输出层与测试使用)。"""
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
    beichi_tuples = detect_beichis(bi_zhongshu, confirmed_bis, macd, df,
                                   level=level, config=config)
    beichis = [bc for bc, _z, _s in beichi_tuples]
    maimaidians = detect_maimaidians(beichi_tuples, level=level)
    lianli = build_lianli_single_level(beichi_tuples, level=level)

    monitor = []
    if len(df):
        current = float(df["close"].iloc[-1])
        latest_zs = zhongshus[-1] if zhongshus else None
        first_buy_low = next((m.pivot_price for m in maimaidians
                              if m.kind == "一买"), None)
        monitor = derive_monitor_levels(current_price=current, zhongshu=latest_zs,
                                        recent_first_buy_low=first_buy_low)

    return {
        "bis": bis, "segments": segments, "zhongshus": zhongshus,
        "beichis": beichis, "maimaidians": maimaidians, "lianli": lianli,
        "monitor": monitor,
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
