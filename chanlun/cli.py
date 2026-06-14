"""模块 11 · CLI(§11.3/11.4):代号/名称 → JSON + 可读报告。

``analyze`` 把规范 OHLCV 串起 包含→分型→笔→线段→中枢,产出 §11.1 输出 dict;
右端未完成结构显式标注(笔 forming / 线段非 CONFIRMED_END / 分型 pending)。
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .config import DEFAULT_CONFIG, Config, market_of
from .monitor import derive_monitor_levels
from .output import build_output, to_json
from .structure.bi import build_bi
from .structure.fractal import detect_fractals
from .structure.inclusion import DOWN, UP, process_inclusion
from .structure.xianduan import Pen, build_segments
from .structure.zhongshu import BI, XIANDUAN, ZUnit, build_zhongshu


def _bi_to_pen(b, idx: int) -> Pen:
    hi = max(b.start_price, b.pivot_price)
    lo = min(b.start_price, b.pivot_price)
    return Pen(direction=b.direction, high=hi, low=lo, idx=idx,
               start_date=b.start_date, end_date=b.pivot_date, bi_id=b.id)


def _units_from_bis(bis) -> list[ZUnit]:
    """仅已确认笔(有 confirm_date)→ 笔中枢单位。"""
    units = []
    for b in bis:
        if b.confirm_date is None:
            continue
        units.append(ZUnit(
            high=max(b.start_price, b.pivot_price),
            low=min(b.start_price, b.pivot_price),
            start_date=b.start_date, start_price=b.start_price,
            confirm_date=b.confirm_date, confirm_price=b.confirm_price,
            direction=b.direction, id=b.id,
        ))
    return units


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
            direction=s.direction, id=s.id,
        ))
    return units


def analyze(
    df: pd.DataFrame, *, symbol: str, market: str | None = None,
    level: str = "daily", data_health=None, snapshot_meta=None,
    config: Config = DEFAULT_CONFIG,
) -> dict:
    """规范 OHLCV → §11.1 输出 dict(包含→分型→笔→线段→中枢 + 监控位)。"""
    merged = process_inclusion(df)
    fractals = detect_fractals(merged, df, level=level)
    bis = build_bi(fractals, merged, level=level)

    pens = [_bi_to_pen(b, i) for i, b in enumerate(bis)]
    machine = build_segments(pens, level=level)
    segments = machine.all_segments()

    bi_zhongshu = build_zhongshu(_units_from_bis(bis), level=level, kind=BI)
    xd_zhongshu = build_zhongshu(_units_from_segments(machine.confirmed),
                                 level=level, kind=XIANDUAN)
    zhongshus = bi_zhongshu + xd_zhongshu

    monitor = []
    if len(df):
        current = float(df["close"].iloc[-1])
        latest_zs = zhongshus[-1] if zhongshus else None
        monitor = derive_monitor_levels(current_price=current, zhongshu=latest_zs)

    return build_output(
        symbol=symbol, level=level, data_health=data_health, snapshot_meta=snapshot_meta,
        bi=bis, xianduan=segments, zhongshu=zhongshus,
        beichi=[], mai_mai_dian=[], lianli=None,
        monitor_levels=monitor, config=config,
    )


def format_report(output: dict) -> str:
    """可读报告(§11.3)。"""
    lines = [
        f"标的: {output['symbol']}  级别: {output['level']}",
        f"spec={output['spec_version']} engine={output['engine_version']} "
        f"config_hash={output['algorithm_config_hash']}",
        f"笔: {len(output['bi'])}  线段: {len(output['xianduan'])}  "
        f"中枢: {len(output['zhongshu'])}",
    ]
    # 右端未确认结构显式呈现(§11.4)
    pending_bi = [b for b in output["bi"] if b.get("status") == "forming"]
    pending_xd = [s for s in output["xianduan"]
                  if s.get("state") != "CONFIRMED_END"]
    if pending_bi:
        lines.append(f"右端未确认笔: {[b['id'] for b in pending_bi]}")
    if pending_xd:
        lines.append(f"右端未确认线段: state={[s['state'] for s in pending_xd]}")
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
