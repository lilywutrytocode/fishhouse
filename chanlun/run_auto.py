"""一键闭环:下载(tdx/eastmoney/akshare)→ 标准化 CSV → analyze → out.json/report.txt。

    python3 -m chanlun.run_auto --market cn --symbol sz300308 \
        --level min30 --start 20250101 --end 20260617 --provider tdx

不重做分析逻辑,直接复用 :func:`chanlun.cli.analyze` / :func:`chanlun.cli.format_report`。
TDX 标 ``adjust=raw``(report 顶部带 banner);eastmoney/akshare 走 qfq 源。
本轮:market=cn、level=daily|min30;无 auto fallback、无 interactive。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .cli import analyze, format_report
from .data.loaders import load_local_csv
from .data.symbols import normalize_cn_symbol
from .output import to_json

_MARKETS = {"cn": "A"}


def _fetch_df(provider: str, sym, market: str, level: str,
              start: str, end: str) -> tuple[pd.DataFrame, str]:
    """按 provider 拉数,返回 (规范 df, adjust)。失败抛异常(由调用方报错退出)。"""
    if provider == "tdx":
        from .data.sources.tdx_source import TdxSource
        res = TdxSource().fetch(sym.exch_symbol, market, level, start=start, end=end)
        return res.df, res.adjust
    if provider in ("eastmoney", "akshare"):
        from .data.fetch import fetch
        from .data.sources.akshare_source import AkshareSource
        from .data.sources.eastmoney_source import EastmoneySource
        src = {"akshare": AkshareSource, "eastmoney": EastmoneySource}[provider]()
        res = fetch(sym.code, market=market, level=level, sources=[src])
        df = _crop(res.df, start, end)
        return df, res.adjust
    raise ValueError(f"未知 provider:{provider}")


def _crop(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = df.index
    if start:
        df = df[idx >= pd.Timestamp(start, tz=idx.tz)]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz=df.index.tz)
                + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)]
    return df


def _save_csv(df: pd.DataFrame, path: Path, level: str) -> None:
    """规范 df → source.csv;daily date=YYYY-MM-DD,min30 date=YYYY-MM-DD HH:MM。"""
    fmt = "%Y-%m-%d" if level == "daily" else "%Y-%m-%d %H:%M"
    out = df.copy()
    out.insert(0, "date", out.index.strftime(fmt))
    cols = ["date", "open", "high", "low", "close", "volume", "amount"]
    out[cols].to_csv(path, index=False)


def run(*, market: str, symbol: str, level: str, start: str, end: str,
        provider: str, out_dir: str | None = None) -> Path:
    if market not in _MARKETS:
        raise ValueError(f"本轮仅支持 market=cn,收到 {market!r}")
    if level not in ("daily", "min30"):
        raise ValueError(f"本轮仅支持 level=daily|min30,收到 {level!r}")
    mk = _MARKETS[market]
    sym = normalize_cn_symbol(symbol)

    df, adjust = _fetch_df(provider, sym, mk, level, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"下载结果为空:{sym.exch_symbol} {level} {start}~{end}")

    out_path = Path(out_dir or f"outputs/{sym.exch_symbol}_{level}_{start}_{end}")
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "source.csv"
    _save_csv(df, csv_path, level)

    canon = load_local_csv(str(csv_path), level=level).df       # 复用标准加载
    output = analyze(canon, symbol=sym.exch_symbol, market=mk, level=level,
                     source="source.csv", adjust=adjust)

    (out_path / "out.json").write_text(to_json(output), encoding="utf-8")
    report = format_report(output)
    (out_path / "report.txt").write_text(report, encoding="utf-8")

    _print_summary(output, out_path, provider, adjust)
    return out_path


def _print_summary(output: dict, out_path: Path, provider: str, adjust: str) -> None:
    h = output.get("data_health") or {}
    print(f"输出目录: {out_path}/  (provider={provider} adjust={adjust})")
    print(f"data_health: status={h.get('status')} "
          f"missing_rate={h.get('missing_rate')} bars={h.get('bars_available')}")
    print(f"结构数量: bi={len(output['bi'])} xianduan={len(output['xianduan'])} "
          f"zhongshu={len(output['zhongshu'])} beichi={len(output['beichi'])} "
          f"mai_mai_dian={len(output['mai_mai_dian'])} "
          f"signal_events={len(output['signal_events'])}")
    primary = [e for e in (output.get("signal_events") or [])
               if e.get("signal_quality") in ("S", "A")]
    print("主信号摘要:")
    if not primary:
        print("  (无 S/A 级主信号)")
    for e in primary:
        print(f"  [{e['signal_quality']}] {str(e.get('pivot_date'))[:10]} "
              f"{e.get('kind')} pivot={e.get('pivot_price')} "
              f"confirm={str(e.get('confirm_date') or '')[:10]}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="一键:下载 → 标准化 CSV → analyze → out.json/report.txt")
    p.add_argument("--market", default="cn", help="市场(本轮仅 cn)")
    p.add_argument("--symbol", required=True, help="代号,如 sz300308 / sh000001 / 300308")
    p.add_argument("--level", default="daily", choices=["daily", "min30"])
    p.add_argument("--start", required=True, help="起始 YYYYMMDD")
    p.add_argument("--end", required=True, help="结束 YYYYMMDD")
    p.add_argument("--provider", default="tdx",
                   choices=["tdx", "eastmoney", "akshare"])
    p.add_argument("--out-dir", default=None, help="输出目录(默认 outputs/{...})")
    args = p.parse_args(argv)
    try:
        run(market=args.market, symbol=args.symbol, level=args.level,
            start=args.start, end=args.end, provider=args.provider,
            out_dir=args.out_dir)
    except Exception as e:
        print(f"[run_auto] 失败: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
