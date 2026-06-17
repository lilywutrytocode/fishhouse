#!/usr/bin/env bash
# v0.1-mvp 一键 demo:000001 上证指数(标准趋势底背驰样本)。
# 产出 out_000001.json + report_000001.txt,并打印结构数量与 2019-01-04 一买·标准事件。
# 不改 chanlun 核心逻辑;只是固化使用方式。
set -euo pipefail
cd "$(dirname "$0")/.."

CSV="chanlun/data/raw/000001/000001_sh_daily_20170601_20190630_ohlcv.csv"
SYMBOL="000001"
JSON="out_000001.json"
REPORT="report_000001.txt"

python3 - "$CSV" "$SYMBOL" "$JSON" "$REPORT" <<'PY'
import sys
from chanlun.data.loaders import load_local_csv
from chanlun.cli import analyze, format_report
from chanlun.output import to_json

csv, symbol, json_path, report_path = sys.argv[1:5]
df = load_local_csv(csv).df
out = analyze(df, symbol=symbol, level="daily", source=csv.split("/")[-1])

with open(json_path, "w", encoding="utf-8") as f:
    f.write(to_json(out))
with open(report_path, "w", encoding="utf-8") as f:
    f.write(format_report(out))

h = out["data_health"]
print(f"symbol={out['symbol']} level={out['level']} "
      f"snapshot={out['data_snapshot_id']}")
print(f"data_health: status={h['status']} missing_rate={h['missing_rate']} "
      f"short_history={h['short_history']}")
print(f"counts: bi={len(out['bi'])} xianduan={len(out['xianduan'])} "
      f"zhongshu={len(out['zhongshu'])} beichi={len(out['beichi'])} "
      f"mai_mai_dian={len(out['mai_mai_dian'])} signal_events={len(out['signal_events'])}")

ev = [e for e in out["signal_events"]
      if str(e.get("pivot_date") or "").startswith("2019-01-04")]
print("\n2019-01-04 一买·标准事件:")
if not ev:
    print("  (未找到 — 检查样本/算法回归)")
for e in ev:
    print(f"  id={e['id']} kind={e['kind']} subkind={e.get('subkind')} "
          f"dir={e['direction']} grade={e.get('beichi_grade')} "
          f"pivot={e['pivot_price']} confirm={e['confirm_date']} "
          f"executable={e['executable_price']} invalidated={e.get('invalidated')}")
print(f"\n输出文件: {json_path}  {report_path}")
PY
