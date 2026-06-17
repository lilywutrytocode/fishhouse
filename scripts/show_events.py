#!/usr/bin/env python3
"""v0.1-mvp 查看器:读 analyze 产出的 out_xxx.json,打印摘要 + 事件流。

用法:
    python3 scripts/show_events.py out_000001.json

不改 chanlun 核心逻辑;只读 JSON,不重算。
"""

from __future__ import annotations

import json
import sys


def _n(x) -> int:
    return len(x) if isinstance(x, list) else 0


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法: python3 scripts/show_events.py <out_xxx.json>", file=sys.stderr)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        out = json.load(f)

    h = out.get("data_health") or {}
    print(f"symbol={out.get('symbol')} level={out.get('level')}")
    print(f"data_snapshot_id={out.get('data_snapshot_id')}")
    print(f"data_health: status={h.get('status')} missing_rate={h.get('missing_rate')} "
          f"missing_count={h.get('missing_count')} short_history={h.get('short_history')}")

    print("counts: "
          f"bi={_n(out.get('bi'))} xianduan={_n(out.get('xianduan'))} "
          f"zhongshu={_n(out.get('zhongshu'))} beichi={_n(out.get('beichi'))} "
          f"mai_mai_dian={_n(out.get('mai_mai_dian'))} "
          f"signal_events={_n(out.get('signal_events'))}")

    events = out.get("signal_events") or []

    print("\n最近 10 条 signal_events:")
    for e in events[-10:]:
        print(f"  id={e.get('id')} kind={e.get('kind')} subkind={e.get('subkind')} "
              f"dir={e.get('direction')} grade={e.get('beichi_grade')} "
              f"pivot={e.get('pivot_price')} confirm={e.get('confirm_date')} "
              f"executable={e.get('executable_price')} "
              f"invalidated={e.get('invalidated')}")

    standard = [e for e in events if e.get("subkind") == "标准"]
    print(f"\nsubkind=标准 的事件({len(standard)} 条):")
    for e in standard:
        print(f"  id={e.get('id')} kind={e.get('kind')} dir={e.get('direction')} "
              f"grade={e.get('beichi_grade')} pivot={e.get('pivot_price')} "
              f"confirm={e.get('confirm_date')} invalidated={e.get('invalidated')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
