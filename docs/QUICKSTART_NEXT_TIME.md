# 下次快速上手指南(5 分钟恢复上下文)

## A. 项目当前状态(一句话)
缠论引擎 v0.1-mvp 已跑通:000001 标准趋势底背驰 demo 通过;report 已按 S/A/B/C/D 降噪;
TDX provider + run_auto 已接入,主要用于 A 股 min30 短周期数据;data_health 对 A 股节假日
误杀已修;当前暂停在 daily raw 的 xianduan 偶数笔 assert 边界。

## B. 本地启动步骤
```bash
cd ~/Claude\ Code/chanlun/fishhouse
git pull
```

## C. 标准 demo 验证
```bash
./scripts/demo_000001.sh
```
验收点:
```text
2019-01-04 一买·标准 = S / primary
2019-03-14 二买/三买 = A / secondary
report 有主信号、监控位、弱信号统计
```

## D. TDX min30 一键运行
```bash
python3 -m chanlun.run_auto \
  --market cn \
  --symbol sz300308 \
  --level min30 \
  --start 20250101 \
  --end 20260617 \
  --provider tdx
```
输出:
```text
outputs/sz300308_min30_20250101_20260617/
  source.csv
  out.json
  report.txt
```
> 需先 `pip install pytdx`,且环境允许出站 TCP 7709(行情服务器)。

## E. 查看结构数量
```bash
python3 - <<'PY'
import json

path = "outputs/sz300308_min30_20250101_20260617/out.json"
with open(path, "r", encoding="utf-8") as f:
    out = json.load(f)

print("health:", out["data_health"]["status"])
print("flags:", out["data_health"].get("flags"))
print("bars:", out["data_snapshot"]["row_count"])
print("bi:", len(out.get("bi", [])))
print("xianduan:", len(out.get("xianduan", [])))
print("zhongshu:", len(out.get("zhongshu", [])))
print("beichi:", len(out.get("beichi", [])))
print("signals:", len(out.get("signal_events", [])))
PY
```
（也可以直接 `python3 scripts/show_events.py outputs/.../out.json`。）

## F. 当前已知边界
```text
1. TDX 数据是 raw,不是 qfq;
2. TDX 更适合 min30 / 分时短周期确认;
3. 长周期 daily/weekly 仍优先用 qfq 源;
4. sz300308 daily raw 当前触发 xianduan assert:
   "线段含笔数应为单数(§5.4 assert), 实得 22"
5. v0.1-mvp 是结构辅助工具,不是自动交易系统。
```
补充:本地若缺 `exchange_calendars`,daily 用 weekday 兜底日历,节假日不再误杀
(标 `APPROX_CALENDAR_WEEKDAY`,至多 WARN);min30 不套 daily session 缺失逻辑。

## G. 下次继续开发的优先级
```text
P1:用 TDX min30 跑 300308 / 300502 / 600989,确认是否稳定出 report;
P2:结合 out.json / report.txt / K线截图做人眼复核;
P3:如果多个股票都遇到 xianduan assert,再修线段边界;
P4:再做 invalidated 拆分 trade_invalidated / structure_invalidated;
P5:最后再考虑回测、dashboard、自动化策略。
```

## 样本数据位置(归档后)
```text
chanlun/data/raw/000001/000001_sh_daily_20170601_20190630_ohlcv.csv   # 默认 demo(qfq)
chanlun/data/raw/300750/300750_qfq_daily_20210101_20230731.csv        # 弱档趋势背驰(qfq)
chanlun/data/raw/300502/300502_daily.csv          # 基准B 长日线(2047 根)
chanlun/data/raw/300502/300502_daily_long.csv     # 同上(历史副本,内容一致)
chanlun/data/raw/300502/300502_daily_short.csv    # 228 根短样本 → SHORT_HISTORY
chanlun/data/raw/300502/300502_30min.csv          # 基准B 30min
chanlun/data/raw/300502/300502_30min_a.csv        # 基准A 30min
chanlun/data/raw/300502/300502_5min.csv           # 5min
```
`chanlun/data/` 根目录现在只放数据层代码;样本统一在 `raw/{symbol}/` 下。
