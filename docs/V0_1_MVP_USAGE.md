# v0.1-mvp 使用说明

确定性缠论引擎的第一个可运行冻结版。**只做结构辅助分析,不是自动交易系统**:
输出的买卖点/背驰/中枢是确定层信号,操作决策与仓位仍由人(或主观层)负责。

## 1. clone / checkout / 确认 commit

```bash
git clone <repo-url> fishhouse
cd fishhouse
git checkout claude/awesome-curie-wf8m3d      # v0.1-mvp 所在分支
git log -1 --format="%h %s"                    # 确认在 v0.1-mvp commit 上
# 期望 HEAD = 6e53c69 (#4.5 data_health + snapshot 接入 analyze)
```

> 轻量 tag `v0.1-mvp` 已在本地指向该 commit;若远程缺该 tag,可在自己的 clone 里
> `git tag v0.1-mvp 6e53c69 && git push origin v0.1-mvp`。

依赖:Python 3.11+、pandas、numpy、exchange_calendars(交易日历;缺失时回落周一~周五日历)。

## 2. 运行 demo

```bash
./scripts/demo_000001.sh
```

产出:
- `out_000001.json` — 完整结构输出(§11.1 schema)
- `report_000001.txt` — 可读报告(§11.3)

终端会打印 data_health、bi/xianduan/zhongshu/beichi/mai_mai_dian/signal_events 数量,
以及 **2019-01-04 一买·标准** 事件(标准趋势底背驰回归锚点)。

查看任意产出 JSON:

```bash
python3 scripts/show_events.py out_000001.json
```

打印:symbol/level/data_health/data_snapshot_id、各结构数量、最近 10 条 signal_events、
以及全部 `subkind=标准` 的事件。

## 3. 为什么 000001 是默认 demo

`000001`(上证指数,2017-06~2019-06)是**标准趋势底背驰**的干净样本:
2019-01-04 一买被识别为 `一买·标准`(主信号,`is_main=True`),data_health=OK,
适合作为「引擎跑通」的回归锚点。

## 4. 其他样本与 data_health 行为

| 文件 | data_health | 说明 |
| --- | --- | --- |
| `chanlun/data/raw/000001/000001_sh_daily_20170601_20190630_ohlcv.csv` | **OK** | 默认 demo,标准趋势底背驰 |
| `chanlun/data/raw/300750/300750_qfq_daily_20210101_20230731.csv` | OK | 弱档趋势背驰样本 |
| `chanlun/data/300502_daily_long.csv` | OK(2047 根) | 强趋势股:趋势背驰 0 次,多数信号后续 `invalidated=True`;**不适合**当「典型背驰」demo,但 data_health 不拒 |
| `chanlun/data/300502_daily.csv` | **SHORT_HISTORY** | 228 根历史不足 → 低置信,仍输出但长历史判定不可全信 |

> 校正:`300502_daily_long.csv` 当前**不会**被 data_health REJECT(缺失率 0、无连续缺失)。
> data_health **REJECT** 只在「连续缺 ≥3 交易日」或「缺失率 >5%」时触发;REJECT 时引擎
> 不输出 confirmed 结构,只诚实给出拒算的 data_health。当前三个真实样本均无此情况。

## 5. 边界声明

- 这是**结构辅助层**:确定层给结构 + 监控位 + 信号,不给「算出来的胜率/收益百分比」。
- 右端最新 K 线附近结构默认未确认;回测/实盘只用 `confirm_date + executable_price`,严禁 `pivot_*`。
- `invalidated` 是**结果字段**(信号失效),不是回测剔除条件 —— 样本不删。
- 不含自动下单、仓位管理、实时行情;v0.1-mvp 仅离线 CSV → JSON/报告。
