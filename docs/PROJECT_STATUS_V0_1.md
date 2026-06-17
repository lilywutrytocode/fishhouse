# 项目状态归档 · v0.1-mvp

## 阶段结论
确定性缠论引擎 v0.1-mvp 已跑通完整结构链路并固化使用方式:离线/在线数据 → 标准 CSV →
analyze → JSON + 降噪报告。当前为**收尾归档**阶段,不再开发新功能;仅余少量已知边界。

## 已完成 checklist
- [x] 结构链路:包含 → 分型 → 笔 → 线段(§5.7 状态机)→ 中枢 → 背驰(趋势局部 A/C + 盘整)
      → 三类买卖点 → 区间套联立 → 事件流(§6.7 去重 + §8.6 重合)。
- [x] §0.5 防未来函数:pivot/confirm/executable;回测触发只用 confirm_date + executable_price。
- [x] 失效结果字段 invalidated(结果字段,不删样本)。
- [x] data_health + 可复现 snapshot 接入 analyze;data_health 对 A 股节假日误杀已修
      (weekday 兜底日历软化为 WARN,标 APPROX_CALENDAR_WEEKDAY;min30 不套 daily 逻辑)。
- [x] 信号分级 S/A/B/C/D + report 三段降噪(主信号 / 监控位 / 弱信号统计)。
- [x] TDX provider(pytdx)+ run_auto 一键闭环(A 股 daily|min30,raw,带 banner)。
- [x] 样本 CSV 归档到 raw/{symbol}/;demo 脚本 + show_events 查看器 + 使用文档。

## 未完成 TODO
- [ ] min30 单日 bar 数过少 → WARN(本轮标 TODO,未实现)。
- [ ] invalidated 拆分 trade_invalidated / structure_invalidated。
- [ ] auto fallback(tdx→eastmoney→akshare)、interactive 模式。
- [ ] 真三级共振样本(周-日-30min)、更多真实样本(需用户 CSV)。
- [ ] 最小回测统计(接口已锁 confirm_date + executable_price,未实现)。
- [ ] dashboard / 自动化策略。

## 已知 bug / 边界
1. TDX 数据为 **raw(不复权)**,仅适合 min30/分时短周期确认;长周期 daily/weekly 用 qfq 源。
2. `sz300308` daily raw 当前触发线段 assert:「线段含笔数应为单数(§5.4 assert), 实得 22」。
   暂停在此边界,本轮不修;若多股复现再统一修线段边界(见 QUICKSTART P3)。
3. 本地缺 `exchange_calendars` 时 daily 走 weekday 近似日历(节假日不误杀,至多 WARN)。
4. data_health REJECT 时不输出 confirmed 结构,report 顶部明确标「数据门禁: REJECT」。
5. v0.1-mvp 为结构辅助工具,不给胜率/收益,不是自动交易系统。

## 使用办法
- 标准 demo:`./scripts/demo_000001.sh`
- 一键下载分析:`python3 -m chanlun.run_auto --market cn --symbol sz300308 --level min30
  --start 20250101 --end 20260617 --provider tdx`
- 查看结果:`python3 scripts/show_events.py outputs/.../out.json`
- 详见 `docs/QUICKSTART_NEXT_TIME.md` 与 `docs/V0_1_MVP_USAGE.md`。

## 下次优先级
P1 TDX min30 跑 300308/300502/600989 验稳 → P2 人眼复核(out.json/report/K线截图)→
P3 修 xianduan 线段边界(若多股复现)→ P4 invalidated 拆分 → P5 回测/dashboard。
