# CLAUDE.md (v1.2)

确定性缠论引擎。本文件是纲领与纪律,**完整判据见 `缠论引擎开发规格.md` (v1.2)**。

## 项目目的
输入(代号/名称)→ JSON + 可读报告:合并 → 分型 → 笔 → 线段 → 中枢 → 背驰 → 三类买卖点 → 区间套/联立,附监控位与未确认状态。

## 来源优先级
主干=中泰框架;模糊/缺失/冲突→缠师 REF;规则已逐条锁定,实现不得擅改;未覆盖边界回报决策人,不静默选边。

## 构建顺序(自底向上,严格按依赖)
`数据 → 包含 → 分型 → 笔 → 线段 → 中枢 → 背驰 → 买卖点 → 区间套/联立 → 概率/LLM → 输出`
每模块完成即跑该模块回归。**线段(§5.7 状态机)最危险,先把状态机跑绿再搭中枢。**

## 模块依赖
```
数据(1)→包含(2)→分型(3)→笔(4)→线段(5,状态机)→中枢(6,笔+线段都建)─┬→背驰(7)→买卖点(8)→联立(9)
联立(9)→概率/LLM(10)→输出(11)
```

## 六条不可破的纪律
1. **★ pivot/confirm/executable(防未来函数)**:每结构带 `pivot_*`(极值)、`confirm_*`(可确认,confirm_price=确认 bar close)、`executable_price`(下一 bar open)。末根 bar→`live_pending` 不入回测。**回测/实盘只用 confirm_date + executable_price,严禁 pivot_***。测试强制断言 `confirm_date >= pivot_date`,需右侧确认者 `confirm_date > pivot_date`。
2. **确定层 vs 主观层不混**:结构+监控位+联立信号=代码确定层(可复现);操作动作/LLM 复核=主观层,贴标,仓位只给强度档,不输出"算出来的"百分比当确定值。
3. **★ LLM 不改结构**:LLM 只进 `review_notes`,不写 `bi/xianduan/zhongshu/beichi/mai_mai_dian` 确定字段,不把 `未确认/待定` 改 `confirmed`。平时 0 token,仅叙事/盲复核/复核建议三情形;留痕标 `非确定·非复现`。
4. **右端未确认**:最新 K 线附近结构默认 `未确认/待定`,显式 emit。`[需判断]` 分支不准启发式糊过去。
5. **数据门禁 + 降级隔离**:健康 REJECT 拒输出已确认结构;30min 用日线内部近似的信号不进最高强度共振,标 `降级共振`、policy 降一档。
6. **★ 线段只有 CONFIRMED_END 才喂中枢**:右端非 CONFIRMED_END 线段标未确认,不入确定性中枢。

## 关键已锁口径(v1.2)
- **笔**:纯新笔(独立、≥1 独立 K、≥5 根、无缺口豁免)。去重**按价格极值取**(最高顶/最低底),**同价取最先**;★ tie-break 只影响 pivot,**不得提前 confirm_date**。
- **线段**:特征序列法 + **§5.7 七态状态机**(FORMING/EXTENDING/BREAK_CANDIDATE/WAIT_SECOND_FEATURE/PENDING_DIRECTION/CONFIRMED_END/INVALIDATED)。每态定义输入/转移/输出/是否喂中枢。
- **中枢**:笔+线段都做;`ZG=min(三高)/ZD=max(三低)`,成立后固定,GG/DD 随延伸刷新。**§6.7 去重事件簇·只在同级别内**(同级同向同类才合并,跨级别都保留;线段中枢>笔中枢>...,confirm_date 早者触发,余进 supporting_signals)。
- **背驰**:三档(标准=主信号 / 面积 / DIF)+ 三态 k=0.9。**C 段 confirmed 才出 confirmed 背驰**,extending 只标提前判/疑似/待确认。盘整背驰取"最近一组同向 A/C"。趋势背驰 0 轴只出 `macd_reset_status` 字段,不硬门槛。
- **买卖点**:一买分标准(趋势背驰)/盘背(类一买),+ pivot/confirm 相对中枢位置。二买 5 步识别第一个次级别回调。三买离开需已确认次级别单位脱离 ZG、回试需确认结束,记 leave/retest id。
- **共振**:真共振=最高强度;**降级共振=降一档**(顶减仓不清仓/底试仓不重仓,真 30min 补齐可升级)。
- **级别**:日+周+30min;周线由日线合成;30min 与日线做一致性校验(>2% 只 REJECT 日-30min 联立,**不碰单级别日线**)。

## 数据(模块 1)
- A+港 akshare(回落东财),美股 yfinance;前复权(**OHLC 全复权**)+ 快照。
- 健康:交易日历判缺失,连续缺 ≥3 / 总缺 >5%(自 analysis_start_date)拒算;停牌≠缺失(SUSPENDED);短历史 SHORT_HISTORY。区分 listed/source/analysis_start_date + `pre_analysis_history_unavailable`。
- 交易制度:时区(沪/港/纽约)带时区存储;半日市按实际 session;美股只用 regular session。
- 标的:A 股 300502/300308/000792;港股 09992/02400;美股 NVDA/SNDK/IONQ/SMMT。

## 输出与版本化
- 顶层带 `spec_version(v1.2)/engine_version/data_snapshot_id/algorithm_config_hash`(区分行情变 vs 规格/代码/快照变)。
- 所有结构带 id/parent_id/source_unit_ids/level/direction/status/pivot_*/confirm_*/executable_price;买卖点带 related_zhongshu/beichi/leave_unit/retest_unit id + supporting_signals。
- 监控位:结构自动派生 + 上下文提示语(价在中枢上/下/内)。
- CLI:代号/名称 → JSON + 报告。

## 测试
缠师 54/70 课(单元)+ 新易盛 300502(实盘)+ 6 类极端(包含嵌套/连续同类分型/右端未确认/30min 降级/中枢>9 段/未来函数断言)。改代码必跑全部;结构变动需决策人确认。

## 可配置阈值(默认值,可调)
MACD 12/26/9;背驰 k=0.9;去重 confirm_date ≤3 根K、confirm_price ≤1%;一致性校验 close 0.5%/high·low 1%/REJECT 2%;后 N 日 5/10/20。
