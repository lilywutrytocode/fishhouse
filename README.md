# fishhouse · 确定性缠论引擎

输入(代号/名称)→ JSON + 可读报告:合并 → 分型 → 笔 → 线段 → 中枢 → 背驰 →
三类买卖点 → 区间套/联立。完整判据见 [`缠论引擎开发规格.md`](缠论引擎开发规格.md) (v1.2)
与 [`CLAUDE.md`](CLAUDE.md)。

构建顺序(自底向上,严格按依赖):
`数据 → 包含 → 分型 → 笔 → 线段 → 中枢 → 背驰 → 买卖点 → 区间套/联立 → 概率/LLM → 输出`

## 当前进度

✅ **模块 1 · 数据层**(§1.1–1.10),其余模块未开始。

| 文件 | 规格 | 内容 |
|---|---|---|
| `chanlun/config.py` | §1.1/1.4/1.5/1.8 | 标的/市场/源/时区/日历映射 + 可配置阈值 |
| `chanlun/data/models.py` | §1.2/1.6/1.8 | 规范 OHLCV schema、枚举、报告/快照数据结构 |
| `chanlun/data/calendars.py` | §1.7/1.8 | 交易日历(A=XSHG / 港=XHKG / 美=XNYS,兜底周内日历) |
| `chanlun/data/sources/` | §1.1/1.2 | akshare(主)/东财(兜底)/yfinance,**惰性 import** |
| `chanlun/data/fetch.py` | §1.1 | 主源→兜底回落 + 前复权校验 |
| `chanlun/data/snapshot.py` | §1.2 | 快照落盘/读取 + 内容派生 `data_snapshot_id` |
| `chanlun/data/health.py` | §1.7 | 数据健康检查(listed/source/analysis 三日期分离) |
| `chanlun/data/weekly.py` | §1.9 | 周线由日线合成(锚定周五,不直读源周线) |
| `chanlun/data/consistency.py` | §1.10 | 日内-日线一致性校验(REJECT 仅作用于联立) |

设计取舍:真实行情源(akshare/yfinance)在适配器内**惰性 import**,因此本包与全部
单元测试**离线可跑、不依赖这些库也不触网**;健康/合成/一致性逻辑全部在 pandas 层面,
测试用合成数据 + 注入式日历/桩源确定性验证。

## 开发

```bash
pip install -e .            # 核心依赖(pandas/numpy/pyarrow)
pip install -e '.[sources]' # 可选:真实行情源 + 交易日历
pip install -e '.[test]'    # pytest
pytest                       # 39 用例,全部离线
```
