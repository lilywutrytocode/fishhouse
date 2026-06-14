"""模块 1 · 数据层。

子模块:
- :mod:`chanlun.data.models`       规范 OHLCV schema、枚举、报告/快照数据结构
- :mod:`chanlun.data.calendars`    交易日历(A/港/美)
- :mod:`chanlun.data.sources`      行情源适配器(akshare / 东财 / yfinance,惰性 import)
- :mod:`chanlun.data.fetch`        拉取编排:主源→兜底、前复权校验、快照
- :mod:`chanlun.data.snapshot`     快照落盘 / 读取 + data_snapshot_id
- :mod:`chanlun.data.health`       §1.7 数据健康检查
- :mod:`chanlun.data.weekly`       §1.9 周线由日线合成
- :mod:`chanlun.data.consistency`  §1.10 日内-日线一致性校验
"""
