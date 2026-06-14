"""行情源适配器。

所有真实源(akshare / 东财 / yfinance)在方法内**惰性 import**,因此本包及
单元测试无需安装这些依赖。源对象遵循 :class:`base.DataSource` 协议,
:mod:`chanlun.data.fetch` 据此做主源→兜底编排,测试可注入桩源。
"""

from .base import DataSource, FetchError, SourceResult

__all__ = ["DataSource", "FetchError", "SourceResult"]
