"""§1.1 拉取编排:主源 → 兜底回落 + 前复权校验。

A/港:akshare 主源,失败/数据不全 → 自动回落东财;美股:仅 yfinance(无兜底)。
源对象可注入(默认按市场用真实适配器),便于单元测试用桩源验证回落逻辑。
"""

from __future__ import annotations

from ..config import MARKET_SOURCES, MARKET_TZ, market_of
from .models import Adjust, validate_canonical
from .sources.base import DataSource, FetchError, SourceResult


def _default_sources(market: str) -> list[DataSource]:
    """按市场返回 [主源, 兜底?] 的真实适配器(惰性 import 在各源内部)。"""
    from .sources.akshare_source import AkshareSource
    from .sources.eastmoney_source import EastmoneySource
    from .sources.yfinance_source import YFinanceSource

    by_name = {
        "akshare": AkshareSource,
        "eastmoney": EastmoneySource,
        "yfinance": YFinanceSource,
    }
    primary, fallback = MARKET_SOURCES[market]
    chain = [by_name[primary]()]
    if fallback is not None:
        chain.append(by_name[fallback]())
    return chain


def _validate_qfq(result: SourceResult) -> None:
    """校验结果为规范 schema 且前复权(全四价)。不合格抛 :class:`FetchError`。"""
    try:
        validate_canonical(result.df)
    except ValueError as e:
        raise FetchError(f"源 {result.source} 返回非规范数据:{e}") from e
    if result.adjust != Adjust.QFQ.value:
        raise FetchError(f"源 {result.source} 非前复权(adjust={result.adjust})")


def fetch(
    symbol: str,
    *,
    market: str | None = None,
    level: str = "daily",
    sources: list[DataSource] | None = None,
) -> SourceResult:
    """拉取单标的×级别的前复权 OHLC,按源链顺序回落。

    返回首个成功且通过前复权校验的 :class:`SourceResult`;全部失败抛
    :class:`FetchError`(聚合各源错误)。
    """
    if market is None:
        market = market_of(symbol)
    chain = sources if sources is not None else _default_sources(market)
    if not chain:
        raise FetchError(f"市场 {market} 无可用源")

    errors: list[str] = []
    for src in chain:
        try:
            result = src.fetch(symbol, market, level)
            _validate_qfq(result)
            return result
        except FetchError as e:
            errors.append(f"{getattr(src, 'name', src)}: {e}")
            continue
    raise FetchError(
        f"{symbol}({market}/{level}) 所有源均失败 → " + " | ".join(errors)
    )
