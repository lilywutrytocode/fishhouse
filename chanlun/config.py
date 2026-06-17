"""全局约定与可配置阈值。

默认值取自 `CLAUDE.md` 「可配置阈值」一节与 `缠论引擎开发规格.md` 模块 1。
凡标 `可配置` 的阈值集中在 :class:`Config`,便于回测时整体替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── §1.1 数据源 / 标的【约定】──────────────────────────────────────────────
# 市场 → 标的列表
SYMBOLS: dict[str, list[str]] = {
    "A": ["300502", "300308", "000792"],
    "HK": ["09992", "02400"],
    "US": ["NVDA", "SNDK", "IONQ", "SMMT"],
}

# 市场 → (主源, 兜底源);兜底为 None 表示无回落(美股仅 yfinance)。
MARKET_SOURCES: dict[str, tuple[str, str | None]] = {
    "A": ("akshare", "eastmoney"),
    "HK": ("akshare", "eastmoney"),
    "US": ("yfinance", None),
}

# ── §1.8 交易制度 / 时区【约定】────────────────────────────────────────────
# 市场 → IANA 时区(沪/港/纽约),带时区存储。
MARKET_TZ: dict[str, str] = {
    "A": "Asia/Shanghai",
    "HK": "Asia/Hong_Kong",
    "US": "America/New_York",
}

# 市场 → exchange_calendars 交易所代码(A=上交所日历,港=港交所,美=NYSE)。
MARKET_CALENDAR: dict[str, str] = {
    "A": "XSHG",
    "HK": "XHKG",
    "US": "XNYS",
}


def market_of(symbol: str) -> str:
    """按 :data:`SYMBOLS` 反查标的所属市场。未知标的抛 ``KeyError``。"""
    for market, syms in SYMBOLS.items():
        if symbol in syms:
            return market
    raise KeyError(f"未知标的(不在 SYMBOLS 中):{symbol!r}")


@dataclass(frozen=True)
class Config:
    """可配置阈值集合。所有字段均有默认值,回测时可整体替换。"""

    # §1.4 MACD 12/26/9,收盘价
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # 背驰容差三态 k=0.9(模块 7,模块 1 不用,集中放此处便于统一管理)
    beichi_k: float = 0.9

    # §6.7 信号去重(模块 6,占位)
    dedup_confirm_bars: int = 3
    dedup_confirm_price_pct: float = 0.01

    # §1.10 日内-日线一致性校验阈值
    consistency_close_pct: float = 0.005   # close 偏差 > 0.5% → WARN
    consistency_highlow_pct: float = 0.01  # high/low 偏差 > 1% → WARN
    consistency_reject_pct: float = 0.02   # 任一核心价偏差 > 2% → REJECT 日-30min 联立

    # 后 N 日(模块 10 回测口径占位)
    after_n_days: tuple[int, ...] = (5, 10, 20)

    # §1.5 数据长度下限(级别 → 最少 bar 数);min30 未规定 → None
    daily_min_length: int = 250
    weekly_min_length: int = 150
    min30_min_length: int | None = None

    # §1.7 历史充足度:可得 bar < short_history_factor × 长度下限 → SHORT_HISTORY
    short_history_factor: float = 1.5

    # §7.1 ★ MACD EMA 暖机:每级别前 macd_warmup_factor × macd_slow 根为暖机区,
    # 不发背驰/买卖点(标 MACD_WARMUP·低置信);analysis_start_date ≥ source_start + 该根数。
    macd_warmup_factor: int = 5

    # §1.7 缺失判定
    missing_consecutive_reject: int = 3   # 连续缺 ≥ 3 交易日 → REJECT
    missing_rate_reject: float = 0.05      # 总缺失率 > 5% → REJECT
    # weekday 兜底日历(无真实交易日历)专用宽松阈值:节假日缺口不误杀,仅明显异常 REJECT
    approx_consec_reject: int = 10         # 连续缺 > 10 交易日(估算)→ REJECT
    approx_missing_rate_reject: float = 0.20  # 缺失率 > 20% → REJECT

    def macd_warmup_bars(self) -> int:
        """EMA 暖机根数 = factor × macd_slow(默认 5×26 = 130)。"""
        return self.macd_warmup_factor * self.macd_slow

    def min_length_for(self, level: str) -> int | None:
        """返回级别对应的长度下限。未知级别返回 None(不做长度判定)。"""
        return {
            "daily": self.daily_min_length,
            "weekly": self.weekly_min_length,
            "min30": self.min30_min_length,
        }.get(level)


DEFAULT_CONFIG = Config()
