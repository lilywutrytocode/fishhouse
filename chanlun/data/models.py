"""规范数据结构:OHLCV schema、枚举、健康/一致性报告、快照元数据。

§1.6 规范 schema:列 ``date, open, high, low, close, volume, amount``,``amount`` 可空。
内部统一表示:**带时区的 ``DatetimeIndex``(名为 ``date``)+ 数值列**,
列顺序固定为 :data:`OHLCV_COLUMNS`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd


# 规范数值列(不含索引 date)
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume", "amount")
# 价格列(前复权应作用于全部四价,§1.2)
PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


class Market(str, Enum):
    A = "A"
    HK = "HK"
    US = "US"


class Level(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MIN30 = "min30"


class Adjust(str, Enum):
    QFQ = "qfq"   # 前复权(钉死,§1.2)
    NONE = "none"


class HealthStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    REJECT = "REJECT"


class ConsistencyStatus(str, Enum):
    OK = "OK"
    WARN = "WARN"
    # ★ 仅作用于日-30min 联立,不 REJECT 单级别日线(§1.10)
    REJECT_LIANLI = "REJECT_LIANLI"


# 健康检查标志位 / 拒绝原因码
FLAG_SHORT_HISTORY = "SHORT_HISTORY"
FLAG_LISTED_DATE_UNKNOWN = "LISTED_DATE_UNKNOWN"
FLAG_PRE_ANALYSIS_HISTORY_UNAVAILABLE = "PRE_ANALYSIS_HISTORY_UNAVAILABLE"
# weekday 兜底日历(无真实交易日历):缺失含节假日,门禁对正常假期缺口软化为 WARN
FLAG_APPROX_CALENDAR_WEEKDAY = "APPROX_CALENDAR_WEEKDAY"

REASON_NO_DATA = "NO_DATA"
REASON_CONSEC_MISSING = "CONSEC_MISSING"
REASON_MISSING_RATE = "MISSING_RATE"
REASON_HAS_MISSING = "HAS_MISSING"


def validate_canonical(df: pd.DataFrame) -> None:
    """校验 DataFrame 是否符合规范 schema,不符合抛 ``ValueError``。

    要求:tz-aware ``DatetimeIndex``(名 ``date``)、严格递增唯一、
    含 :data:`OHLCV_COLUMNS` 全部列、价列无缺失、high≥low。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("索引必须是 DatetimeIndex")
    if df.index.tz is None:
        raise ValueError("索引必须带时区(§1.8)")
    if df.index.name != "date":
        raise ValueError("索引名必须是 'date'")
    if not df.index.is_monotonic_increasing:
        raise ValueError("索引必须按时间严格递增")
    if not df.index.is_unique:
        raise ValueError("索引存在重复时间戳")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少规范列:{missing}")
    for c in PRICE_COLUMNS + ("volume",):
        if df[c].isna().any():
            raise ValueError(f"列 {c!r} 不允许缺失")
    if (df["high"] < df["low"]).any():
        raise ValueError("存在 high < low 的异常 bar")


def to_canonical(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """把宽松输入整理为规范 schema(不改数值,只规整结构 + 列顺序 + 时区)。

    输入可用 ``date`` 列或已有 DatetimeIndex;``amount`` 缺失则补 ``NaN``。
    """
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    if out.index.tz is None:
        out.index = out.index.tz_localize(tz)
    else:
        out.index = out.index.tz_convert(tz)
    if "amount" not in out.columns:
        out["amount"] = pd.NA
    out = out[list(OHLCV_COLUMNS)]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


@dataclass(frozen=True)
class SnapshotMeta:
    """快照元数据(只读,保证可复现,§1.2)。"""

    symbol: str
    market: str
    level: str
    source: str          # 实际生效的源(主源或兜底)
    adjust: str          # 复权口径(qfq)
    fetch_date: str      # 拉取日 ISO date
    tz: str
    row_count: int
    first_date: str | None
    last_date: str | None
    data_snapshot_id: str  # 内容哈希派生,详见 snapshot.compute_snapshot_id


@dataclass
class HealthReport:
    """§1.7 数据健康检查结果。"""

    symbol: str
    market: str
    level: str
    status: str  # HealthStatus

    listed_date: date | None
    source_start_date: date | None
    analysis_start_date: date | None
    pre_analysis_history_unavailable: bool

    # 统计(均自 analysis_start_date 起、交易所交易日历内)
    expected_sessions: int      # 期望交易日数(已剔停牌)
    present_bars: int           # 实际存在 bar 数
    suspended_days: int         # 停牌日数(≠ 缺失)
    missing_days: int
    missing_rate: float
    max_consecutive_missing: int

    # 历史充足度
    bars_available: int
    required_length: int | None
    short_history: bool
    low_confidence: bool        # 趋势背驰/多中枢/区间套等长历史判定标低置信

    reasons: list[str] = field(default_factory=list)   # 拒绝/告警原因码
    flags: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return self.status == HealthStatus.REJECT.value


@dataclass
class ConsistencyReport:
    """§1.10 日内-日线一致性校验结果。"""

    symbol: str
    status: str  # ConsistencyStatus
    compared_days: int
    max_close_dev: float
    max_high_dev: float
    max_low_dev: float
    warn_days: list[date] = field(default_factory=list)
    reject_days: list[date] = field(default_factory=list)

    @property
    def reject_daily_30min_lianli(self) -> bool:
        """是否禁止用此 30min 做日-30min 联立/共振。

        ★ 仅作用于联立,**不** REJECT 单级别日线分析(§1.10)。
        """
        return self.status == ConsistencyStatus.REJECT_LIANLI.value
