"""A 股代号规范化(带交易所前缀 + TDX 市场号 + 指数识别)。

避免 ``000001`` 同时可能是上证指数与平安银行:本模块只按确定规则补前缀,
不确定时清晰报错,绝不猜错后继续跑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CnSymbol:
    exch_symbol: str   # 带前缀,如 sz300308 / sh000001
    tdx_market: int    # TDX 市场号:1=上交所(sh),0=深交所(sz)
    code: str          # 6 位代码
    is_index: bool     # 指数(sh000xxx / sz399xxx)→ 走 get_index_bars


def _market_for_bare(code: str) -> str:
    """纯 6 位代码补交易所:6→sh,0/3→sz,其它清晰报错。"""
    head = code[0]
    if head == "6":
        return "sh"
    if head in ("0", "3"):
        return "sz"
    raise ValueError(
        f"无法判断 {code!r} 所属交易所(仅支持 6→sh / 0,3→sz);"
        "如为北交所或指数请显式传带前缀代号(如 sh000001 / sz399001)。"
    )


def normalize_cn_symbol(s: str) -> CnSymbol:
    """规范化 A 股代号 → :class:`CnSymbol`。

    - 带前缀 ``sh*`` / ``sz*`` 原样使用;
    - 纯 6 位:``6``→sh、``0/3``→sz,其它报错(不猜);
    - 指数:``sh000xxx`` / ``sz399xxx`` → ``is_index=True``。
    """
    if not s or not isinstance(s, str):
        raise ValueError(f"无效代号:{s!r}")
    raw = s.strip().lower()
    if raw[:2] in ("sh", "sz"):
        exch, code = raw[:2], raw[2:]
    else:
        code = raw
        exch = _market_for_bare(code)
    if not (len(code) == 6 and code.isdigit()):
        raise ValueError(f"代码必须为 6 位数字:{s!r}")

    tdx_market = 1 if exch == "sh" else 0
    is_index = (exch == "sh" and code.startswith("000")) or (
        exch == "sz" and code.startswith("399"))
    return CnSymbol(exch_symbol=f"{exch}{code}", tdx_market=tdx_market,
                    code=code, is_index=is_index)
