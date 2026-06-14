"""模块 11 · 监控位(§11.2)【确定性 + 提示语】

从结构派生(不手填):``{价格, 档位(caution/reassessment/target), 结构来源, 提示语}``。
提示语按现价相对中枢位置生成:
- 价在中枢**上方**:ZG = 三买防守/回中枢判断位;跌破 ZG = 三买失败/回中枢(≠三卖)。
- 价在中枢**下方**:ZD = 三卖回抽确认位;反抽不过 ZD = 三卖候选。
- 价在中枢**内部**:ZG/ZD 仅震荡边界,**不出**三买/三卖提示。
- reassessment = 最近一买低点(跌破=一买失效);target = 前高/GG/趋势背驰预期回跌位(B 段)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .structure.maimaidian import ABOVE, BELOW, INSIDE, relation_to_zhongshu

CAUTION, REASSESSMENT, TARGET = "caution", "reassessment", "target"


@dataclass
class MonitorLevel:
    price: float
    tier: str               # caution / reassessment / target
    source: str             # 结构来源
    hint: str               # 上下文提示语


def derive_monitor_levels(
    *,
    current_price: float,
    zhongshu=None,
    recent_first_buy_low: float | None = None,
    prev_high: float | None = None,
) -> list[MonitorLevel]:
    """由(最近)中枢 + 现价派生监控位与提示语。"""
    levels: list[MonitorLevel] = []

    if zhongshu is not None:
        rel = relation_to_zhongshu(current_price, zhongshu.ZD, zhongshu.ZG)
        src = f"中枢 {getattr(zhongshu, 'id', '')}".strip()
        if rel == ABOVE:
            levels.append(MonitorLevel(
                price=zhongshu.ZG, tier=CAUTION, source=src,
                hint="价在中枢上方:ZG=三买防守/回中枢判断位;跌破 ZG=三买失败/回中枢(不等于三卖)",
            ))
        elif rel == BELOW:
            levels.append(MonitorLevel(
                price=zhongshu.ZD, tier=CAUTION, source=src,
                hint="价在中枢下方:ZD=三卖回抽确认位;反抽不过 ZD=三卖候选",
            ))
        else:  # INSIDE
            levels.append(MonitorLevel(
                price=zhongshu.ZG, tier=CAUTION, source=src,
                hint="价在中枢内部:ZG/ZD 仅震荡边界,不出三买/三卖提示",
            ))
            levels.append(MonitorLevel(
                price=zhongshu.ZD, tier=CAUTION, source=src,
                hint="价在中枢内部:ZG/ZD 仅震荡边界,不出三买/三卖提示",
            ))

    if recent_first_buy_low is not None:
        levels.append(MonitorLevel(
            price=recent_first_buy_low, tier=REASSESSMENT, source="最近一买低点",
            hint="跌破=一买失效",
        ))

    target_price = prev_high if prev_high is not None else getattr(zhongshu, "GG", None)
    if target_price is not None:
        levels.append(MonitorLevel(
            price=target_price, tier=TARGET, source="前高/GG/趋势背驰预期回跌位",
            hint="目标位:前高/GG/趋势背驰预期回跌位(B 段)",
        ))

    return levels
