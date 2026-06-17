"""模块 11 · 输出 schema 与版本化(§11.1)。

顶层带 ``spec_version / engine_version / data_snapshot_id / algorithm_config_hash``
(区分行情变 vs 规格/代码/快照变)。所有结构含 §0.6 通用字段;买卖点带 ``related_*`` +
``supporting_signals``。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, fields, is_dataclass
from enum import Enum

import pandas as pd

from . import SPEC_VERSION, __version__
from .config import DEFAULT_CONFIG, Config


def _jsonable(o):
    """递归转 JSON 友好:Timestamp→ISO,dataclass/Enum/np→原生。"""
    if o is None or isinstance(o, (str, bool, int, float)):
        return o
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if is_dataclass(o) and not isinstance(o, type):
        return _jsonable(asdict(o))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "item"):              # numpy 标量
        return o.item()
    return str(o)


def algorithm_config_hash(config: Config = DEFAULT_CONFIG) -> str:
    """对可配置阈值求 sha256 前 16 位:区分"规格/代码"变更。"""
    payload = json.dumps(_jsonable(asdict(config)), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _serialize(obj, *, parent_id=None, source_alias: str | None = None,
               source_unit_ids=None) -> dict:
    """结构对象 → dict,补齐 §0.6 通用字段 parent_id / source_unit_ids。"""
    d = _jsonable(obj)
    if not isinstance(d, dict):
        return d
    d.setdefault("parent_id", parent_id)
    if source_unit_ids is not None:
        d["source_unit_ids"] = _jsonable(source_unit_ids)
    elif source_alias and source_alias in d:
        d["source_unit_ids"] = d[source_alias]
    d.setdefault("source_unit_ids", [])
    return d


def serialize_zhongshu(z) -> dict:
    return _serialize(z, source_alias="member_unit_ids")


def serialize_beichi(b) -> dict:
    return _serialize(b, source_unit_ids=[x for x in (b.a_unit_id, b.c_unit_id) if x])


def build_output(
    *,
    symbol: str,
    level: str = "daily",
    data_health=None,
    snapshot_meta=None,
    data_snapshot=None,
    data_snapshot_id: str | None = None,
    bi: list | None = None,
    xianduan: list | None = None,
    zhongshu: list | None = None,
    beichi: list | None = None,
    mai_mai_dian: list | None = None,
    lianli=None,
    monitor_levels: list | None = None,
    signal_events: list | None = None,
    min30_consistency=None,
    macd_warmup=None,
    config: Config = DEFAULT_CONFIG,
) -> dict:
    """组装顶层输出 dict(§11.1 + §10.2 事件流),含版本化字段。"""
    # data_snapshot_id 优先用显式传入(内容派生),否则回落 snapshot_meta(§1.2)。
    snapshot_id = data_snapshot_id or getattr(snapshot_meta, "data_snapshot_id", None)
    return {
        "spec_version": SPEC_VERSION,
        "engine_version": __version__,
        "data_snapshot_id": snapshot_id,
        "data_snapshot": _jsonable(data_snapshot) if data_snapshot is not None else None,
        "algorithm_config_hash": algorithm_config_hash(config),
        "symbol": symbol,
        "level": level,
        "data_health": _jsonable(data_health) if data_health is not None else None,
        "bi": [_serialize(b) for b in (bi or [])],
        "xianduan": [_serialize(s) for s in (xianduan or [])],
        "zhongshu": [serialize_zhongshu(z) for z in (zhongshu or [])],
        "beichi": [serialize_beichi(b) for b in (beichi or [])],
        "mai_mai_dian": [_serialize(m) for m in (mai_mai_dian or [])],
        "lianli": _jsonable(lianli) if lianli is not None else None,
        "monitor_levels": [_jsonable(m) for m in (monitor_levels or [])],
        "signal_events": [_serialize(e) for e in (signal_events or [])],   # §10.2 事件流
        "min30_consistency": min30_consistency,                            # §1.10 30min 门禁
        "macd_warmup": _jsonable(macd_warmup) if macd_warmup is not None else None,
    }


# §11.1/§10.2 顶层必备键(供 schema 完整性校验)
REQUIRED_TOP_KEYS = (
    "spec_version", "engine_version", "data_snapshot_id", "data_snapshot",
    "algorithm_config_hash", "data_health", "bi", "xianduan", "zhongshu", "beichi",
    "mai_mai_dian", "lianli", "monitor_levels", "signal_events", "min30_consistency",
    "macd_warmup",
)

# 事件流每条必备字段(§0.6 通用 + §10.2 扩展)
REQUIRED_EVENT_FIELDS = (
    "id", "parent_id", "source_unit_ids", "level", "direction", "status",
    "pivot_date", "pivot_price", "confirm_date", "confirm_price", "executable_price",
    "kind", "downgraded", "invalidated", "beichi_grade", "supporting_signals",
)


def output_schema_complete(output: dict) -> bool:
    """顶层键齐 + 买卖点含 label + 事件流每条含必备字段。"""
    if not all(k in output for k in REQUIRED_TOP_KEYS):
        return False
    if not all("label" in m for m in output["mai_mai_dian"]):
        return False
    if not all(all(f in e for f in REQUIRED_EVENT_FIELDS) for e in output["signal_events"]):
        return False
    return True


def to_json(output: dict, *, indent: int = 2) -> str:
    return json.dumps(output, ensure_ascii=False, indent=indent)
