"""run_auto + TDX provider 最小回归(不联网)。

联网 smoke(sz300308 min30/daily)手动跑,不入 CI。
"""

from __future__ import annotations

import json

import pytest

from chanlun.data.symbols import normalize_cn_symbol


# ── 1. symbol 规范化 ─────────────────────────────────────────────────────────
def test_symbol_normalize_bare_and_prefixed():
    assert normalize_cn_symbol("300308").exch_symbol == "sz300308"
    assert normalize_cn_symbol("600989").exch_symbol == "sh600989"
    assert normalize_cn_symbol("sh000001").exch_symbol == "sh000001"
    # 交易所号 + 指数识别
    a = normalize_cn_symbol("sh000001")
    assert a.tdx_market == 1 and a.is_index is True
    b = normalize_cn_symbol("sz300308")
    assert b.tdx_market == 0 and b.is_index is False


def test_symbol_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_cn_symbol("400001")        # 非 6/0/3 开头 → 不猜
    with pytest.raises(ValueError):
        normalize_cn_symbol("30030")         # 非 6 位


# ── 2. TDX 数据转换(注入假 bars,不联网)──────────────────────────────────────
def _fake_bars(level):
    base = "2025-01-02 10:00" if level == "min30" else "2025-01-02"
    out = []
    # 故意乱序 + 一条重复,验证排序/去重
    seq = ["2025-01-03", "2025-01-02", "2025-01-02", "2025-01-06"]
    if level == "min30":
        seq = ["2025-01-02 10:30", "2025-01-02 10:00",
               "2025-01-02 10:00", "2025-01-02 11:00"]
    for i, dt in enumerate(seq):
        out.append({"datetime": dt, "open": 10 + i, "high": 11 + i,
                    "low": 9 + i, "close": 10.5 + i, "vol": 100 + i})  # 无 amount
    return out


def test_tdx_transform_columns_sorted_dedup():
    from chanlun.data.sources.tdx_source import TdxSource
    src = TdxSource()
    for level in ("daily", "min30"):
        df = src._to_frame(_fake_bars(level), level)
        assert list(df.columns) == ["date", "open", "high", "low",
                                    "close", "volume", "amount"]
        assert df["date"].is_monotonic_increasing          # 升序
        assert df["date"].is_unique                          # 去重
        assert (df["amount"] == 0).all()                     # amount 缺失填 0


def test_tdx_empty_raises():
    from chanlun.data.sources.base import FetchError
    from chanlun.data.sources.tdx_source import TdxSource
    src = TdxSource()
    # 直接走 fetch 的空数据分支:用桩替换内部分页返回空
    src._connect = lambda api: "stub"            # type: ignore
    src._page_all = lambda *a, **k: []            # type: ignore

    import sys
    import types
    fake = types.ModuleType("pytdx.hq")
    fake.TdxHq_API = lambda **k: types.SimpleNamespace(disconnect=lambda: None)
    sys.modules["pytdx"] = types.ModuleType("pytdx")
    sys.modules["pytdx.hq"] = fake
    try:
        with pytest.raises(FetchError):
            src.fetch("sz300308", "A", "daily", start="20250101", end="20260101")
    finally:
        sys.modules.pop("pytdx.hq", None)
        sys.modules.pop("pytdx", None)


# ── 3. run_auto 输出文件(provider 打桩为本地样本,不联网)────────────────────
def test_run_auto_writes_outputs(tmp_path, monkeypatch):
    from chanlun.data.loaders import load_local_csv
    import chanlun.run_auto as ra

    sample = load_local_csv(
        "chanlun/data/raw/000001/000001_sh_daily_20170601_20190630_ohlcv.csv",
        level="daily").df

    def _stub(provider, sym, market, level, start, end):
        return sample.copy(), "raw"             # 模拟 tdx raw

    monkeypatch.setattr(ra, "_fetch_df", _stub)
    out_dir = ra.run(market="cn", symbol="sh000001", level="daily",
                     start="20170601", end="20190630", provider="tdx",
                     out_dir=str(tmp_path / "o"))

    assert (out_dir / "source.csv").exists()
    assert (out_dir / "out.json").exists()
    assert (out_dir / "report.txt").exists()
    assert (out_dir / "source.csv").read_text().splitlines()[0] == \
        "date,open,high,low,close,volume,amount"

    out = json.loads((out_dir / "out.json").read_text())
    assert out["data_snapshot"]["adjust"] == "raw"
    assert out["data_snapshot"]["confidence"] == "lower_than_qfq_daily"

    report = (out_dir / "report.txt").read_text()
    assert report.startswith("TDX raw data;")          # raw banner 顶部
    assert "主信号:" in report and "弱信号统计:" in report
