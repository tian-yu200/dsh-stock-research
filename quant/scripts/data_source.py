# -*- coding: utf-8 -*-
"""
统一数据源模块：多源容错抓取 A 股/基金数据
东财(push2his)有反爬，自动降级 腾讯 -> 新浪
输出统一列: date, open, high, low, close, volume, amount, turnover
"""
import os

os.environ.setdefault("NO_PROXY", "*")  # 绕过系统代理直连国内数据源

import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

import akshare as ak
import pandas as pd

from market_data import DailyBarSchemaError, EmptyDailyBarResponse, normalize_daily_bars

class NoDailyBarsError(RuntimeError):
    """Independent sources agreed that the requested window has no bars."""

    def __init__(self, symbol: str, sources: list[str]) -> None:
        self.symbol = symbol
        self.sources = tuple(sources)
        joined = ", ".join(sources)
        super().__init__(f"{symbol} returned no daily bars from: {joined}")


# 解析类异常来自上游载荷形态或接口代码本身，秒级重试不会改变结果，
# 熔断与跳过重试都以“异常类型”为粒度，避免一次反爬/解析故障烧光时间预算。
_DETERMINISTIC_EXCEPTIONS = (AttributeError, KeyError, TypeError, IndexError)
_SOURCE_BREAKER_THRESHOLD = 20
_BREAKER_STATE: dict[str, dict[str, Any]] = {}


def _breaker_allows(name: str) -> bool:
    state = _BREAKER_STATE.get(name)
    return not (state and state.get("disabled"))


def _breaker_record_success(name: str) -> None:
    _BREAKER_STATE.pop(name, None)


def _breaker_record_failure(name: str, exc: BaseException) -> None:
    etype = type(exc).__name__
    state = _BREAKER_STATE.setdefault(
        name, {"etype": etype, "count": 0, "disabled": False}
    )
    if state["etype"] != etype:
        state.update({"etype": etype, "count": 1, "disabled": False})
        return
    state["count"] += 1
    if not state["disabled"] and state["count"] >= _SOURCE_BREAKER_THRESHOLD:
        state["disabled"] = True
        print(f"  [熔断] {name}源连续 {state['count']} 次 {etype}，本次运行内跳过该源")


def _market_prefix(symbol: str) -> str:
    """Return the exchange prefix expected by quote-history adapters."""
    symbol = str(symbol).strip()
    if symbol.startswith(("4", "8", "920")):
        return "bj"
    if symbol.startswith(("5", "6", "9")):
        return "sh"
    return "sz"


def _em(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start, end_date=end, adjust="hfq",
    )
    return normalize_daily_bars(df, source=f"eastmoney stock {symbol}")


def _tx(symbol: str, start: str, end: str) -> pd.DataFrame:
    prefixed = _market_prefix(symbol) + symbol
    df = ak.stock_zh_a_hist_tx(
        symbol=prefixed, start_date=start, end_date=end, adjust="hfq",
    )
    return normalize_daily_bars(df, source=f"tencent stock {symbol}")


def _sina(symbol: str, start: str, end: str) -> pd.DataFrame:
    prefixed = _market_prefix(symbol) + symbol
    df = ak.stock_zh_a_daily(
        symbol=prefixed, start_date=start, end_date=end, adjust="hfq",
    )
    return normalize_daily_bars(df, source=f"sina stock {symbol}")


_THS_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "http://q.10jqka.com.cn/"}
_THS_TIMEOUT_SECONDS = 10


def _ths_get_lines(path: str) -> list[str] | None:
    """Fetch one THS v6 line payload; 404/missing data means no rows for it."""
    url = "http://d.10jqka.com.cn/v6/line/" + path
    req = urllib.request.Request(url, headers=_THS_HEADERS)
    raw: bytes | None = None
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(req, timeout=_THS_TIMEOUT_SECONDS).read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code < 500 or attempt == 2:
                raise
            time.sleep(1.5)
    text = raw.decode("utf-8", errors="replace")  # type: ignore[union-attr]
    marker = '"data":"'
    if marker not in text:
        return None
    payload = text.split(marker, 1)[1].rsplit('"', 2)[0]
    return [line for line in payload.split(";") if line and line.split(",")[0].isdigit()]


def _ths_parse(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        c = line.split(",")
        if len(c) < 7 or not c[1] or not c[4]:
            continue
        d = c[0]
        try:
            rows.append(
                {
                    "date": f"{d[0:4]}-{d[4:6]}-{d[6:8]}",
                    "open": float(c[1]),
                    "high": float(c[2]),
                    "low": float(c[3]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                    "amount": float(c[6]) if c[6] else float("nan"),
                    "turnover": (float(c[7]) / 100.0) if len(c) > 7 and c[7] else float("nan"),
                }
            )
        except (ValueError, IndexError):
            continue
    return pd.DataFrame(rows)


def _ths(symbol: str, start: str, end: str) -> pd.DataFrame:
    """同花顺 v6 日线兜底源：/02 为后复权，金额单位元，920 北交所代码在 hs_ 命名空间。"""
    lines: list[str] = []
    for year in range(int(start[:4]), datetime.now().year + 1):
        part = _ths_get_lines(f"hs_{symbol}/02/{year}.js")
        if part:
            lines.extend(part)
    last = _ths_get_lines(f"hs_{symbol}/02/last.js")
    if last:
        lines.extend(last)
    if not lines:
        raise EmptyDailyBarResponse(f"ths stock {symbol}: no daily-bar payload")
    frame = _ths_parse(lines)
    if frame.empty:
        raise EmptyDailyBarResponse(f"ths stock {symbol}: empty daily-bar response")
    df = normalize_daily_bars(frame, source=f"ths stock {symbol}")
    df = df.loc[
        df["date"].between(
            pd.to_datetime(start, format="%Y%m%d"), pd.to_datetime(end, format="%Y%m%d")
        )
    ].reset_index(drop=True)
    if df.empty:
        raise EmptyDailyBarResponse(f"ths stock {symbol}: no bars in window")
    df.attrs["akshare_source"] = "同花顺"
    df.attrs["akshare_interface"] = "ths_line"
    df.attrs["adjustment"] = "hfq"
    print(f"  [{symbol}] ths line rows={len(df)}")
    return df


_SOURCES = [("东财", _em), ("腾讯", _tx), ("新浪", _sina)]


def _sources_for(symbol: str) -> list[tuple[str, Any]]:
    """920 北交所代码追加同花顺兜底：东财/腾讯当前不可用或解析不了北交所载荷。"""
    if str(symbol).startswith("920"):
        return _SOURCES + [("同花顺", _ths)]
    return _SOURCES


def fetch_a_stock_daily(symbol: str, start="20200101", end="20991231") -> pd.DataFrame:
    """Try independent sources and classify a confirmed no-bar window."""
    last_err = None
    empty_sources: list[str] = []
    schema_errors: list[DailyBarSchemaError] = []
    skipped_sources: list[str] = []
    for name, fn in _sources_for(symbol):
        if not _breaker_allows(name):
            skipped_sources.append(name)
            print(f"  [{symbol}] {name}源熔断跳过")
            continue
        try:
            df = fn(symbol, start, end)
            if df is not None and len(df) > 0:
                _breaker_record_success(name)
                print(f"  [{symbol}] 使用数据源: {name} ({len(df)} 行)")
                df.attrs["akshare_interface"] = (
                    "stock_zh_a_hist"
                    if name == "东财"
                    else "stock_zh_a_hist_tx"
                    if name == "腾讯"
                    else "stock_zh_a_daily"
                )
                df.attrs["akshare_source"] = name
                df.attrs["adjustment"] = "hfq"
                try:
                    from price_guards import validate_daily_bars as _pg_validate
                    from price_guards import ths_crosscheck as _pg_crosscheck
                    from price_guards import ths_patch_recent as _pg_patch
                    _pg_c, _pg_b, _pg_r = _pg_validate(df, source=name, symbol=symbol)
                    if len(_pg_b) > 0 and len(df) >= 20:
                        _pg_chk = _pg_crosscheck(symbol, df)
                        if _pg_chk.get("conflict"):
                            df, _pg_rep = _pg_patch(symbol, df)
                            print("  [" + symbol + "] 同花顺复核修复 " + str(_pg_rep.get("patched", 0)) + " 行 (" + str(_pg_chk.get("detail")) + ")")
                            df.attrs["ths_repaired"] = _pg_rep.get("patched", 0)
                        else:
                            print("  [" + symbol + "] 本地验货报警 " + str(len(_pg_b)) + " 行，复核无分歧，保留待查")
                except Exception as _pg_exc:
                    print("  [" + symbol + "] 同花顺复核跳过: " + type(_pg_exc).__name__)
                return df
            empty_sources.append(name)
            print(f"  [{symbol}] {name}源返回空行情")
        except EmptyDailyBarResponse:
            empty_sources.append(name)
            print(f"  [{symbol}] {name}源返回空行情")
        except DailyBarSchemaError as exc:
            schema_errors.append(exc)
            last_err = exc
            print(f"  [{symbol}] {name}源结构异常: {exc.failure_class}")
        except Exception as e:
            _breaker_record_failure(name, e)
            last_err = e
            print(f"  [{symbol}] {name}源失败: {type(e).__name__}")
    if len(empty_sources) >= 2:
        raise NoDailyBarsError(symbol, empty_sources) from last_err
    if schema_errors:
        raise schema_errors[-1] from last_err
    if last_err is None and skipped_sources:
        error = RuntimeError(f"{symbol} 所有数据源已熔断跳过: {', '.join(skipped_sources)}")
        error.failure_class = "all_sources_failed"
        raise error
    error = RuntimeError(f"{symbol} 所有数据源均失败: {last_err}")
    error.failure_class = "all_sources_failed"
    if isinstance(last_err, _DETERMINISTIC_EXCEPTIONS):
        error.deterministic = True
    raise error


def fetch_etf_daily(symbol: str, start="20200101", end="20991231") -> pd.DataFrame:
    """场内ETF: 东财 -> 新浪"""
    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        df = normalize_daily_bars(df, source=f"eastmoney ETF {symbol}")
        df.attrs["akshare_interface"] = "fund_etf_hist_em"
        df.attrs["akshare_source"] = "东财"
        df.attrs["adjustment"] = "qfq"
        print(f"  [ETF {symbol}] 使用数据源: 东财 ({len(df)} 行)")
        return df
    except Exception as exc:
        print(f"  [ETF {symbol}] 东财源失败: {type(exc).__name__}")
    prefixed = _market_prefix(symbol) + symbol
    df = ak.fund_etf_hist_sina(symbol=prefixed)
    df = normalize_daily_bars(df, source=f"sina ETF {symbol}")
    df.attrs["akshare_interface"] = "fund_etf_hist_sina"
    df.attrs["akshare_source"] = "新浪"
    df.attrs["adjustment"] = "qfq"
    start_date = pd.to_datetime(start, format="%Y%m%d")
    end_date = pd.to_datetime(end, format="%Y%m%d")
    df = df.loc[df["date"].between(start_date, end_date)].reset_index(drop=True)
    print(f"  [ETF {symbol}] 使用数据源: 新浪 ({len(df)} 行)")
    return df


def fetch_otc_fund_nav(symbol: str) -> pd.DataFrame:
    """场外基金单位净值"""
    df = ak.fund_open_fund_info_em(symbol=symbol, indicator="单位净值走势")
    df.attrs["akshare_interface"] = "fund_open_fund_info_em"
    df.attrs["akshare_source"] = "东财"
    df.attrs["adjustment"] = "unit_nav"
    print(f"  [场外 {symbol}] 东财净值 ({len(df)} 行)")
    return df
