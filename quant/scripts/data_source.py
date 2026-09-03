# -*- coding: utf-8 -*-
"""
统一数据源模块：多源容错抓取 A 股/基金数据
东财(push2his)有反爬，自动降级 腾讯 -> 新浪
输出统一列: date, open, high, low, close, volume, amount, turnover
"""
import os

os.environ.setdefault("NO_PROXY", "*")  # 绕过系统代理直连国内数据源

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


def _market_prefix(symbol: str) -> str:
    """Return the exchange prefix expected by quote-history adapters."""
    symbol = str(symbol).strip()
    if symbol.startswith(("4", "8")):
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


_SOURCES = [("东财", _em), ("腾讯", _tx), ("新浪", _sina)]


def fetch_a_stock_daily(symbol: str, start="20200101", end="20991231") -> pd.DataFrame:
    """Try independent sources and classify a confirmed no-bar window."""
    last_err = None
    empty_sources: list[str] = []
    schema_errors: list[DailyBarSchemaError] = []
    for name, fn in _SOURCES:
        try:
            df = fn(symbol, start, end)
            if df is not None and len(df) > 0:
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
            last_err = e
            print(f"  [{symbol}] {name}源失败: {type(e).__name__}")
    if len(empty_sources) >= 2:
        raise NoDailyBarsError(symbol, empty_sources) from last_err
    if schema_errors:
        raise schema_errors[-1] from last_err
    raise RuntimeError(f"{symbol} 所有数据源均失败: {last_err}")


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
