"""技术面信号层（自 research.py 逐字搬迁，docs/重构计划.md 阶段 1）。

跨模块名字来源约定：_number、_iso_date 等定义在 research.py 的工具经
``research.`` 属性访问，使 patch.object(research, ...) 对搬入的函数继续生效；
其余名字为模块内自有。
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from . import research

def _dated_close_series(frame: pd.DataFrame) -> pd.Series:
    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["close"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates)
    return series.dropna().sort_index()


def _drawdown_recovery(close: pd.Series) -> dict[str, Any]:
    """Describe the largest drawdown and the current recovery state.

    Recovery durations use calendar days between available observation dates.
    This keeps the value meaningful for both daily bars and less frequent fund
    NAV observations.
    """
    series = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    if series.empty:
        return {
            "max_drawdown": None,
            "max_drawdown_peak_date": None,
            "max_drawdown_trough_date": None,
            "max_drawdown_duration_days": None,
            "max_drawdown_recovery_date": None,
            "max_drawdown_recovery_days": None,
            "current_drawdown": None,
            "current_drawdown_peak_date": None,
            "current_drawdown_duration_days": None,
            "current_drawdown_recovered": False,
        }

    running_peak = series.cummax()
    drawdown = series / running_peak - 1.0
    trough_date = drawdown.idxmin()
    trough_value = float(drawdown.loc[trough_date])
    peak_value = float(running_peak.loc[trough_date])
    peak_candidates = series.loc[:trough_date]
    peak_date = peak_candidates[peak_candidates == peak_value].index[-1]

    recovery_candidates = series.loc[trough_date:]
    recovery_candidates = recovery_candidates[recovery_candidates >= peak_value * (1.0 - 1e-12)]
    recovery_date = recovery_candidates.index[0] if not recovery_candidates.empty else None

    latest_date = series.index[-1]
    latest_peak_value = float(running_peak.iloc[-1])
    current_peak_candidates = series.loc[:latest_date]
    current_peak_date = current_peak_candidates[
        current_peak_candidates == latest_peak_value
    ].index[-1]
    current_recovered = bool(float(series.iloc[-1]) >= latest_peak_value * (1.0 - 1e-12))
    current_duration = (
        0
        if current_recovered
        else max(0, int((latest_date - current_peak_date).days))
    )

    return {
        "max_drawdown": research._number(trough_value),
        "max_drawdown_peak_date": research._iso_date(peak_date),
        "max_drawdown_trough_date": research._iso_date(trough_date),
        "max_drawdown_duration_days": max(
            0,
            int((trough_date - peak_date).days),
        ),
        "max_drawdown_recovery_date": research._iso_date(recovery_date),
        "max_drawdown_recovery_days": (
            max(0, int((recovery_date - trough_date).days))
            if recovery_date is not None
            else None
        ),
        "current_drawdown": research._number(float(series.iloc[-1]) / latest_peak_value - 1.0),
        "current_drawdown_peak_date": research._iso_date(current_peak_date),
        "current_drawdown_duration_days": current_duration,
        "current_drawdown_recovered": current_recovered,
    }


def _relative_returns(
    asset: pd.Series,
    benchmark: pd.Series | None,
    periods: tuple[int, ...] = (20, 60, 252),
) -> dict[str, float | None]:
    """Return multi-period relative returns on a shared observation calendar."""
    if benchmark is None:
        return {f"relative_return_{period}d_vs_benchmark": None for period in periods}
    asset_series = pd.to_numeric(asset, errors="coerce").dropna()
    benchmark_series = pd.to_numeric(benchmark, errors="coerce").dropna()
    asset_series.index = pd.to_datetime(asset_series.index)
    benchmark_series.index = pd.to_datetime(benchmark_series.index)
    common = asset_series.index.intersection(benchmark_series.index)
    common = common.sort_values()
    if common.empty:
        return {f"relative_return_{period}d_vs_benchmark": None for period in periods}
    asset_series = asset_series.loc[common]
    benchmark_series = benchmark_series.loc[common]
    result: dict[str, float | None] = {}
    for period in periods:
        key = f"relative_return_{period}d_vs_benchmark"
        result[key] = (
            research._number(
                (asset_series.iloc[-1] / asset_series.iloc[-period - 1])
                / (benchmark_series.iloc[-1] / benchmark_series.iloc[-period - 1])
                - 1.0
            )
            if len(common) > period
            else None
        )
    return result


def _liquidity_metrics(frame: pd.DataFrame, window: int = 20) -> dict[str, Any]:
    """Summarize ETF liquidity while exposing unavailable fields explicitly."""
    metrics: dict[str, Any] = {"liquidity_window_days": min(len(frame), window)}
    for column, key in (("amount", "amount"), ("volume", "volume")):
        values = (
            pd.to_numeric(frame[column], errors="coerce").tail(window).dropna()
            if column in frame
            else pd.Series(dtype=float)
        )
        metrics[f"average_{key}_{window}d"] = research._number(values.mean()) if not values.empty else None
        metrics[f"{key}_data_status"] = "available" if not values.empty else "unavailable"
    metrics["liquidity_status"] = (
        "available"
        if metrics["amount_data_status"] == "available"
        or metrics["volume_data_status"] == "available"
        else "unavailable"
    )
    return metrics


def _market_metrics(frame: pd.DataFrame, *, fund: bool = False) -> dict[str, Any]:
    close = _dated_close_series(frame)
    if close.empty:
        raise ValueError("no close/NAV values")
    returns = close.pct_change()
    latest = float(close.iloc[-1])

    def ret(period: int) -> float | None:
        return (
            research._number(close.iloc[-1] / close.iloc[-period - 1] - 1.0)
            if len(close) > period
            else None
        )

    volatility = research._number(returns.tail(60).std(ddof=1) * math.sqrt(252))
    trailing_drawdown = _drawdown_recovery(close.tail(252))["max_drawdown"]
    drawdown = _drawdown_recovery(close)
    ma20 = research._number(close.tail(20).mean()) if len(close) >= 20 else None
    ma60 = research._number(close.tail(60).mean()) if len(close) >= 60 else None
    ma200 = research._number(close.tail(200).mean()) if len(close) >= 200 else None
    trend_score = (
        (ret(20) or 0.0) * 0.45
        + (ret(60) or 0.0) * 0.35
        + (ret(252) or 0.0) * 0.20
    )
    if ma20 and latest > ma20:
        trend_score += 0.05
    if ma60 and latest > ma60:
        trend_score += 0.05
    return {
        "latest_value": latest,
        "return_20d": ret(20),
        "return_60d": ret(60),
        "return_252d": ret(252),
        "volatility_60d_annualized": volatility,
        "max_drawdown": drawdown["max_drawdown"],
        "max_drawdown_252d": trailing_drawdown,
        "max_drawdown_peak_date": drawdown["max_drawdown_peak_date"],
        "max_drawdown_trough_date": drawdown["max_drawdown_trough_date"],
        "max_drawdown_duration_days": drawdown["max_drawdown_duration_days"],
        "max_drawdown_recovery_date": drawdown["max_drawdown_recovery_date"],
        "max_drawdown_recovery_days": drawdown["max_drawdown_recovery_days"],
        "current_drawdown": drawdown["current_drawdown"],
        "current_drawdown_peak_date": drawdown["current_drawdown_peak_date"],
        "current_drawdown_duration_days": drawdown["current_drawdown_duration_days"],
        "current_drawdown_recovered": drawdown["current_drawdown_recovered"],
        "ma20": ma20,
        "ma60": ma60,
        "ma200": ma200,
        "trend_score": research._number(trend_score),
        "series_type": "net_value" if fund else "daily_close",
    }


def _peer_comparison(
    symbol: str,
    metrics_by_symbol: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    period = next(
        (
            candidate
            for candidate in (252, 60, 20)
            if sum(
                values.get(f"return_{candidate}d") is not None
                for values in metrics_by_symbol.values()
            )
            >= 2
        ),
        None,
    )
    if period is None:
        return {
            "peer_comparison_status": "unavailable",
            "peer_comparison_reason": "at least two same-category assets with comparable returns are required",
            "peer_count": 0,
        }
    peer_returns = pd.Series(
        {
            peer_symbol: values.get(f"return_{period}d")
            for peer_symbol, values in metrics_by_symbol.items()
            if values.get(f"return_{period}d") is not None
        },
        dtype=float,
    ).dropna()
    if symbol not in peer_returns.index or len(peer_returns) < 2:
        return {
            "peer_comparison_status": "unavailable",
            "peer_comparison_reason": "the asset has no comparable return in the peer period",
            "peer_count": max(0, int(len(peer_returns) - (symbol in peer_returns.index))),
        }
    value = float(peer_returns.loc[symbol])
    peers = peer_returns.drop(index=symbol)
    return {
        "peer_comparison_status": "available",
        "peer_comparison_period_days": period,
        "peer_count": int(len(peers)),
        f"peer_average_return_{period}d": research._number(peers.mean()),
        f"excess_return_{period}d_vs_peer_mean": research._number(value - peers.mean()),
        f"peer_return_{period}d_percentile": research._number(float((peer_returns <= value).mean())),
        f"peer_rank_{period}d": int(peer_returns.rank(ascending=False, method="min").loc[symbol]),
        f"peer_universe_return_{period}d": research._number(peer_returns.mean()),
    }


def _technical_asset_analysis(
    asset_type: str,
    frames: Mapping[str, pd.DataFrame],
    facts: list[dict[str, Any]],
    benchmark: pd.Series | None = None,
    peer_frames: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    peer_frames = peer_frames or frames
    metric_frames = {
        symbol: _market_metrics(frame, fund=asset_type == "otc_fund")
        for symbol, frame in peer_frames.items()
    }
    fact_by_symbol = {
        str(item.get("symbol")): item
        for item in facts
        if item.get("symbol") is not None
    }
    signals: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        metrics = dict(metric_frames[symbol])
        close = _dated_close_series(frame)
        relative = _relative_returns(close, benchmark)
        metrics.update(relative)
        metrics["benchmark_symbol"] = "510300" if benchmark is not None else None
        metrics["benchmark_data_status"] = (
            "available"
            if benchmark is not None and any(value is not None for value in relative.values())
            else "unavailable"
        )
        metrics.update(_peer_comparison(symbol, metric_frames))
        relative_20d = relative["relative_return_20d_vs_benchmark"]
        if relative_20d is not None:
            metrics["trend_score"] = research._number(
                (metrics["trend_score"] or 0.0) + relative_20d * 0.25
            )
        peer_period = metrics.get("peer_comparison_period_days")
        peer_excess = (
            metrics.get(f"excess_return_{int(peer_period)}d_vs_peer_mean")
            if peer_period is not None
            else None
        )
        if peer_excess is not None:
            metrics["trend_score"] = research._number(
                (metrics["trend_score"] or 0.0) + peer_excess * 0.20
            )
        if asset_type in {"stock", "etf"}:
            metrics.update(_liquidity_metrics(frame))
        signal = {
            "symbol": symbol,
            "name": fact_by_symbol.get(symbol, {}).get("name"),
            **metrics,
        }
        signals.append(signal)
    signals.sort(key=lambda item: item.get("trend_score") or -math.inf, reverse=True)
    return {
        "asset_type": asset_type,
        "method": "trend_return_volatility_drawdown_relative_benchmark_peer_comparison",
        "signals": signals,
        "data_end_date": max(
            (item["data_end_date"] for item in facts if item.get("data_end_date")),
            default=None,
        ),
    }
