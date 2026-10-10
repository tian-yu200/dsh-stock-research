"""Train Qlib on the isolated under-CNY50 pool and simulate a CNY6,000 account."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd
import yaml

from qlib.data import D

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.qlib_model import (  # noqa: E402
    build_dataset,
    init_qlib,
    limit_up_threshold,
    segment_prediction,
    train_lgbm,
)

DEFAULT_CONFIG = ROOT / "configs" / "qlib_under50_6000.yaml"
NAMES = {
    "SZ000651": "格力电器",
    "SH600887": "伊利股份",
    "SH600900": "长江电力",
    "SH600886": "国投电力",
    "SH600919": "江苏银行",
    "SH600938": "中国海油",
    "SZ002142": "宁波银行",
    "SZ300628": "亿联网络",
    "SZ000895": "双汇发展",
    "SZ002078": "太阳纸业",
    "SZ002352": "顺丰控股",
    "SH601225": "陕西煤业",
    "SH601919": "中远海控",
    "SH600690": "海尔智家",
    "SH600066": "宇通客车",
    "SH601598": "中国外运",
}

@dataclass(frozen=True)
class Allocation:
    quantities: dict[str, int]
    invested: float
    estimated_buy_cost: float
    reserve: float
    objective: float

def project_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else ROOT / path

def order_cost(value: float, rate: float, minimum: float) -> float:
    return 0.0 if value <= 0 else max(minimum, value * rate)

def choose_allocation(
    scores: pd.Series,
    prices: pd.Series,
    equity: float,
    config: dict,
    *,
    require_two: bool = True,
 ) -> Allocation:
    """Enumerate lot-constrained portfolios, favouring high ranks and balanced weights."""
    lot = int(config["lot_size"])
    max_positions = int(config["max_positions"])
    max_weight = float(config["max_position_weight"])
    reserve_ratio = float(config["reserve_cash_ratio"])
    depth = int(config["candidate_depth"])
    open_rate = float(config["open_cost"])
    minimum = float(config["min_cost"])
    usable = equity * (1.0 - reserve_ratio)
    ranked = scores.dropna().sort_values(ascending=False)
    ranked = ranked[ranked.index.isin(prices.dropna().index)].head(depth)
    if ranked.empty:
        return Allocation({}, 0.0, 0.0, equity, -math.inf)

    rank_utility = {symbol: (len(ranked) - rank) / len(ranked) for rank, symbol in enumerate(ranked.index)}
    options: list[tuple[str, list[int]]] = []
    for symbol in ranked.index:
        price = float(prices[symbol])
        max_lots = int(min(usable, equity * max_weight) // (price * lot))
        if max_lots > 0:
            options.append((symbol, list(range(1, max_lots + 1))))
    best: Allocation | None = None
    min_positions = 2 if require_two and len(options) >= 2 and max_positions >= 2 else 1
    for count in range(min_positions, max_positions + 1):
        for selected in itertools.combinations(options, count):
            symbols = [item[0] for item in selected]
            for lots in itertools.product(*(item[1] for item in selected)):
                quantities = {symbol: number * lot for symbol, number in zip(symbols, lots)}
                values = {symbol: quantities[symbol] * float(prices[symbol]) for symbol in symbols}
                invested = sum(values.values())
                costs = sum(order_cost(value, open_rate, minimum) for value in values.values())
                if invested + costs > usable or invested <= 0:
                    continue
                weights = np.array(list(values.values()), dtype=float) / equity
                rank_score = sum(rank_utility[symbol] * values[symbol] for symbol in symbols) / invested
                balance_penalty = float(np.std(weights))
                cash_penalty = max(0.0, 0.75 - invested / equity)
                objective = rank_score - 0.35 * balance_penalty - 0.75 * cash_penalty
                allocation = Allocation(quantities, invested, costs, equity - invested - costs, objective)
                if best is None or allocation.objective > best.objective:
                    best = allocation
    if best is None and require_two:
        return choose_allocation(scores, prices, equity, config, require_two=False)
    return best or Allocation({}, 0.0, 0.0, equity, -math.inf)

def prediction_metrics(prediction: pd.Series, labels: pd.Series) -> dict[str, float]:
    frame = pd.concat({"score": prediction, "label": labels}, axis=1).dropna()
    daily = frame.groupby(level="datetime").apply(
        lambda data: pd.Series({
            "ic": data["score"].corr(data["label"]),
            "rank_ic": data["score"].corr(data["label"], method="spearman"),
        }),
        include_groups=False,
    ).dropna(how="all")
    rank_std = daily["rank_ic"].std(ddof=1)
    return {
        "observations": int(len(frame)),
        "days": int(len(daily)),
        "mean_ic": float(daily["ic"].mean()),
        "mean_rank_ic": float(daily["rank_ic"].mean()),
        "rank_ic_ir": float(daily["rank_ic"].mean() / rank_std) if rank_std and not np.isnan(rank_std) else float("nan"),
        "rank_ic_positive_ratio": float((daily["rank_ic"] > 0).mean()),
    }

def forward_group_metrics(prediction: pd.Series, close: pd.DataFrame, horizon: int) -> dict[str, float]:
    future = close.shift(-horizon) / close - 1.0
    rows = []
    for date, values in prediction.groupby(level="datetime"):
        if date not in future.index:
            continue
        scores = values.droplevel("datetime").dropna()
        realised = future.loc[date].reindex(scores.index).dropna()
        scores = scores.reindex(realised.index)
        if len(scores) < 8:
            continue
        count = max(1, len(scores) // 4)
        ordered = scores.sort_values()
        bottom = realised.reindex(ordered.head(count).index).mean()
        top = realised.reindex(ordered.tail(count).index).mean()
        rows.append((date, top, bottom, top - bottom))
    data = pd.DataFrame(rows, columns=["date", "top", "bottom", "spread"]).set_index("date")
    if data.empty:
        return {"periods": 0, "top_mean": float("nan"), "bottom_mean": float("nan"), "spread_mean": float("nan"), "spread_positive_ratio": float("nan")}
    return {
        "periods": int(len(data)),
        "top_mean": float(data["top"].mean()),
        "bottom_mean": float(data["bottom"].mean()),
        "spread_mean": float(data["spread"].mean()),
        "spread_positive_ratio": float((data["spread"] > 0).mean()),
    }

def performance_metrics(equity: pd.Series) -> dict[str, float]:
    equity = equity.dropna()
    returns = equity.pct_change().dropna()
    if len(equity) < 2 or returns.empty:
        return {key: float("nan") for key in ("total_return", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown")}
    years = len(returns) / 252.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annual = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(returns.std(ddof=1) * math.sqrt(252))
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if returns.std(ddof=1) else float("nan")
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "annualized_return": annual,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }

def monthly_signal_dates(prediction: pd.Series) -> list[pd.Timestamp]:
    dates = pd.DatetimeIndex(prediction.index.get_level_values("datetime").unique()).sort_values()
    return list(pd.Series(dates, index=dates).groupby(dates.to_period("M")).last())

MAX_OPEN_CHASE = 0.05  # 开盘较信号收盘涨幅超过 5% 属追价，当次不买（报告"不追价"条款）

def simulate_account(
    prediction: pd.Series,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    config: dict,
 ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    initial_cash = float(config["initial_cash"])
    close_cost = float(config["close_cost"])
    open_cost = float(config["open_cost"])
    minimum = float(config["min_cost"])
    calendar = closes.index.sort_values()
    signals = monthly_signal_dates(prediction)
    execution_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for signal_date in signals:
        later = calendar[calendar > signal_date]
        if len(later):
            execution_map[later[0]] = signal_date
    cash = initial_cash
    holdings: dict[str, int] = {}
    records = []
    trades = []
    total_turnover = 0.0
    total_cost = 0.0
    for date in calendar:
        if date in execution_map:
            signal_date = execution_map[date]
            signal_scores = prediction.xs(signal_date, level="datetime")
            signal_prices = closes.loc[signal_date]
            equity_at_signal = cash + sum(qty * float(signal_prices.get(symbol, np.nan)) for symbol, qty in holdings.items())
            allocation = choose_allocation(signal_scores, signal_prices, equity_at_signal, config)
            targets = allocation.quantities
            for symbol, old_qty in list(holdings.items()):
                target_qty = targets.get(symbol, 0)
                sell_qty = max(0, old_qty - target_qty)
                price = float(opens.at[date, symbol]) if symbol in opens.columns else float("nan")
                if sell_qty and np.isfinite(price):
                    value = sell_qty * price
                    fee = order_cost(value, close_cost, minimum)
                    cash += value - fee
                    total_turnover += value
                    total_cost += fee
                    holdings[symbol] = old_qty - sell_qty
                    trades.append({"signal_date": signal_date, "trade_date": date, "symbol": symbol, "side": "sell", "quantity": sell_qty, "price": price, "value": value, "cost": fee})
            for symbol, target_qty in targets.items():
                old_qty = holdings.get(symbol, 0)
                buy_qty = max(0, target_qty - old_qty)
                price = float(opens.at[date, symbol]) if symbol in opens.columns else float("nan")
                if not buy_qty:
                    continue
                if not np.isfinite(price):
                    trades.append({"signal_date": signal_date, "trade_date": date, "symbol": symbol, "side": "skip_buy", "quantity": 0, "price": price, "value": 0.0, "cost": 0.0, "reason": "missing_open"})
                    continue
                reference = float(signal_prices.get(symbol, np.nan))
                if np.isfinite(reference) and reference > 0:
                    change = price / reference - 1.0
                    if change >= limit_up_threshold(symbol):
                        trades.append({"signal_date": signal_date, "trade_date": date, "symbol": symbol, "side": "skip_buy", "quantity": 0, "price": price, "value": 0.0, "cost": 0.0, "reason": "limit_up"})
                        continue
                    if change > MAX_OPEN_CHASE:
                        trades.append({"signal_date": signal_date, "trade_date": date, "symbol": symbol, "side": "skip_buy", "quantity": 0, "price": price, "value": 0.0, "cost": 0.0, "reason": "open_gap_over_5pct"})
                        continue
                while buy_qty > 0:
                    value = buy_qty * price
                    fee = order_cost(value, open_cost, minimum)
                    if value + fee <= cash:
                        break
                    buy_qty -= int(config["lot_size"])
                if buy_qty > 0:
                    value = buy_qty * price
                    fee = order_cost(value, open_cost, minimum)
                    cash -= value + fee
                    total_turnover += value
                    total_cost += fee
                    holdings[symbol] = old_qty + buy_qty
                    trades.append({"signal_date": signal_date, "trade_date": date, "symbol": symbol, "side": "buy", "quantity": buy_qty, "price": price, "value": value, "cost": fee})
            holdings = {symbol: qty for symbol, qty in holdings.items() if qty > 0}
        market_value = 0.0
        for symbol, qty in holdings.items():
            price = closes.at[date, symbol] if symbol in closes.columns else np.nan
            if pd.notna(price):
                market_value += qty * float(price)
        records.append({"date": date, "cash": cash, "market_value": market_value, "equity": cash + market_value, "positions": len(holdings)})
    report = pd.DataFrame(records).set_index("date")
    trade_frame = pd.DataFrame(trades)
    executed = (
        int((trade_frame["side"] != "skip_buy").sum())
        if "side" in trade_frame.columns
        else 0
    )
    metrics = performance_metrics(report["equity"])
    metrics.update({
        "total_cost": total_cost,
        "turnover_value": total_turnover,
        "turnover_ratio": total_turnover / report["equity"].mean() / max(1.0, len(report) / 252.0),
        "trade_orders": executed,
        "ending_equity": float(report["equity"].iloc[-1]),
    })
    return report, trade_frame, metrics

def current_risk_metrics(closes: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    history = closes.loc[:date]
    returns = history.pct_change(fill_method=None)
    return pd.DataFrame({
        "close": history.iloc[-1],
        "return_20d": history.iloc[-1] / history.shift(20).iloc[-1] - 1.0,
        "return_60d": history.iloc[-1] / history.shift(60).iloc[-1] - 1.0,
        "volatility_60d": returns.tail(60).std(ddof=1) * math.sqrt(252),
        "drawdown_252d": history.iloc[-1] / history.tail(252).max() - 1.0,
        "above_ma200": history.iloc[-1] > history.tail(200).mean(),
    })

def fmt_pct(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:.2%}"

def write_report(
    path: pathlib.Path,
    config: dict,
    valid_metrics: dict,
    test_metrics: dict,
    group_metrics: dict,
    strategy_metrics: dict,
    benchmark_metrics: dict,
    gate_checks: dict[str, bool],
    latest: pd.DataFrame,
    allocation: Allocation,
 ) -> None:
    passed = all(gate_checks.values())
    lines = [
        "# 本地 Qlib：50元以下候选与6000元持仓规划",
        "",
        "## 结论",
        "",
        f"- 旧方案已作废；本报告只采用独立候选池的本地 Qlib 结果。",
        f"- 量化门槛：**{'通过' if passed else '未通过'}**。未通过时不执行个股组合，资金保持现金或使用宽基ETF替代。",
        f"- 数据截止：2026-08-24；股票池16只，另含沪深300ETF作为比较基准。",
        "",
        "## 无泄漏设计",
        "",
        f"- 训练：{config['dataset']['segments']['train'][0]} 至 {config['dataset']['segments']['train'][1]}。",
        f"- 验证：{config['dataset']['segments']['valid'][0]} 至 {config['dataset']['segments']['valid'][1]}。",
        f"- 测试：{config['dataset']['segments']['test'][0]} 至 {config['dataset']['segments']['test'][1]}。",
        "- 标签为收盘后形成信号对应的未来20个交易日收益；训练、验证、测试之间预留超过20个交易日。",
        "- LightGBM仅用训练集拟合、验证集早停；测试集不参与调参。月末信号在下一交易日开盘执行。",
        "",
        "## 模型与样本外结果",
        "",
        f"- 验证 RankIC：{valid_metrics['mean_rank_ic']:.4f}；RankIC IR：{valid_metrics['rank_ic_ir']:.4f}。",
        f"- 测试 RankIC：{test_metrics['mean_rank_ic']:.4f}；RankIC IR：{test_metrics['rank_ic_ir']:.4f}；正值比例：{fmt_pct(test_metrics['rank_ic_positive_ratio'])}。",
        f"- 测试期Top四分位20日平均收益：{fmt_pct(group_metrics['top_mean'])}；Bottom四分位：{fmt_pct(group_metrics['bottom_mean'])}；多空差：{fmt_pct(group_metrics['spread_mean'])}。",
        f"- 6000元月度策略含成本年化：{fmt_pct(strategy_metrics['annualized_return'])}；最大回撤：{fmt_pct(strategy_metrics['max_drawdown'])}；Sharpe：{strategy_metrics['sharpe']:.3f}。",
        f"- 沪深300ETF同期年化：{fmt_pct(benchmark_metrics['annualized_return'])}；最大回撤：{fmt_pct(benchmark_metrics['max_drawdown'])}。",
        f"- 策略累计交易成本：{strategy_metrics['total_cost']:.2f}元；订单数：{strategy_metrics['trade_orders']}。",
        "",
        "## 门槛检查",
        "",
    ]
    for name, value in gate_checks.items():
        lines.append(f"- {'通过' if value else '未通过'}：{name}")
    lines.extend(["", "## 2026-08-24 当前信号", "", "| 排名 | 代码 | 名称 | Qlib分数 | 收盘价 | 20日收益 | 60日年化波动 | 252日回撤 |", "|---:|---|---|---:|---:|---:|---:|---:|"])
    for rank, (symbol, row) in enumerate(latest.iterrows(), 1):
        lines.append(f"| {rank} | {symbol[2:]} | {NAMES.get(symbol, symbol)} | {row['score']:.5f} | {row['close']:.2f} | {fmt_pct(row['return_20d'])} | {fmt_pct(row['volatility_60d'])} | {fmt_pct(row['drawdown_252d'])} |")
    lines.extend(["", "## 6000元整数手方案", ""])
    if passed and allocation.quantities:
        for symbol, quantity in allocation.quantities.items():
            price = float(latest.at[symbol, "close"])
            lines.append(f"- {NAMES.get(symbol, symbol)}（{symbol[2:]}）：{quantity}股，按{price:.2f}元估算{quantity * price:.2f}元。")
        lines.append(f"- 预计买入费用：{allocation.estimated_buy_cost:.2f}元；预计保留现金：{allocation.reserve:.2f}元。")
        lines.append("- 分两次执行：首个交易日先买排名最高的一只；10个交易日后仅在两只仍位于模型前半且组合未触发失效条件时补齐第二只。")
    else:
        lines.append("- 不执行个股买入。6000元保持现金；若必须保持市场敞口，仅考虑低费率宽基ETF，且不把它表述为Qlib个股推荐。")
    lines.extend([
        "",
        "## 成本、风险与失效条件",
        "",
        "- A股按100股一手；买入费率0.03%、卖出费率及印花税合计按0.13%，每笔最低5元；不使用融资。",
        "- 任一拟买股票开盘较信号收盘上涨超过5%、停牌或无法成交，当次不追价。",
        "- 月末复核；连续两个月跌出模型前半、公司出现重大财务/监管风险，或组合回撤达到15%，停止加仓并重新运行研究。",
        "- 16只候选是经价格和基本可交易性预筛后的有限样本，存在幸存者偏差；历史样本外结果不构成收益承诺。",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(config_path: pathlib.Path) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    result_dir = project_path(config["results"]["output_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    init_qlib(project_path(config["qlib_init"]["provider_uri"]), kernels=2)

    dataset_config = config["dataset"]
    horizon = int(dataset_config["label_horizon"])
    dataset = build_dataset(
        instruments=dataset_config["instruments"],
        start_time=dataset_config["start_time"],
        end_time=dataset_config["end_time"],
        fit_start_time=dataset_config["fit_start_time"],
        fit_end_time=dataset_config["fit_end_time"],
        label_horizon=horizon,
        segments=dataset_config["segments"],
        backtest_fields={"$open": "BACKTEST_OPEN", "$close": "BACKTEST_CLOSE"},
    )
    model = train_lgbm(dataset, config["model"], verbose_eval=25)

    predictions = {}
    metrics = {}
    for segment in ("valid", "test"):
        prediction, labels = segment_prediction(dataset, model, segment)
        predictions[segment] = prediction
        metrics[segment] = prediction_metrics(prediction, labels)
        prediction.to_csv(result_dir / f"predictions_{segment}.csv", encoding="utf-8-sig")

    stock_raw = dataset.handler.fetch(
        selector=slice(dataset_config["start_time"], dataset_config["end_time"]),
        level="datetime",
        col_set="backtest",
        data_key="raw",
    )
    opens = stock_raw["BACKTEST_OPEN"].unstack("instrument").sort_index()
    closes = stock_raw["BACKTEST_CLOSE"].unstack("instrument").sort_index()
    test_prediction = predictions["test"]
    group_metrics = forward_group_metrics(test_prediction, closes, horizon)
    test_start = pd.Timestamp(dataset_config["segments"]["test"][0])
    sim_closes = closes.loc[test_start:]
    sim_opens = opens.reindex(sim_closes.index)
    portfolio, trades, strategy_metrics = simulate_account(test_prediction, sim_opens, sim_closes, config["simulation"])
    portfolio.to_csv(result_dir / "account_daily.csv", encoding="utf-8-sig")
    trades.to_csv(result_dir / "trades.csv", index=False, encoding="utf-8-sig")

    benchmark_symbol = config["simulation"]["benchmark"]
    benchmark_raw = D.features(
        [benchmark_symbol],
        ["$close"],
        start_time=dataset_config["segments"]["test"][0],
        end_time=dataset_config["segments"]["test"][1],
    )
    benchmark = benchmark_raw["$close"].droplevel("instrument").sort_index().dropna()
    benchmark_equity = benchmark / benchmark.iloc[0] * float(config["simulation"]["initial_cash"])
    benchmark_metrics = performance_metrics(benchmark_equity)

    latest_date = test_prediction.index.get_level_values("datetime").max()
    latest_scores = test_prediction.xs(latest_date, level="datetime").sort_values(ascending=False)
    risks = current_risk_metrics(closes, latest_date)
    latest = risks.join(latest_scores.rename("score"), how="inner").sort_values("score", ascending=False)
    allocation = choose_allocation(latest["score"], latest["close"], float(config["simulation"]["initial_cash"]), config["simulation"])
    latest.to_csv(result_dir / "latest_signal.csv", encoding="utf-8-sig")

    gate = config["gate"]
    gate_checks = {
        f"测试RankIC >= {gate['min_test_rank_ic']}": metrics["test"]["mean_rank_ic"] >= float(gate["min_test_rank_ic"]),
        f"测试RankIC IR >= {gate['min_rank_ic_ir']}": metrics["test"]["rank_ic_ir"] >= float(gate["min_rank_ic_ir"]),
        f"含成本年化收益 >= {gate['min_net_annualized_return']:.0%}": strategy_metrics["annualized_return"] >= float(gate["min_net_annualized_return"]),
        f"最大回撤 >= {gate['max_drawdown_floor']:.0%}": strategy_metrics["max_drawdown"] >= float(gate["max_drawdown_floor"]),
    }
    gate_passed = all(gate_checks.values())
    if gate_passed:
        executable_allocation = pd.DataFrame([
            {
                "asset_type": "stock",
                "symbol": symbol,
                "name": NAMES.get(symbol, symbol),
                "quantity": quantity,
                "price": latest.at[symbol, "close"],
                "amount": quantity * latest.at[symbol, "close"],
                "status": "executable",
            }
            for symbol, quantity in allocation.quantities.items()
        ])
    else:
        executable_allocation = pd.DataFrame([
            {
                "asset_type": "cash",
                "symbol": "CNY",
                "name": "现金",
                "quantity": 1,
                "price": float(config["simulation"]["initial_cash"]),
                "amount": float(config["simulation"]["initial_cash"]),
                "status": "model_gate_failed_no_stock_trade",
            }
        ])
    executable_allocation.to_csv(
        result_dir / "current_allocation.csv", index=False, encoding="utf-8-sig"
    )

    output = {
        "data_end": str(latest_date.date()),
        "valid_model": metrics["valid"],
        "test_model": metrics["test"],
        "test_groups": group_metrics,
        "strategy": strategy_metrics,
        "benchmark": benchmark_metrics,
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "allocation": {
            "quantities": allocation.quantities if gate_passed else {},
            "invested": allocation.invested if gate_passed else 0.0,
            "buy_cost": allocation.estimated_buy_cost if gate_passed else 0.0,
            "reserve": allocation.reserve if gate_passed else float(config["simulation"]["initial_cash"]),
        },
        "shadow_allocation_not_executable": {
            "quantities": allocation.quantities,
            "invested": allocation.invested,
            "buy_cost": allocation.estimated_buy_cost,
            "reserve": allocation.reserve,
        },
        "training_best_iteration": int(model.model.best_iteration),
    }
    (result_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    with (result_dir / "run_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    report_path = project_path(config["results"]["report_path"])
    write_report(report_path, config, metrics["valid"], metrics["test"], group_metrics, strategy_metrics, benchmark_metrics, gate_checks, latest, allocation)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Report: {report_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())

if __name__ == "__main__":
    main()
