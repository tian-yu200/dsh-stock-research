"""Thin daily decision-dashboard renderer (read-only snapshots only).

Reads the last successful snapshot (``current.json``) and the rolling
preliminary pool (``rolling_scan.json``), renders a bounded human-readable
``daily.md`` plus a bounded ``daily.json`` aggregate, and optionally files a
DSH inbox notification.  It never touches the fetch DB, the Qlib provider,
the network, or ``current.json`` itself.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

try:  # `python -m assistant.report_daily`
    from .notifications import add_notification
    from .paths import (
        ASSISTANT_RESULTS_DIR,
        atomic_write_json,
        atomic_write_text,
        ensure_directories,
        read_json,
    )
except ImportError:  # direct `python assistant/report_daily.py` fallback
    from assistant.notifications import add_notification  # type: ignore
    from assistant.paths import (  # type: ignore
        ASSISTANT_RESULTS_DIR,
        atomic_write_json,
        atomic_write_text,
        ensure_directories,
        read_json,
    )

DAILY_MD_NAME = "daily.md"
DAILY_JSON_NAME = "daily.json"

POOL_FIELDS = (
    "rank",
    "asset_type",
    "symbol",
    "name",
    "trend_score",
    "return_20d",
    "return_60d",
    "volatility_60d_annualized",
    "max_drawdown",
    "current_drawdown",
    "average_amount_20d",
    "data_end_date",
)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "--"


def _fmt_num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _summarize_checks(checks: Any) -> tuple[int, int, list[str]]:
    if not isinstance(checks, dict):
        return 0, 0, []
    failed = sorted(str(k) for k, v in checks.items() if not v)
    total = len(checks)
    return total - len(failed), total, failed


def _pick_pool_top(rolling: dict[str, Any] | None, top_n: int) -> list[dict[str, Any]]:
    if not isinstance(rolling, dict):
        return []
    pool = rolling.get("pool")
    if not isinstance(pool, list):
        return []
    ordered = sorted(
        (r for r in pool if isinstance(r, dict)),
        key=lambda r: (r.get("rank") is None, r.get("rank") or 0),
    )
    slim: list[dict[str, Any]] = []
    for row in ordered[: max(0, top_n)]:
        slim.append({k: row.get(k) for k in POOL_FIELDS})
    return slim


def _render_daily_md(
    current: dict[str, Any] | None,
    rolling: dict[str, Any] | None,
    pool_top: list[dict[str, Any]],
    generated_at: str,
) -> str:
    lines: list[str] = []
    lines.append("# DSH 每日决策仪表盘")
    lines.append("")
    lines.append(f"生成时间：{generated_at}")
    lines.append("")
    if not isinstance(current, dict):
        lines.append("状态：尚无成功研究结果（current.json 缺失）。")
        lines.append("请先运行 daily/monthly 或定向 standard 研究，成功后本页自动更新。")
        lines.append("")
        lines.append("> 仅供研究，不构成投资建议。")
        return "\n".join(lines) + "\n"

    run_id = str(current.get("run_id", "--"))
    data_end = str(current.get("data_end_date", "--"))
    status = str(current.get("status", "--"))
    decision = current.get("decision_support", {}) if isinstance(current.get("decision_support"), dict) else {}
    gates = current.get("rule_gates", {}) if isinstance(current.get("rule_gates"), dict) else {}
    portfolio = gates.get("portfolio", {}) if isinstance(gates.get("portfolio"), dict) else decision.get("portfolio_gate", {})
    if not isinstance(portfolio, dict):
        portfolio = {}
    stock_gate = gates.get("stock", {}) if isinstance(gates.get("stock"), dict) else {}
    if not isinstance(stock_gate, dict):
        stock_gate = {}

    targets = decision.get("target_weights", {}) if isinstance(decision.get("target_weights"), dict) else {}
    cash = decision.get("cash_weight", portfolio.get("cash_weight", "--"))
    turnover = portfolio.get("turnover", "--")
    gate_passed = portfolio.get("passed", "--")
    p_ok, p_total, p_failed = _summarize_checks(portfolio.get("checks", {}))
    s_ok, s_total, s_failed = _summarize_checks(stock_gate.get("checks", {}))

    lines.append(f"研究 run_id：`{run_id}` ｜ 数据截止：`{data_end}` ｜ 状态：`{status}`")
    lines.append("")
    lines.append("## 组合结论（只读成功快照，不含下单指令）")
    lines.append("")
    lines.append(f"- 组合闸门：`{gate_passed}`（{p_ok}/{p_total} 通过{('，失败：' + '、'.join(p_failed)) if p_failed else ''}）")
    try:
        lines.append(f"- 现金比例：`{float(cash):.2f}` ｜ 换手：`{float(turnover):.2f}`")
    except (TypeError, ValueError):
        lines.append(f"- 现金比例：`{cash}` ｜ 换手：`{turnover}`")
    lines.append(f"- 假设资金：`{decision.get('hypothetical_capital', '--')}` ｜ 口径：{decision.get('allocation_basis', '--')}")
    lines.append(f"- 自动下单：`{decision.get('auto_order', False)}` ｜ 券商连接：`{decision.get('broker_connected', False)}`")
    lines.append("")
    if targets:
        lines.append("| 标的 | 目标权重 |")
        lines.append("|---|---|")
        for sym in sorted(targets):
            try:
                lines.append(f"| `{sym}` | `{float(targets[sym]):.2f}` |")
            except (TypeError, ValueError):
                lines.append(f"| `{sym}` | `{targets[sym]}` |")
        lines.append("")
    else:
        lines.append("目标权重：空（模型门槛未通过或风控拦截时保持空仓/候选，属预期行为）。")
        lines.append("")

    asset_paths = decision.get("asset_paths", {}) if isinstance(decision.get("asset_paths"), dict) else {}
    if asset_paths:
        lines.append("## 分资产动作")
        lines.append("")
        for asset_type in sorted(asset_paths):
            node = asset_paths[asset_type] if isinstance(asset_paths[asset_type], dict) else {}
            lines.append(
                f"- {asset_type}：{node.get('action', '--')}（{node.get('status', '--')}），"
                f"现金 {node.get('cash_weight', '--')}，理由：{node.get('reason', '--')}"
            )
        lines.append("")

    lines.append("## 股票模型门槛")
    lines.append("")
    lines.append(f"- 通过：`{stock_gate.get('passed', '--')}`（{s_ok}/{s_total} 通过{('，失败：' + '、'.join(s_failed)) if s_failed else ''}）")
    metrics = stock_gate.get("model_metrics", {}) if isinstance(stock_gate.get("model_metrics"), dict) else {}
    if metrics:
        lines.append(
            f"- 模型：{metrics.get('model_name', 'Qlib Alpha158 + LightGBM')} ｜ "
            f"缓存命中：`{metrics.get('cache_hit', '--')}` ｜ "
            f"RankIC：`{_fmt_num(metrics.get('mean_rank_ic'), 4)}` ｜ "
            f"RankIC_IR：`{_fmt_num(metrics.get('rank_ic_ir'), 4)}` ｜ "
            f"多空超额：`{_fmt_pct(metrics.get('top_bottom_excess_return'))}` ｜ "
            f"回撤：`{_fmt_pct(metrics.get('test_max_drawdown'))}`"
        )
    lines.append("")

    lines.append("## 抓取/初选池（preliminary，只读观察）")
    lines.append("")
    if isinstance(rolling, dict):
        cov = rolling.get("coverage", {}) if isinstance(rolling.get("coverage"), dict) else {}
        lines.append(
            f"- 覆盖：`{_fmt_pct(cov.get('coverage'))}`（{cov.get('success', '--')}/{cov.get('total', '--')}）｜ "
            f"本轮已扫：`{rolling.get('analyzed_this_pass', '--')}` ｜ "
            f"待扫：`{rolling.get('pending_remaining', '--')}` ｜ "
            f"合格：`{rolling.get('eligible_count', '--')}`"
        )
        if rolling.get("disclaimer"):
            lines.append(f"- 说明：{rolling.get('disclaimer')}")
        lines.append("")
        if pool_top:
            lines.append(f"### 初选 Top-{len(pool_top)}（按 rank 截断，非最终建议）")
            lines.append("")
            lines.append("| rank | 代码 | 名称 | 趋势分 | 20日涨跌 | 60日涨跌 | 当前回撤 | 20日均额 | 截止 |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
            for row in pool_top:
                lines.append(
                    f"| {row.get('rank', '--')} | `{row.get('symbol', '--')}` | "
                    f"{row.get('name', '--')} | {_fmt_num(row.get('trend_score'), 1)} | "
                    f"{_fmt_pct(row.get('return_20d'))} | {_fmt_pct(row.get('return_60d'))} | "
                    f"{_fmt_pct(row.get('current_drawdown'))} | {_fmt_num(row.get('average_amount_20d'), 0)} | "
                    f"{row.get('data_end_date', '--')} |"
                )
            lines.append("")
    else:
        lines.append("rolling_scan.json 缺失：全市场抓取未运行或仍在首轮，初选池暂为空。")
        lines.append("")

    lines.append("## 运行溯源")
    lines.append("")
    lines.append(f"- 运行目录：`{current.get('result_path', '--')}`")
    paths = current.get("paths", {}) if isinstance(current.get("paths"), dict) else {}
    for key in ("manifest", "summary", "model_signals", "rule_gates", "decision_support", "logs"):
        if paths.get(key):
            lines.append(f"- {key}：`{paths.get(key)}`")
    lines.append("")
    lines.append("> 仅供学习和研究使用，不构成任何投资建议。失败、超时或校验失败时保留上一份成功快照。")
    lines.append("")
    return "\n".join(lines) + "\n"


def render_daily(
    current: dict[str, Any] | None,
    rolling: dict[str, Any] | None,
    top_n: int,
    generated_at: str,
) -> tuple[str, dict[str, Any]]:
    """Pure render: no filesystem, no notification. Easy to unit test."""
    if not isinstance(current, dict):
        current = None
    if not isinstance(rolling, dict):
        rolling = None
    pool_top = _pick_pool_top(rolling, top_n)
    markdown = _render_daily_md(current, rolling, pool_top, generated_at)
    decision = current.get("decision_support", {}) if isinstance(current, dict) else {}
    if not isinstance(decision, dict):
        decision = {}
    targets = decision.get("target_weights", {}) if isinstance(decision, dict) else {}
    if not isinstance(targets, dict):
        targets = {}
    cov = rolling.get("coverage", {}) if isinstance(rolling, dict) else {}
    if not isinstance(cov, dict):
        cov = {}
    payload: dict[str, Any] = {
        "generated_at": generated_at,
        "run_id": current.get("run_id") if isinstance(current, dict) else None,
        "status": current.get("status") if isinstance(current, dict) else "no_successful_run",
        "data_end_date": current.get("data_end_date") if isinstance(current, dict) else None,
        "target_count": len(targets),
        "target_weights": dict(targets),
        "cash_weight": decision.get("cash_weight") if isinstance(decision, dict) else None,
        "coverage": cov.get("coverage"),
        "pending_remaining": rolling.get("pending_remaining") if isinstance(rolling, dict) else None,
        "pool_top": pool_top,
    }
    return markdown, payload


def build_daily(targets_top_n: int = 10) -> dict[str, Any]:
    ensure_directories()
    generated_at = datetime.now(timezone.utc).isoformat()
    current = read_json(ASSISTANT_RESULTS_DIR / "current.json", None)
    rolling = read_json(ASSISTANT_RESULTS_DIR / "rolling_scan.json", None)
    markdown, payload = render_daily(current, rolling, targets_top_n, generated_at)
    atomic_write_text(ASSISTANT_RESULTS_DIR / DAILY_MD_NAME, markdown)
    payload = dict(payload)
    payload["daily_md"] = str(ASSISTANT_RESULTS_DIR / DAILY_MD_NAME)
    payload["daily_json"] = str(ASSISTANT_RESULTS_DIR / DAILY_JSON_NAME)
    atomic_write_json(ASSISTANT_RESULTS_DIR / DAILY_JSON_NAME, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render read-only daily dashboard from snapshots.")
    parser.add_argument("--top-n", type=int, default=10, help="Rolling pool rows to include (1-30).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--notify", dest="notify", action="store_true", default=True)
    group.add_argument("--no-notify", dest="notify", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    top_n = int(args.top_n)
    if top_n < 1 or top_n > 30:
        print(json.dumps({"ok": False, "error": "top-n must be between 1 and 30"}, ensure_ascii=True))
        return 2
    payload = build_daily(top_n)
    notified: dict[str, Any] | None = None
    if args.notify:
        title_bits = [str(payload.get("data_end_date") or "na"), f"{payload.get('target_count', 0)} targets"]
        notified = add_notification(
            "daily_report",
            "daily dashboard " + " / ".join(title_bits),
            "run=%s targets=%s coverage=%s pending=%s" % (
                payload.get("run_id"), payload.get("target_count"),
                payload.get("coverage"), payload.get("pending_remaining"),
            ),
            run_id=payload.get("run_id"),
            paths={"daily_md": payload["daily_md"], "daily_json": payload["daily_json"]},
        )
    summary = {
        "ok": True,
        "run_id": payload.get("run_id"),
        "data_end_date": payload.get("data_end_date"),
        "target_count": payload.get("target_count"),
        "pool_top": len(payload.get("pool_top", [])),
        "coverage": payload.get("coverage"),
        "notified": bool(notified),
        "daily_md": payload.get("daily_md"),
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
