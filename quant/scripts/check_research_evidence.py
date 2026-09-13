"""复现《量化系统评估报告-20260911.md》的全部运行台账证据（只读）。

用法（在任意目录）：
    D:/dsh-gupiao/quant/.venv/Scripts/python.exe quant/scripts/check_research_evidence.py

本脚本只读文件，不写入、不删除、不联网、不启动任何研究进程。
输出量有界（每个运行一行汇总），便于快速核对报告 §2/§3 的断言。
"""
from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EVIDENCE_FILES = [
    "quant/docs/研究管线不变量.md",
    "quant/results/assistant/current.json",
    "quant/results/assistant/rolling_scan.json",
    "quant/logs/notifications.json",
    "quant/data/fetch_state.sqlite3",
    "quant/qlib_data/akshare_data/manifest.json",
    "quant/qlib_data/akshare_data/calendars/day.txt",
]

RUNS = REPO / "quant/results/assistant/runs"


def _fmt(value, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return "None" if value is None else str(value)[:26]


def check_files() -> int:
    print("=" * 78)
    print("[1] 关键证据文件是否存在（相对仓库根：%s）" % REPO)
    print("=" * 78)
    missing = 0
    for rel in EVIDENCE_FILES:
        path = REPO / rel
        if path.is_file():
            stat = path.stat()
            stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"  存在  {stat.st_size:>10,} B  {stamp}  {rel}")
        else:
            print(f"  缺失  {'':>10}  {'':<16}  {rel}")
            missing += 1
    print(f"\n  缺失 {missing} / {len(EVIDENCE_FILES)}")
    if missing:
        print("  提示：quant/results、quant/logs、quant/data、quant/qlib_data 均在 .gitignore 内。")
        print("        在 git worktree 或任何按 .gitignore 过滤的视图中，这些目录必然为空。")
    return missing


def check_runs() -> None:
    print()
    print("=" * 78)
    print("[2] 2026-09 全部运行的台账（manifest + rule_gates + model_metadata）")
    print("=" * 78)
    month = RUNS / "2026/09"
    if not month.is_dir():
        print(f"  目录不存在：{month}")
        print("  → 如果你看到的正是这个结果，说明当前所在的工作树不是 D:/dsh-gupiao。")
        return
    header = ("run", "mode", "depth", "status", "IC", "IR", "maxDD", "netRet", "turn", "scores", "False检查项")
    print("  %-15s %-9s %-8s %-8s %7s %6s %8s %8s %6s %7s %s" % header)
    successful = []
    for day in sorted(p for p in month.iterdir() if p.is_dir()):
        for run in sorted(p for p in day.iterdir() if p.is_dir()):
            manifest = run / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                man = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                continue
            gates, model = {}, {}
            try:
                gates = json.loads((run / "rule_gates.json").read_text(encoding="utf-8"))
                model = (gates.get("stock") or {}).get("model_metrics") or {}
            except Exception:
                pass
            scores, failed = "?", "?"
            try:
                meta = json.loads((run / "model_metadata.json").read_text(encoding="utf-8"))
                scores = len((meta.get("stock_model") or {}).get("scores") or {})
            except Exception:
                pass
            try:
                failed = [k for k, v in (gates["stock"]["checks"]).items() if v is False]
            except Exception:
                pass
            status = man.get("status")
            # 2026-09-13 P0-3：status 是三态 success/partial/failed，
            # partial 同样是"已发布"（就绪类别已落盘、被拦类别归零标注）。
            if status in {"success", "partial"}:
                successful.append((man.get("finished_at"), run.name))
            print("  %-15s %-9s %-8s %-8s %7s %6s %8s %8s %6s %7s %s" % (
                run.name[:15], str(man.get("mode"))[:9], str(man.get("depth"))[:8], str(status)[:8],
                _fmt(model.get("mean_rank_ic")), _fmt(model.get("rank_ic_ir"), 3),
                _fmt(model.get("test_max_drawdown"), 3), _fmt(model.get("net_annualized_return"), 3),
                _fmt(model.get("mean_daily_turnover"), 3), scores, failed,
            ))
    print()
    ok = sorted(x for x in successful if x[0])
    if ok:
        print("  最后一次已发布的运行（status=success/partial）：")
        print("    finished_at = %s" % ok[-1][0])
        print("    run_id      = %s" % ok[-1][1])
    else:
        print("  未发现任何 status=success/partial 的运行。")


def check_scores_detail() -> None:
    print()
    print("=" * 78)
    print("[3] 最新运行的 scores 明细（报告 P0-1 的核心证据）")
    print("=" * 78)
    month = RUNS / "2026/09"
    if not month.is_dir():
        return
    latest = None
    for day in sorted(p for p in month.iterdir() if p.is_dir()):
        for run in sorted(p for p in day.iterdir() if p.is_dir()):
            if (run / "model_metadata.json").is_file():
                latest = run
    if latest is None:
        print("  未找到 model_metadata.json")
        return
    path = latest / "model_metadata.json"
    raw = path.read_bytes()
    stock = json.loads(raw.decode("utf-8")).get("stock_model") or {}
    scores = stock.get("scores") or {}
    print(f"  运行目录     : {latest}")
    print(f"  sha256       : {hashlib.sha256(raw).hexdigest()}")
    print(f"  scores 数量  : {len(scores)}")
    print(f"  评分明细     : {sorted(scores.items())}")
    print(f"  cache_write_skipped : {stock.get('cache_write_skipped')}  {stock.get('cache_skip_reason')}")
    print(f"  test_max_drawdown   : {_fmt(stock.get('test_max_drawdown'))}")
    print(f"  benchmark.max_drawdown : {_fmt((stock.get('benchmark') or {}).get('max_drawdown'))}")
    bj = [s for s in scores if str(s).startswith(("92", "43", "83", "87"))]
    print(f"  其中北交所代码        : {len(bj)} / {len(scores)}（pool_exclude_bj=true 会在取权重前剔除）")


def check_gate_config() -> None:
    print()
    print("=" * 78)
    print("[4] 门槛值：生产配置 vs DEFAULTS（报告 P1-6）")
    print("=" * 78)
    prod = REPO / "quant/configs/assistant.yaml"
    defaults = REPO / "quant/assistant/config.py"
    for label, path, needle in (
        ("生产 assistant.yaml", prod, "max_excess_test_drawdown"),
        ("DEFAULTS config.py", defaults, "max_excess_test_drawdown"),
    ):
        if not path.is_file():
            print(f"  {label}: 文件缺失")
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if needle in line:
                print(f"  {label:<22}: {line.strip()}")


def main() -> int:
    print()
    print("复核《量化系统评估报告-20260911.md》运行台账证据（只读）")
    print(f"仓库根：{REPO}")
    print()
    check_files()
    check_runs()
    check_scores_detail()
    check_gate_config()
    print()
    print("=" * 78)
    print("以上输出即可复核报告 §2 / §3 / §5 的全部运行层断言。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
