"""Run one fixed-path stock assistant research workflow.

This is the only command-line entry point used by the DSH launcher and the
Windows scheduled task.  It deliberately accepts no paths, commands, or
output-directory overrides.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.paths import read_json, run_directory  # noqa: E402
from assistant.research import run_research  # noqa: E402


CODE_PATTERN = re.compile(r"^[0-9]{6}$")
MODES = ("daily", "monthly", "on_demand")
ASSET_SCOPES = ("all", "stock", "etf", "otc_fund")
DEPTHS = ("quick", "standard", "full")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-path DSH stock/ETF/OTC fund research workflow."
    )
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--asset-scope", choices=ASSET_SCOPES)
    parser.add_argument(
        "--depth",
        choices=DEPTHS,
        default=None,
        help=(
            "Research depth: quick reads cached data only; standard refreshes "
            "targeted symbols; full refreshes the whole selected universe. "
            "Defaults to standard for daily/monthly and the configured "
            "default_depth for on_demand."
        ),
    )
    parser.add_argument(
        "--queued-run-id",
        default=None,
        help="Run the fixed request stored in one validated queued research manifest.",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional list of at most 100 six-digit market codes.",
    )
    return parser


def _validate_symbols(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if not values:
        raise ValueError("symbols must be omitted or contain at least one code")
    if len(values) > 100:
        raise ValueError("at most 100 symbols are allowed")
    if any(CODE_PATTERN.fullmatch(value) is None for value in values):
        raise ValueError("symbols must be six-digit market codes")
    if len(set(values)) != len(values):
        raise ValueError("symbols must be unique")
    return values


def _queued_request(
    run_id: str,
) -> tuple[str, str, dict[str, list[str]] | None, str]:
    manifest = read_json(run_directory(run_id) / "manifest.json", {})
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise ValueError("queued research manifest does not match run_id")
    mode = str(manifest.get("mode", ""))
    asset_scope = str(manifest.get("asset_scope", ""))
    if mode not in MODES or asset_scope not in ASSET_SCOPES:
        raise ValueError("queued research manifest has an invalid mode or asset scope")
    depth = str(manifest.get("depth") or "standard")
    if depth not in DEPTHS:
        raise ValueError("queued research manifest has an invalid depth")
    if manifest.get("full_universe", True):
        return mode, asset_scope, None, depth
    raw_routes = manifest.get("symbols", {})
    if not isinstance(raw_routes, dict):
        raise ValueError("queued targeted research manifest has invalid symbol routes")
    routes = {
        str(asset_type): _validate_symbols([str(value) for value in values]) or []
        for asset_type, values in raw_routes.items()
        if isinstance(values, list) and values
    }
    if not routes:
        raise ValueError("queued targeted research manifest has no symbols")
    return mode, asset_scope, routes, depth


def _run_fixed_research(
    *,
    mode: str,
    asset_scope: str,
    symbols: list[str] | None = None,
    symbols_by_type: dict[str, list[str]] | None = None,
    depth: str | None = None,
    force_sync: bool = False,
    run_id_override: str | None = None,
    run_dir_override: Path | None = None,
 ) -> dict[str, Any]:
    captured = io.StringIO()
    previous_skip = os.environ.get("DSH_STOCK_SKIP_FETCH")
    if force_sync:
        os.environ["DSH_STOCK_SKIP_FETCH"] = "1"
    try:
        with contextlib.redirect_stdout(captured):
            return run_research(
                mode=mode,
                asset_scope=asset_scope,
                symbols=symbols,
                depth=depth,
                _force_sync=force_sync,
                _symbols_by_type=symbols_by_type,
                _run_id_override=run_id_override,
                _run_dir_override=run_dir_override,
            )
    finally:
        noisy_output = captured.getvalue()
        if noisy_output:
            print(noisy_output, file=sys.stderr, end="")
        if force_sync:
            if previous_skip is None:
                os.environ.pop("DSH_STOCK_SKIP_FETCH", None)
            else:
                os.environ["DSH_STOCK_SKIP_FETCH"] = previous_skip


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认 GBK；结果里的中文与子进程输出残留的 U+FFFD
    # 可能无法编码。降级替换而不是让最后一行 print 崩掉。
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    args = _parser().parse_args(argv)
    try:
        if args.queued_run_id is not None:
            if (
                args.mode is not None
                or args.asset_scope is not None
                or args.symbols is not None
                or args.depth is not None
            ):
                raise ValueError("queued-run-id cannot be combined with request arguments")
            mode, asset_scope, symbols_by_type, depth = _queued_request(args.queued_run_id)
            result = _run_fixed_research(
                mode=mode,
                asset_scope=asset_scope,
                symbols_by_type=symbols_by_type,
                depth=depth,
                force_sync=True,
                run_id_override=args.queued_run_id,
                run_dir_override=run_directory(args.queued_run_id),
            )
        else:
            if args.mode is None or args.asset_scope is None:
                raise ValueError("mode and asset-scope are required")
            result = _run_fixed_research(
                mode=args.mode,
                asset_scope=args.asset_scope,
                symbols=_validate_symbols(args.symbols),
                depth=args.depth,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default))
        return 0 if result.get("status") in {"success", "queued", "partial"} else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
