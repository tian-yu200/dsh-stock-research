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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed-path DSH stock/ETF/OTC fund research workflow."
    )
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--asset-scope", choices=ASSET_SCOPES)
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


def _queued_request(run_id: str) -> tuple[str, str, dict[str, list[str]] | None]:
    manifest = read_json(run_directory(run_id) / "manifest.json", {})
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise ValueError("queued research manifest does not match run_id")
    mode = str(manifest.get("mode", ""))
    asset_scope = str(manifest.get("asset_scope", ""))
    if mode not in MODES or asset_scope not in ASSET_SCOPES:
        raise ValueError("queued research manifest has an invalid mode or asset scope")
    if manifest.get("full_universe", True):
        return mode, asset_scope, None
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
    return mode, asset_scope, routes


def _run_fixed_research(
    *,
    mode: str,
    asset_scope: str,
    symbols: list[str] | None = None,
    symbols_by_type: dict[str, list[str]] | None = None,
    force_sync: bool = False,
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
                _force_sync=force_sync,
                _symbols_by_type=symbols_by_type,
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
    args = _parser().parse_args(argv)
    try:
        if args.queued_run_id is not None:
            if args.mode is not None or args.asset_scope is not None or args.symbols is not None:
                raise ValueError("queued-run-id cannot be combined with request arguments")
            mode, asset_scope, symbols_by_type = _queued_request(args.queued_run_id)
            result = _run_fixed_research(
                mode=mode,
                asset_scope=asset_scope,
                symbols_by_type=symbols_by_type,
                force_sync=True,
            )
        else:
            if args.mode is None or args.asset_scope is None:
                raise ValueError("mode and asset-scope are required")
            result = _run_fixed_research(
                mode=args.mode,
                asset_scope=args.asset_scope,
                symbols=_validate_symbols(args.symbols),
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
