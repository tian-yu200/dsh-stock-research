"""Fixed-tool stdio MCP server for the private stock research assistant."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Callable

import anyio
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant import (  # noqa: E402
    ack_notification,
    get_current_snapshot,
    get_rolling_scan,
    get_run_status,
    list_notifications,
    run_research,
)

LOGGER = logging.getLogger("stock_assistant_mcp")
SYMBOL_PATTERN = r"^(?:SH|SZ)?[0-9]{6}$"
RUN_ID_PATTERN = r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
NOTIFICATION_ID_PATTERN = r"^n-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"

TOOLS = [
    types.Tool(
        name="run_research",
        description=(
            "Submit a fixed-path stock, ETF, or OTC fund research workflow. "
            "When data refresh is enabled, both targeted and full-universe requests "
            "return a run_id promptly and continue in a bounded background worker. "
            "Targeted requests only check and refresh the supplied symbols. "
            "This never connects to a broker or places orders."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["daily", "monthly", "on_demand"]},
                "asset_scope": {
                    "type": "string",
                    "enum": ["all", "stock", "etf", "otc_fund"],
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string", "pattern": SYMBOL_PATTERN},
                    "minItems": 1,
                    "maxItems": 100,
                    "uniqueItems": True,
                },
            },
            "required": ["mode", "asset_scope"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="get_current_snapshot",
        description="Read the latest successfully published research snapshot.",
        inputSchema={
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "enum": ["stock", "etf", "otc_fund"],
                },
                "symbol": {"type": "string", "pattern": SYMBOL_PATTERN},
            },
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="get_rolling_scan",
        description=(
            "Read the rolling stock pool built incrementally while a full-market "
            "scan is running. Preliminary and order-biased until the scan "
            "completes; final recommendations still come from get_current_snapshot."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "enum": ["stock", "etf", "otc_fund"],
                },
                "symbol": {"type": "string", "pattern": SYMBOL_PATTERN},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="get_run_status",
        description=(
            "Read one validated research run ID, or the current/recent run when omitted. "
            "Returns compact lifecycle and progress fields. Queued, running, partial, "
            "and stalled are non-success states and do not guarantee fresh data."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            },
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="list_notifications",
        description="List a bounded, optionally filtered page of research notifications.",
        inputSchema={
            "type": "object",
            "properties": {
                "unread_only": {"type": "boolean"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "since": {"type": "string"},
                "run_id": {"type": "string", "pattern": RUN_ID_PATTERN},
            },
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="ack_notification",
        description="Acknowledge one validated notification ID after user confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "notification_id": {
                    "type": "string",
                    "pattern": NOTIFICATION_ID_PATTERN,
                },
            },
            "required": ["notification_id"],
            "additionalProperties": False,
        },
    ),
]

TOOL_NAMES = {tool.name for tool in TOOLS}
SCHEMAS = {tool.name: tool.input_schema for tool in TOOLS}


def _matches(value: Any, pattern: str) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def validate_tool_arguments(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool: {name}")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    schema = SCHEMAS[name]
    properties = schema.get("properties", {})
    unknown = sorted(set(arguments) - set(properties))
    if unknown:
        raise ValueError(f"unexpected arguments: {unknown}")
    missing = sorted(set(schema.get("required", [])) - set(arguments))
    if missing:
        raise ValueError(f"missing required arguments: {missing}")

    if name == "run_research":
        if arguments["mode"] not in {"daily", "monthly", "on_demand"}:
            raise ValueError("invalid mode")
        if arguments["asset_scope"] not in {"all", "stock", "etf", "otc_fund"}:
            raise ValueError("invalid asset_scope")
        symbols = arguments.get("symbols")
        if symbols is not None:
            if (
                not isinstance(symbols, list)
                or not symbols
                or len(symbols) > 100
                or any(not _matches(symbol, SYMBOL_PATTERN) for symbol in symbols)
                or len(set(symbols)) != len(symbols)
            ):
                raise ValueError("symbols must be a unique list of at most 100 market symbols")
    elif name == "get_current_snapshot":
        asset_type = arguments.get("asset_type")
        if asset_type is not None and asset_type not in {"stock", "etf", "otc_fund"}:
            raise ValueError("invalid asset_type")
        symbol = arguments.get("symbol")
        if symbol is not None and not _matches(symbol, SYMBOL_PATTERN):
            raise ValueError("invalid symbol")
    elif name == "get_rolling_scan":
        asset_type = arguments.get("asset_type")
        if asset_type is not None and asset_type not in {"stock", "etf", "otc_fund"}:
            raise ValueError("invalid asset_type")
        symbol = arguments.get("symbol")
        if symbol is not None and not _matches(symbol, SYMBOL_PATTERN):
            raise ValueError("invalid symbol")
        limit = arguments.get("limit", 30)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
    elif name == "get_run_status":
        if "run_id" in arguments and not _matches(arguments.get("run_id"), RUN_ID_PATTERN):
            raise ValueError("invalid run_id")
    elif name == "list_notifications":
        if "unread_only" in arguments and not isinstance(arguments["unread_only"], bool):
            raise ValueError("unread_only must be boolean")
        limit = arguments.get("limit", 10)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if "since" in arguments and not isinstance(arguments["since"], str):
            raise ValueError("since must be an ISO-8601 timestamp string")
        if "run_id" in arguments and not _matches(arguments.get("run_id"), RUN_ID_PATTERN):
            raise ValueError("invalid run_id")
    elif name == "ack_notification":
        if not _matches(arguments.get("notification_id"), NOTIFICATION_ID_PATTERN):
            raise ValueError("invalid notification_id")
    return arguments


def dispatch_tool(name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
    values = validate_tool_arguments(name, arguments)
    calls: dict[str, Callable[[], dict[str, Any]]] = {
        "run_research": lambda: run_research(
            mode=values["mode"],
            asset_scope=values["asset_scope"],
            symbols=values.get("symbols"),
        ),
        "get_current_snapshot": lambda: get_current_snapshot(
            asset_type=values.get("asset_type"),
            symbol=values.get("symbol"),
        ),
        "get_rolling_scan": lambda: get_rolling_scan(
            values.get("asset_type"),
            values.get("symbol"),
            limit=values.get("limit", 30),
        ),
        "get_run_status": lambda: get_run_status(values.get("run_id")),
        "list_notifications": lambda: list_notifications(
            values.get("unread_only", False),
            limit=values.get("limit", 10),
            since=values.get("since"),
            run_id=values.get("run_id"),
        ),
        "ack_notification": lambda: ack_notification(values["notification_id"]),
    }
    return calls[name]()


async def _list_tools(_context: Any, _params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def _call_tool(_context: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    try:
        result = await anyio.to_thread.run_sync(
            lambda: dispatch_tool(params.name, params.arguments)
        )
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(result, ensure_ascii=False, sort_keys=True),
                )
            ],
            structuredContent=result,
            isError=False,
        )
    except Exception as exc:
        LOGGER.exception("tool call failed: %s", params.name)
        error = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        return types.CallToolResult(
            content=[
                types.TextContent(
                    type="text",
                    text=json.dumps(error, ensure_ascii=False, sort_keys=True),
                )
            ],
            structuredContent=error,
            isError=True,
        )


SERVER = Server(
    "stock_assistant",
    version="1.0.0",
    description="Private fixed-path stock, ETF, and OTC fund research tools.",
    on_list_tools=_list_tools,
    on_call_tool=_call_tool,
)


async def serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await SERVER.run(
            read_stream,
            write_stream,
            SERVER.create_initialization_options(),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    anyio.run(serve)


if __name__ == "__main__":
    main()
