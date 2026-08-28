from __future__ import annotations

import pathlib
import sys
import unittest
from unittest.mock import patch

# unittest discovery can add the quant project root to sys.path. That root
# contains the application package named "mcp", which must not shadow the
# third-party MCP SDK used by this integration test.
ROOT = pathlib.Path(__file__).resolve().parents[1]
project_root = ROOT.resolve()
filtered_path = []
for entry in sys.path:
    candidate = pathlib.Path.cwd().resolve() if not entry else pathlib.Path(entry).resolve()
    if candidate == project_root:
        continue
    filtered_path.append(entry)
sys.path[:] = filtered_path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
SERVER = ROOT / "mcp" / "server.py"


class McpStdioTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_handshake_tools_and_fixed_argument_boundaries(self) -> None:
        parameters = StdioServerParameters(
            command=str(PYTHON),
            args=[str(SERVER)],
            cwd=str(ROOT),
        )

        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                self.assertEqual(initialized.server_info.name, "stock_assistant")

                listed = await session.list_tools()
                names = [tool.name for tool in listed.tools]
                self.assertEqual(
                    names,
                    [
                        "run_research",
                        "get_current_snapshot",
                        "get_run_status",
                        "list_notifications",
                        "ack_notification",
                    ],
                )

                snapshot = await session.call_tool("get_current_snapshot", {})
                self.assertFalse(snapshot.is_error)

                run_status = await session.call_tool("get_run_status", {})
                self.assertFalse(run_status.is_error)

                notifications = await session.call_tool(
                    "list_notifications",
                    {"unread_only": True},
                )
                self.assertFalse(notifications.is_error)

                acknowledgement = await session.call_tool(
                    "ack_notification",
                    {"notification_id": "n-20260826T000000Z-00000000"},
                )
                self.assertFalse(acknowledgement.is_error)

                rejected_run = await session.call_tool(
                    "run_research",
                    {
                        "mode": "invalid",
                        "asset_scope": "stock",
                    },
                )
                self.assertTrue(rejected_run.is_error)

                rejected_extra_argument = await session.call_tool(
                    "get_run_status",
                    {"output_dir": "D:/outside"},
                )
                self.assertTrue(rejected_extra_argument.is_error)

                rejected_empty_symbols = await session.call_tool(
                    "run_research",
                    {
                        "mode": "on_demand",
                        "asset_scope": "stock",
                        "symbols": [],
                    },
                )
                self.assertTrue(rejected_empty_symbols.is_error)

                rejected_notification_limit = await session.call_tool(
                    "list_notifications",
                    {"limit": 0},
                )
                self.assertTrue(rejected_notification_limit.is_error)


import importlib.util


def _load_server_module():
    """Import mcp/server.py in-process without shadowing the SDK.

    The third-party SDK is imported first (sys.path is already filtered at
    module top) and stays cached in sys.modules, so server.py's
    ``from mcp import types`` resolves to the SDK even while the project
    root is temporarily on sys.path for the ``assistant`` package.
    """
    cached = sys.modules.get("stock_mcp_server_under_test")
    if cached is not None:
        return cached

    import mcp as sdk_mcp  # noqa: F401  (pins the SDK in sys.modules)

    saved_path = list(sys.path)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        spec = importlib.util.spec_from_file_location(
            "stock_mcp_server_under_test", str(SERVER)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    sys.modules["stock_mcp_server_under_test"] = module
    return module


class McpToolContractUnitTests(unittest.TestCase):
    """In-process contract tests for tool argument validation and dispatch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _load_server_module()

    def test_unknown_tool_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.server.validate_tool_arguments("delete_universe", {})

    def test_every_tool_accepts_its_minimal_arguments(self) -> None:
        cases = {
            "run_research": {"mode": "daily", "asset_scope": "all"},
            "get_current_snapshot": {},
            "get_run_status": {},
            "list_notifications": {},
            "ack_notification": {"notification_id": "n-20260829T000000Z-00000000"},
        }
        for name, arguments in cases.items():
            with self.subTest(tool=name):
                self.assertEqual(
                    self.server.validate_tool_arguments(name, arguments),
                    arguments,
                )

    def test_run_research_rejects_duplicate_and_overlong_symbols(self) -> None:
        base = {"mode": "on_demand", "asset_scope": "stock"}
        with self.assertRaises(ValueError):
            self.server.validate_tool_arguments(
                "run_research",
                {**base, "symbols": ["600519", "600519"]},
            )
        with self.assertRaises(ValueError):
            self.server.validate_tool_arguments(
                "run_research",
                {**base, "symbols": ["600519"] * 101},
            )

    def test_dispatch_routes_validated_arguments_to_assistant_functions(self) -> None:
        with patch.object(
            self.server,
            "run_research",
            return_value={"run_id": "20260829T000000Z-00000000"},
        ) as mock_run:
            result = self.server.dispatch_tool(
                "run_research",
                {"mode": "on_demand", "asset_scope": "etf", "symbols": ["510300"]},
            )
        self.assertEqual(result["run_id"], "20260829T000000Z-00000000")
        mock_run.assert_called_once_with(
            mode="on_demand",
            asset_scope="etf",
            symbols=["510300"],
        )

    def test_dispatch_applies_documented_defaults(self) -> None:
        with patch.object(
            self.server,
            "list_notifications",
            return_value={"records": []},
        ) as mock_list:
            self.server.dispatch_tool("list_notifications", {})
        mock_list.assert_called_once_with(False, limit=10, since=None, run_id=None)

    def test_dispatch_of_unknown_tool_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.server.dispatch_tool("delete_universe", {})


if __name__ == "__main__":
    unittest.main()
