from __future__ import annotations

import pathlib
import sys
import unittest

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


if __name__ == "__main__":
    unittest.main()
