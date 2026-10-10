# -*- coding: utf-8 -*-
"""Tests for the T5 silent-exception tracer and its wired sites.

Covers the shared helper (parseable JSON line, silent on unwritable path,
5 MB rotation) and at least one real site (``rolling_scan._read_state`` with a
corrupt state file) asserting a log line while the return value is unchanged.
"""
from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import silent_trace


class SilentTraceHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "silent-exceptions.log"
        self._old = os.environ.get("DSH_SILENT_TRACE_PATH")
        os.environ["DSH_SILENT_TRACE_PATH"] = str(self.path)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("DSH_SILENT_TRACE_PATH", None)
        else:
            os.environ["DSH_SILENT_TRACE_PATH"] = self._old
        self.tmp.cleanup()

    def test_writes_parseable_line(self):
        try:
            raise ValueError("boom " * 200)
        except Exception as exc:
            silent_trace.trace("mod", "site.x", exc, symbol="000875", job_id="j-1")
        record = json.loads(self.path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["module"], "mod")
        self.assertEqual(record["site"], "site.x")
        self.assertEqual(record["exc_type"], "ValueError")
        self.assertEqual(record["symbol"], "000875")
        self.assertEqual(record["job_id"], "j-1")
        self.assertLessEqual(len(record["message"]), 300)
        self.assertIn("ts", record)

    def test_silent_when_path_unwritable(self):
        os.environ["DSH_SILENT_TRACE_PATH"] = "Z:/definitely/not/here/x.log"
        try:
            raise RuntimeError("nope")
        except Exception as exc:
            silent_trace.trace("m", "s", exc)  # must not raise

    def test_rotates_when_over_max_bytes(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"x" * (silent_trace.MAX_BYTES + 1))
        try:
            raise ValueError("rotate")
        except Exception as exc:
            silent_trace.trace("m", "s", exc)
        backup = self.path.with_name(self.path.name + ".1")
        self.assertTrue(backup.exists())
        record = json.loads(self.path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["exc_type"], "ValueError")


class RollingScanSiteTests(unittest.TestCase):
    def test_read_state_traces_and_return_unchanged(self):
        # Real wired site: the exclude-cross-border branch's totals recompute.
        # Force that branch on and make the manifest load fail; assert a log line
        # is written while the returned rows/totals are unchanged.
        import sqlite3
        from unittest import mock
        from assistant import rolling_scan, research
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = pathlib.Path(tmp.name) / "fetch_state.sqlite3"
        con = sqlite3.connect(str(path))
        con.execute("CREATE TABLE assets(asset_type TEXT, symbol TEXT, "
                    "data_end_date TEXT, rows INT, source TEXT, status TEXT)")
        con.execute("INSERT INTO assets VALUES('stock','000001','2026-09-30',1,'x','success')")
        con.commit()
        con.close()
        log_path = pathlib.Path(tmp.name) / "silent.log"
        old = os.environ.get("DSH_SILENT_TRACE_PATH")
        os.environ["DSH_SILENT_TRACE_PATH"] = str(log_path)
        self.addCleanup(lambda: os.environ.__setitem__("DSH_SILENT_TRACE_PATH", old)
                        if old is not None else os.environ.pop("DSH_SILENT_TRACE_PATH", None))
        with mock.patch.object(research, "load_config",
                               return_value={"research": {"pool_exclude_cross_border": True}}),                 mock.patch.object(rolling_scan, "load_universe_manifest",
                                  side_effect=RuntimeError("manifest boom")):
            rows, totals = rolling_scan._read_state(path)
        # Control flow unchanged: the success row is still returned.
        self.assertEqual([r["symbol"] for r in rows], ["000001"])
        self.assertEqual(totals, {"stock": {"total": 1, "success": 1}})
        self.assertTrue(log_path.exists())
        record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
        self.assertEqual(record["module"], "rolling_scan")
        self.assertIn("_read_state", record["site"])


if __name__ == "__main__":
    unittest.main()
