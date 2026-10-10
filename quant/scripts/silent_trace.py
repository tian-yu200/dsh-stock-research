# -*- coding: utf-8 -*-
"""Shared silent-exception tracer (T5).

Appends one JSON line per caught-and-ignored exception to
``quant/logs/silent-exceptions.log``. The whole body is wrapped in a silent
try/except: tracing must never raise, never print, and never write to stderr
(queueing/fetch_state run inside the MCP stdio server whose stdout is the
protocol; the job ledger misreads worker stderr). Control flow is unchanged:
callers keep their ``pass``.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
MESSAGE_LIMIT = 300

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "logs" / "silent-exceptions.log"


def log_path() -> Path:
    override = os.environ.get("DSH_SILENT_TRACE_PATH")
    return Path(override) if override else _DEFAULT_PATH


def _rotate(path: Path) -> None:
    try:
        if path.exists() and path.stat().st_size >= MAX_BYTES:
            backup = path.with_name(path.name + ".1")
            if backup.exists():
                backup.unlink()
            path.replace(backup)
    except Exception:
        pass


def trace(module: str, site: str, exc: BaseException | None = None,
          *, symbol: str | None = None, job_id: str | None = None,
          message: str | None = None) -> None:
    """Append one JSON line describing a silently-caught exception."""
    try:
        if message is None:
            message = str(exc) if exc is not None else ""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "module": str(module),
            "site": str(site),
            "exc_type": type(exc).__name__ if exc is not None else "None",
            "message": message[:MESSAGE_LIMIT],
        }
        if symbol:
            record["symbol"] = str(symbol)
        if job_id:
            record["job_id"] = str(job_id)
        path = log_path()
        _rotate(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    except Exception:
        pass
