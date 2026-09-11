r"""Fixed, auditable paths for the stock assistant.

No caller-provided path is accepted by the research service.  All persistent
state is resolved below ``<仓库根>/quant``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUANT_ROOT = PROJECT_ROOT / "quant"
CONFIG_DIR = QUANT_ROOT / "configs"
DATA_DIR = QUANT_ROOT / "data"
QLIB_PROVIDER_DIR = QUANT_ROOT / "qlib_data" / "akshare_data"
ASSISTANT_RESULTS_DIR = QUANT_ROOT / "results" / "assistant"
RUNS_DIR = ASSISTANT_RESULTS_DIR / "runs"
LOGS_DIR = QUANT_ROOT / "logs"
NOTIFICATIONS_PATH = LOGS_DIR / "notifications.json"
LOCK_PATH = LOGS_DIR / "research.lock"

RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")
SYMBOL_RE = re.compile(r"^(?:SH|SZ|BJ)?[0-9]{6}$", re.IGNORECASE)
NOTIFICATION_ID_RE = re.compile(r"^n-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


def ensure_directories() -> None:
    for path in (
        CONFIG_DIR,
        DATA_DIR / "stocks",
        DATA_DIR / "funds",
        DATA_DIR / "benchmarks",
        QLIB_PROVIDER_DIR,
        ASSISTANT_RESULTS_DIR,
        RUNS_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


_LEGACY_QUANT_ROOT_MARKER = "dsh-股票/quant"


def relocate_legacy_path(value: str | os.PathLike[str]) -> Path | None:
    """Map a path persisted under the pre-migration repo root onto this checkout.

    The repository moved from ``D:\\dsh-股票`` to ``D:\\dsh-gupiao``; result
    snapshots written before the move (e.g. ``current.json``) still carry the
    old absolute prefix. Only the well-known legacy quant-root marker is
    remapped, and only onto a location inside the current ``QUANT_ROOT``;
    anything else returns ``None`` so the fixed-path boundary keeps rejecting
    genuinely out-of-project references.
    """
    text = str(value).replace("\\", "/")
    index = text.lower().find(_LEGACY_QUANT_ROOT_MARKER)
    if index < 0:
        return None
    suffix = text[index + len(_LEGACY_QUANT_ROOT_MARKER):].lstrip("/")
    candidate = QUANT_ROOT.joinpath(*suffix.split("/")) if suffix else QUANT_ROOT
    try:
        candidate.resolve().relative_to(QUANT_ROOT.resolve())
    except ValueError:
        return None
    return candidate


def fixed_path(value: str | os.PathLike[str], *, base: Path = QUANT_ROOT) -> Path:
    """Resolve a configured relative path and reject paths outside the project."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(QUANT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"path must stay inside {QUANT_ROOT}: {value}") from exc
    return resolved


def validate_symbol(value: str) -> str:
    symbol = str(value).strip().upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"invalid market symbol: {value}")
    return symbol[2:] if symbol[:2] in {"SH", "SZ", "BJ"} else symbol


def validate_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("invalid run_id")
    return run_id


def validate_notification_id(value: str) -> str:
    notification_id = str(value).strip()
    if not NOTIFICATION_ID_RE.fullmatch(notification_id):
        raise ValueError("invalid notification_id")
    return notification_id


def run_directory(run_id: str) -> Path:
    run_id = validate_run_id(run_id)
    date = run_id[:8]
    candidate = (
        RUNS_DIR
        / date[:4]
        / date[4:6]
        / date[6:8]
        / run_id
    ).resolve()
    try:
        candidate.relative_to(RUNS_DIR.resolve())
    except ValueError as exc:
        raise FileNotFoundError(f"run not found: {run_id}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"run not found: {run_id}")
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    path = fixed_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 2026-09-10 状态写入重试：run_research_worker._write_status 与
    # research._touch_status 并发 os.replace 同一 status.json 时，
    # Windows 报 PermissionError WinError 5 把运行误标 failed
    #（summary/model 已生成）。重试不改原子语义与心跳语义。
    last_error: OSError | None = None
    for attempt in range(5):
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            except OSError:
                pass
            time.sleep(0.05 * (2 ** attempt))
        finally:
            if os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
    assert last_error is not None
    raise last_error


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
    )


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
