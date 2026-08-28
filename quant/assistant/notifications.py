"""Small JSON notification inbox consumed by the DSH assistant."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .paths import NOTIFICATIONS_PATH, atomic_write_json, ensure_directories, read_json


def _load() -> list[dict[str, Any]]:
    ensure_directories()
    value = read_json(NOTIFICATIONS_PATH, [])
    return value if isinstance(value, list) else []


def _save(records: list[dict[str, Any]]) -> None:
    atomic_write_json(NOTIFICATIONS_PATH, records[-500:])


def add_notification(
    kind: str,
    title: str,
    message: str,
    *,
    run_id: str | None = None,
    paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    record = {
        "notification_id": f"n-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "title": title,
        "message": message,
        "run_id": run_id,
        "paths": paths or {},
        "acknowledged": False,
    }
    records = _load()
    records.append(record)
    _save(records)
    return record


def list_records(
    unread_only: bool = False,
    *,
    limit: int = 10,
    since: str | None = None,
    run_id: str | None = None,
 ) -> list[dict[str, Any]]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    since_at: datetime | None = None
    if since:
        try:
            since_at = datetime.fromisoformat(str(since).replace("Z", "+00:00"))
            if since_at.tzinfo is None:
                since_at = since_at.replace(tzinfo=timezone.utc)
            since_at = since_at.astimezone(timezone.utc)
        except ValueError as exc:
            raise ValueError("since must be an ISO-8601 timestamp") from exc
    records = _load()
    selected: list[dict[str, Any]] = []
    for record in reversed(records):
        if unread_only and record.get("acknowledged", False):
            continue
        if run_id is not None and record.get("run_id") != run_id:
            continue
        if since_at is not None:
            try:
                created_at = datetime.fromisoformat(
                    str(record.get("created_at", "")).replace("Z", "+00:00")
                )
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                created_at = created_at.astimezone(timezone.utc)
            except ValueError:
                continue
            if created_at < since_at:
                continue
        selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def acknowledge(notification_id: str) -> dict[str, Any]:
    records = _load()
    for record in records:
        if record.get("notification_id") == notification_id:
            record["acknowledged"] = True
            record["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
            _save(records)
            return {"ok": True, "notification": record}
    return {"ok": False, "error": f"notification not found: {notification_id}"}
