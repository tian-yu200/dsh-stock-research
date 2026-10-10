"""Persistent state for incremental, resumable market-data collection.

The state database deliberately stores metadata only.  Market data remains in
the existing CSV files so the current research and Qlib export paths keep their
file contract.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import silent_trace

UTC = timezone.utc
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "fetch_state.sqlite3"
QUANT_ROOT_F = Path(__file__).resolve().parents[1]
UNIVERSE_PATH_F = QUANT_ROOT_F / "data" / "universe.json"

_CROSS_BORDER_KEYWORDS_F = (
    "标普", "纳斯达克", "纳指", "道琼斯", "美国", "美股", "美债", "美元",
    "恒生", "恒指", "港股", "沪港深", "港通", "中概", "海外", "全球", "环球",
    "亚太", "日经", "东京", "德国", "法国", "欧洲", "新兴", "金砖", "REIT", "QDII",
    "韩国", "中韩", "越南", "英国",
)

def _is_cross_border_f(name: str | None, asset_type: str | None = None) -> bool:
    if asset_type == "stock":
        return False
    if not name:
        return False
    low = str(name).lower()
    for kw in _CROSS_BORDER_KEYWORDS_F:
        if kw.lower() in low:
            return True
    return False

def _load_manifest_names_f() -> dict[str, dict[str, str]]:
    try:
        import json as _j
        data = _j.loads(UNIVERSE_PATH_F.read_text(encoding="utf-8"))
        out = {}
        for at in ("etf", "otc_fund"):
            recs = data.get(at) or []
            d = {}
            for item in recs:
                if isinstance(item, dict) and item.get("symbol") and item.get("name"):
                    d[str(item["symbol"])] = str(item["name"])
            out[at] = d
        return out
    except Exception:
        return {"etf": {}, "otc_fund": {}}

def _should_filter_progress() -> bool:
    try:
        import yaml as _yaml
        cfg = _yaml.safe_load((QUANT_ROOT_F / "configs" / "assistant.yaml").read_text(encoding="utf-8")) or {}
        research = (cfg.get("research") or {})
        return bool(research.get("universe_exclude_cross_border") or research.get("pool_exclude_cross_border"))
    except Exception:
        return False


def process_alive(pid: int | None) -> bool:
    """Return True when ``pid`` refers to a live process.

    ``os.kill(pid, 0)`` is **not** a liveness probe on Windows: it maps to
    ``TerminateProcess`` and raises ``OSError(winerror=87, errno=22)`` for live
    processes the caller did not spawn, so any "exception means dead" check
    reports unrelated live workers as dead.  Windows therefore asks the kernel
    directly (``OpenProcess`` + ``GetExitCodeProcess``); access denied means the
    process exists but belongs to someone else, which counts as alive.
    """
    try:
        value = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(value, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    access_denied = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, value)
    if not handle:
        return ctypes.get_last_error() == access_denied
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(UTC).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


class FetchState:
    """Small SQLite-backed queue with leases safe for multiple worker processes."""

    def __init__(self, path: Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    asset_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    data_end_date TEXT,
                    last_success_date TEXT,
                    last_attempt_at TEXT,
                    next_retry_at TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    source TEXT,
                    last_error TEXT,
                    failure_class TEXT,
                    failure_details_json TEXT,
                    rows INTEGER,
                    lease_owner TEXT,
                    lease_until TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (asset_type, symbol)
                );

                CREATE INDEX IF NOT EXISTS idx_assets_due
                    ON assets (asset_type, next_retry_at, data_end_date, lease_until);

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    asset_scope TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    total INTEGER NOT NULL DEFAULT 0,
                    completed INTEGER NOT NULL DEFAULT 0,
                    pending INTEGER NOT NULL DEFAULT 0,
                    failed INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    coverage REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(assets)").fetchall()
            }
            if "failure_class" not in columns:
                connection.execute("ALTER TABLE assets ADD COLUMN failure_class TEXT")
            if "failure_details_json" not in columns:
                connection.execute("ALTER TABLE assets ADD COLUMN failure_details_json TEXT")

    @staticmethod
    def _symbols(values: Iterable[str]) -> list[str]:
        return sorted({str(value).strip().upper() for value in values if str(value).strip()})

    def register_assets(
        self,
        asset_type: str,
        symbols: Iterable[str],
        *,
        data_end_dates: dict[str, str | None] | None = None,
    ) -> int:
        """Insert new assets without overwriting progress for existing assets."""
        values = self._symbols(symbols)
        now = iso_now()
        data_end_dates = data_end_dates or {}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for symbol in values:
                cached_end = data_end_dates.get(symbol)
                connection.execute(
                    """
                    INSERT INTO assets (
                        asset_type, symbol, data_end_date, last_success_date,
                        status, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(asset_type, symbol) DO UPDATE SET
                        data_end_date = CASE
                            WHEN assets.data_end_date IS NULL THEN excluded.data_end_date
                            ELSE assets.data_end_date
                        END,
                        last_success_date = CASE
                            WHEN assets.last_success_date IS NULL THEN excluded.last_success_date
                            ELSE assets.last_success_date
                        END,
                        updated_at = excluded.updated_at
                    """,
                    (
                        asset_type,
                        symbol,
                        cached_end,
                        cached_end,
                        "pending",
                        now,
                    ),
                )
            connection.commit()
        return len(values)

    def assets_needing_cache_backfill(
        self,
        asset_type: str,
        symbols: Iterable[str],
    ) -> list[str]:
        """Return selected assets that do not yet have a persisted end date."""
        values = self._symbols(symbols)
        if not values:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol
                FROM assets
                WHERE asset_type = ? AND data_end_date IS NOT NULL
                """,
                (asset_type,),
            ).fetchall()
        complete = {str(row["symbol"]) for row in rows}
        return [symbol for symbol in values if symbol not in complete]

    def reconcile_cached_assets(
        self,
        asset_type: str,
        data_end_dates: Mapping[str, str | None],
    ) -> int:
        """Accept CSV progress that is newer than the durable queue state."""
        values = {
            str(symbol).strip().upper(): str(end_date)
            for symbol, end_date in data_end_dates.items()
            if str(symbol).strip() and end_date
        }
        if not values:
            return 0
        updated = 0
        now = iso_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for symbol, cached_end in values.items():
                cursor = connection.execute(
                    """
                    UPDATE assets
                    SET data_end_date = ?,
                        last_success_date = ?,
                        next_retry_at = NULL,
                        status = 'success',
                        last_error = NULL,
                        failure_class = NULL,
                        failure_details_json = NULL,
                        lease_owner = NULL,
                        lease_until = NULL,
                        updated_at = ?
                    WHERE asset_type = ? AND symbol = ?
                      AND status != 'running'
                      AND ? > MAX(
                          COALESCE(data_end_date, ''),
                          COALESCE(last_success_date, '')
                      )
                    """,
                    (cached_end, cached_end, now, asset_type, symbol, cached_end),
                )
                updated += max(0, int(cursor.rowcount))
            connection.commit()
        return updated

    def lifecycle_candidates(
        self,
        asset_type: str,
        symbols: Iterable[str],
        *,
        failure_class: str = "empty_window",
        min_attempts: int = 2,
    ) -> list[str]:
        """Return repeatedly empty assets eligible for catalog confirmation."""
        selected = set(self._symbols(symbols))
        if not selected:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT symbol
                FROM assets
                WHERE asset_type = ?
                  AND (
                      status = 'inactive'
                      OR (
                          status = 'failed'
                          AND failure_class = ?
                          AND attempt_count >= ?
                      )
                  )
                ORDER BY symbol
                """,
                (asset_type, failure_class, max(1, int(min_attempts))),
            ).fetchall()
        return [str(row["symbol"]) for row in rows if row["symbol"] in selected]

    def apply_lifecycle_snapshot(
        self,
        asset_type: str,
        snapshot: Mapping[str, Any],
    ) -> dict[str, int]:
        """Apply confirmed inactivity and revive symbols seen as active."""
        active_value = snapshot.get("active", [])
        inactive_value = snapshot.get("inactive_candidates", [])
        active = self._symbols(
            active_value if isinstance(active_value, (list, tuple, set)) else []
        )
        active_set = set(active)
        inactive = [
            symbol
            for symbol in self._symbols(
                inactive_value
                if isinstance(inactive_value, (list, tuple, set))
                else []
            )
            if symbol not in active_set
        ]
        evidence = {
            "generated_at": snapshot.get("generated_at"),
            "sources": snapshot.get("sources", {}),
        }
        details = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        result = {"inactivated": 0, "revived": 0}
        now = iso_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for symbol in active:
                cursor = connection.execute(
                    """
                    UPDATE assets
                    SET status = 'pending',
                        next_retry_at = NULL,
                        last_error = NULL,
                        failure_class = NULL,
                        failure_details_json = NULL,
                        lease_owner = NULL,
                        lease_until = NULL,
                        updated_at = ?
                    WHERE asset_type = ? AND symbol = ?
                      AND status = 'inactive'
                    """,
                    (now, asset_type, symbol),
                )
                result["revived"] += max(0, int(cursor.rowcount))
            if snapshot.get("complete") is True:
                for symbol in inactive:
                    cursor = connection.execute(
                        """
                        UPDATE assets
                        SET status = 'inactive',
                            next_retry_at = NULL,
                            last_error = 'absent from exchange, Eastmoney, and Sina catalogs',
                            failure_class = 'inactive_catalog',
                            failure_details_json = ?,
                            lease_owner = NULL,
                            lease_until = NULL,
                            updated_at = ?
                        WHERE asset_type = ? AND symbol = ?
                          AND status = 'failed'
                          AND failure_class = 'empty_window'
                          AND attempt_count >= 2
                        """,
                        (details, now, asset_type, symbol),
                    )
                    result["inactivated"] += max(0, int(cursor.rowcount))
            connection.commit()
        return result

    def claim_due(
        self,
        asset_type: str,
        *,
        symbols: Iterable[str] | None = None,
        limit: int = 100,
        target_date: str,
        lease_seconds: int = 1800,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Claim at most ``limit`` due assets and return their prior state."""
        if limit < 1:
            return []
        owner = owner or f"{asset_type}-{uuid.uuid4().hex}"
        current = now or utc_now()
        now_text = iso_now(current)
        lease_until = iso_now(current + timedelta(seconds=max(1, lease_seconds)))
        selected = self._symbols(symbols or [])
        # 2026-09-09 空转修复：跨境排除必须在 SQL LIMIT 之前生效。
        # 旧逻辑先 LIMIT 3 再 Python 过滤，头部连续跨境时每轮返回 []，
        # 租约不动、头部不变，无限空转。现对 selected 预过滤，
        # LIMIT 后全是可抓标的；与 progress() 全量读后过滤口径一致。
        # stock 永不过滤，缺名 fail-open，config 默认 false 保测试。
        if _should_filter_progress() and asset_type in ("etf", "otc_fund") and selected:
            try:
                _names = _load_manifest_names_f().get(asset_type, {})
                _kept = [
                    s for s in selected
                    if not _is_cross_border_f(_names.get(s, ""), asset_type=asset_type)
                ]
                selected = _kept
                if not selected:
                    return []
            except Exception as exc:
                silent_trace.trace("fetch_state", "claim_due.cross_border_filter", exc)
                pass
        # A newer target date may bypass a prior success refresh once.
        clauses = [
            "asset_type = ?",
            "status NOT IN ('quarantined', 'inactive')",
            "(lease_until IS NULL OR lease_until < ?)",
            """(
                (next_retry_at IS NULL OR next_retry_at <= ?)
                OR (
                    status = 'success'
                    AND (last_attempt_at IS NULL OR last_attempt_at < ?)
                )
            )""",
            "MAX(COALESCE(data_end_date, ''), COALESCE(last_success_date, '')) < ?",
        ]
        parameters: list[Any] = [
            asset_type,
            now_text,
            now_text,
            target_date,
            target_date,
        ]
        if selected:
            placeholders = ",".join("?" for _ in selected)
            clauses.append(f"symbol IN ({placeholders})")
            parameters.extend(selected)
        where = " AND ".join(clauses)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"""
                SELECT asset_type, symbol, data_end_date, last_success_date,
                       attempt_count, status, source, last_error
                FROM assets
                WHERE {where}
                ORDER BY
                    CASE status WHEN 'failed' THEN 1 ELSE 0 END,
                    CASE WHEN MAX(COALESCE(data_end_date, ''), COALESCE(last_success_date, '')) = '' THEN 0 ELSE 1 END,
                    MAX(COALESCE(data_end_date, ''), COALESCE(last_success_date, '')),
                    symbol
                LIMIT ?
                """,
                [*parameters, int(limit)],
            ).fetchall()
            # selected 预过滤后，此处仅作 symbols=None 路径的兜底；
            # 生产基金通道恒传 pool，正常情况下为无操作。
            if not selected and _should_filter_progress() and asset_type in ("etf", "otc_fund"):
                try:
                    _names_all = _load_manifest_names_f().get(asset_type, {})
                    rows = [
                        r for r in rows
                        if not _is_cross_border_f(_names_all.get(str(r["symbol"]), ""), asset_type=asset_type)
                    ]
                except Exception as exc:
                    silent_trace.trace("fetch_state", "progress.cross_border_filter", exc)
                    pass
            for row in rows:
                connection.execute(
                    """
                    UPDATE assets
                    SET last_attempt_at = ?,
                        next_retry_at = NULL,
                        attempt_count = attempt_count + 1,
                        status = 'running',
                        lease_owner = ?,
                        lease_until = ?,
                        updated_at = ?
                    WHERE asset_type = ? AND symbol = ?
                    """,
                    (
                        now_text,
                        owner,
                        lease_until,
                        now_text,
                        row["asset_type"],
                        row["symbol"],
                    ),
                )
            connection.commit()
        return [dict(row) | {"lease_owner": owner} for row in rows]

    def reclaim_orphaned_leases(self) -> int:
        """Return leases held by dead worker PIDs to the pending queue."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT asset_type, symbol, lease_owner
                FROM assets
                WHERE status = 'running' AND lease_owner IS NOT NULL
                """
            ).fetchall()
            orphaned = []
            for row in rows:
                match = re.search(r"-(\d+)$", str(row["lease_owner"]))
                if not match:
                    continue
                if not process_alive(int(match.group(1))):
                    orphaned.append((row["asset_type"], row["symbol"]))
            if not orphaned:
                return 0
            connection.execute("BEGIN IMMEDIATE")
            for asset_type, symbol in orphaned:
                connection.execute(
                    """
                    UPDATE assets
                    SET status = 'pending',
                        lease_owner = NULL,
                        lease_until = NULL,
                        next_retry_at = NULL,
                        updated_at = ?
                    WHERE asset_type = ? AND symbol = ? AND status = 'running'
                    """,
                    (iso_now(), asset_type, symbol),
                )
            connection.commit()
            return len(orphaned)

    def release_leases(
        self,
        asset_type: str,
        symbols: Iterable[str],
        *,
        owner: str,
    ) -> int:
        """Return unstarted claims owned by this worker to the pending queue."""
        values = self._symbols(symbols)
        if not values:
            return 0
        placeholders = ",".join("?" for _ in values)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE assets
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_until = NULL,
                    next_retry_at = NULL,
                    updated_at = ?
                WHERE asset_type = ?
                  AND symbol IN ({placeholders})
                  AND status = 'running'
                  AND lease_owner = ?
                """,
                [iso_now(), asset_type, *values, owner],
            )
            return max(0, int(cursor.rowcount))

    def mark_success(
        self,
        asset_type: str,
        symbol: str,
        *,
        data_end_date: str | None,
        source: str | None,
        rows: int | None,
        refresh_interval_seconds: int,
        verified_through: str | None = None,
        owner: str | None = None,
    ) -> bool:
        now = utc_now()
        next_retry = now + timedelta(seconds=max(0, refresh_interval_seconds))
        verified_date = verified_through or data_end_date
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assets
                SET data_end_date = ?,
                    last_success_date = ?,
                    next_retry_at = ?,
                    status = 'success',
                    source = ?,
                    last_error = NULL,
                    failure_class = NULL,
                    failure_details_json = NULL,
                    rows = ?,
                    lease_owner = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE asset_type = ? AND symbol = ?
                  AND (? IS NULL OR lease_owner = ?)
                """,
                (
                    data_end_date,
                    verified_date,
                    iso_now(next_retry),
                    source,
                    rows,
                    iso_now(now),
                    asset_type,
                    symbol,
                    owner,
                    owner,
                ),
            )
        return cursor.rowcount > 0

    def advance_due_for_stale(
        self,
        *,
        before_date: str,
        asset_types: Iterable[str] | None = None,
        now: datetime | None = None,
    ) -> int:
        """把 data_end_date 落后 before_date 的成功标的强制置为到期。

        收盘后补数专用：mark_success 的刷新计时器（next_retry_at = 成功
        时刻 + refresh_interval）与 A 股收盘不对齐——在数据源发布当日
        K 线之前抓到"确认无新数据"的标的会再睡满一个周期，滞后逐日
        复利。这里清空落后标的的 next_retry_at，让下一轮 wave 立即
        认领。只动 status='success' 且租约已释放的行；本就已到期的行
        不重复计数。返回本次被强制到期的标的数。
        """
        current = iso_now(now or utc_now())
        query = """
            UPDATE assets
            SET next_retry_at = NULL,
                updated_at = ?
            WHERE status = 'success'
              AND data_end_date IS NOT NULL
              AND data_end_date < ?
              AND (lease_until IS NULL OR lease_until <= ?)
              AND next_retry_at IS NOT NULL
              AND next_retry_at > ?
        """
        params: list[Any] = [current, before_date, current, current]
        if asset_types is not None:
            types = sorted({str(value) for value in asset_types})
            if not types:
                return 0
            placeholders = ",".join("?" for _ in types)
            query += f" AND asset_type IN ({placeholders})"
            params.extend(types)
        with self._connect() as connection:
            cursor = connection.execute(query, params)
            return max(0, int(cursor.rowcount))

    def mark_failure(
        self,
        asset_type: str,
        symbol: str,
        *,
        error: str,
        backoff_base_seconds: int,
        backoff_max_seconds: int,
        failure_class: str | None = None,
        field_anomalies: Mapping[str, Any] | None = None,
        owner: str | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT attempt_count FROM assets
                WHERE asset_type = ? AND symbol = ?
                """,
                (asset_type, symbol),
            ).fetchone()
            attempts = int(row["attempt_count"]) if row else 1
        exponent = max(0, min(attempts - 1, 10))
        delay = min(
            max(1, int(backoff_max_seconds)),
            max(1, int(backoff_base_seconds)) * (2**exponent),
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assets
                SET next_retry_at = ?,
                    status = 'failed',
                    last_error = ?,
                    failure_class = ?,
                    failure_details_json = ?,
                    lease_owner = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE asset_type = ? AND symbol = ?
                  AND (? IS NULL OR lease_owner = ?)
                """,
                (
                    iso_now(now + timedelta(seconds=delay)),
                    str(error)[:2000],
                    failure_class,
                    json.dumps(field_anomalies, ensure_ascii=False, sort_keys=True) if field_anomalies else None,
                    iso_now(now),
                    asset_type,
                    symbol,
                    owner,
                    owner,
                ),
            )

    def mark_quarantined(
        self,
        asset_type: str,
        symbol: str,
        *,
        error: str,
        failure_class: str | None = None,
        field_anomalies: Mapping[str, Any] | None = None,
        owner: str | None = None,
    ) -> None:
        """Remove a deterministic bad response from automatic retries."""
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE assets
                SET next_retry_at = NULL,
                    status = 'quarantined',
                    last_error = ?,
                    failure_class = ?,
                    failure_details_json = ?,
                    lease_owner = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE asset_type = ? AND symbol = ?
                  AND (? IS NULL OR lease_owner = ?)
                """,
                (
                    str(error)[:2000],
                    failure_class,
                    json.dumps(field_anomalies, ensure_ascii=False, sort_keys=True) if field_anomalies else None,
                    iso_now(),
                    asset_type,
                    symbol,
                    owner,
                    owner,
                ),
            )

    def requeue_quarantined(
        self,
        asset_type: str,
        *,
        symbols: Iterable[str] | None = None,
        failure_class: str = "schema_conflict",
    ) -> int:
        """Return only matching quarantined assets to the durable queue."""
        selected = self._symbols(symbols or [])
        clauses = [
            "asset_type = ?",
            "status = 'quarantined'",
            "failure_class = ?",
        ]
        parameters: list[Any] = [asset_type, failure_class]
        if selected:
            placeholders = ",".join("?" for _ in selected)
            clauses.append(f"symbol IN ({placeholders})")
            parameters.extend(selected)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE assets
                SET status = 'pending',
                    next_retry_at = NULL,
                    last_error = NULL,
                    failure_class = NULL,
                    failure_details_json = NULL,
                    lease_owner = NULL,
                    lease_until = NULL,
                    updated_at = ?
                WHERE {' AND '.join(clauses)}
                """,
                [iso_now(), *parameters],
            )
        return max(0, int(cursor.rowcount))
    def progress(
        self,
        asset_types: Iterable[str],
        *,
        symbols: dict[str, Iterable[str]] | None = None,
        target_date: str | Mapping[str, str],
        max_age_days: int | None = None,
    ) -> dict[str, Any]:
        selected_types = tuple(dict.fromkeys(str(value) for value in asset_types))
        if isinstance(target_date, Mapping):
            target_dates = {
                asset_type: str(target_date[asset_type])
                for asset_type in selected_types
                if asset_type in target_date
            }
            missing_targets = [
                asset_type for asset_type in selected_types if asset_type not in target_dates
            ]
            if missing_targets:
                raise ValueError(
                    f"target dates missing for asset types: {', '.join(missing_targets)}"
                )
            legacy_target_date = max(target_dates.values(), default="")
        else:
            legacy_target_date = str(target_date)
            target_dates = {
                asset_type: legacy_target_date for asset_type in selected_types
            }
        target_days = {
            asset_type: datetime.fromisoformat(value).date()
            for asset_type, value in target_dates.items()
        }

        selected = {
            asset_type: set(self._symbols((symbols or {}).get(asset_type, [])))
            for asset_type in selected_types
        }
        rows: list[sqlite3.Row] = []
        with self._connect() as connection:
            for asset_type in selected_types:
                clauses = ["asset_type = ?"]
                params: list[Any] = [asset_type]
                if selected[asset_type]:
                    placeholders = ",".join("?" for _ in selected[asset_type])
                    clauses.append(f"symbol IN ({placeholders})")
                    params.extend(sorted(selected[asset_type]))
                rows.extend(
                    connection.execute(
                        f"SELECT * FROM assets WHERE {' AND '.join(clauses)}",
                        params,
                    ).fetchall()
                )
        # 2026-09-07 跨境排除：不计入统计（软排除，stock 永不过滤）
        if _should_filter_progress():
            try:
                names_by_type = _load_manifest_names_f()
                filtered = []
                for r in rows:
                    at = str(r["asset_type"])
                    if at not in ("etf", "otc_fund"):
                        filtered.append(r)
                    else:
                        nm = names_by_type.get(at, {}).get(str(r["symbol"]), "")
                        if not _is_cross_border_f(nm, asset_type=at):
                            filtered.append(r)
                rows = filtered
            except Exception as exc:
                silent_trace.trace("fetch_state", "progress.recompute_filter", exc)
                pass
        completed = 0
        inactive = 0
        failed = 0
        stale = 0
        pending = 0
        by_type: dict[str, dict[str, Any]] = {}
        for row in rows:
            asset_type = row["asset_type"]
            asset_target_date = target_dates[asset_type]
            target_day = target_days[asset_type]
            data_end_date = row["data_end_date"]
            last_success_date = row["last_success_date"]
            end_date = max(
                (value for value in (data_end_date, last_success_date) if value),
                default=None,
            )
            inactive_flag = row["status"] == "inactive"
            complete = inactive_flag or bool(end_date and end_date >= asset_target_date)
            inactive += int(inactive_flag)
            if complete:
                completed += 1
            else:
                pending += 1
            if row["status"] in {"failed", "quarantined"}:
                failed += 1
            stale_flag = False
            if max_age_days is not None and not inactive_flag:
                if not end_date:
                    stale_flag = True
                else:
                    try:
                        age = (target_day - datetime.fromisoformat(end_date).date()).days
                    except ValueError:
                        age = max_age_days + 1
                    stale_flag = age > max_age_days
                if stale_flag:
                    stale += 1
            group = by_type.setdefault(
                asset_type,
                {
                    "total": 0,
                    "completed": 0,
                    "inactive": 0,
                    "pending": 0,
                    "failed": 0,
                    "stale": 0,
                    "target_date": asset_target_date,
                },
            )
            group["total"] += 1
            group["completed"] += int(complete)
            group["inactive"] += int(inactive_flag)
            group["pending"] += int(not complete)
            group["failed"] += int(row["status"] in {"failed", "quarantined"})
            if max_age_days is not None:
                group["stale"] += int(stale_flag)
        total = len(rows)
        return {
            "total": total,
            "completed": completed,
            "inactive": inactive,
            "pending": pending,
            "failed": failed,
            "stale": stale,
            "coverage": completed / total if total else 0.0,
            "target_date": legacy_target_date,
            "target_dates": target_dates,
            "by_asset_type": by_type,
            "status": "complete" if total and completed == total else "partial",
        }

    def create_job(
        self,
        job_id: str,
        *,
        kind: str,
        asset_scope: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = iso_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO jobs (
                    job_id, kind, asset_scope, status, created_at, metadata_json
                )
                VALUES (?, ?, ?, 'queued', ?, ?)
                """,
                (
                    job_id,
                    kind,
                    asset_scope,
                    now,
                    json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
        return self.get_job(job_id) or {"job_id": job_id, "status": "queued"}

    def claim_job(self, job_id: str, *, stale_after_seconds: int = 3600) -> bool:
        """Atomically claim a queued job, recovering an abandoned old worker."""
        now = utc_now()
        now_text = iso_now(now)
        stale_before = iso_now(
            now - timedelta(seconds=max(1, int(stale_after_seconds)))
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, started_at FROM jobs WHERE job_id = ?
                """,
                (job_id,),
            ).fetchone()
            if row is None:
                connection.rollback()
                return False
            status = str(row["status"])
            started_at = row["started_at"]
            reclaimable = status == "running" and (
                not started_at or started_at < stale_before
            )
            if status not in {"queued", "partial"} and not reclaimable:
                connection.rollback()
                return False
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, finished_at = NULL
                WHERE job_id = ?
                """,
                (now_text, job_id),
            )
            connection.commit()
        return True

    def update_job(self, job_id: str, **values: Any) -> dict[str, Any] | None:
        allowed = {
            "status",
            "started_at",
            "finished_at",
            "total",
            "completed",
            "pending",
            "failed",
            "stale",
            "coverage",
            "last_error",
            "metadata_json",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "metadata" in values:
            updates["metadata_json"] = json.dumps(
                values["metadata"],
                ensure_ascii=False,
                sort_keys=True,
            )
        if not updates:
            return self.get_job(job_id)
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = ?",
                [*updates.values(), job_id],
            )
        return self.get_job(job_id)

    def update_job_if(
        self,
        job_id: str,
        *,
        expected_status: str | None = None,
        expected_started_at: str | None = None,
        **values: Any,
    ) -> bool:
        """Update a job only while its observed worker generation is unchanged."""
        allowed = {
            "status",
            "started_at",
            "finished_at",
            "total",
            "completed",
            "pending",
            "failed",
            "stale",
            "coverage",
            "last_error",
            "metadata_json",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        if "metadata" in values:
            updates["metadata_json"] = json.dumps(
                values["metadata"],
                ensure_ascii=False,
                sort_keys=True,
            )
        if not updates:
            return False
        assignments = ", ".join(f"{key} = ?" for key in updates)
        conditions = ["job_id = ?"]
        parameters: list[Any] = [*updates.values(), job_id]
        if expected_status is not None:
            conditions.append("status = ?")
            parameters.append(expected_status)
        if expected_started_at is not None:
            conditions.append("started_at = ?")
            parameters.append(expected_started_at)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            result = connection.execute(
                f"UPDATE jobs SET {assignments} WHERE {' AND '.join(conditions)}",
                parameters,
            )
            connection.commit()
        return result.rowcount == 1

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        value = dict(row)
        try:
            value["metadata"] = json.loads(value.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            value["metadata"] = {}
        return value

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [self.get_job(row["job_id"]) for row in rows if self.get_job(row["job_id"])]

    def list_assets(
        self,
        asset_types: Iterable[str] | None = None,
        *,
        statuses: Iterable[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return asset state rows for monitoring and operator tooling."""
        # Asset type names are semantic keys, not market symbols.  Preserve
        # their spelling so callers can query the lowercase schema values.
        types = sorted(
            {
                str(value).strip()
                for value in (asset_types or [])
                if str(value).strip()
            }
        )
        status_values = sorted(
            {str(value).strip() for value in (statuses or []) if str(value).strip()}
        )
        clauses: list[str] = []
        parameters: list[Any] = []
        if types:
            placeholders = ",".join("?" for _ in types)
            clauses.append(f"asset_type IN ({placeholders})")
            parameters.extend(types)
        if status_values:
            placeholders = ",".join("?" for _ in status_values)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status_values)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM assets
                {where}
                ORDER BY updated_at DESC, asset_type, symbol
                LIMIT ?
                """,
                [*parameters, max(1, int(limit))],
            ).fetchall()
        return [dict(row) for row in rows]

    def defer_retry(
        self,
        asset_type: str,
        symbol: str,
        *,
        next_retry_at: str,
        reason: str | None = None,
    ) -> bool:
        """Delay a failed asset without changing its attempt counter."""
        now = iso_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE assets
                SET next_retry_at = ?,
                    last_error = COALESCE(?, last_error),
                    updated_at = ?
                WHERE asset_type = ? AND symbol = ? AND status = 'failed'
                """,
                (
                    next_retry_at,
                    reason,
                    now,
                    asset_type,
                    symbol,
                ),
            )
        return cursor.rowcount > 0
