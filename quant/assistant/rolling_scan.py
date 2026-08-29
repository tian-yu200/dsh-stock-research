"""滚动扫描分析层：全市场抓取进行中"边扫边算"的初选股票池。

按波次增量计算单票技术面指标（signals._market_metrics），把得分写入独立的
``data/rolling_scan.sqlite3``，在已扫描子集内维护一个滚动初选池并记录入池/
出池事件，最后原子发布 ``results/assistant/rolling_scan.json`` 快照。

设计边界：
- 单票得分只依赖该票自身的 CSV，不随扫描进度漂移，保证事件流可回放；
  横截面排名（peer/benchmark/qlib 模型）仍属于覆盖率达标后 run_research
  的完整研究，本模块输出一律标记 ``preliminary: true``。
- 只以只读方式访问 fetch_state.sqlite3 的 assets 表判断"哪些票已成功、
  数据截止日是否更新"，不参与抓取队列写入，避免与抓取进程竞争。
- 数据行变更（data_end_date 变化）触发重算，未变则沿用已有得分。
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from . import data_layer, research, signals
from .paths import atomic_write_json, fixed_path

try:
    from scripts.universe import load_universe_manifest, manifest_names
except ImportError:  # pragma: no cover - 直接以脚本方式运行时的回退
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from universe import load_universe_manifest, manifest_names


DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "state_db": "data/rolling_scan.sqlite3",
    "snapshot_path": "results/assistant/rolling_scan.json",
    "pool_asset_types": ["stock"],
    "pool_size": 30,
    "min_sample": 200,
    "min_rows": 120,
    "min_average_amount_20d": 30000000.0,
    "stale_after_days": 15,
    "max_symbols_per_pass": 400,
    "score_table_limit": 500,
}

SCORE_FIELDS = (
    "trend_score",
    "return_20d",
    "return_60d",
    "return_252d",
    "volatility_60d_annualized",
    "max_drawdown",
    "current_drawdown",
    "ma20",
    "ma60",
    "ma200",
    "average_amount_20d",
    "average_volume_20d",
)

_EVENT_HISTORY_LIMIT = 50


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _settings(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_SETTINGS)
    merged.update((config or {}).get("rolling_scan") or {})
    pool_types = merged.get("pool_asset_types")
    if isinstance(pool_types, str):
        merged["pool_asset_types"] = [pool_types]
    return merged


def _load_frame(asset_type: str, symbol: str) -> pd.DataFrame:
    path = data_layer._data_file(asset_type, symbol)
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    columns = {str(column).strip(): column for column in raw.columns}
    date_column = next(
        (columns[name] for name in ("date", "日期", "净值日期") if name in columns),
        None,
    )
    close_column = next(
        (columns[name] for name in ("close", "单位净值") if name in columns),
        None,
    )
    if date_column is None or close_column is None:
        raise ValueError(f"{path}: missing date/close columns")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_column], errors="coerce"),
            "close": pd.to_numeric(raw[close_column], errors="coerce"),
        }
    )
    for name in ("volume", "amount"):
        if name in columns:
            frame[name] = pd.to_numeric(raw[columns[name]], errors="coerce")
    return (
        frame.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _read_state(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read success rows and per-type totals from the fetch state (read-only)."""
    try:
        uri = f"{path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=30)
    except sqlite3.OperationalError:
        return [], {}
    try:
        connection.row_factory = sqlite3.Row
        totals = {
            str(row["asset_type"]): {
                "total": int(row["total"]),
                "success": int(row["success"] or 0),
            }
            for row in connection.execute(
                """
                SELECT asset_type,
                       COUNT(*) AS total,
                       SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success
                FROM assets GROUP BY asset_type
                """
            ).fetchall()
        }
        rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT asset_type, symbol, data_end_date FROM assets
                WHERE status = 'success'
                ORDER BY asset_type, symbol
                """
            ).fetchall()
        ]
    finally:
        connection.close()
    return rows, totals


def _coverage(totals: dict[str, dict[str, int]]) -> dict[str, Any]:
    total = sum(int(value["total"]) for value in totals.values())
    success = sum(int(value["success"]) for value in totals.values())
    return {
        "by_asset_type": totals,
        "total": total,
        "success": success,
        "coverage": round(success / total, 4) if total else 0.0,
    }


class RollingScanStore:
    """SQLite persistence for per-symbol rolling scores, pool events, and meta."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.execute("PRAGMA journal_mode = WAL")
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    asset_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    data_end_date TEXT,
                    rows INTEGER,
                    name TEXT,
                    quality_status TEXT,
                    analyzed_at TEXT NOT NULL,
                    trend_score REAL,
                    return_20d REAL,
                    return_60d REAL,
                    return_252d REAL,
                    volatility_60d_annualized REAL,
                    max_drawdown REAL,
                    current_drawdown REAL,
                    ma20 REAL,
                    ma60 REAL,
                    ma200 REAL,
                    average_amount_20d REAL,
                    average_volume_20d REAL,
                    PRIMARY KEY (asset_type, symbol)
                );

                CREATE TABLE IF NOT EXISTS pool_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    job_id TEXT,
                    event TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    rank INTEGER,
                    score REAL,
                    reason TEXT
                );

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def scored_end_dates(self) -> dict[tuple[str, str], str | None]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT asset_type, symbol, data_end_date FROM scores"
            ).fetchall()
        return {
            (str(row["asset_type"]), str(row["symbol"])): row["data_end_date"]
            for row in rows
        }

    def all_scores(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            return [
                dict(row) for row in connection.execute("SELECT * FROM scores").fetchall()
            ]

    def upsert_scores(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        columns = [
            "asset_type",
            "symbol",
            "data_end_date",
            "rows",
            "name",
            "quality_status",
            "analyzed_at",
            *SCORE_FIELDS,
        ]
        statement = """
            INSERT INTO scores ({columns})
            VALUES ({placeholders})
            ON CONFLICT(asset_type, symbol) DO UPDATE SET
                {updates}
        """.format(
            columns=", ".join(columns),
            placeholders=", ".join("?" for _ in columns),
            updates=", ".join(
                f"{column} = excluded.{column}" for column in columns[2:]
            ),
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for record in records:
                connection.execute(
                    statement,
                    [record.get(column) for column in columns],
                )
            connection.commit()

    def append_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.executemany(
                """
                INSERT INTO pool_events (
                    created_at, job_id, event, asset_type, symbol, rank, score, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event["created_at"],
                        event.get("job_id"),
                        event["event"],
                        event["asset_type"],
                        event["symbol"],
                        event.get("rank"),
                        event.get("score"),
                        event.get("reason"),
                    )
                    for event in events
                ],
            )
            connection.commit()

    def recent_events(self, limit: int = _EVENT_HISTORY_LIMIT) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, job_id, event, asset_type, symbol, rank, score, reason
                FROM pool_events ORDER BY seq DESC LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_meta(self, key: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = ?", (key,)
            ).fetchone()
        return str(row["value"]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO meta (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )


def _analyze_one(
    row: dict[str, Any],
    names: dict[str, str],
) -> dict[str, Any]:
    asset_type = str(row["asset_type"])
    symbol = str(row["symbol"])
    record: dict[str, Any] = {
        "asset_type": asset_type,
        "symbol": symbol,
        "data_end_date": row.get("data_end_date"),
        "rows": None,
        "name": names.get(symbol),
        "quality_status": None,
        "analyzed_at": utc_now_iso(),
        **{field: None for field in SCORE_FIELDS},
    }
    try:
        frame = _load_frame(asset_type, symbol)
        metrics: dict[str, Any] = dict(
            signals._market_metrics(frame, fund=asset_type == "otc_fund")
        )
        if asset_type in {"stock", "etf"}:
            metrics.update(signals._liquidity_metrics(frame))
    except Exception as exc:
        record["quality_status"] = f"error: {type(exc).__name__}: {exc}"[:200]
        return record
    record.update(
        {
            "data_end_date": research._iso_date(frame["date"].iloc[-1]),
            "rows": int(len(frame)),
            "quality_status": "ok",
            **{field: metrics.get(field) for field in SCORE_FIELDS},
        }
    )
    return record


def _rank_scores(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    today: date,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return eligible rows ranked by trend_score plus ineligibility reasons."""
    pool_types = set(settings["pool_asset_types"])
    min_rows = int(settings["min_rows"])
    min_amount = float(settings["min_average_amount_20d"])
    stale_after = int(settings["stale_after_days"])
    eligible: list[dict[str, Any]] = []
    ineligible: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        if str(row["asset_type"]) not in pool_types:
            continue
        key = f"{row['asset_type']}:{row['symbol']}"
        reasons: list[str] = []
        if row.get("trend_score") is None:
            reasons.append("缺少技术面得分")
        if row.get("rows") is None or int(row["rows"]) < min_rows:
            reasons.append(f"历史数据不足 {min_rows} 行")
        end_date = row.get("data_end_date")
        age: int | None = None
        if end_date:
            try:
                age = (today - date.fromisoformat(str(end_date))).days
            except ValueError:
                age = None
        if age is None or age > stale_after:
            reasons.append("数据过期")
        amount = row.get("average_amount_20d")
        if amount is None or float(amount) < min_amount:
            reasons.append(f"20日日均成交额低于 {min_amount:.0f} 元")
        if reasons:
            ineligible[key] = "；".join(reasons)
        else:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (-float(row["trend_score"]), str(row["symbol"]))
    )
    return eligible, ineligible


def _pool_item(row: dict[str, Any], rank: int) -> dict[str, Any]:
    item: dict[str, Any] = {
        "rank": rank,
        "asset_type": row["asset_type"],
        "symbol": row["symbol"],
        "name": row.get("name"),
    }
    item.update({field: row.get(field) for field in SCORE_FIELDS})
    item.update(
        {
            "data_end_date": row.get("data_end_date"),
            "rows": row.get("rows"),
            "analyzed_at": row.get("analyzed_at"),
        }
    )
    return item


def run_rolling_pass(
    job_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Run one incremental rolling-scan pass and publish the snapshot.

    Analyzes symbols whose fetch succeeded and whose ``data_end_date`` changed
    since the last pass, re-ranks the eligible subset, diffs the pool against
    the previous pass to append enter/exit events, and atomically rewrites the
    rolling snapshot JSON. Never touches the fetch queue.
    """
    config = config or research.load_config()
    settings = _settings(config)
    started = time.monotonic()
    state_rows, totals = _read_state(Path(config["fetch"]["state_db"]))
    coverage = _coverage(totals)
    store = RollingScanStore(fixed_path(settings["state_db"]))

    try:
        manifest = load_universe_manifest(research.UNIVERSE_PATH)
    except Exception:
        manifest = None
    names_by_type = {
        asset_type: manifest_names(manifest, asset_type)
        for asset_type in ("stock", "etf", "otc_fund")
    }

    scored = store.scored_end_dates()
    pending = [
        row
        for row in state_rows
        if scored.get((str(row["asset_type"]), str(row["symbol"])))
        != row.get("data_end_date")
    ]
    pending_total = len(pending)
    cap = pending_total if max_symbols is not None else settings["max_symbols_per_pass"]
    selected = pending[: max(0, int(cap))]
    analyzed = [_analyze_one(row, names_by_type.get(str(row["asset_type"]), {})) for row in selected]
    store.upsert_scores(analyzed)

    all_rows = store.all_scores()
    eligible, ineligible = _rank_scores(all_rows, settings, datetime.now(timezone.utc).date())
    min_sample = int(settings["min_sample"])
    pool_size = int(settings["pool_size"])
    pool_status = "active" if len(eligible) >= min_sample else "accumulating"
    pool_keys = [
        f"{row['asset_type']}:{row['symbol']}" for row in eligible[:pool_size]
    ] if pool_status == "active" else []

    events: list[dict[str, Any]] = []
    previous_raw = store.get_meta("pool_state")
    previous = json.loads(previous_raw) if previous_raw else []
    if pool_status == "active":
        rank_by_key = {
            f"{row['asset_type']}:{row['symbol']}": rank
            for rank, row in enumerate(eligible, start=1)
        }
        new_set = set(pool_keys)
        previous_set = set(previous)
        for rank, key in enumerate(pool_keys, start=1):
            if key in previous_set:
                continue
            asset_type, symbol = key.split(":", 1)
            events.append(
                {
                    "created_at": utc_now_iso(),
                    "job_id": job_id,
                    "event": "enter",
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "rank": rank,
                    "score": eligible[rank - 1].get("trend_score"),
                    "reason": f"trend_score={float(eligible[rank - 1]['trend_score']):.4f}，初选排名第 {rank} 位",
                }
            )
        for key in previous:
            if key in new_set:
                continue
            asset_type, symbol = key.split(":", 1)
            reason = ineligible.get(key)
            if not reason:
                rank = rank_by_key.get(key)
                reason = (
                    f"初选排名第 {rank} 位，跌出前 {pool_size}"
                    if rank
                    else "不再符合入池条件"
                )
            events.append(
                {
                    "created_at": utc_now_iso(),
                    "job_id": job_id,
                    "event": "exit",
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "rank": rank_by_key.get(key),
                    "score": None,
                    "reason": reason,
                }
            )
    store.append_events(events)
    store.set_meta("pool_state", json.dumps(pool_keys))
    store.set_meta("last_pass_at", utc_now_iso())
    if job_id:
        store.set_meta("last_job_id", str(job_id))

    snapshot_path = fixed_path(settings["snapshot_path"])
    snapshot = {
        "status": "ok",
        "pool_status": pool_status,
        "preliminary": True,
        "job_id": job_id,
        "updated_at": utc_now_iso(),
        "coverage": coverage,
        "scored_total": len(all_rows),
        "eligible_count": len(eligible),
        "analyzed_this_pass": len(analyzed),
        "pending_remaining": pending_total - len(selected),
        "pool": [
            _pool_item(row, rank)
            for rank, row in enumerate(eligible[:pool_size], start=1)
        ]
        if pool_status == "active"
        else [],
        "recent_events": store.recent_events(_EVENT_HISTORY_LIMIT),
        "score_table": [
            _pool_item(row, rank)
            for rank, row in enumerate(eligible[: int(settings["score_table_limit"])], start=1)
        ],
        "settings": {
            key: settings[key]
            for key in (
                "pool_asset_types",
                "pool_size",
                "min_sample",
                "min_rows",
                "min_average_amount_20d",
                "stale_after_days",
            )
        },
        "disclaimer": (
            "初步名单：仅基于已扫描标的的单票技术面得分，扫描顺序有偏、随进度滚动变化；"
            "最终建议以全市场覆盖后的完整研究（qlib 模型与风控闸门）为准。"
        ),
    }
    atomic_write_json(snapshot_path, snapshot)
    return {
        "status": "ok",
        "job_id": job_id,
        "pool_status": pool_status,
        "analyzed_this_pass": len(analyzed),
        "pending_remaining": pending_total - len(selected),
        "scored_total": len(all_rows),
        "eligible_count": len(eligible),
        "pool_size": len(pool_keys),
        "events_added": len(events),
        "coverage": coverage,
        "snapshot_path": str(snapshot_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
