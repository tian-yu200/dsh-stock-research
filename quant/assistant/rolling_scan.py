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
- 数据行变更（data_end_date / rows / source 任一变化）触发重算：CSV 被整体重写
  （如跨源复权基准切换）时 data_end_date 往往不变，只有内容指纹能识别，
  这是 2026-09-02 毒分数事故（499 条 r20=31756% 的分数永不重算）的教训。
- 快照按 data_end_date 相对全市场基准交易日的滞后分层层展示覆盖率；
  落后 >= max_trading_lag_days 个交易日的标的踢出海选池。
"""
from __future__ import annotations

import bisect
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
from .cross_border import is_cross_border
from .paths import atomic_write_json, fixed_path

try:
    from scripts.universe import load_universe_manifest, manifest_names
except ImportError:  # pragma: no cover - 直接以脚本方式运行时的回退
    scripts_root = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from universe import load_universe_manifest, manifest_names

try:
    from scripts import silent_trace
except ImportError:  # pragma: no cover - 直接以脚本方式运行时的回退
    import silent_trace


DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "state_db": "data/rolling_scan.sqlite3",
    "snapshot_path": "results/assistant/rolling_scan.json",
    "pool_asset_types": ["stock", "etf"],
    "pool_size": 30,
    "min_sample": 200,
    "min_rows": 120,
    "min_average_amount_20d": 30000000.0,
    "stale_after_days": 15,
    "max_symbols_per_pass": 800,
    "score_table_limit": 500,
    # 行情新鲜度：落后基准交易日达到该数值的标的踢出海选池。
    "max_trading_lag_days": 2,
    # 物理合理性门槛：A股日线（后复权）不可能出现的常态外数值，
    # 一旦出现几乎必然是复权基准断裂残留（2026-09-02 事故特征）。
    "max_abs_return_20d": 1.5,
    "max_abs_return_60d": 1.5,
    "max_volatility_60d": 3.0,
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
        .loc[lambda value: value["close"].gt(0)]
        .sort_values("date", kind="stable")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )


def _file_revision(asset_type: str, symbol: str) -> str:
    path = data_layer._data_file(asset_type, symbol)
    revision = []
    for candidate in (path, path.with_suffix(".meta.json")):
        try:
            stat = candidate.stat()
            revision.append(f"{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            revision.append("missing")
    return "|".join(revision)


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
                SELECT asset_type, symbol, data_end_date, rows, source FROM assets
                WHERE status = 'success'
                ORDER BY asset_type, symbol
                """
            ).fetchall()
        ]
    finally:
        connection.close()
    # 2026-09-07 跨境排除：etf/otc_fund 美港QDII不计入统计（复用 pool_exclude_cross_border）
    try:
        cfg = research.load_config()  # type: ignore[attr-defined]
    except Exception:
        cfg = {}
    research_cfg = (cfg.get("research") or {}) if isinstance(cfg, dict) else {}
    exclude_rs = bool(research_cfg.get("universe_exclude_cross_border", False) or research_cfg.get("pool_exclude_cross_border", False))
    if exclude_rs:
        from pathlib import Path as _P
        try:
            manifest = load_universe_manifest(research.UNIVERSE_PATH)
            # 按类型取 name 字典
            import itertools
            names_by_type = {}
            for at in ("etf", "otc_fund"):
                try:
                    names_by_type[at] = manifest_names(manifest, at)
                except Exception:
                    names_by_type[at] = {}
            kws = research_cfg.get("cross_border_keywords") or None
            # 过滤 rows
            filtered_rows = []
            for r in rows:
                at = str(r["asset_type"])
                if at not in ("etf", "otc_fund"):
                    filtered_rows.append(r)
                else:
                    nm = names_by_type.get(at, {}).get(str(r["symbol"]), "")
                    if not is_cross_border(nm, asset_type=at, keywords=kws):
                        filtered_rows.append(r)
            rows = filtered_rows
            # 重算 totals 排除跨境分母：重查全量 assets 并按 name 过滤（fail-open）
            try:
                import sqlite3 as _sql
                uri2 = f"{research.DATA_DIR / 'fetch_state.sqlite3'}"
                # 使用之前的 path 参数对应的 fetch_state 路径
                fetch_path = path if "fetch_state" in str(path) else (research.DATA_DIR / "fetch_state.sqlite3")
                # rolling _read_state的path是fetch_state.sqlite3
                try:
                    uri_all = f"{fetch_path.resolve().as_uri()}?mode=ro"
                    conn2 = _sql.connect(uri_all, uri=True, timeout=30)
                    conn2.row_factory = _sql.Row
                    all_rows = conn2.execute("SELECT asset_type, symbol FROM assets").fetchall()
                    conn2.close()
                    new_totals = {}
                    for at in set(str(r["asset_type"]) for r in all_rows):
                        nms = names_by_type.get(at, {})
                        cnt_total = 0
                        for r in all_rows:
                            if str(r["asset_type"]) != at:
                                continue
                            nm = nms.get(str(r["symbol"]), "")
                            if at in ("etf", "otc_fund") and is_cross_border(nm, asset_type=at, keywords=kws):
                                continue
                            cnt_total += 1
                        cnt_succ = len([r for r in rows if str(r["asset_type"])==at])
                        new_totals[at] = {"total": cnt_total, "success": cnt_succ}
                    # 保留 stock 等未过滤类型的 totals 若新计算为空则用原值
                    for at, val in list(totals.items()):
                        if at not in new_totals:
                            new_totals[at] = val
                    totals = new_totals
                except Exception:
                    # 退化：仅修正 success
                    for at in list(totals.keys()):
                        if at in ("etf", "otc_fund"):
                            totals[at]["success"] = len([r for r in rows if str(r["asset_type"])==at])
            except Exception as exc:
                silent_trace.trace("rolling_scan", "_read_state.totals_recompute", exc)
                pass
        except Exception as exc:
            silent_trace.trace("rolling_scan", "_read_state.read", exc)
            pass
    for row in rows:
        row["file_revision"] = _file_revision(str(row["asset_type"]), str(row["symbol"]))
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
                    source TEXT,
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
            # 2026-09 事故后新增的内容指纹列：旧库平滑迁移。
            existing_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(scores)")
            }
            if "source" not in existing_columns:
                connection.execute("ALTER TABLE scores ADD COLUMN source TEXT")
            if "file_revision" not in existing_columns:
                connection.execute("ALTER TABLE scores ADD COLUMN file_revision TEXT")

    def scored_end_dates(self) -> dict[tuple[str, str], str | None]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT asset_type, symbol, data_end_date FROM scores"
            ).fetchall()
        return {
            (str(row["asset_type"]), str(row["symbol"])): row["data_end_date"]
            for row in rows
        }

    def scored_fingerprints(
        self,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Scored state plus file revision, including same-date CSV corrections."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT asset_type, symbol, data_end_date, rows, source, file_revision FROM scores"
            ).fetchall()
        return {
            (str(row["asset_type"]), str(row["symbol"])): {
                "end_date": row["data_end_date"],
                "rows": row["rows"],
                "source": row["source"],
                "file_revision": row["file_revision"],
            }
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
            "source",
            "file_revision",
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


def _cached_calendar_dates(basis: str) -> set[str]:
    """读取本地交易日历缓存（只读、不联网）；缺失或未覆盖基准日时返回空集。

    交易日历缓存在量化内核里可随时离线重建（scripts/trading_calendar.py），
    所以这里宁可退化为"用全市场观测日期近似交易日"，也不引入网络依赖。
    """
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "trade_calendar.csv"
    try:
        raw = pd.read_csv(path)
    except Exception:
        return set()
    values = pd.to_datetime(raw.iloc[:, 0], errors="coerce").dropna()
    dates = sorted({value.strftime("%Y-%m-%d") for value in values})
    if not dates or dates[-1] < basis:
        return set()
    return {value for value in dates if value <= basis}


def _trading_lag_map(
    state_rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[str | None, dict[tuple[str, str], int], str]:
    """计算每只标的相对全市场基准交易日的交易日滞后。

    基准日 = 池内资产类型（股票/ETF）已成功标的的最大 data_end_date。
    返回 ``(基准日, {(asset_type, symbol): lag}, 日历来源)``。
    """
    pool_types = {str(value) for value in settings["pool_asset_types"]}
    dated = [row for row in state_rows if row.get("data_end_date")]
    if not dated:
        return None, {}, "none"
    pool_dates = [
        str(row["data_end_date"]) for row in dated if str(row["asset_type"]) in pool_types
    ]
    basis = max(pool_dates, default=max(str(row["data_end_date"]) for row in dated))
    observed = {str(row["data_end_date"]) for row in dated}
    calendar = _cached_calendar_dates(basis)
    union = sorted(calendar | observed)
    if basis not in union:
        union.append(basis)
        union.sort()
    index_by_date = {value: position for position, value in enumerate(union)}
    basis_position = index_by_date[basis]
    lags: dict[tuple[str, str], int] = {}
    for row in dated:
        value = str(row["data_end_date"])
        position = index_by_date.get(value)
        if position is None:
            position = bisect.bisect_left(union, value)
        lags[(str(row["asset_type"]), str(row["symbol"]))] = max(0, basis_position - position)
    return basis, lags, "calendar_cache" if calendar else "observed_only"


_FRESHNESS_LAYERS: tuple[tuple[str, str, int, int | None], ...] = (
    ("current", "当日", 0, 0),
    ("lag1", "落后1个交易日", 1, 1),
    ("lag2", "落后2个交易日", 2, 2),
    ("lag3_4", "落后3-4个交易日", 3, 4),
    ("lag5_plus", "落后5个交易日以上", 5, None),
)


def _freshness_summary(
    state_rows: list[dict[str, Any]],
    lags: dict[tuple[str, str], int],
    basis: str | None,
    calendar_source: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """按 data_end_date 滞后分层汇总覆盖率（供界面展示与告警）。"""
    max_lag = int(settings["max_trading_lag_days"])
    total = len(state_rows)

    def bucket(lag: int | None) -> str:
        if lag is None:
            return "unknown"
        for key, _, low, high in _FRESHNESS_LAYERS:
            if lag >= low and (high is None or lag <= high):
                return key
        return "unknown"

    counts: dict[str, int] = {key: 0 for key, _, _, _ in _FRESHNESS_LAYERS}
    counts["unknown"] = 0
    by_type: dict[str, dict[str, int]] = {}
    stale_count = 0
    for row in state_rows:
        asset_type = str(row["asset_type"])
        lag = lags.get((asset_type, str(row["symbol"])))
        if lag is not None and lag >= max_lag:
            stale_count += 1
        key = bucket(lag)
        counts[key] += 1
        by_type.setdefault(asset_type, {})[key] = by_type.get(asset_type, {}).get(key, 0) + 1
    layers = [
        {
            "key": key,
            "label": label,
            "min_lag": low,
            "max_lag": high,
            "count": counts[key],
            "ratio": round(counts[key] / total, 4) if total else 0.0,
        }
        for key, label, low, high in _FRESHNESS_LAYERS
    ]
    layers.append(
        {
            "key": "unknown",
            "label": "日期缺失",
            "min_lag": None,
            "max_lag": None,
            "count": counts["unknown"],
            "ratio": round(counts["unknown"] / total, 4) if total else 0.0,
        }
    )
    return {
        "basis_date": basis,
        "basis_note": "池内资产类型已成功标的的最大 data_end_date",
        "calendar_source": calendar_source,
        "total": total,
        "stale_after_trading_days": max_lag,
        "stale_count": stale_count,
        "stale_ratio": round(stale_count / total, 4) if total else 0.0,
        "layers": layers,
        "by_asset_type": by_type,
    }


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
        "source": row.get("source"),
        "file_revision": row.get("file_revision"),
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
    lags: dict[tuple[str, str], int] | None = None,
) -> dict[str, tuple[list[dict[str, Any]], dict[str, str]]]:
    """Rank eligible rows within each pool asset type by trend_score.

    Returns ``{asset_type: (ranked_rows, ineligibility_reasons)}`` so every
    pool type opens its own top-N once it has enough eligible samples.
    """
    lags = lags or {}
    return {
        str(asset_type): _rank_one_type(
            [dict(row) for row in rows if str(row["asset_type"]) == str(asset_type)],
            settings,
            today,
            lags,
        )
        for asset_type in settings["pool_asset_types"]
    }


def _rank_one_type(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    today: date,
    lags: dict[tuple[str, str], int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    min_rows = int(settings["min_rows"])
    min_amount = float(settings["min_average_amount_20d"])
    stale_after = int(settings["stale_after_days"])
    max_trading_lag = int(settings["max_trading_lag_days"])
    lags = lags or {}
    eligible: list[dict[str, Any]] = []
    ineligible: dict[str, str] = {}
    for row in rows:
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
        lag = lags.get((str(row["asset_type"]), str(row["symbol"])))
        if lag is not None and lag >= max_trading_lag:
            reasons.append(f"行情落后 {lag} 个交易日")
        amount = row.get("average_amount_20d")
        if amount is None or float(amount) < min_amount:
            reasons.append(f"20日日均成交额低于 {min_amount:.0f} 元")
        # 物理合理性门槛：后复权日线不可能出现的常态外数值，几乎必然是
        # 复权基准断裂残留（2026-09-02 事故），绝不能当成趋势机会。
        for field, limit, label in (
            ("return_20d", settings["max_abs_return_20d"], "20日收益"),
            ("return_60d", settings["max_abs_return_60d"], "60日收益"),
        ):
            value = row.get(field)
            if value is not None and abs(float(value)) > float(limit):
                reasons.append(f"{label}超出物理合理范围")
        volatility = row.get("volatility_60d_annualized")
        if (
            volatility is not None
            and float(volatility) > float(settings["max_volatility_60d"])
        ):
            reasons.append("60日波动率超出物理合理范围")
        if reasons:
            ineligible[key] = "；".join(reasons)
        else:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (-float(row["trend_score"]), str(row["symbol"]))
    )
    return eligible, ineligible


def _pool_item(
    row: dict[str, Any],
    rank: int,
    lag: int | None = None,
) -> dict[str, Any]:
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
            "trading_lag_days": lag,
        }
    )
    return item


def _fingerprint_changed(
    fingerprint: dict[str, Any] | None,
    state_row: dict[str, Any],
) -> bool:
    """判断已打分指纹与抓取状态是否不一致（不一致则需要重算）。

    只在两侧都有值时比较 rows/source：NULL 侧（历史记录或异常记录）
    不参与比较，避免错误记录每轮 pass 被无意义地重算。
    """
    if fingerprint is None:
        return True
    revision = state_row.get("file_revision")
    if revision is not None and fingerprint.get("file_revision") != revision:
        return True
    if fingerprint.get("end_date") != state_row.get("data_end_date"):
        return True
    scored_rows = fingerprint.get("rows")
    state_rows_count = state_row.get("rows")
    if scored_rows is not None and state_rows_count is not None:
        if int(scored_rows) != int(state_rows_count):
            return True
    scored_source = fingerprint.get("source")
    state_source = state_row.get("source")
    if scored_source is not None and state_source is not None:
        if str(scored_source) != str(state_source):
            return True
    return False


def run_rolling_pass(
    job_id: str | None = None,
    *,
    config: dict[str, Any] | None = None,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Run one incremental rolling-scan pass and publish the snapshot.

    Analyzes symbols whose fetch succeeded and whose data fingerprint
    (``data_end_date`` + ``rows`` + ``source``) changed since the last pass,
    re-ranks the eligible subset, diffs the pool against the previous pass to
    append enter/exit events, and atomically rewrites the rolling snapshot
    JSON. Never touches the fetch queue.
    """
    config = config or research.load_config()
    settings = _settings(config)
    started = time.monotonic()
    state_rows, totals = _read_state(fixed_path(config["fetch"]["state_db"]))
    coverage = _coverage(totals)
    basis_date, lags, calendar_source = _trading_lag_map(state_rows, settings)
    freshness = _freshness_summary(state_rows, lags, basis_date, calendar_source, settings)
    store = RollingScanStore(fixed_path(settings["state_db"]))

    try:
        manifest = load_universe_manifest(research.UNIVERSE_PATH)
    except Exception:
        manifest = None
    names_by_type = {
        asset_type: manifest_names(manifest, asset_type)
        for asset_type in ("stock", "etf", "otc_fund")
    }

    scored = store.scored_fingerprints()
    pending = [
        row
        for row in state_rows
        if _fingerprint_changed(scored.get((str(row["asset_type"]), str(row["symbol"]))), row)
    ]
    # 池内资产类型优先分析，避免被未入池类型（如场外基金）的大批积压饿死。
    pool_first = {str(value) for value in settings["pool_asset_types"]}
    pending.sort(
        key=lambda row: (
            0 if str(row["asset_type"]) in pool_first else 1,
            str(row["asset_type"]),
            str(row["symbol"]),
        )
    )
    pending_total = len(pending)
    cap = pending_total if max_symbols is not None else settings["max_symbols_per_pass"]
    selected = pending[: max(0, int(cap))]
    analyzed = [_analyze_one(row, names_by_type.get(str(row["asset_type"]), {})) for row in selected]
    store.upsert_scores(analyzed)

    all_rows = store.all_scores()
    ranking = _rank_scores(
        all_rows, settings, datetime.now(timezone.utc).date(), lags
    )
    min_sample = int(settings["min_sample"])
    pool_size = int(settings["pool_size"])
    eligible_count = sum(len(ranked) for ranked, _ in ranking.values())
    eligible_by_type = {
        str(asset_type): len(ranked)
        for asset_type, (ranked, _) in ranking.items()
    }
    ineligible: dict[str, str] = {}
    for _, type_ineligible in ranking.values():
        ineligible.update(type_ineligible)
    active_types = {
        str(asset_type)
        for asset_type, (ranked, _) in ranking.items()
        if len(ranked) >= min_sample
    }
    pool_status = "active" if active_types else "accumulating"

    pool_keys: list[str] = []
    rank_by_key: dict[str, int] = {}
    rows_by_key: dict[str, dict[str, Any]] = {}
    for asset_type, (ranked, _) in ranking.items():
        if str(asset_type) not in active_types:
            continue
        for rank, row in enumerate(ranked, start=1):
            key = f"{row['asset_type']}:{row['symbol']}"
            rank_by_key[key] = rank
            if rank <= pool_size:
                pool_keys.append(key)
                rows_by_key[key] = row

    events: list[dict[str, Any]] = []
    previous_raw = store.get_meta("pool_state")
    previous = json.loads(previous_raw) if previous_raw else []
    if pool_status == "active":
        new_set = set(pool_keys)
        previous_set = set(previous)
        for key in pool_keys:
            if key in previous_set:
                continue
            asset_type, symbol = key.split(":", 1)
            row = rows_by_key[key]
            rank = rank_by_key[key]
            events.append(
                {
                    "created_at": utc_now_iso(),
                    "job_id": job_id,
                    "event": "enter",
                    "asset_type": asset_type,
                    "symbol": symbol,
                    "rank": rank,
                    "score": row.get("trend_score"),
                    "reason": (
                        f"trend_score={float(row['trend_score']):.4f}，"
                        f"{asset_type} 初选排名第 {rank} 位"
                    ),
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
        "eligible_count": eligible_count,
        "eligible_by_type": eligible_by_type,
        "analyzed_this_pass": len(analyzed),
        "pending_remaining": pending_total - len(selected),
        "pool": [
            _pool_item(
                rows_by_key[key],
                rank_by_key[key],
                lags.get((key.split(":", 1)[0], key.split(":", 1)[1])),
            )
            for key in pool_keys
        ],
        "recent_events": store.recent_events(_EVENT_HISTORY_LIMIT),
        "score_table": [
            _pool_item(
                row,
                rank,
                lags.get((str(row["asset_type"]), str(row["symbol"]))),
            )
            for asset_type, (ranked, _) in ranking.items()
            for rank, row in enumerate(
                ranked[: int(settings["score_table_limit"])], start=1
            )
        ],
        "freshness": freshness,
        "settings": {
            key: settings[key]
            for key in (
                "pool_asset_types",
                "pool_size",
                "min_sample",
                "min_rows",
                "min_average_amount_20d",
                "stale_after_days",
                "max_trading_lag_days",
                "max_abs_return_20d",
                "max_abs_return_60d",
                "max_volatility_60d",
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
        "eligible_count": eligible_count,
        "eligible_by_type": eligible_by_type,
        "pool_size": len(pool_keys),
        "events_added": len(events),
        "coverage": coverage,
        "freshness": freshness,
        "snapshot_path": str(snapshot_path),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
