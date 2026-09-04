# -*- coding: utf-8 -*-
"""Bounded, resumable A-share daily-data worker.

Each invocation processes only due symbols until the batch/time budget is
exhausted. CSV files remain the source of truth for market data; SQLite keeps
the durable queue and retry state.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from data_source import NoDailyBarsError, fetch_a_stock_daily
from market_data import DailyBarSchemaError
from fetch_state import DEFAULT_DB_PATH, FetchState
from fetch_utils import (
    RateLimiter,
    atomic_write_frame,
    atomic_write_json,
    cached_end_date,
    merge_daily_frames,
    next_start_date,
    install_request_timeout,
    target_date,
)
from universe import (
    discover_stock_universe,
    discover_stock_lifecycle,
    load_universe_manifest,
    manifest_symbols,
    normalize_symbol,
    write_universe_manifest,
)


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "stocks"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_PATH = DATA_DIR.parent / "universe.json"
DEFAULT_POOL = ["600519", "000001", "300750", "600036", "000858"]
LIFECYCLE_CACHE_SECONDS = 86400
INCOMPLETE_LIFECYCLE_CACHE_SECONDS = 3600


def is_allowed(code: str) -> bool:
    """Return whether a six-digit code belongs to a mainland A-share board."""
    symbol = normalize_symbol(code)
    if symbol is None:
        return False
    return symbol.startswith(
        (
            "000",
            "001",
            "002",
            "003",
            "300",
            "301",
            "302",
            "600",
            "601",
            "603",
            "605",
            "688",
            "689",
            "4",
            "8",
            "92",
        )
    )


def _manifest_is_authoritative() -> bool:
    manifest = load_universe_manifest(UNIVERSE_PATH)
    return bool(
        manifest
        and (
            manifest.get("complete")
            or manifest.get("stock_universe_complete", False)
        )
        and manifest_symbols(manifest, "stock")
    )


def discover_pool(
    data_dir: pathlib.Path = DATA_DIR,
    *,
    online: bool = False,
) -> list[str]:
    """Discover the complete cached catalog before making a network call."""
    if (
        not online
        and data_dir.resolve() == DATA_DIR.resolve()
        and _manifest_is_authoritative()
    ):
        manifest = load_universe_manifest(UNIVERSE_PATH) or {}
        return [
            item["symbol"]
            for item in manifest.get("stock", [])
            if isinstance(item, dict) and is_allowed(item.get("symbol", ""))
        ]
    if online:
        try:
            return [
                item["symbol"]
                for item in discover_stock_universe()
                if is_allowed(item["symbol"])
            ]
        except Exception as exc:
            print(f"[股票目录] 在线发现失败，回退本地缓存: {type(exc).__name__}")
    discovered = sorted(
        path.stem
        for path in data_dir.glob("*.csv")
        if len(path.stem) == 6 and path.stem.isdigit() and is_allowed(path.stem)
    )
    return discovered or list(DEFAULT_POOL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbols",
        nargs="*",
        help="可选的六位股票代码；省略时处理完整目录中的到期标的",
    )
    parser.add_argument("--workers", type=int, default=3, help="并发抓取数")
    parser.add_argument("--batch-size", type=int, default=100, help="单批最多领取的标的数")
    parser.add_argument(
        "--time-budget",
        type=float,
        default=600.0,
        help="本次运行最多抓取秒数；小于等于0表示不限制",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅用于验收时限制抓取数量")
    parser.add_argument("--target-date", default=None, help="数据目标日期，默认当前 UTC 日期")
    parser.add_argument("--state-db", type=pathlib.Path, default=DEFAULT_DB_PATH, help="增量状态 SQLite 文件")
    parser.add_argument("--retries", type=int, default=2, help="单标的网络重试次数")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="进程内请求速率，单位请求/秒")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="单次 HTTP 请求连接/读取超时秒数")
    parser.add_argument("--backoff-base", type=float, default=2.0, help="失败指数退避起始秒数")
    parser.add_argument("--backoff-max", type=float, default=120.0, help="失败指数退避最大秒数")
    parser.add_argument("--lease-seconds", type=int, default=1800, help="单标的租约秒数")
    parser.add_argument("--job-id", default=None, help="可选的研究任务 ID，用于刷新任务进度")
    parser.add_argument(
        "--requeue-failure-class",
        default=None,
        help="显式重放指定失败分类的隔离股票，例如 schema_conflict；默认不重放",
    )
    return parser


def _data_path(code: str) -> pathlib.Path:
    return DATA_DIR / f"{code}.csv"


def _write_success(code: str, frame: Any, path: pathlib.Path) -> dict[str, Any]:
    merged = merge_daily_frames(path, frame, source=f"stock {code}")
    atomic_write_frame(merged, path)
    latest_date = pd.to_datetime(merged["date"], errors="coerce").max()
    if pd.isna(latest_date):
        raise DailyBarSchemaError.for_source(
            f"stock {code}",
            "no valid date remains after price guards",
            failure_class="invalid_required_values",
            field_anomalies={"date": {"invalid_rows": int(len(merged))}},
        )
    metadata = {
        "asset_type": "stock",
        "symbol": code,
        "source": frame.attrs.get("akshare_source"),
        "interface": frame.attrs.get("akshare_interface"),
        "adjustment": frame.attrs.get("adjustment", "hfq"),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_end_date": pd.Timestamp(latest_date).date().isoformat(),
        "rows": int(len(merged)),
    }
    if frame.attrs.get("field_anomalies"):
        metadata["field_anomalies"] = frame.attrs["field_anomalies"]
    atomic_write_json(metadata, path.with_suffix(".meta.json"))
    return metadata


def _fetch_one(
    item: dict[str, Any],
    *,
    state: FetchState,
    limiter: RateLimiter,
    target: str,
    retries: int,
    backoff_base: float,
    backoff_max: float,
    lease_owner: str,
    deadline: float | None,
) -> tuple[str, bool, dict[str, Any] | None, str | None, dict[str, Any] | None]:
    code = str(item["symbol"])
    path = _data_path(code)
    start = next_start_date(
        cached_end_date(path) or item.get("data_end_date"),
        "20200101",
    )
    end = target.replace("-", "")
    attempts = max(1, int(retries) + 1)
    last_error: str | None = None
    confirmed_empty = False
    failure_info: dict[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        if deadline is not None and time.monotonic() >= deadline:
            last_error = "worker deadline reached before next request"
            failure_info = {"failure_class": "worker_deadline"}
            break
        try:
            limiter.wait()
            if deadline is not None and time.monotonic() >= deadline:
                last_error = "worker deadline reached while waiting for rate limit"
                failure_info = {"failure_class": "worker_deadline"}
                break
            frame = fetch_a_stock_daily(code, start=start, end=end)
            metadata = _write_success(code, frame, path)
            state.mark_success(
                "stock",
                code,
                data_end_date=metadata["data_end_date"],
                source=metadata.get("source"),
                rows=metadata["rows"],
                refresh_interval_seconds=86400,
                verified_through=target,
                owner=lease_owner,
            )
            return code, True, metadata, None, None
        except DailyBarSchemaError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            failure_info = {
                "failure_class": exc.failure_class,
                "field_anomalies": exc.field_anomalies,
            }
            state.mark_quarantined(
                "stock",
                code,
                error=last_error,
                failure_class=exc.failure_class,
                field_anomalies=exc.field_anomalies,
                owner=lease_owner,
            )
            return code, False, None, last_error, failure_info
        except NoDailyBarsError as exc:
            cached_end = cached_end_date(path)
            if cached_end:
                state.mark_success(
                    "stock",
                    code,
                    data_end_date=cached_end,
                    source="verified-empty-window",
                    rows=None,
                    refresh_interval_seconds=86400,
                    verified_through=target,
                    owner=lease_owner,
                )
                metadata = {
                    "asset_type": "stock",
                    "symbol": code,
                    "source": "verified-empty-window",
                    "data_end_date": cached_end,
                    "rows": None,
                    "verified_through": target,
                    "empty_sources": list(exc.sources),
                }
                return code, True, metadata, None, None
            last_error = f"{type(exc).__name__}: {exc}"
            failure_info = {"failure_class": "empty_window"}
            # 三个独立数据源一致确认无行情：重试不会改变结果，
            # 用 24 小时退避，避免每轮 wave 都重新确认全部死代码。
            confirmed_empty = True
            break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            failure_info = {"failure_class": "upstream_error"}
            if attempt < attempts:
                delay = min(
                    backoff_max,
                    max(0.0, backoff_base) * (2 ** (attempt - 1)),
                )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        last_error = "worker deadline reached before retry"
                        failure_info = {"failure_class": "worker_deadline"}
                        break
                    delay = min(delay, remaining)
                if delay:
                    time.sleep(delay)
    if confirmed_empty:
        failure_backoff_base = failure_backoff_max = 86400
    else:
        failure_backoff_base = max(1, int(backoff_base))
        failure_backoff_max = max(1, int(backoff_max))
    state.mark_failure(
        "stock",
        code,
        error=last_error or "unknown fetch error",
        backoff_base_seconds=failure_backoff_base,
        backoff_max_seconds=failure_backoff_max,
        failure_class=(failure_info or {}).get("failure_class"),
        field_anomalies=(failure_info or {}).get("field_anomalies"),
        owner=lease_owner,
    )
    return code, False, None, last_error, failure_info


def _refresh_manifest(
    existing: dict[str, Any] | None,
    *,
    limited: bool,
    requested: int,
    ok: int,
    failed: int,
) -> None:
    try:
        catalog = discover_stock_universe()
    except Exception as exc:
        print(f"[目录] 代码目录缓存失败: {type(exc).__name__}: {exc}")
        return
    existing = existing or {}
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "AkShare",
        "stock": catalog,
        "stock_universe_complete": not limited,
        "stock_refresh": {
            "requested": requested,
            "success": ok,
            "failed": failed,
            "limited": limited,
        },
    }
    for asset_type in ("etf", "otc_fund"):
        if asset_type in existing:
            snapshot[asset_type] = existing[asset_type]
        flag = f"{asset_type}_universe_complete"
        if flag in existing:
            snapshot[flag] = existing[flag]
    snapshot["counts"] = {
        asset_type: len(snapshot.get(asset_type, []))
        for asset_type in ("stock", "etf", "otc_fund")
    }
    snapshot["complete"] = all(
        snapshot.get(f"{asset_type}_universe_complete", False)
        for asset_type in ("stock", "etf", "otc_fund")
    )
    write_universe_manifest(UNIVERSE_PATH, snapshot)
    print(f"[目录] A股代码 {len(catalog)} 个，已缓存到 {UNIVERSE_PATH}")


def _lifecycle_snapshot_is_fresh(snapshot: dict[str, Any]) -> bool:
    generated_at = snapshot.get("generated_at")
    if not generated_at:
        return False
    try:
        generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    ttl = (
        LIFECYCLE_CACHE_SECONDS
        if snapshot.get("complete") is True
        else INCOMPLETE_LIFECYCLE_CACHE_SECONDS
    )
    age = (datetime.now(timezone.utc) - generated.astimezone(timezone.utc)).total_seconds()
    return 0 <= age < ttl


def _reconcile_stock_lifecycle(
    state: FetchState,
    codes: list[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    candidates = state.lifecycle_candidates("stock", codes)
    if not candidates:
        return {
            "candidate_count": 0,
            "checked_count": 0,
            "complete": True,
            "cached": False,
            "inactivated": 0,
            "revived": 0,
            "unresolved_count": 0,
            "deferred_count": 0,
            "errors": {},
        }

    cached_value = manifest.get("stock_lifecycle")
    cached_snapshot = cached_value if isinstance(cached_value, dict) else {}
    cached = _lifecycle_snapshot_is_fresh(cached_snapshot)
    if cached:
        snapshot = cached_snapshot
    else:
        try:
            snapshot = discover_stock_lifecycle(candidates)
        except Exception as exc:
            snapshot = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "complete": False,
                "candidates": candidates,
                "active": [],
                "unresolved_candidates": candidates,
                "inactive_candidates": [],
                "sources": {},
                "errors": {"discovery": f"{type(exc).__name__}: {exc}"},
            }
        write_universe_manifest(UNIVERSE_PATH, {"stock_lifecycle": snapshot})

    candidate_set = set(candidates)
    checked = candidate_set & {
        str(value) for value in snapshot.get("candidates", [])
    }
    scoped_snapshot = dict(snapshot)
    scoped_snapshot["candidates"] = sorted(checked)
    for field in ("active", "unresolved_candidates", "inactive_candidates"):
        scoped_snapshot[field] = sorted(
            candidate_set & {str(value) for value in snapshot.get(field, [])}
        )
    applied = state.apply_lifecycle_snapshot("stock", scoped_snapshot)
    result = {
        "candidate_count": len(candidates),
        "checked_count": len(checked),
        "complete": snapshot.get("complete") is True,
        "cached": cached,
        "inactivated": applied["inactivated"],
        "revived": applied["revived"],
        "unresolved_count": len(scoped_snapshot["unresolved_candidates"]),
        "deferred_count": len(set(candidates) - checked),
        "errors": snapshot.get("errors", {}),
    }
    print(
        "[生命周期] 候选 {candidate_count}，检查 {checked_count}，"
        "停用 {inactivated}，恢复 {revived}，待复核 {unresolved_count}，"
        "延后 {deferred_count}".format(**result)
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    target = target_date(args.target_date)
    install_request_timeout(float(args.request_timeout))
    explicit = bool(args.symbols)
    normalized = [normalize_symbol(value) for value in (args.symbols or [])]
    if any(value is None or not is_allowed(value) for value in normalized):
        raise ValueError("symbols must be valid mainland A-share codes")
    codes = [value for value in normalized if value is not None]
    if not codes:
        codes = discover_pool(online=False)
        if not _manifest_is_authoritative():
            codes = discover_pool(online=True)
    if args.limit is not None:
        codes = codes[: args.limit]

    state = FetchState(args.state_db)
    state.reclaim_orphaned_leases()
    data_end_dates = {code: cached_end_date(_data_path(code)) for code in codes}
    state.register_assets("stock", codes, data_end_dates=data_end_dates)
    cache_reconciled = state.reconcile_cached_assets("stock", data_end_dates)
    if cache_reconciled:
        print(f"[缓存协调] 从 CSV 恢复 {cache_reconciled} 个已落盘标的")
    manifest = load_universe_manifest(UNIVERSE_PATH) or {}
    lifecycle = _reconcile_stock_lifecycle(state, codes, manifest)
    requeued = 0
    if args.requeue_failure_class:
        requeued = state.requeue_quarantined(
            "stock",
            symbols=codes,
            failure_class=str(args.requeue_failure_class),
        )
        print(
            f"[重放] 已将 {requeued} 个 {args.requeue_failure_class} 隔离标的放回队列"
        )
    lease_owner = (
        f"stock-worker-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{os.getpid()}"
    )
    limiter = RateLimiter(float(args.rate_limit))
    workers = max(1, min(int(args.workers), 8))
    batch_size = max(1, min(int(args.batch_size), 500))
    deadline = (
        None
        if float(args.time_budget) <= 0
        else time.monotonic() + float(args.time_budget)
    )
    ok = 0
    fail = 0
    source_by_symbol: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    error_classes: dict[str, int] = {}
    field_anomalies: list[dict[str, Any]] = []
    claimed_limit = min(batch_size, workers * 2)

    while deadline is None or time.monotonic() < deadline:
        due = state.claim_due(
            "stock",
            symbols=codes,
            limit=claimed_limit,
            target_date=target,
            lease_seconds=max(30, int(args.lease_seconds)),
            owner=lease_owner,
        )
        if not due:
            break
        handled: set[Future[tuple[str, bool, dict[str, Any] | None, str | None, dict[str, Any] | None]]] = set()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fetch_one,
                    item,
                    state=state,
                    limiter=limiter,
                    target=target,
                    retries=args.retries,
                    backoff_base=args.backoff_base,
                    backoff_max=args.backoff_max,
                    lease_owner=lease_owner,
                    deadline=deadline,
                ): item["symbol"]
                for item in due
            }
            pending = set(futures)
            while pending:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    break
                done, pending = wait(
                    pending,
                    timeout=remaining,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    break
                for future in done:
                    handled.add(future)
                    code, succeeded, metadata, error, failure_info = future.result()
                    if succeeded and metadata:
                        ok += 1
                        source_by_symbol[code] = metadata
                        anomalies = metadata.get("field_anomalies")
                        if anomalies:
                            field_anomalies.append({"symbol": code, "fields": anomalies})
                        print(f"[完成] {code} -> {_data_path(code)}")
                    else:
                        fail += 1
                        errors[code] = error or "unknown error"
                        if failure_info:
                            failure_class = failure_info.get("failure_class")
                            if failure_class:
                                error_classes[failure_class] = error_classes.get(failure_class, 0) + 1
                            anomalies = failure_info.get("field_anomalies")
                            if anomalies:
                                field_anomalies.append({"symbol": code, "fields": anomalies})
                        print(f"[失败] {code}: {errors[code]}")
            cancelled_symbols = [
                str(futures[future])
                for future in pending
                if future.cancel()
            ]
            state.release_leases(
                "stock",
                cancelled_symbols,
                owner=lease_owner,
            )
        for future, symbol in futures.items():
            if future in handled or future.cancelled():
                continue
            code, succeeded, metadata, error, failure_info = future.result()
            if succeeded and metadata:
                ok += 1
                source_by_symbol[code] = metadata
                anomalies = metadata.get("field_anomalies")
                if anomalies:
                    field_anomalies.append({"symbol": code, "fields": anomalies})
                print(f"[完成] {code} -> {_data_path(code)}")
            else:
                fail += 1
                errors[code] = error or "unknown error"
                if failure_info:
                    failure_class = failure_info.get("failure_class")
                    if failure_class:
                        error_classes[failure_class] = error_classes.get(failure_class, 0) + 1
                    anomalies = failure_info.get("field_anomalies")
                    if anomalies:
                        field_anomalies.append({"symbol": code, "fields": anomalies})
                print(f"[失败] {code}: {errors[code]}")
        if deadline is not None and time.monotonic() >= deadline:
            break

    progress = state.progress(
        ("stock",),
        symbols={"stock": codes},
        target_date=target,
    )
    manifest = load_universe_manifest(UNIVERSE_PATH) or {}
    catalog_complete = bool(
        manifest.get("complete") or manifest.get("stock_universe_complete", False)
    )
    limited = bool(args.limit is not None)
    if not explicit and args.limit is None:
        _refresh_manifest(
            manifest,
            limited=False,
            requested=len(codes),
            ok=ok,
            failed=fail,
        )
    status = "complete" if progress["status"] == "complete" else "partial"
    if fail and ok == 0 and progress["completed"] == 0:
        status = "failed"
    summary = {
        "kind": "stock",
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "failed": fail,
        "requested": len(codes),
        "processed": ok + fail,
        "pending": progress["pending"],
        "progress": progress,
        "catalog_coverage": {
            "complete": catalog_complete and args.limit is None,
            "catalog_count": len(manifest.get("stock", [])),
            "requested_count": len(codes),
        },
        "interfaces": sorted(
            {
                value.get("interface")
                for value in source_by_symbol.values()
                if value.get("interface")
            }
        ),
        "source_by_symbol": source_by_symbol,
        "errors": errors,
        "error_classes": error_classes,
        "field_anomalies": field_anomalies,
        "cache_reconciled": cache_reconciled,
        "lifecycle": lifecycle,
        "requeued": requeued,
        "state_db": str(pathlib.Path(args.state_db).resolve()),
        "limited": limited,
        "partial_failure": bool(fail or progress["pending"] or limited),
        "time_budget_exhausted": bool(
            deadline is not None
            and time.monotonic() >= deadline
            and progress["pending"]
        ),
    }
    print("ASSISTANT_FETCH_SUMMARY=" + json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if status != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
