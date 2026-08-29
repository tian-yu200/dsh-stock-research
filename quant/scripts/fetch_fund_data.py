# -*- coding: utf-8 -*-
"""Bounded, resumable ETF and OTC-fund data worker."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

import akshare as ak
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from data_source import fetch_etf_daily, fetch_otc_fund_nav
from fetch_state import DEFAULT_DB_PATH, FetchState
from fetch_utils import (
    RateLimiter,
    atomic_write_frame,
    atomic_write_json,
    cached_end_date,
    merge_daily_frames,
    FundNavValidationError,
    merge_otc_frames,
    next_start_date,
    install_request_timeout,
    target_date,
)
from universe import (
    discover_etf_universe,
    discover_otc_fund_universe,
    load_universe_manifest,
    manifest_symbols,
    normalize_symbol,
    write_universe_manifest,
)


DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "funds"
DATA_DIR.mkdir(parents=True, exist_ok=True)
UNIVERSE_PATH = DATA_DIR.parent / "universe.json"
ETF_POOL = ["510300", "510500", "159915"]
OTC_POOL = ["110020", "001594"]
# 池小于该阈值（定向小任务）时直接逐只抓取，不值得花一次全市场快照请求。
_BULK_MIN_POOL = 50
# 快照只用来推进已有数据的尾端；距目标日超过该天数的深度过期基金
# 交给逐只路径补全历史，避免批量追加制造净值序列空洞。
_BULK_MAX_GAP_DAYS = 10


def _manifest_complete(asset_type: str) -> bool:
    manifest = load_universe_manifest(UNIVERSE_PATH)
    return bool(
        manifest
        and (
            manifest.get("complete")
            or manifest.get(f"{asset_type}_universe_complete", False)
        )
        and manifest_symbols(manifest, asset_type)
    )


def discover_pools(
    data_dir: pathlib.Path = DATA_DIR,
    *,
    online: bool = False,
) -> tuple[list[str], list[str]]:
    """Discover complete ETF/OTC pools from the manifest before the network."""
    if not online and data_dir.resolve() == DATA_DIR.resolve():
        manifest = load_universe_manifest(UNIVERSE_PATH) or {}
        etf_symbols = (
            manifest_symbols(manifest, "etf")
            if _manifest_complete("etf")
            else []
        )
        otc_symbols = (
            manifest_symbols(manifest, "otc_fund")
            if _manifest_complete("otc_fund")
            else []
        )
        if etf_symbols and otc_symbols:
            return etf_symbols, otc_symbols
    if online:
        try:
            etf_catalog = discover_etf_universe()
            etf_symbols = {item["symbol"] for item in etf_catalog}
            otc_catalog = discover_otc_fund_universe(exclude_symbols=etf_symbols)
            return (
                sorted(etf_symbols),
                sorted(item["symbol"] for item in otc_catalog),
            )
        except Exception as exc:
            print(f"[基金目录] 在线发现失败，回退本地缓存: {type(exc).__name__}")
    etfs = sorted(
        path.stem.removeprefix("etf_")
        for path in data_dir.glob("etf_*.csv")
        if len(path.stem.removeprefix("etf_")) == 6
        and path.stem.removeprefix("etf_").isdigit()
    )
    otc_funds = sorted(
        path.stem.removeprefix("fund_")
        for path in data_dir.glob("fund_*.csv")
        if len(path.stem.removeprefix("fund_")) == 6
        and path.stem.removeprefix("fund_").isdigit()
    )
    return etfs or list(ETF_POOL), otc_funds or list(OTC_POOL)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, help="并发抓取数")
    parser.add_argument("--batch-size", type=int, default=100, help="单类单批最多领取的标的数")
    parser.add_argument("--time-budget", type=float, default=600.0, help="本次运行最多抓取秒数；小于等于0表示不限制")
    parser.add_argument("--limit", type=int, default=None, help="仅用于验收时限制每类抓取数量")
    parser.add_argument("--target-date", default=None, help="数据目标日期，默认当前 UTC 日期")
    parser.add_argument("--state-db", type=pathlib.Path, default=DEFAULT_DB_PATH, help="增量状态 SQLite 文件")
    parser.add_argument("--retries", type=int, default=2, help="单标的网络重试次数")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="进程内请求速率，单位请求/秒")
    parser.add_argument("--request-timeout", type=float, default=30.0, help="单次 HTTP 请求连接/读取超时秒数")
    parser.add_argument("--backoff-base", type=float, default=2.0, help="失败指数退避起始秒数")
    parser.add_argument("--backoff-max", type=float, default=120.0, help="失败指数退避最大秒数")
    parser.add_argument("--lease-seconds", type=int, default=1800, help="单标的租约秒数")
    parser.add_argument("--refresh-catalog", action="store_true", help="强制重新发现 ETF/场外基金目录")
    parser.add_argument(
        "--bulk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="场外基金池足够大时，先用一次全市场净值快照推进已有基金的尾端",
    )
    parser.add_argument("--asset-type", choices=["etf", "otc_fund"], default=None, help="只处理一个基金类别")
    parser.add_argument("symbols", nargs="*", help="可选的代码子集；配合 --asset-type 使用")
    parser.add_argument("--job-id", default=None, help="可选的研究任务 ID，用于刷新任务进度")
    return parser


def _path(kind: str, code: str) -> pathlib.Path:
    return DATA_DIR / f"{'etf_' if kind == 'etf' else 'fund_'}{code}.csv"


def _write_success(kind: str, code: str, frame: Any, path: pathlib.Path) -> dict[str, Any]:
    if kind == "etf":
        merged = merge_daily_frames(path, frame, source=f"ETF {code}")
    else:
        merged = merge_otc_frames(path, frame, source=f"OTC fund {code}")
    atomic_write_frame(merged, path)
    metadata = {
        "asset_type": kind,
        "symbol": code,
        "source": frame.attrs.get("akshare_source"),
        "interface": frame.attrs.get("akshare_interface"),
        "adjustment": frame.attrs.get(
            "adjustment",
            "qfq" if kind == "etf" else "unit_nav",
        ),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data_end_date": merged["date"].max().date().isoformat(),
        "rows": int(len(merged)),
    }
    atomic_write_json(metadata, path.with_suffix(".meta.json"))
    return metadata


def _bulk_otc_snapshot() -> dict[str, list[tuple[str, float]]] | None:
    """一次请求取全市场基金最新两个净值日的单位净值；失败返回 None，由逐只路径兜底。"""
    try:
        raw = ak.fund_open_fund_daily_em()
    except Exception as exc:
        print(f"[批量] 全市场净值快照失败，回退逐只: {type(exc).__name__}: {exc}")
        return None
    dated_columns = sorted(
        (match.group(1), column)
        for column in raw.columns
        if (match := re.fullmatch(r"(\d{4}-\d{2}-\d{2})-单位净值", str(column)))
    )
    if not dated_columns:
        print("[批量] 快照缺少带日期的单位净值列，回退逐只")
        return None
    codes = raw["基金代码"].astype(str).str.strip()
    snapshot: dict[str, list[tuple[str, float]]] = {}
    for date_str, column in dated_columns:
        values = pd.to_numeric(raw[column], errors="coerce")
        for code, nav in zip(codes, values):
            if pd.isna(nav):
                continue
            snapshot.setdefault(code, []).append((date_str, float(nav)))
    return snapshot


def _bulk_refresh_otc(
    *,
    pool: list[str],
    state: FetchState,
    lease_owner: str,
    deadline: float | None,
) -> tuple[int, int]:
    """用一次全市场快照推进已回填基金的净值尾端，返回 (成功数, 处理数)。

    快照每基金最多两个最新净值点，因此只处理已有 CSV 且新点能近似
    连续接上现有尾端的基金；深度过期基金留给逐只路径补全历史。
    """
    snapshot = _bulk_otc_snapshot()
    if snapshot is None:
        return 0, 0
    ok = 0
    processed = 0
    max_gap = pd.Timedelta(days=_BULK_MAX_GAP_DAYS)
    for code in pool:
        if deadline is not None and time.monotonic() >= deadline:
            break
        points = snapshot.get(str(code))
        if not points:
            continue
        path = _path("otc_fund", str(code))
        last = cached_end_date(path)
        if last is None:
            continue
        fresh = [(date_str, nav) for date_str, nav in points if date_str > last]
        if not fresh:
            continue
        if pd.Timestamp(fresh[0][0]) - pd.Timestamp(last) > max_gap:
            continue
        processed += 1
        frame = pd.DataFrame(fresh, columns=["净值日期", "单位净值"])
        frame.attrs.update(
            {
                "akshare_interface": "fund_open_fund_daily_em",
                "akshare_source": "东财(批量)",
                "adjustment": "unit_nav",
            }
        )
        try:
            metadata = _write_success("otc_fund", str(code), frame, path)
        except FundNavValidationError:
            continue
        state.mark_success(
            "otc_fund",
            str(code),
            data_end_date=metadata["data_end_date"],
            source=metadata.get("source"),
            rows=metadata["rows"],
            refresh_interval_seconds=5 * 86400,
            owner=lease_owner,
        )
        ok += 1
        if ok % 2000 == 0:
            print(f"[批量] 已推进 {ok} 只基金净值")
    return ok, processed


def _fetch_one(
    kind: str,
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
) -> tuple[str, str, bool, dict[str, Any] | None, str | None]:
    code = str(item["symbol"])
    path = _path(kind, code)
    start = next_start_date(
        cached_end_date(path) or item.get("data_end_date"),
        "20200101",
    )
    end = target.replace("-", "")
    attempts = max(1, int(retries) + 1)
    last_error: str | None = None
    for attempt in range(1, attempts + 1):
        if deadline is not None and time.monotonic() >= deadline:
            last_error = "worker deadline reached before next request"
            break
        try:
            limiter.wait()
            frame = (
                fetch_etf_daily(code, start=start, end=end)
                if kind == "etf"
                else fetch_otc_fund_nav(code)
            )
            metadata = _write_success(kind, code, frame, path)
            state.mark_success(
                kind,
                code,
                data_end_date=metadata["data_end_date"],
                source=metadata.get("source"),
                rows=metadata["rows"],
                refresh_interval_seconds=86400 if kind == "etf" else 5 * 86400,
                owner=lease_owner,
            )
            return kind, code, True, metadata, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, FundNavValidationError):
                state.mark_quarantined(
                    kind,
                    code,
                    error=last_error,
                    owner=lease_owner,
                )
                return kind, code, False, None, last_error
            if attempt < attempts:
                delay = min(
                    backoff_max,
                    max(0.0, backoff_base) * (2 ** (attempt - 1)),
                )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or delay >= remaining:
                        break
                if delay:
                    time.sleep(delay)
    state.mark_failure(
        kind,
        code,
        error=last_error or "unknown fetch error",
        backoff_base_seconds=max(1, int(backoff_base)),
        backoff_max_seconds=max(1, int(backoff_max)),
        owner=lease_owner,
    )
    return kind, code, False, None, last_error


def _refresh_manifest(
    existing: dict[str, Any] | None,
    *,
    limited: bool,
    requested_etf: int,
    requested_otc: int,
    ok: int,
    failed: int,
) -> None:
    try:
        etf_catalog = discover_etf_universe()
        etf_symbols = {item["symbol"] for item in etf_catalog}
        otc_catalog = discover_otc_fund_universe(exclude_symbols=etf_symbols)
    except Exception as exc:
        print(f"[目录] 基金目录缓存失败: {type(exc).__name__}: {exc}")
        return
    existing = existing or {}
    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "AkShare",
        "etf": etf_catalog,
        "otc_fund": otc_catalog,
        "etf_universe_complete": not limited,
        "otc_fund_universe_complete": not limited,
        "fund_refresh": {
            "requested_etf": requested_etf,
            "requested_otc_fund": requested_otc,
            "success": ok,
            "failed": failed,
            "limited": limited,
        },
    }
    if "stock" in existing:
        snapshot["stock"] = existing["stock"]
    if "stock_universe_complete" in existing:
        snapshot["stock_universe_complete"] = existing["stock_universe_complete"]
    snapshot["counts"] = {
        asset_type: len(snapshot.get(asset_type, []))
        for asset_type in ("stock", "etf", "otc_fund")
    }
    snapshot["complete"] = all(
        snapshot.get(f"{asset_type}_universe_complete", False)
        for asset_type in ("stock", "etf", "otc_fund")
    )
    write_universe_manifest(UNIVERSE_PATH, snapshot)
    print(
        f"[目录] ETF {len(etf_catalog)} 个，场外基金 {len(otc_catalog)} 个，"
        f"已缓存到 {UNIVERSE_PATH}"
    )


def _register_pool(
    state: FetchState,
    kind: str,
    pool: list[str],
) -> None:
    candidates = state.assets_needing_cache_backfill(kind, pool)
    data_end_dates = {}
    for code in candidates:
        path = _path(kind, code)
        if path.exists():
            data_end_dates[code] = cached_end_date(path)
    state.register_assets(kind, candidates, data_end_dates=data_end_dates)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    target = target_date(args.target_date)
    install_request_timeout(float(args.request_timeout))
    full_catalog = not args.refresh_catalog and _manifest_complete("etf") and _manifest_complete("otc_fund")
    etf_pool, otc_pool = discover_pools(
        online=bool(args.refresh_catalog or not full_catalog)
    )
    if args.symbols:
        normalized = [normalize_symbol(value) for value in args.symbols]
        if any(value is None for value in normalized):
            raise ValueError("symbols must be six-digit market codes")
        selected = [value for value in normalized if value is not None]
        if args.asset_type == "etf":
            etf_pool, otc_pool = selected, []
        elif args.asset_type == "otc_fund":
            etf_pool, otc_pool = [], selected
        else:
            etf_set = set(etf_pool)
            otc_set = set(otc_pool)
            etf_pool = [value for value in selected if value in etf_set]
            otc_pool = [value for value in selected if value in otc_set]
            unmatched = sorted(set(selected) - set(etf_pool) - set(otc_pool))
            if unmatched:
                raise ValueError(f"symbols not found in the discovered fund pools: {unmatched}")
    if args.limit is not None:
        etf_pool = etf_pool[: args.limit]
        otc_pool = otc_pool[: args.limit]
    if args.asset_type == "etf":
        otc_pool = []
    elif args.asset_type == "otc_fund":
        etf_pool = []

    state = FetchState(args.state_db)
    state.reclaim_orphaned_leases()
    _register_pool(state, "etf", etf_pool)
    _register_pool(state, "otc_fund", otc_pool)
    lease_owner = (
        f"fund-worker-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{os.getpid()}"
    )
    limiter = RateLimiter(float(args.rate_limit))
    install_request_timeout(float(args.request_timeout))
    workers = max(1, min(int(args.workers), 8))
    batch_size = max(1, min(int(args.batch_size), 500))
    deadline = (
        None
        if float(args.time_budget) <= 0
        else time.monotonic() + float(args.time_budget)
    )
    ok = 0
    fail = 0
    bulk_ok = 0
    bulk_processed = 0
    source_by_symbol: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    if args.bulk and args.limit is None and len(otc_pool) >= _BULK_MIN_POOL:
        bulk_ok, bulk_processed = _bulk_refresh_otc(
            pool=otc_pool,
            state=state,
            lease_owner=lease_owner,
            deadline=deadline,
        )
        print(f"[批量] 快照推进完成: 成功 {bulk_ok} / 处理 {bulk_processed}")

    while deadline is None or time.monotonic() < deadline:
        did_work = False
        for kind, pool in (("etf", etf_pool), ("otc_fund", otc_pool)):
            if not pool:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                break
            due = state.claim_due(
                kind,
                symbols=pool,
                limit=min(batch_size, workers),
                target_date=target,
                lease_seconds=max(30, int(args.lease_seconds)),
                owner=lease_owner,
            )
            if not due:
                continue
            did_work = True
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _fetch_one,
                        kind,
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
                for future in as_completed(futures):
                    result_kind, code, succeeded, metadata, error = future.result()
                    if succeeded and metadata:
                        ok += 1
                        source_by_symbol[f"{result_kind}:{code}"] = metadata
                        print(f"[完成] {result_kind}:{code} -> {_path(result_kind, code)}")
                    else:
                        fail += 1
                        errors[f"{result_kind}:{code}"] = error or "unknown error"
                        print(f"[失败] {result_kind}:{code}: {errors[f'{result_kind}:{code}']}")
        if not did_work:
            break

    selected_types = tuple(
        kind
        for kind, pool in (("etf", etf_pool), ("otc_fund", otc_pool))
        if pool
    )
    progress = state.progress(
        selected_types,
        symbols={"etf": etf_pool, "otc_fund": otc_pool},
        target_date=target,
    )
    manifest = load_universe_manifest(UNIVERSE_PATH) or {}
    limited = args.limit is not None
    if args.refresh_catalog or not full_catalog:
        _refresh_manifest(
            manifest,
            limited=limited,
            requested_etf=len(etf_pool),
            requested_otc=len(otc_pool),
            ok=ok,
            failed=fail,
        )
    catalog_complete = {
        "etf": bool(manifest.get("complete") or manifest.get("etf_universe_complete", False)),
        "otc_fund": bool(manifest.get("complete") or manifest.get("otc_fund_universe_complete", False)),
    }
    status = "complete" if progress["status"] == "complete" else "partial"
    if fail and ok == 0 and progress["completed"] == 0:
        status = "failed"
    summary = {
        "kind": "funds",
        "status": status,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "ok": ok,
        "failed": fail,
        "bulk_refresh": {"ok": bulk_ok, "processed": bulk_processed},
        "requested": len(etf_pool) + len(otc_pool),
        "processed": ok + fail,
        "pending": progress["pending"],
        "progress": progress,
        "catalog_coverage": {
            "etf": {
                "complete": catalog_complete["etf"] and args.limit is None,
                "catalog_count": len(manifest.get("etf", [])),
                "requested_count": len(etf_pool),
            },
            "otc_fund": {
                "complete": catalog_complete["otc_fund"] and args.limit is None,
                "catalog_count": len(manifest.get("otc_fund", [])),
                "requested_count": len(otc_pool),
            },
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
        "field_anomalies": [],
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
