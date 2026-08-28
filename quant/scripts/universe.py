"""Discover and persist the complete A-share, ETF, and OTC-fund universe."""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import akshare as ak
except ImportError:  # pragma: no cover - the production venv provides AkShare
    ak = None


ASSET_TYPES = ("stock", "etf", "otc_fund")
CODE_RE = re.compile(r"^\d{6}$")

_CODE_ALIASES = (
    "code",
    "代码",
    "基金代码",
    "证券代码",
    "证券代码",
    "A股代码",
)
_NAME_ALIASES = (
    "name",
    "名称",
    "基金简称",
    "基金名称",
    "证券简称",
    "证券名称",
    "A股简称",
)


def _clean_column(value: object) -> str:
    return re.sub(r"\s+", "", str(value).strip()).lower()


def _pick_column(frame: pd.DataFrame, aliases: Iterable[str]) -> object | None:
    normalized = {_clean_column(column): column for column in frame.columns}
    for alias in aliases:
        found = normalized.get(_clean_column(alias))
        if found is not None:
            return found
    for column in frame.columns:
        value = _clean_column(column)
        if any(_clean_column(alias) in value for alias in aliases):
            return column
    return None


def normalize_symbol(value: object) -> str | None:
    """Return a six-digit market code, handling numeric CSV/API values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if not text.isdigit() or len(text) > 6:
        return None
    text = text.zfill(6)
    return text if CODE_RE.fullmatch(text) else None


def infer_exchange(symbol: str) -> str:
    """Infer the mainland exchange used by Qlib and market-data adapters."""
    if symbol.startswith(("4", "8")):
        return "BJ"
    if symbol.startswith(("5", "6", "9")):
        return "SH"
    return "SZ"


def _catalog(
    frame: pd.DataFrame,
    *,
    asset_type: str,
    source: str,
    exclude_symbols: set[str] | None = None,
) -> list[dict[str, str]]:
    if frame is None or frame.empty:
        raise ValueError(f"{source}: empty universe response")
    code_column = _pick_column(frame, _CODE_ALIASES)
    name_column = _pick_column(frame, _NAME_ALIASES)
    if code_column is None:
        code_column = frame.columns[0]
    if name_column is None and len(frame.columns) > 1:
        name_column = frame.columns[1]

    excluded = exclude_symbols or set()
    records: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        symbol = normalize_symbol(row.get(code_column))
        if symbol is None or symbol in excluded:
            continue
        name = ""
        if name_column is not None:
            value = row.get(name_column)
            if value is not None and not pd.isna(value):
                name = str(value).strip()
        records[symbol] = {
            "symbol": symbol,
            "name": name,
            "asset_type": asset_type,
            "exchange": infer_exchange(symbol),
            "source": source,
        }
    if not records:
        raise ValueError(f"{source}: no six-digit symbols found")
    return [records[symbol] for symbol in sorted(records)]


def _require_akshare(client: Any | None) -> Any:
    client = client or ak
    if client is None:
        raise RuntimeError("AkShare is not installed")
    return client


def discover_stock_universe(client: Any | None = None) -> list[dict[str, str]]:
    client = _require_akshare(client)
    return _catalog(
        client.stock_info_a_code_name(),
        asset_type="stock",
        source="ak.stock_info_a_code_name",
    )


def discover_etf_universe(client: Any | None = None) -> list[dict[str, str]]:
    client = _require_akshare(client)
    errors: list[str] = []
    for name, call in (
        ("ak.fund_etf_spot_em", lambda: client.fund_etf_spot_em()),
        (
            "ak.fund_etf_category_sina",
            lambda: client.fund_etf_category_sina(symbol="ETF基金"),
        ),
        (
            "ak.fund_etf_category_ths",
            lambda: client.fund_etf_category_ths(symbol="ETF"),
        ),
    ):
        try:
            return _catalog(call(), asset_type="etf", source=name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("ETF universe discovery failed: " + "; ".join(errors))


def discover_otc_fund_universe(
    client: Any | None = None,
    *,
    exclude_symbols: set[str] | None = None,
) -> list[dict[str, str]]:
    client = _require_akshare(client)
    errors: list[str] = []
    for name, call in (
        ("ak.fund_open_fund_daily_em", lambda: client.fund_open_fund_daily_em()),
        ("ak.fund_name_em", lambda: client.fund_name_em()),
    ):
        try:
            return _catalog(
                call(),
                asset_type="otc_fund",
                source=name,
                exclude_symbols=exclude_symbols,
            )
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("OTC fund universe discovery failed: " + "; ".join(errors))


def discover_universe(client: Any | None = None) -> dict[str, Any]:
    """Discover all three universes while retaining per-category failures."""
    client = _require_akshare(client)
    result: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "AkShare",
        "complete": False,
        "errors": {},
        "stock": [],
        "etf": [],
        "otc_fund": [],
    }
    try:
        result["stock"] = discover_stock_universe(client)
    except Exception as exc:
        result["errors"]["stock"] = f"{type(exc).__name__}: {exc}"
    try:
        result["etf"] = discover_etf_universe(client)
    except Exception as exc:
        result["errors"]["etf"] = f"{type(exc).__name__}: {exc}"
    etf_symbols = {item["symbol"] for item in result["etf"]}
    try:
        result["otc_fund"] = discover_otc_fund_universe(
            client,
            exclude_symbols=etf_symbols,
        )
    except Exception as exc:
        result["errors"]["otc_fund"] = f"{type(exc).__name__}: {exc}"
    result["counts"] = {
        asset_type: len(result[asset_type])
        for asset_type in ASSET_TYPES
    }
    for asset_type in ASSET_TYPES:
        result[f"{asset_type}_universe_complete"] = bool(
            result[asset_type] and asset_type not in result["errors"]
        )
    result["complete"] = all(
        result[f"{asset_type}_universe_complete"]
        for asset_type in ASSET_TYPES
    )
    return result


def load_universe_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_universe_manifest(
    path: Path,
    snapshot: dict[str, Any],
    *,
    merge_existing: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {}
    if merge_existing:
        existing = load_universe_manifest(path)
        if existing:
            value.update(existing)
    value.update(snapshot)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_asset_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write the adjustment/NAV provenance next to a downloaded CSV."""
    metadata_path = path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_asset_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def manifest_symbols(manifest: dict[str, Any] | None, asset_type: str) -> list[str]:
    if not manifest:
        return []
    records = manifest.get(asset_type, [])
    if not isinstance(records, list):
        return []
    symbols = {
        item.get("symbol")
        for item in records
        if isinstance(item, dict) and normalize_symbol(item.get("symbol"))
    }
    return sorted(symbols)


def manifest_names(manifest: dict[str, Any] | None, asset_type: str) -> dict[str, str]:
    if not manifest:
        return {}
    records = manifest.get(asset_type, [])
    if not isinstance(records, list):
        return {}
    return {
        str(item["symbol"]): str(item.get("name") or "")
        for item in records
        if isinstance(item, dict) and normalize_symbol(item.get("symbol"))
    }
