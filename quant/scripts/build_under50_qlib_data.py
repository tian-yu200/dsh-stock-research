"""Build the isolated Qlib provider used by the under-CNY50 research."""
from __future__ import annotations

import argparse
import pathlib

from qlib_export import SourceGroup, build_qlib_provider, validate_provider

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = (
    "000651",
    "600887",
    "600900",
    "600886",
    "600919",
    "600938",
    "002142",
    "300628",
    "000895",
    "002078",
    "002352",
    "601225",
    "601919",
    "600690",
    "600066",
    "601598",
)

def source_groups() -> tuple[SourceGroup, ...]:
    stock_dir = ROOT / "data" / "stocks"
    groups = [SourceGroup("stocks", stock_dir, f"{symbol}.csv") for symbol in CANDIDATES]
    groups.append(SourceGroup("benchmark", ROOT / "data" / "funds", "etf_510300.csv"))
    return tuple(groups)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=pathlib.Path,
        default=ROOT / "qlib_data" / "under50_data",
        help="isolated Qlib provider directory",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    summary = build_qlib_provider(args.target, source_groups(), replace=args.replace)
    validated = validate_provider(args.target)
    if validated.markets.get("stocks") != len(CANDIDATES):
        raise ValueError(f"expected {len(CANDIDATES)} stocks, got {validated.markets}")
    print("Under-50 Qlib provider built and validated:")
    print(f"  directory: {validated.target_dir}")
    print(f"  calendar: {validated.calendar_start} -> {validated.calendar_end} ({validated.calendar_rows})")
    print(f"  instruments: {validated.instruments}; groups: {validated.markets}")
    assert summary.instruments == validated.instruments

if __name__ == "__main__":
    main()
