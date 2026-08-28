"""Build and validate the AkShare-backed Qlib provider."""
from __future__ import annotations

import argparse
import pathlib

from qlib_export import build_qlib_provider, default_source_groups, validate_provider

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=pathlib.Path,
        default=ROOT / "qlib_data" / "akshare_data",
        help="Qlib provider directory",
    )
    parser.add_argument("--replace", action="store_true", help="replace an existing provider atomically")
    args = parser.parse_args()

    summary = build_qlib_provider(args.target, default_source_groups(ROOT), replace=args.replace)
    validated = validate_provider(args.target)
    print("AkShare Qlib 数据已构建并通过校验:")
    print(f"  目录: {validated.target_dir}")
    print(f"  日历: {validated.calendar_start} -> {validated.calendar_end} ({validated.calendar_rows} 日)")
    print(f"  标的: {validated.instruments}; 分组: {validated.markets}")
    assert summary.instruments == validated.instruments


if __name__ == "__main__":
    main()
