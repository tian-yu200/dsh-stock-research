"""Build and validate the AkShare-backed Qlib provider."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile

from qlib_export import build_qlib_provider, default_source_groups

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _ProgressWriter:
    """Atomically publish a bounded view of the provider build progress."""

    _FIELDS = (
        "phase",
        "phase_seq",
        "completed",
        "total",
        "updated_at",
        "cache_hit",
        "source_fingerprint_digest",
    )

    def __init__(self, path: pathlib.Path) -> None:
        self.path = pathlib.Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, payload: dict[str, object]) -> None:
        compact = {key: payload[key] for key in self._FIELDS if key in payload}
        temporary_path: pathlib.Path | None = None
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        temporary_path = pathlib.Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    compact,
                    handle,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=pathlib.Path,
        default=ROOT / "qlib_data" / "akshare_data",
        help="Qlib provider directory",
    )
    parser.add_argument("--replace", action="store_true", help="replace an existing provider atomically")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="ignore the provider source fingerprint cache",
    )
    parser.add_argument(
        "--progress-file",
        type=pathlib.Path,
        default=None,
        help="atomically updated bounded build progress JSON path",
    )
    args = parser.parse_args()

    progress_callback = _ProgressWriter(args.progress_file) if args.progress_file else None
    summary = build_qlib_provider(
        args.target,
        default_source_groups(ROOT),
        replace=args.replace,
        skip_if_unchanged=not args.force_rebuild,
        progress_callback=progress_callback,
    )
    print("AkShare Qlib 数据已命中缓存并通过校验:" if summary.cache_hit else "AkShare Qlib 数据已构建并通过校验:")
    print(f"  目录: {summary.target_dir}")
    print(f"  日历: {summary.calendar_start} -> {summary.calendar_end} ({summary.calendar_rows} 日)")
    print(f"  标的: {summary.instruments}; 分组: {summary.markets}")
    print(
        "QLIB_BUILD_SUMMARY="
        + json.dumps(
            {
                "cache_hit": summary.cache_hit,
                "source_fingerprint_digest": summary.source_fingerprint,
                "instruments": summary.instruments,
                "calendar_end": summary.calendar_end,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
