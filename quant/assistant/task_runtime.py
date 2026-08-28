"""Record failures that happen outside the research state machine."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.notifications import add_notification  # noqa: E402


KINDS = ("scheduled_task_failed", "scheduled_task_timeout", "scheduled_task_start_failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one fixed-path task notification.")
    parser.add_argument("--kind", choices=KINDS, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    record = add_notification(
        args.kind,
        args.title,
        args.message,
        run_id=args.run_id,
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
