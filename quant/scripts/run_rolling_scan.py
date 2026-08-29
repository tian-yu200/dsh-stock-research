"""Run one incremental rolling-scan pass outside the fetch worker.

Normal operation hooks this into run_research_worker's wave loop; this CLI
exists for manual passes and for backfilling the rolling pool from data that
already exists on disk (pass a large --max-symbols to lift the per-pass cap).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.config import load_config
from assistant.rolling_scan import run_rolling_pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", default=None, help="label passes with a research job ID")
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=None,
        help="override the per-pass symbol cap; omit to use the configured cap",
    )
    args = parser.parse_args(argv)
    config = load_config()
    summary = run_rolling_pass(
        job_id=args.job_id,
        config=config,
        max_symbols=args.max_symbols,
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0 if summary.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
