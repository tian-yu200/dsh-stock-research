"""Offline multi-window stability check (never part of the daily pipeline).

Runs ``assistant.qlib_model.multi_window_stability_check`` outside the
scheduler/worker so extra full trainings can never blow the 7200s worker
budget (see quant/docs/\u7814\u7a76\u7ba1\u7ebf\u4e0d\u53d8\u91cf.md \u00a74).
Reads windows from the independent ``stability_check`` config section --
never ``research.stability_windows`` -- and writes a JSON report. The daily
path is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

QUANT_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ROOT))

from assistant.config import load_config
from assistant.paths import atomic_write_json, fixed_path


class _PeakRssSampler:
    """Sample own + children peak RSS in a daemon thread (psutil)."""

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self._interval = max(0.5, float(interval_seconds))
        self._peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            import psutil
        except ImportError:
            return
        try:
            root = psutil.Process()
        except Exception:
            return
        while not self._stop.wait(self._interval):
            try:
                total = root.memory_info().rss
                for child in root.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        continue
                self._peak = max(self._peak, total)
            except Exception:
                continue

    def __enter__(self) -> "_PeakRssSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()

    @property
    def peak_gb(self) -> float | None:
        return (self._peak / (1024.0**3)) if self._peak else None


def _parse_windows(raw: str) -> list[list[str]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--windows is not valid JSON: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("--windows must be a non-empty JSON list of [start, end] pairs")
    parsed: list[list[str]] = []
    for item in value:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(edge, str) and edge for edge in item)
        ):
            raise ValueError(f"bad window {item!r}: expected [\"YYYY-MM-DD\", \"YYYY-MM-DD\"]")
        parsed.append([str(item[0]), str(item[1])])
    return parsed


def _jsonable(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows",
        default=None,
        help='JSON list of [start, end] pairs overriding stability_check.windows, '
        'e.g. \'[["2023-04-01","2024-03-29"]]\'',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the execution plan without training anything",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="run only the first N configured windows (cost probe without JSON quoting)",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="override the JSON report path",
    )
    args = parser.parse_args(argv)

    config = load_config()
    section = config.get("stability_check") or {}
    windows = _parse_windows(args.windows) if args.windows else [list(w) for w in (section.get("windows") or [])]
    if args.max_windows is not None:
        if args.max_windows < 1:
            raise ValueError("--max-windows must be positive")
        windows = windows[: args.max_windows]
    if not windows:
        print(json.dumps({"status": "error", "error": "no windows configured"}, ensure_ascii=True, sort_keys=True))
        return 2
    threshold = float(section.get("threshold", 0.02))
    segment = str(section.get("segment", "test"))
    dataset_config = config["dataset"]
    report_path = fixed_path(args.report_path or section.get("report_path") or "results/assistant/stability_report.json")

    plan = {
        "status": "planned",
        "windows": windows,
        "windows_count": len(windows),
        "segment": segment,
        "threshold": threshold,
        "dataset": {
            "start_time": dataset_config.get("start_time"),
            "fit_start_time": dataset_config.get("fit_start_time"),
            "fit_end_time": dataset_config.get("fit_end_time"),
        },
        "report_path": str(report_path),
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, sort_keys=True))
        return 0

    # Heavy imports only on real runs so --dry-run stays instant.
    # Mirror production env (qlib_model._run_qlib_stock_model): thread caps plus
    # MLflow file-store opt-out, otherwise train_lgbm dies in MlflowException.
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        f"file:///{(Path(config['paths']['logs_dir']) / 'mlruns').as_posix()}",
    )
    from assistant import qlib_model
    from assistant.provider_lock import ProviderLock

    started = time.monotonic()
    with _PeakRssSampler() as sampler:
        provider = Path(config["paths"]["qlib_provider"]).resolve()
        lock = ProviderLock(provider, purpose="offline stability check (single window trial)").acquire(
            timeout_seconds=float(config.get("runtime", {}).get("provider_lock_timeout_seconds", 30.0))
        )
        try:
            qlib_model.init_qlib(provider, kernels=1)
            from qlib.data import D

            calendar = D.calendar(freq="day")
            if not len(calendar):
                raise ValueError("Qlib provider calendar is empty")
            import pandas as pd

            provider_end = pd.Timestamp(calendar[-1]).date().isoformat()
            label_horizon = int(config["research"]["stock_label_horizon"])
            end_time = min(str(dataset_config["segments"]["test"][1]), provider_end)
            segments = {
                name: [values[0], min(values[1], end_time)]
                for name, values in dataset_config["segments"].items()
            }
            prep_started = time.monotonic()
            dataset = qlib_model.build_dataset(
                instruments="stocks",
                start_time=dataset_config["start_time"],
                end_time=end_time,
                fit_start_time=dataset_config["fit_start_time"],
                fit_end_time=dataset_config["fit_end_time"],
                label_horizon=label_horizon,
                segments=segments,
            )
            prep_seconds = time.monotonic() - prep_started
            check_started = time.monotonic()
            stability = qlib_model.multi_window_stability_check(
                dataset,
                config["model"],
                windows,
                segment=segment,
                threshold=threshold,
            )
            check_seconds = time.monotonic() - check_started
        finally:
            lock.release()
    total_seconds = time.monotonic() - started
    report = _jsonable(
        {
            **plan,
            "status": "ok",
            "stability": stability,
            "timings": {
                "dataset_prepare_seconds": round(prep_seconds, 1),
                "stability_check_seconds": round(check_seconds, 1),
                "total_seconds": round(total_seconds, 1),
            },
            "peak_rss_gb": round(sampler.peak_gb, 3) if sampler.peak_gb is not None else None,
        }
    )
    assert isinstance(report, dict)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
