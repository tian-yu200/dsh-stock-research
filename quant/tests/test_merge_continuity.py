"""基准连续性守卫（merge_daily_frames）与 worker 全量重取降级的测试。

背景：2026-09-02 增量抓取跨源拼接把 ~200 元的后复权行拼到 ~3 元历史上，
60 倍台阶毒化全部下游指标。守卫必须在写盘前拒收这种拼接，
worker 捕获后应挪开本地 CSV 降级为同源全量重取。
"""
from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fetch_stock_data as stock_worker
from fetch_utils import (
    BaselineDiscontinuityError,
    atomic_write_frame,
    merge_daily_frames,
)

HISTORY_DATES = list(pd.bdate_range("2026-01-05", periods=20).strftime("%Y-%m-%d"))
NEW_DATES = list(pd.bdate_range("2026-02-02", periods=2).strftime("%Y-%m-%d"))


def _series(dates: list[str], closes: list[float], *, open_values=None) -> pd.DataFrame:
    opens = open_values if open_values is not None else [close * 1.01 for close in closes]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": opens,
            "high": [close * 1.02 for close in closes],
            "low": [close * 0.98 for close in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
            "amount": [close * 1000.0 for close in closes],
            "turnover": [1.0] * len(closes),
        }
    )


class _FakeLimiter:
    def wait(self) -> None:
        return None


class _FetchStateSpy:
    def __init__(self) -> None:
        self.successes: list[tuple[str, dict]] = []
        self.failures: list[tuple[str, dict]] = []

    def mark_success(self, kind: str, symbol: str, **kwargs) -> None:
        self.successes.append((symbol, kwargs))

    def mark_failure(self, kind: str, symbol: str, **kwargs) -> None:
        self.failures.append((symbol, kwargs))


class MergeContinuityTests(unittest.TestCase):
    def test_normal_append_with_same_basis_passes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="merge-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "000001.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [3.0] * 20), source="test"),
                path,
            )
            merged = merge_daily_frames(
                path, _series(NEW_DATES, [3.03, 3.06]), source="test"
            )
            self.assertEqual(len(merged), 22)
            self.assertEqual(float(merged["close"].iloc[-1]), 3.06)

    def test_boundary_basis_break_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="merge-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "600654.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [3.0] * 20), source="test"),
                path,
            )
            # 9-02 事故形态：另一种复权口径的完整行（~200 元）拼到 ~3 元历史之后
            # （normalize 层会先拦掉非正价格行，所以能走到守卫的毒行都是完整行）
            poisoned = _series(NEW_DATES, [200.0, 201.0])
            with self.assertRaises(BaselineDiscontinuityError):
                merge_daily_frames(path, poisoned, source="test")
            # 拒收时不得写盘：本地历史保持原样
            on_disk = pd.read_csv(path)
            self.assertEqual(len(on_disk), 20)
            self.assertEqual(float(on_disk["close"].iloc[-1]), 3.0)

    def test_downward_basis_break_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="merge-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "600654.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [200.0] * 20), source="test"),
                path,
            )
            with self.assertRaises(BaselineDiscontinuityError):
                merge_daily_frames(path, _series(NEW_DATES, [3.0, 3.0]), source="test")

    def test_overlap_median_deviation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="merge-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "600654.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [3.0] * 20), source="test"),
                path,
            )
            # 同一批日期、另一种口径（重叠无新增日期）也要被拦下
            overlap = _series(HISTORY_DATES[-10:], [200.0] * 10)
            with self.assertRaises(BaselineDiscontinuityError):
                merge_daily_frames(path, overlap, source="test")


class WorkerFallbackTests(unittest.TestCase):
    def test_worker_downgrades_to_full_refetch_on_discontinuity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worker-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "600654.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [3.0] * 20), source="test"),
                path,
            )
            poisoned_increment = _series(NEW_DATES, [200.0, 201.0])
            clean_full = pd.concat(
                [_series(HISTORY_DATES, [200.0] * 20), _series(NEW_DATES, [201.0, 202.0])],
                ignore_index=True,
            )
            state = _FetchStateSpy()
            with patch.object(
                stock_worker,
                "fetch_a_stock_daily",
                side_effect=[poisoned_increment, clean_full],
            ), patch.object(stock_worker, "_data_path", return_value=path):
                code, ok, metadata, error, failure = stock_worker._fetch_one(
                    {"symbol": "600654", "data_end_date": None},
                    state=state,
                    limiter=_FakeLimiter(),
                    target="2026-02-03",
                    retries=2,
                    backoff_base=0,
                    backoff_max=0,
                    lease_owner="test-owner",
                    deadline=None,
                )
            self.assertTrue(ok)
            self.assertIsNone(error)
            self.assertEqual(len(state.successes), 1)
            self.assertEqual(metadata["rows"], 22)
            # 坏历史已被干净的全量序列替换，备份已清理
            on_disk = pd.read_csv(path)
            self.assertEqual(len(on_disk), 22)
            self.assertEqual(float(on_disk["close"].iloc[-1]), 202.0)
            self.assertFalse(
                path.with_name(path.name + ".discont.bak").exists(),
                "全量重取成功后应删除备份",
            )

    def test_worker_restores_backup_when_full_refetch_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worker-", dir=ROOT / "tests") as directory:
            path = pathlib.Path(directory) / "600654.csv"
            atomic_write_frame(
                merge_daily_frames(path, _series(HISTORY_DATES, [3.0] * 20), source="test"),
                path,
            )
            poisoned_increment = _series(NEW_DATES, [200.0, 201.0])
            state = _FetchStateSpy()
            with patch.object(
                stock_worker,
                "fetch_a_stock_daily",
                side_effect=[poisoned_increment, RuntimeError("network down")],
            ), patch.object(stock_worker, "_data_path", return_value=path):
                code, ok, metadata, error, failure = stock_worker._fetch_one(
                    {"symbol": "600654", "data_end_date": None},
                    state=state,
                    limiter=_FakeLimiter(),
                    target="2026-02-03",
                    retries=1,
                    backoff_base=0,
                    backoff_max=0,
                    lease_owner="test-owner",
                    deadline=None,
                )
            self.assertFalse(ok)
            self.assertEqual(len(state.failures), 1)
            # 全量重取失败：还原原始 CSV，毒拼接绝不落盘
            on_disk = pd.read_csv(path)
            self.assertEqual(len(on_disk), 20)
            self.assertEqual(float(on_disk["close"].iloc[-1]), 3.0)
            self.assertFalse(path.with_name(path.name + ".discont.bak").exists())


if __name__ == "__main__":
    unittest.main()
