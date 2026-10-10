# -*- coding: utf-8 -*-
"""Tests for the one-off quarantine normaliser (T1).

Covers: date format normalisation (YYYY-MM-DD), exact-duplicate removal,
date-conflict retention (same date, differing rows are all kept), backup on
apply, and the dry-run default that must not write.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from normalize_quarantine import iter_quarantine_files, main, normalise_frame


def _write(path: pathlib.Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class QuarantineNormaliseTests(unittest.TestCase):
    def _frame(self):
        import pandas as pd
        return pd.DataFrame(
            [
                ["2026-09-04 00:00:00", 7.03, 7.03, 260226, 127095100, 0.0078, "gap"],
                ["2026-09-04", 7.03, 7.03, 260226, 127095100, 0.0078, "gap"],
                ["2026-09-07 00:00:00", 7.02, 7.02, 239192, 116012600, 0.0072, "gap"],
            ],
            columns=["date", "open", "close", "volume", "amount", "turnover", "quarantine_reasons"],
        )

    def test_normalise_frame_formats_and_dedupes_exact_rows(self):
        out, stats = normalise_frame(self._frame())
        self.assertEqual(list(out["date"]), ["2026-09-04", "2026-09-07"])
        self.assertEqual(stats["exact_dupes"], 1)
        self.assertEqual(stats["mixed_format_rows"], 2)

    def test_date_conflict_rows_are_kept_and_counted(self):
        import pandas as pd
        frame = pd.DataFrame(
            [
                ["2026-09-04 00:00:00", 7.03, 7.03, 260226, 127095100, 0.0078, "a"],
                ["2026-09-04", 7.11, 7.11, 999, 999999, 0.001, "b"],
            ],
            columns=["date", "open", "close", "volume", "amount", "turnover", "quarantine_reasons"],
        )
        out, stats = normalise_frame(frame)
        self.assertEqual(len(out), 2)
        self.assertEqual(stats["exact_dupes"], 0)
        self.assertEqual(stats["date_conflicts"], 1)

    def test_dry_run_is_default_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            qdir = root / "stocks" / "quarantine"
            qdir.mkdir(parents=True)
            path = qdir / "T00001.quarantine.csv"
            _write(path, [
                ["date", "open", "close", "volume", "amount", "turnover", "quarantine_reasons"],
                ["2026-09-04 00:00:00", 7.03, 7.03, 260226, 127095100, 0.0078, "gap"],
            ])
            before = path.read_text(encoding="utf-8")
            self.assertEqual(main(["--root", str(root)]), 0)
            self.assertEqual(path.read_text(encoding="utf-8"), before)
            self.assertFalse((qdir / "T00001.quarantine.csv.pre-normalize.bak").exists())

    def test_apply_normalises_backs_up_and_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            qdir = root / "stocks" / "quarantine"
            qdir.mkdir(parents=True)
            path = qdir / "T00001.quarantine.csv"
            _write(path, [
                ["date", "open", "close", "volume", "amount", "turnover", "quarantine_reasons"],
                ["2026-09-04 00:00:00", 7.03, 7.03, 260226, 127095100, 0.0078, "gap"],
                ["2026-09-04", 7.03, 7.03, 260226, 127095100, 0.0078, "gap"],
            ])
            self.assertEqual(main(["--root", str(root), "--apply"]), 0)
            self.assertTrue((qdir / "T00001.quarantine.csv.pre-normalize.bak").exists())
            import pandas as pd
            out = pd.read_csv(path, dtype=str)
            self.assertEqual(list(out["date"]), ["2026-09-04"])

    def test_iter_finds_nested_quarantine_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            qdir = root / "funds" / "quarantine"
            qdir.mkdir(parents=True)
            (qdir / "X.quarantine.csv").write_text("date\n", encoding="utf-8")
            found = [p.name for p in iter_quarantine_files(root)]
            self.assertEqual(found, ["X.quarantine.csv"])


if __name__ == "__main__":
    unittest.main()
