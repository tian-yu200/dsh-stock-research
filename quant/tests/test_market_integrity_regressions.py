from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from assistant import data_layer, research, rolling_scan
import fetch_fund_data
from fetch_utils import BaselineDiscontinuityError


class MarketIntegrityRegressionTests(unittest.TestCase):
    def test_legacy_sina_qfq_label_is_not_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'etf_159320.csv'
            path.with_suffix('.meta.json').write_text(json.dumps({
                'adjustment': 'qfq', 'interface': 'fund_etf_hist_sina',
            }), encoding='utf-8')
            self.assertFalse(data_layer._adjustment_check(path, 'etf')['passed'])

    def test_adjusted_tail_cannot_validate_raw_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'etf_159320.csv'
            frame = pd.DataFrame({
                'date': pd.date_range('2026-09-01', periods=3),
                'open': [1., 1., 1.], 'high': [1., 1., 1.],
                'low': [1., 1., 1.], 'close': [1., 1., 1.],
                'volume': [100., 100., 100.], 'amount': [100., 100., 100.],
            })
            frame.to_csv(path, index=False)
            path.with_suffix('.meta.json').write_text(json.dumps({
                'adjustment': 'qfq', 'interface': 'fund_etf_hist_sina',
            }), encoding='utf-8')
            before = path.read_bytes()
            frame.attrs.update(adjustment='qfq', akshare_interface='stock_zh_a_hist_tx')
            with self.assertRaises(BaselineDiscontinuityError):
                fetch_fund_data._write_success('etf', '159320', frame, path)
            self.assertEqual(path.read_bytes(), before)

    def test_file_revision_invalidates_equal_date_rows_and_source(self):
        fingerprint = {'end_date': '2026-09-04', 'rows': 100,
                       'source': 'test', 'file_revision': 'old'}
        row = {'data_end_date': '2026-09-04', 'rows': 100,
               'source': 'test', 'file_revision': 'new'}
        self.assertTrue(rolling_scan._fingerprint_changed(fingerprint, row))
        fingerprint['file_revision'] = 'new'
        self.assertFalse(rolling_scan._fingerprint_changed(fingerprint, row))

    def test_rolling_loader_discards_nonpositive_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'stocks').mkdir()
            pd.DataFrame({'date': ['2026-09-01', '2026-09-02', '2026-09-03'],
                          'close': [10., 0., 11.]}).to_csv(root / 'stocks/600000.csv', index=False)
            with patch.object(research, 'DATA_DIR', root):
                frame = rolling_scan._load_frame('stock', '600000')
            self.assertEqual(frame['close'].tolist(), [10., 11.])


if __name__ == '__main__':
    unittest.main()
