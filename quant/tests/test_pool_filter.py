from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.research import _pool_quality_filter


def _config(**overrides) -> dict:
    research = {
        "pool_exclude_st": True,
        "min_listing_age_calendar_days": 90,
        "min_avg_amount_20d": 10_000_000.0,
        "pool_exclude_cross_border": True,
    }
    research.update(overrides)
    return {"research": research}


def _fact(name="", start_date="2015-01-05", avg_amount=50_000_000.0) -> dict:
    return {
        "name": name,
        "start_date": start_date,
        "checks": {"liquidity": {"average_amount_20d": avg_amount}},
    }


def _signal(symbol: str) -> dict:
    return {"symbol": symbol, "qlib_score": 0.1}


class PoolQualityFilterTests(unittest.TestCase):
    def test_st_and_delisting_names_are_excluded(self) -> None:
        signals = [_signal("600000"), _signal("301117"), _signal("000003"), _signal("000004")]
        facts = {
            "600000": _fact("浦发银行"),
            "301117": _fact("ST佳缘"),
            "000003": _fact("*ST某某"),
            "000004": _fact("某某退"),
        }
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-04")
        self.assertEqual([s["symbol"] for s in eligible], ["600000"])
        self.assertEqual(
            [(e["symbol"], e["reason"]) for e in exclusions],
            [("301117", "st_or_delisting"), ("000003", "st_or_delisting"), ("000004", "st_or_delisting")],
        )

    def test_recent_listing_is_excluded(self) -> None:
        signals = [_signal("600000"), _signal("301999")]
        facts = {
            "600000": _fact("浦发银行", start_date="2015-01-05"),
            "301999": _fact("某次新股", start_date="2026-08-10"),
        }
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-04")
        self.assertEqual([s["symbol"] for s in eligible], ["600000"])
        self.assertEqual(exclusions[0]["reason"], "new_listing")

    def test_low_liquidity_is_excluded(self) -> None:
        signals = [_signal("600000"), _signal("920999")]
        facts = {
            "600000": _fact("浦发银行"),
            "920999": _fact("某低流动性", avg_amount=2_500_000.0),
        }
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-04")
        self.assertEqual([s["symbol"] for s in eligible], ["600000"])
        self.assertEqual(exclusions[0]["reason"], "low_liquidity")

    def test_missing_data_fails_open(self) -> None:
        signals = [_signal("600000")]
        facts = {"600000": {"symbol": "600000"}}
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-04")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(exclusions, [])

    def test_filter_disabled_by_config_keeps_everything(self) -> None:
        signals = [_signal("301117")]
        facts = {"301117": _fact("ST佳缘")}
        config = _config(pool_exclude_st=False, min_avg_amount_20d=0.0, min_listing_age_calendar_days=0)
        eligible, exclusions = _pool_quality_filter(signals, facts, config, "2026-09-04")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(exclusions, [])

    def test_cross_border_names_are_excluded(self) -> None:
        signals = [_signal("159502"), _signal("513310"), _signal("513290"), _signal("004320")]
        facts = {
            "159502": _fact("标普生物科技ETF嘉实"),
            "513310": _fact("中韩半导体ETF华泰柏瑞"),
            "513290": _fact("纳指生物科技ETF汇添富"),
            "004320": _fact("前海开源沪港深乐享生活"),
        }
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-06")
        self.assertEqual([s["symbol"] for s in eligible], ["513310"])
        self.assertEqual(
            [(e["symbol"], e["reason"]) for e in exclusions],
            [
                ("159502", "cross_border"),
                ("513290", "cross_border"),
                ("004320", "cross_border"),
            ],
        )

    def test_cross_border_filter_disabled_keeps_qdii(self) -> None:
        signals = [_signal("159502")]
        facts = {"159502": _fact("标普生物科技ETF嘉实")}
        config = _config(pool_exclude_cross_border=False)
        eligible, exclusions = _pool_quality_filter(signals, facts, config, "2026-09-06")
        self.assertEqual(len(eligible), 1)
        self.assertEqual(exclusions, [])

    def test_exclusions_preserve_rank_order(self) -> None:
        signals = [_signal(f"6{i:05d}") for i in range(6)]
        facts = {
            "600000": _fact("甲"),
            "600001": _fact("ST乙"),
            "600002": _fact("丙"),
            "600003": _fact("丁"),
            "600004": _fact("ST戊"),
            "600005": _fact("己"),
        }
        eligible, exclusions = _pool_quality_filter(signals, facts, _config(), "2026-09-04")
        self.assertEqual([s["symbol"] for s in eligible], ["600000", "600002", "600003", "600005"])
        self.assertEqual(
            [e["symbol"] for e in exclusions],
            ["600001", "600004"],
        )


if __name__ == "__main__":
    unittest.main()
