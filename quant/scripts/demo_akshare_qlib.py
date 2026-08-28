"""Read-only smoke test for the AkShare-backed Qlib provider."""
from __future__ import annotations

import argparse
import multiprocessing
import pathlib

import qlib
from qlib.config import REG_CN
from qlib.data import D

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PROVIDER = ROOT / "qlib_data" / "akshare_data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        type=pathlib.Path,
        default=DEFAULT_PROVIDER,
        help="AkShare Qlib provider directory",
    )
    parser.add_argument(
        "--symbol",
        default="SH600519",
        help="Qlib instrument to query",
    )
    args = parser.parse_args()

    provider = args.provider.resolve()
    if not provider.exists():
        raise FileNotFoundError(
            f"Qlib provider does not exist: {provider}. "
            "Run scripts/build_qlib_data.py first."
        )

    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1)
    instruments = D.instruments(market="all")
    symbols = D.list_instruments(
        instruments=instruments,
        start_time="2000-01-01",
        end_time="2099-12-31",
        freq="day",
        as_list=True,
    )
    if args.symbol not in symbols:
        raise ValueError(
            f"{args.symbol} is not in provider; available: {symbols}"
        )

    calendar = D.calendar(freq="day")
    query_start = calendar[max(0, len(calendar) - 10)]
    query_end = calendar[-1]
    latest = D.features(
        [args.symbol],
        ["$open", "$high", "$low", "$close", "$vwap", "$volume", "$factor"],
        start_time=query_start,
        end_time=query_end,
        freq="day",
    ).tail(5)
    if latest.empty:
        raise ValueError(f"Qlib returned no rows for {args.symbol}")

    print(f"Qlib version: {qlib.__version__}")
    print(f"Provider: {provider}")
    print(f"Instruments ({len(symbols)}): {', '.join(symbols)}")
    print(f"\nLatest rows for {args.symbol}:")
    print(latest)
    print("\nRead-only AkShare -> Qlib verification passed.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
