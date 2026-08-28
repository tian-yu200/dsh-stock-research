"""Run the complete AkShare-to-Qlib research workflow."""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "configs" / "qlib_akshare_config.yaml"

def _run(*arguments: object) -> None:
    command = [sys.executable, *(str(argument) for argument in arguments)]
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fetch", action="store_true", help="使用现有 CSV，不访问 AkShare")
    parser.add_argument("--skip-backtest", action="store_true", help="只更新并转换数据")
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG, help="回测 YAML 配置")
    args = parser.parse_args()

    if not args.skip_fetch:
        _run(SCRIPTS / "fetch_stock_data.py")
        _run(SCRIPTS / "fetch_fund_data.py")
    _run(SCRIPTS / "build_qlib_data.py", "--replace")
    if not args.skip_backtest:
        _run(SCRIPTS / "run_backtest.py", "--config", args.config.resolve())

if __name__ == "__main__":
    main()
