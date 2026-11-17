from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(prog="insgnss")
    parser.add_argument("--runs", type=int, default=50, help="Monte Carlo runs")
    args = parser.parse_args()
    from .report import run

    print(json.dumps(run(args.runs), indent=2))


if __name__ == "__main__":
    main()
