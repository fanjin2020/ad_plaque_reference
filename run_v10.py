#!/usr/bin/env python3
"""Run all five v10 experiments from the public workbooks."""
import argparse
import os
from pathlib import Path
import sys
import traceback

from run import environment, locked_packages


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results_v10")
    parser.add_argument("--input-dir", type=Path, help="Directory containing the four checksum-matched workbooks")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--check", type=Path, help="Verify a completed output directory")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.11, 3.12 or 3.13")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    from paper4.source_inputs import verify_output

    try:
        if args.check:
            print(f"PASS: {verify_output(args.check)} files verified in {args.check.resolve()}")
            return 0
        pins = locked_packages()
        status = environment(pins, Path(__file__).resolve())
        if status is not None:
            return status
        for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[key] = "1"
        os.environ["MPLBACKEND"] = "Agg"
        from paper4.v10.workflow import run

        result = run(args, pins)
        print(f"PASS: five v10 experiments completed and {verify_output(result)} files verified.")
        print(result)
        return 0
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as error:
        if os.environ.get("PAPER4_DEBUG") == "1":
            traceback.print_exc()
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
