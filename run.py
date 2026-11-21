#!/usr/bin/env python3
"""Reproduce the Paper 4 JPR v9 analysis, tables and figures in one command."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
import venv


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def locked_packages():
    return dict(line.split("==", 1) for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines() if line)


def installed(pins):
    try:
        return all(version(package) == expected for package, expected in pins.items())
    except PackageNotFoundError:
        return False


def environment(pins, entrypoint=Path(__file__).resolve()):
    """Use the current interpreter when it already matches the lock."""
    if installed(pins):
        return None
    cache = ROOT / ".paper4-env"
    python = cache / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    lock_path = cache / "requirements.sha256"
    stamp = sha256(REQUIREMENTS.read_bytes()).hexdigest()
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(cache)
    if not lock_path.is_file() or lock_path.read_text(encoding="ascii") != stamp:
        command = [str(python), "-m", "pip", "install", "--only-binary=:all:", "-r", str(REQUIREMENTS)]
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError("Could not install the locked Python packages.\n" + "\n".join(process.stdout.splitlines()[-12:]))
        lock_path.write_text(stamp, encoding="ascii")
    if Path(sys.prefix).resolve() == cache.resolve():
        raise RuntimeError("The local Python environment does not match requirements.txt")
    return subprocess.call([str(python), str(entrypoint), *sys.argv[1:]])


def reproduce(args, pins):
    started = time.monotonic()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    from paper4.publication.results import generate
    from paper4.reproduce_analyses import run
    from paper4.source_inputs import acquire, output_hashes, sha256_file, verify_output
    from paper4.reference_extensions.io import clean
    dest = args.out.resolve()
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = dest.with_name(dest.name + "_" + stamp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper4-work-", dir=dest.parent) as path:
        work = Path(path)
        print("Verifying six public inputs", flush=True)
        data, records = acquire(work, args.input_dir)
        output = work / "final"
        output.mkdir()
        print("Rebuilding paired analyses and running simulations", flush=True)
        with redirect_stdout(io.StringIO()):
            analyses = run(work, data, args.workers)
        print("Generating tables and figures and checking numerical results", flush=True)
        with redirect_stdout(io.StringIO()):
            tables = generate(analyses, data, output)
            from paper4.validation import validate_figures, validate_numbers
            numbers = validate_numbers(tables)
            figures = validate_figures(output)
        report = {
            "status": "passed",
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "data_source": "publisher_downloads" if args.input_dir is None else "checksum_verified_local_copies",
            "public_source_bytes": sum(item["bytes"] for item in records),
            "inputs": records,
            "annotation_sha256": {
                source.name: sha256_file(source) for source in sorted((ROOT / "paper4" / "plaque_validation").glob("uniprot_*.json"))
            },
            "environment": {"python": platform.python_version(), "platform": platform.platform(), **pins},
            "tables": {"main": 3, "supplement_reading_tables": 3,
                       "numbered_supplement": 83, "data_s1": 18, "data_s2": 12},
            "figures": figures,
            "numeric_checks": numbers,
            "empirical_simulation_checks": analyses.empirical_check,
            "phenotype_and_genotype_checks": analyses.followup_check,
            "outputs_sha256": output_hashes(output),
        }
        (output / "verification.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        verify_output(output)
        output.rename(dest)
    print(f"PASS: {len(figures)} figures, 3 main tables, 83 numbered supplementary tables and Data S1–S2.")
    print(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results", help="Final output directory")
    parser.add_argument("--input-dir", type=Path, help="Use six already downloaded, checksum-matched public files")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Simulation workers")
    parser.add_argument("--check", type=Path, help="Verify a completed output directory")
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.11, 3.12 or 3.13")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    try:
        if args.check:
            from paper4.source_inputs import verify_output
            count = verify_output(args.check)
            print(f"PASS: {count} files verified in {args.check.resolve()}")
            return 0
        pins = locked_packages()
        child = environment(pins)
        if child is not None:
            return child
        reproduce(args, pins)
    except (AssertionError, KeyError, OSError, RuntimeError, ValueError) as exc:
        if os.environ.get("PAPER4_DEBUG") == "1":
            traceback.print_exc()
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
