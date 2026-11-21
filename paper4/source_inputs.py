"""Download and verify the public input versions used by the article."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
STUDY_DOWNLOAD_LIMIT = 10_000_000_000


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_file(path, record):
    if (not path.is_file() or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]):
        raise ValueError(f"Input size or SHA-256 mismatch for {path.name}")


def download(folder, record):
    path = folder / record["name"]
    if path.exists():
        try:
            check_file(path, record)
            return
        except ValueError:
            pass
    request = Request(record["url"], headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response, partial.open("wb") as stream:
                size = 0
                while chunk := response.read(256 * 1024):
                    size += len(chunk)
                    if size > record["bytes"]:
                        raise ValueError("Received more bytes than specified by the input manifest")
                    stream.write(chunk)
            check_file(partial, record)
            os.replace(partial, path)
            return
        except (OSError, URLError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(
                    f"Public download failed for {path.name}: {exc}. "
                    f"Source: {record['url']}. "
                    "A verified copy can be supplied with --input-dir."
                ) from exc
            time.sleep(attempt + 1)
        finally:
            partial.unlink(missing_ok=True)


def acquire(work, input_dir=None):
    records = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    if sum(r["bytes"] for r in records) > STUDY_DOWNLOAD_LIMIT:
        raise ValueError("Input manifest exceeds the public download limit")
    if input_dir is None:
        folder = work / "inputs"
        folder.mkdir()
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda r: download(folder, r), records))
    else:
        folder = input_dir.resolve()
    for record in records:
        check_file(folder / record["name"], record)
    return folder, records


def output_hashes(path):
    return {p.relative_to(path).as_posix(): sha256_file(p)
            for p in sorted(path.rglob("*")) if p.is_file() and p.name != "verification.json"}


def verify_output(path):
    report = json.loads((path / "verification.json").read_text(encoding="utf-8"))
    if report["status"] != "passed":
        raise ValueError("Verification status is incomplete")
    expected, actual = report["outputs_sha256"], output_hashes(path)
    if actual != expected:
        differences = sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
        raise ValueError("Output file manifest differs: " + ", ".join(differences))
    return len(actual)
