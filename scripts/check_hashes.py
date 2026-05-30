from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")
    failures: list[str] = []
    checked = 0
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rel = row["relative_path"]
            path = ROOT / rel
            if not path.exists():
                failures.append(f"missing: {rel}")
                continue
            expected_size = int(row["file_size_bytes"])
            actual_size = path.stat().st_size
            if actual_size != expected_size:
                failures.append(f"size mismatch: {rel} expected {expected_size} got {actual_size}")
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                failures.append(f"sha256 mismatch: {rel}")
            checked += 1
    if failures:
        print("HASH_CHECK: FAIL")
        for item in failures:
            print(item)
        raise SystemExit(1)
    print(f"HASH_CHECK: PASS ({checked} files)")


if __name__ == "__main__":
    main()
