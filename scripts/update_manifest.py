#!/usr/bin/env python3
"""Regenerate the release file inventory and SHA-256 manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
INVENTORY = ROOT / "FILE_INVENTORY.csv"
EXCLUDED = {MANIFEST.name, INVENTORY.name}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if ".git" in relative.parts or relative.as_posix() in EXCLUDED:
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def main() -> None:
    records: list[tuple[str, int, str]] = []
    for path in release_files():
        relative = path.relative_to(ROOT).as_posix()
        records.append((relative, path.stat().st_size, sha256(path)))

    with INVENTORY.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        writer.writerows(records)

    MANIFEST.write_text(
        "".join(f"{digest}  {relative}\n" for relative, _, digest in records),
        encoding="utf-8",
    )
    total_bytes = sum(size for _, size, _ in records)
    print(f"Wrote {len(records):,} records ({total_bytes:,} bytes)")


if __name__ == "__main__":
    main()
