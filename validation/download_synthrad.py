"""Download a small, multi-centre subset of SynthRAD2023 without the 11 GB ZIP."""

from __future__ import annotations

import shutil
from pathlib import Path

from remotezip import RemoteZip

ARCHIVE_URL = "https://zenodo.org/api/records/7260705/files/Task2.zip/content"
CASES = (
    "2PA003",
    "2PA024",
    "2PA056",
    "2PB004",
    "2PB043",
    "2PB119",
    "2PC002",
    "2PC003",
    "2PC037",
)
FILES = ("cbct.nii.gz", "ct.nii.gz", "mask.nii.gz")
OUTPUT_ROOT = Path(__file__).parents[1] / ".validation-data" / "synthrad2023"
CHUNK_SIZE = 8 * 1024 * 1024


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with RemoteZip(ARCHIVE_URL) as archive:
        for case in CASES:
            case_dir = OUTPUT_ROOT / case
            case_dir.mkdir(exist_ok=True)
            for filename in FILES:
                output = case_dir / filename
                if output.is_file():
                    continue
                member = f"Task2/pelvis/{case}/{filename}"
                partial = output.with_name(f"{output.name}.part")
                with archive.open(member) as source, partial.open("wb") as target:
                    shutil.copyfileobj(source, target, length=CHUNK_SIZE)
                partial.replace(output)
                print(
                    f"Downloaded {case}/{filename} ({output.stat().st_size:,} bytes)",
                    flush=True,
                )


if __name__ == "__main__":
    main()
