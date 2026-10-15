"""Download and safely extract the official AllenAI SciFact data archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen

OFFICIAL_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
REQUIRED_FILES = (
    "corpus.jsonl",
    "claims_train.jsonl",
    "claims_dev.jsonl",
    "claims_test.jsonl",
)


def has_complete_dataset(data_dir: Path) -> bool:
    """Return whether all expected original JSONL files are present."""

    return all((data_dir / filename).is_file() for filename in REQUIRED_FILES)


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise tarfile.TarError(f"Archive contains unsafe path: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise tarfile.TarError(f"Archive contains unsupported link or device: {member.name}")
    return members


def _find_extracted_data_dir(staging_dir: Path) -> Path:
    candidates = [
        directory
        for directory in (staging_dir, *staging_dir.rglob("data"))
        if directory.is_dir() and has_complete_dataset(directory)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Archive did not contain exactly one directory with the official SciFact JSONL files"
        )
    return candidates[0]


def download_archive(url: str, destination: Path) -> str:
    """Download an archive and return its SHA-256 digest."""

    digest = hashlib.sha256()
    try:
        with urlopen(url, timeout=60) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
    except (OSError, URLError) as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download official SciFact archive from {url}: {error}"
        ) from error
    return digest.hexdigest()


def install_archive(archive_path: Path, data_dir: Path) -> None:
    """Extract the archive into ``data_dir`` after path-safety checks."""

    with tempfile.TemporaryDirectory(prefix="argus-scifact-") as temporary:
        staging_dir = Path(temporary)
        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(staging_dir, members=_safe_members(archive), filter="data")
        except (OSError, tarfile.TarError) as error:
            raise RuntimeError(
                f"Could not extract SciFact archive {archive_path}: {error}"
            ) from error

        extracted_data_dir = _find_extracted_data_dir(staging_dir)
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        if data_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing data directory: {data_dir}")
        shutil.copytree(extracted_data_dir, data_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official AllenAI SciFact dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/scifact"))
    parser.add_argument(
        "--url", default=OFFICIAL_URL, help="Official archive URL (default: %(default)s)"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    if has_complete_dataset(data_dir):
        print(f"SciFact data already present at {data_dir}; no download needed.")
        return 0
    if data_dir.exists():
        print(
            f"Data directory exists but is incomplete: {data_dir}. "
            "Move it aside before running this command.",
            file=sys.stderr,
        )
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="argus-scifact-download-") as temporary:
            archive_path = Path(temporary) / "data.tar.gz"
            checksum = download_archive(args.url, archive_path)
            install_archive(archive_path, data_dir)
        provenance = {
            "source_url": args.url,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": checksum,
        }
        (data_dir / "PROVENANCE.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    except RuntimeError as error:
        print(f"SciFact download failed: {error}", file=sys.stderr)
        return 1

    print(f"SciFact data installed at {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
