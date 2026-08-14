#!/usr/bin/env python3
"""Fetch real 3GPP specifications from the public 3GPP archive.

The bundled corpus is a condensed excerpt so the project runs offline. To
evaluate on the genuine article, use this script — it pulls the published ZIPs
from https://www.3gpp.org/ftp/Specs/archive/, unpacks the .doc/.docx inside,
and drops them in the corpus directory.

    python scripts/download_3gpp.py                 # the default 5G core set
    python scripts/download_3gpp.py 23.501 38.331   # specific specifications
    python scripts/download_3gpp.py --list

Note on .doc: some older specs ship as legacy Word .doc, which python-docx
cannot read. The script reports these; convert them with LibreOffice:

    soffice --headless --convert-to docx --outdir data/corpus data/corpus/*.doc
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import requests

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from config import settings  # noqa: E402

ARCHIVE_ROOT = "https://www.3gpp.org/ftp/Specs/archive"
TIMEOUT = 120

DEFAULT_SPECS = {
    "23.501": "System architecture for the 5G System",
    "23.502": "Procedures for the 5G System",
    "24.501": "NAS protocol for 5GS (Stage 3)",
    "33.501": "Security architecture and procedures for 5G System",
    "38.300": "NR and NG-RAN Overall Description",
    "38.331": "NR Radio Resource Control (RRC) protocol specification",
    "38.321": "NR Medium Access Control (MAC) protocol specification",
    "23.503": "Policy and charging control framework for the 5G System",
}


def _series(spec: str) -> str:
    return spec.split(".")[0]


def _list_versions(spec: str) -> list[str]:
    """Scrape the spec's archive directory for available version ZIPs."""
    url = f"{ARCHIVE_ROOT}/{_series(spec)}_series/{spec}/"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    # Filenames look like 23501-h90.zip  (h = Rel-17, 9.0 = version)
    pattern = re.compile(rf"{spec.replace('.', '')}-([a-z0-9]+)\.zip", re.IGNORECASE)
    return sorted({m.group(0) for m in pattern.finditer(response.text)})


def download_spec(spec: str, destination: Path) -> list[Path]:
    versions = _list_versions(spec)
    if not versions:
        print(f"  {spec}: no versions found in the archive")
        return []

    # Archive listings sort lexically; the last entry is the newest version.
    latest = versions[-1]
    url = f"{ARCHIVE_ROOT}/{_series(spec)}_series/{spec}/{latest}"
    print(f"  {spec}: downloading {latest}")

    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()

    written: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for name in archive.namelist():
            suffix = Path(name).suffix.lower()
            if suffix not in {".doc", ".docx", ".pdf"}:
                continue
            target = destination / f"{spec.replace('.', '')}-{Path(name).name}"
            target.write_bytes(archive.read(name))
            written.append(target)
            note = "  (legacy .doc - convert to .docx before ingesting)" if suffix == ".doc" else ""
            print(f"    -> {target.name} ({target.stat().st_size // 1024} KB){note}")

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 3GPP specifications.")
    parser.add_argument("specs", nargs="*", help="Spec numbers, e.g. 23.501 38.331")
    parser.add_argument("--out", type=Path, default=settings.corpus_dir)
    parser.add_argument("--list", action="store_true", help="List the default specification set.")
    args = parser.parse_args()

    if args.list:
        print("Default specification set:\n")
        for spec, title in DEFAULT_SPECS.items():
            print(f"  TS {spec:<8} {title}")
        return 0

    specs = args.specs or list(DEFAULT_SPECS)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(specs)} specification(s) into {args.out}\n")

    failures = []
    for spec in specs:
        if not re.fullmatch(r"\d{2}\.\d{3}", spec):
            print(f"  {spec}: skipped (expected NN.NNN format)")
            failures.append(spec)
            continue
        try:
            download_spec(spec, args.out)
        except requests.HTTPError as exc:
            print(f"  {spec}: HTTP error - {exc}")
            failures.append(spec)
        except Exception as exc:
            print(f"  {spec}: {exc.__class__.__name__} - {exc}")
            failures.append(spec)

    print("\nDone. Next: python scripts/ingest.py")
    if failures:
        print(f"Failed: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
