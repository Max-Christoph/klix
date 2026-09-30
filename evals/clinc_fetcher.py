"""Automated fetcher and SHA256 verifier for the CLINC150 dataset.

Downloads data_full.json directly from the official clinc/oos-eval GitHub repository.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

CLINC_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
EXPECTED_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
TARGET_DIR = Path(__file__).parent / "data" / "clinc"
TARGET_FILE = TARGET_DIR / "data_full.json"


def fetch_clinc_data(force: bool = False) -> Path:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not force:
        # Check existing hash
        h = hashlib.sha256(TARGET_FILE.read_bytes()).hexdigest()
        if h == EXPECTED_SHA256:
            print(f"CLINC dataset already exists and verified: {TARGET_FILE}")
            return TARGET_FILE
        else:
            print(f"Existing file hash mismatch ({h} != {EXPECTED_SHA256}). Re-downloading...")

    print(f"Downloading CLINC150 dataset from {CLINC_URL}...")
    req = urllib.request.Request(CLINC_URL, headers={"User-Agent": "klix-eval/1.0"})
    with urllib.request.urlopen(req) as resp, open(TARGET_FILE, "wb") as f:
        f.write(resp.read())

    # Verify SHA256
    actual_hash = hashlib.sha256(TARGET_FILE.read_bytes()).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        TARGET_FILE.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded file failed SHA256 check!\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    print(f"Successfully downloaded and verified: {TARGET_FILE} ({TARGET_FILE.stat().st_size} bytes)")
    return TARGET_FILE


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    try:
        fetch_clinc_data(force=force_download)
    except Exception as e:
        print(f"Error fetching CLINC data: {e}", file=sys.stderr)
        sys.exit(1)
