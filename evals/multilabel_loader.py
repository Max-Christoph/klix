"""Loader for GoEmotions multi-label dataset (HF google-research-datasets/go_emotions).

Downloads the auto-converted parquet files directly from Hugging Face,
normalizes each row into:
    {"id": str, "text": str, "labels": list[str], "options": list[str]}
and writes to `evals/data/bespoke/go_emotions.{split}.jsonl`.

Following Klix's benchmark architecture:
- Zero repository bloat (files are saved to gitignored `evals/data/bespoke/`).
- Parquet read with pyarrow (imported locally, not at top level).
- Canonical 28 emotion taxonomy from Google Research GoEmotions.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "bespoke"
USER_AGENT = "klix-eval/0.11 (multilabel eval loader; repo: klix)"

GO_EMOTIONS_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/train/0000.parquet",
    "test": "https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/refs%2Fconvert%2Fparquet/simplified/test/0000.parquet",
}

EMOTION_NAMES = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]


def _fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    """GET with exponential backoff. Raises on failure."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep((10, 30, 60)[min(attempt, 2)])
                continue
            raise RuntimeError(f"GET failed {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"GET failed for {url}: {exc}") from exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"unreachable: {url}")


def fetch_parquet_split(split: str) -> list[dict]:
    """Fetch and decode GoEmotions split into Python dicts."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    if split not in GO_EMOTIONS_PARQUET_URLS:
        raise ValueError(f"Unknown split {split!r}, must be one of {list(GO_EMOTIONS_PARQUET_URLS)}")

    url = GO_EMOTIONS_PARQUET_URLS[split]
    payload = _fetch(url)
    table = pq.read_table(io.BytesIO(payload))
    return table.to_pylist()


def normalize_go_emotions_row(row: dict) -> dict:
    """Transform raw GoEmotions row into uniform multi-label record."""
    raw_labels = row.get("labels", [])
    # raw_labels is an iterable of integer indices 0..27
    labels = [EMOTION_NAMES[idx] for idx in raw_labels if 0 <= idx < len(EMOTION_NAMES)]
    return {
        "id": str(row["id"]),
        "text": str(row["text"]).strip(),
        "labels": sorted(labels),
        "options": list(EMOTION_NAMES),
    }


def download_go_emotions(target_dir: Path | None = None) -> dict[str, Path]:
    """Download train and test splits, write JSONL, return paths."""
    out_dir = target_dir or DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split in ("train", "test"):
        filename = "go_emotions.train.jsonl" if split == "train" else "go_emotions.jsonl"
        dest = out_dir / filename
        print(f"Fetching GoEmotions {split} split from Hugging Face...")
        rows = fetch_parquet_split(split)
        records = [normalize_go_emotions_row(r) for r in rows if str(r.get("text", "")).strip()]

        with open(dest, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"Wrote {len(records)} records to {dest}")
        paths[split] = dest

    return paths


def load_go_emotions_dataset(split: str = "test", target_dir: Path | None = None) -> list[dict]:
    """Load cached GoEmotions records, downloading if missing."""
    out_dir = target_dir or DATA_DIR
    filename = "go_emotions.train.jsonl" if split == "train" else "go_emotions.jsonl"
    dest = out_dir / filename
    if not dest.exists():
        download_go_emotions(target_dir=out_dir)

    records: list[dict] = []
    with open(dest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    download_go_emotions()
