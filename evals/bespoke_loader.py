"""Load the bespoke/JevBench comparison datasets into a uniform JSONL format.

WHY THIS FILE EXISTS, AND WHAT IT IS NOT
----------------------------------------
The three datasets here (MASSIVE, PAWS, banking77) belong to tasks klix was not
built for: spoken-language intent routing, paraphrase identification, and bank
customer-service intents. klix is a support/process text classifier whose
`Choice` head needs **anchor texts**, and none of these datasets ship anchors.

So this is a **repurposing**, exactly as `evals/run_jevbench.py` already is for
JevBench's `choice` tier. The numbers produced from these files are NOT
comparable to published Tev1 / Nimble / JevBench figures — there is no shared
System-One benchmark, as the JevBench source itself states. Every result file
carries a `caveat` field saying so. See `evals/data/bespoke/PROVENANCE.md`.

Record format (one JSON object per line), identical across all three sources:

    {"id": str, "text": str, "label": str, "options": list[str]}

Loaders
-------
One channel for everything: the HF `refs/convert/parquet` branch, read with
pyarrow. Verified available for all three datasets, and the only workable path:

  * `datasets` itself would fail here. `paws` and `banking77` are legacy
    script-based datasets that `datasets` >= 3 no longer loads, and
    `qanastek/MASSIVE` is not served by the datasets-server at all (HTTP 501).
  * the datasets-server `/rows` endpoint covers MASSIVE, but needs 116 paged
    requests per split (100 rows/page) and takes minutes per language. The
    parquet file for the same split downloads in ~1 s (11 514 rows measured).
    `fetch_rows_api` is kept below as a documented fallback for datasets that
    have no parquet branch.

Run as
    uv run --with pyarrow python -m evals.bespoke_loader --dataset ...
pyarrow is deliberately NOT imported at module top level (a test asserts this)
so the module stays importable without it.

Why not `datasets.load_dataset`?
--------------------------------
`datasets` is absent from this project's environment on purpose (klix ships three
runtime deps; evals should not add a fourth just to read three files). It would
also have failed here, per the two facts above — both verified before writing this.

DATA IS NOT VENDORED
--------------------
Nothing is copied into the repository. Files land in `evals/data/bespoke/` and
are gitignored (decision §7.4); the repo keeps this loader plus PROVENANCE.md.
A network failure aborts loudly — it never falls back to a stale cached file,
because that would silently change what "8000 cases" means.

Run:
    uv run --with pyarrow python -m evals.bespoke_loader --dataset massive --lang de
    uv run --with pyarrow python -m evals.bespoke_loader --dataset massive --lang en
    uv run --with pyarrow python -m evals.bespoke_loader --dataset banking77
    uv run --with pyarrow python -m evals.bespoke_loader --dataset paws
    uv run --with pyarrow python -m evals.bespoke_loader --all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "bespoke"

PARQUET_BRANCH = "refs%2Fconvert%2Fparquet"
HF_RESOLVE = "https://huggingface.co/datasets/{dataset}/resolve/{branch}/{path}"
ROWS_API = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "klix-eval/0.10 (bespoke eval loader; repo: klix)"

PAWS_OPTIONS = ["paraphrase", "not_paraphrase"]
PAWS_SEP = " [SEP] "


# --------------------------------------------------------------------------
# normalisation (pure, tested without network)
# --------------------------------------------------------------------------

def normalize_massive(row: dict, options: list[str]) -> dict:
    """MASSIVE row -> uniform record.

    `options` is the full intent list for the dataset (60), passed in rather
    than derived per row so that every record carries an identical option list —
    a per-row option set would make the task easier than it is.
    """
    label = row["label"]
    if label not in options:
        raise ValueError(f"label {label!r} not among options")
    return {
        "id": str(row["id"]),
        "text": row["text"],
        "label": label,
        "options": list(options),
    }


def intents_from_rows(rows: list[dict]) -> list[str]:
    """The option list, taken from the data — never typed by hand.

    Used on the `train` split, which carries all 60 intents; the `test` split is
    ordered by class, so a small sample of it does not.
    """
    return sorted({r["label"].strip() for r in rows if r.get("label")})


def normalize_paws_pair(row: dict) -> dict:
    """PAWS sentence pair -> uniform record (join strategy §7.3, variant a).

    PAWS is a *pair* task; the uniform format has one `text` field. Joining with
    a marker keeps the adversarial property intact — the scrambled word order
    that makes PAWS hard lives in the relationship between the two sentences,
    and dropping either side would destroy it.
    """
    label = row["label"]
    if label in (0, "0"):
        name = "not_paraphrase"
    elif label in (1, "1"):
        name = "paraphrase"
    else:
        raise ValueError(f"unexpected PAWS label {label!r}")
    return {
        "id": str(row["id"]),
        "text": f"{row['sentence1']}{PAWS_SEP}{row['sentence2']}",
        "label": name,
        "options": list(PAWS_OPTIONS),
    }


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

def _fetch(url: str, retries: int = 4, timeout: int = 120) -> bytes:
    """GET with backoff. Raises on final failure — never returns partial data."""
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


def _parquet_url(dataset: str, config: str, split: str, shard: str = "0000.parquet") -> str:
    path = f"{config}/{split}/{shard}"
    return HF_RESOLVE.format(dataset=dataset, branch=PARQUET_BRANCH, path=path)


def fetch_parquet_rows(dataset: str, config: str, split: str) -> list[dict]:
    """Read one parquet file from the auto-converted HF branch.

    pyarrow is imported here, not at module level, so importing this module does
    not require it.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415  (deliberate: eval-only dep)

    url = _parquet_url(dataset, config, split)
    payload = _fetch(url)
    table = pq.read_table(io.BytesIO(payload))
    return table.to_pylist()


def fetch_parquet_label_names(dataset: str, config: str, split: str) -> list[str] | None:
    """The class names behind an integer label column, or None if not encoded.

    banking77 stores `label` as int64 and carries the 77 class names ONLY in the
    parquet schema metadata (`huggingface` -> features -> label -> names). Reading
    the rows alone yields integers 0..76, which look like perfectly valid labels
    until you notice they are indices — the first version of this loader scored
    `accuracy=None` for exactly that reason.

    The order of `names` was cross-checked against the dataset's own README listing
    (77 entries, identical except index 53 which the metadata spells
    "reverted_card_payment?"); the metadata is used because it is what the column
    actually indexes into.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    payload = _fetch(_parquet_url(dataset, config, split))
    table = pq.read_table(io.BytesIO(payload))
    raw = (table.schema.metadata or {}).get(b"huggingface")
    if not raw:
        return None
    try:
        info = json.loads(raw.decode("utf-8"))
        names = info["info"]["features"]["label"]["names"]
    except (KeyError, ValueError, UnicodeDecodeError):
        return None
    return list(names) if names else None


def fetch_rows_api(dataset: str, config: str, split: str,
                   page: int = 100, pace: float = 0.0) -> list[dict]:
    """Read a split through the datasets-server /rows endpoint (stdlib only)."""
    rows: list[dict] = []
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        url = (f"{ROWS_API}?dataset={urllib.parse.quote(dataset, safe='')}"
               f"&config={urllib.parse.quote(config, safe='')}"
               f"&split={urllib.parse.quote(split, safe='')}"
               f"&offset={offset}&length={page}")
        payload = json.loads(_fetch(url).decode("utf-8"))
        total = payload.get("num_rows_total")
        batch = payload.get("rows", [])
        if not batch:
            break
        rows.extend(item["row"] for item in batch)
        offset += len(batch)
        if pace:
            time.sleep(pace)
    if total is not None and len(rows) != total:
        raise RuntimeError(f"{dataset}:{config}:{split}: got {len(rows)} of {total} rows")
    return rows


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)  # atomic: a half-written file must never look complete


def load_massive(lang: str, split: str = "test") -> tuple[Path, Path]:
    """MASSIVE `test` for one language, plus its `train` rows for few-shot anchors.

    Both splits are written: the `test` file is what gets scored, the `train`
    file is what `evals/bespoke_anchors.few_shot_anchors()` draws the k=3 anchors
    from. Keeping them in separate files is what makes the no-leakage check
    possible at all — one file could not be checked against itself.

    Note: MASSIVE's own `train`/`test` split has no deliberate sentence-level
    separation guarantee, so a low overlap count is reported, not forbidden.
    """
    config = lang  # configs are `de` / `en`, not `de-DE` / `en-US`

    train_rows = fetch_parquet_rows("mteb/amazon_massive_intent", config, "train")
    options = intents_from_rows(train_rows)
    print(f"  {lang}: {len(options)} intents from train ({len(train_rows)} rows)")
    if len(options) != 60:
        print(f"  WARNING: expected 60 intents, found {len(options)}")

    train_out = DATA_DIR / f"massive_intent.{lang}.train.jsonl"
    _write_jsonl(train_out, [
        {"id": str(r["id"]), "text": r["text"], "label": r["label"], "options": options}
        for r in train_rows
    ])

    test_rows = fetch_parquet_rows("mteb/amazon_massive_intent", config, split)
    records = [normalize_massive(r, options) for r in test_rows]
    # normalize_massive raises on an unknown label, so reaching here means the
    # test split only uses intents that the option list carries.
    out = DATA_DIR / f"massive_intent.{lang}.jsonl"
    _write_jsonl(out, records)
    print(f"  {lang}: {len(records)} test records -> {out}")
    print(f"  {lang}: {len(train_rows)} train records -> {train_out}")
    return out, train_out


def resolve_banking77_label(value, class_names: list[str] | None) -> str:
    """Turn banking77's integer label index into its class name.

    The dataset ships `label` as an integer index; the name lives only in the
    parquet schema metadata. A loader that skips this step produces records whose
    `label` is `11` while `options` holds strings — every prediction then looks
    like a parse failure, and the run reports `accuracy=None` with full coverage
    of nothing. That is exactly what happened on the first attempt, so this
    conversion is a named, tested function rather than an inline expression.

    Passing `class_names=None` is an error, not a fallback: guessing names from
    indices would be inventing labels.
    """
    if isinstance(value, str) and not value.isdigit():
        return value  # already a name (some republications do this)
    if class_names is None:
        raise ValueError(
            "banking77 label is an integer index but the parquet schema carries no "
            "class names — cannot resolve the label without inventing one"
        )
    idx = int(value)
    if not 0 <= idx < len(class_names):
        raise ValueError(f"banking77 label index {idx} outside 0..{len(class_names) - 1}")
    return class_names[idx]


def load_banking77(split: str = "test") -> tuple[Path, Path]:
    """banking77 `test` + `train` (the latter for few-shot anchors, §7.2)."""
    class_names = fetch_parquet_label_names("banking77", "default", split)
    if class_names is None:
        raise RuntimeError("banking77: no class names in the parquet metadata — refusing "
                           "to write records with integer labels")
    rows = fetch_parquet_rows("banking77", "default", split)
    train = fetch_parquet_rows("banking77", "default", "train")

    # The option list must be the *full* class set, not just the classes that
    # happen to appear in `test` — otherwise the task is easier than the 77 the
    # dataset claims.
    observed = {resolve_banking77_label(r["label"], class_names) for r in rows}
    observed |= {resolve_banking77_label(r["label"], class_names) for r in train}
    options = sorted(observed)
    print(f"  banking77: {len(options)} classes (train {len(train)} / {split} {len(rows)})")
    if len(options) != 77:
        print(f"  WARNING: expected 77 classes, found {len(options)}")

    def _recs(source: list[dict]) -> list[dict]:
        out = []
        for r in source:
            label = resolve_banking77_label(r["label"], class_names)
            if label not in options:
                raise ValueError(f"label {label!r} missing from class list")
            out.append({
                "id": str(r.get("id", len(out))),
                "text": r["text"],
                "label": label,
                "options": list(options),
            })
        return out

    train_out = DATA_DIR / "banking77.train.jsonl"
    _write_jsonl(train_out, _recs(train))
    out = DATA_DIR / "banking77.jsonl"
    _write_jsonl(out, _recs(rows))
    print(f"  banking77: {len(rows)} test records -> {out}")
    print(f"  banking77: {len(train)} train records -> {train_out}")
    return out, train_out


def load_paws(split: str = "test") -> Path:
    rows = fetch_parquet_rows("paws", "labeled_final", split)
    records = [normalize_paws_pair(r) for r in rows]
    counts = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    # PAWS labeled_final is NOT 50/50; printing the split keeps that visible so
    # nobody later reads a 50% accuracy as "chance level".
    print(f"  paws: label distribution {counts}")
    out = DATA_DIR / "paws_adversarial.jsonl"
    _write_jsonl(out, records)
    print(f"  paws: {len(records)} records -> {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["massive", "banking77", "paws"])
    ap.add_argument("--lang", choices=["de", "en"], default=None,
                    help="for --dataset massive (default: both)")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if not args.all and not args.dataset:
        ap.error("pass --dataset ... or --all")

    written: list[Path] = []
    if args.all or args.dataset == "massive":
        for lang in ([args.lang] if args.lang else ["de", "en"]):
            written.extend(load_massive(lang))
    if args.all or args.dataset == "banking77":
        written.extend(load_banking77())
    if args.all or args.dataset == "paws":
        written.append(load_paws())

    print("\nwritten:")
    for p in written:
        n = sum(1 for _ in p.open(encoding="utf-8"))
        print(f"  {p}  {n} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
