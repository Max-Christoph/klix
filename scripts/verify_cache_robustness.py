"""Verify the build script's SPARQL cache is corruption-tolerant + atomic.

Checks the two hardening behaviours added after a build died with an
unexplained JSONDecodeError:
  1. a VALID cache entry is used without touching the network
  2. a CORRUPT entry is treated as a miss and refetched (never fatal)
  3. writes leave no .tmp files behind (atomic os.replace)

Run: uv run python scripts/verify_cache_robustness.py
"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_glossary", REPO / "scripts" / "build_default_glossary.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

d = pathlib.Path(tempfile.mkdtemp(prefix="klix_cache_test_"))
bg.CACHE_DIR = d
QUERY = "SELECT ?x WHERE { ?x ?y ?z } LIMIT 1"
key = hashlib.sha256(QUERY.encode("utf-8")).hexdigest()[:16]
cache_file = d / f"{key}.json"

failures = []

# 1. valid entry -> served from cache, exactly as stored
cache_file.write_text(json.dumps([{"x": {"value": "cached"}}]), encoding="utf-8")
got = bg.sparql(QUERY)
ok = got == [{"x": {"value": "cached"}}]
print(f"1. valid cache hit        : {got}  [{'OK' if ok else 'FAIL'}]")
if not ok:
    failures.append("valid cache hit")

# 2. corrupt entry -> self-heals by refetching, and stays usable afterwards
cache_file.write_text('{"results": {"bindings": [', encoding="utf-8")
rows = bg.sparql(QUERY)
ok = isinstance(rows, list) and len(rows) >= 1
print(f"2. corrupt entry recovers : {len(rows)} rows refetched  [{'OK' if ok else 'FAIL'}]")
if not ok:
    failures.append("corrupt entry recovery")

# 3. atomic writes -> no .tmp debris
tmp_left = [p.name for p in d.glob("*.tmp")]
ok = not tmp_left
print(f"3. no .tmp leftovers      : {tmp_left or 'none'}  [{'OK' if ok else 'FAIL'}]")
if not ok:
    failures.append("tmp leftovers")

# 4. the healed entry is valid JSON on disk
try:
    json.loads(cache_file.read_text(encoding="utf-8"))
    print("4. healed entry is valid  : OK")
except Exception as exc:  # noqa: BLE001
    print(f"4. healed entry is valid  : FAIL ({exc})")
    failures.append("healed entry invalid")

print()
print("RESULT:", "all checks passed" if not failures else f"FAILURES: {failures}")
sys.exit(1 if failures else 0)
