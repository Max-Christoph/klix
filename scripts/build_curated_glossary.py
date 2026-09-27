"""Rebuild the curated glossary data files from the two content modules.

Pipeline, in order — each step is a gate that stops the build:

  1. `merge_sources` unions the sources and raises on any term collision
     (one term -> one concept; the index is flat, so a duplicate is an
     ambiguous mapping, never a merge detail)
  2. `Glossary.validate()` reports structural findings (duplicates, circular
     mappings, homographs) — must be empty
  3. `Glossary.assert_valid()` raises on the ambiguous-mapping subset
  4. the produced document is validated against the shipped JSON Schema

Writes:
  src/klix/data/curated_glossary.json   the concepts, versioned document format
  src/klix/data/curated_domains.json    the domain of each concept, as tags

Run: uv run python scripts/build_curated_glossary.py
"""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from klix.glossary import Glossary  # noqa: E402

OUT_GLOSSARY = REPO / "src" / "klix" / "data" / "curated_glossary.json"
OUT_DOMAINS = REPO / "src" / "klix" / "data" / "curated_domains.json"
SCHEMA = REPO / "src" / "klix" / "data" / "glossary.schema.json"

PROVENANCE = {"source": "klix curated (hand-written)", "license": "MIT"}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    core = _load(REPO / "scripts" / "curated_glossary_data.py", "curated_core")
    if "curated_glossary_ext" not in sys.modules:
        _load(REPO / "scripts" / "curated_glossary_ext.py", "curated_glossary_ext")

    domains = core.all_domains()

    print("=" * 78)
    print("BUILD: curated glossary")
    print("=" * 78)
    for d, m in domains.items():
        terms = sum(len(t) for v in m.values() for t in v.values())
        print(f"  source {d:14} {len(m):3d} concepts  {terms:4d} terms")

    # -- gate 1 ------------------------------------------------------------
    merged = core.merged()            # raises ValueError on a term collision
    print(f"\n  [1] merge_sources      OK  ({len(merged)} concepts)")

    # -- gates 2 + 3 -------------------------------------------------------
    g = Glossary(dict(merged))
    findings = g.validate()
    if findings:
        print(f"  [2] validate()         FAIL  {len(findings)} findings")
        for f in findings[:10]:
            print(f"        {f['kind']}: {f['message']}")
        return 1
    print("  [2] validate()         OK  (0 findings)")
    g.assert_valid()                  # raises GlossaryConflict
    print("  [3] assert_valid()     OK  (0 ambiguous mappings)")

    # -- write -------------------------------------------------------------
    tagmap = core.tags()
    glossary_doc = {
        "schema_version": 1,
        **PROVENANCE,
        "description": ("Curated DE/EN vocabulary for support, process and mail "
                        "routing: manufacturing, IT, everyday office."),
        "concepts": {k: {**v, "tags": tagmap.get(k, [])} for k, v in merged.items()},
    }
    domains_doc = {
        "schema_version": 1,
        "glossary_source": PROVENANCE,
        "description": ("Domain of each curated concept. Domains are free metadata "
                        "tags: they take no part in validation or merging."),
        "concepts": tagmap,
    }

    # -- gate 4 ------------------------------------------------------------
    try:
        import jsonschema
    except ImportError:
        print("  [4] schema validation SKIPPED (jsonschema not installed)")
    else:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        errs = list(jsonschema.Draft202012Validator(schema).iter_errors(glossary_doc))
        if errs:
            print(f"  [4] schema validation FAIL  {len(errs)} errors")
            for e in errs[:10]:
                print(f"        {list(e.path)}: {e.message}")
            return 1
        print("  [4] schema validation OK  (0 errors)")

    OUT_GLOSSARY.write_text(
        json.dumps(glossary_doc, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8")
    OUT_DOMAINS.write_text(
        json.dumps(domains_doc, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8")

    n_terms = sum(len(t) for v in merged.values() for t in v.values())
    print(f"\n  wrote {OUT_GLOSSARY.relative_to(REPO)}   "
          f"({len(merged)} concepts, {n_terms} terms, "
          f"{OUT_GLOSSARY.stat().st_size / 1024:.1f} KB)")
    print(f"  wrote {OUT_DOMAINS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
