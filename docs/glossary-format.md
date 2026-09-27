# The klix glossary format

A klix glossary answers one question: **which words mean the same concept?** It
is a flat lookup — `word -> concept` — because that is what the routing engine
needs to bridge a query in one language to anchors written in another.

This document is the format specification. You can write a glossary by hand; you
do not need klix's build pipeline, and you do not need to modify klix's source to
use your own. The machine-readable schema is
[`src/klix/data/glossary.schema.json`](../src/klix/data/glossary.schema.json).

## Two accepted shapes

### Bare concept map (original, still supported)

```json
{
  "conveyor":  {"de": ["foerderband", "förderband"], "en": ["conveyor"]},
  "urgent":    {"de": ["dringend", "eilig"],         "en": ["urgent", "asap"]}
}
```

Every top-level key is a concept. No version, no provenance. Fine for a quick
local file; use the versioned form for anything you share.

### Versioned document (preferred)

```json
{
  "schema_version": 1,
  "source": "curated",
  "license": "MIT",
  "description": "Manufacturing, IT and everyday office vocabulary.",
  "concepts": {
    "conveyor": {
      "de": ["foerderband", "förderband"],
      "en": ["conveyor", "conveyor belt"],
      "tags": ["manufacturing"]
    }
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | Format version. Currently `1`. A **newer** version is refused rather than parsed optimistically. |
| `concepts` | yes | Concept key → language map. |
| `source` | no | Where the glossary came from. |
| `license` | no | Licence of this glossary. Recorded verbatim. |
| `description` | no | Human-readable summary. |
| `tags` (per concept) | no | Free metadata labels. |

`glossary_source` / `glossary_license` are accepted as alternative spellings of
`source` / `license`, and `glossary_source` may also be an object
`{"source": ..., "license": ...}`.

## Concepts

A concept key is a stable identifier you choose — `conveyor`, `urgent`,
`password`. It also acts as a search term when it looks like a plain word (no
underscores), so `{"conveyor": {...}}` matches the English word `conveyor` even
if you forget to list it under `"en"`.

Concept names must not contain whitespace, and `tags` is reserved.

## Languages

**Any number of languages is allowed, and there is no fixed pair.** The engine
iterates whatever keys it finds; a concept may carry one language or ten:

```json
{
  "conveyor": {
    "de": ["förderband"],
    "en": ["conveyor"],
    "fr": ["convoyeur"],
    "pl": ["przenośnik"]
  }
}
```

Keys are ISO-639-1 (or `de-AT`-style variants). Unknown keys are not rejected —
a two-letter code the engine has never seen is still just a language bucket.

## The one rule: a term belongs to exactly one concept

klix's index is **flat**: one term resolves to one concept. So the same surface
word under two concepts is an **ambiguous mapping**, and ambiguity is the failure
mode that matters — the query is silently routed to the wrong concept rather
than failing visibly.

Therefore: the same term must not appear under two concepts, and must not appear
twice within one concept. This holds **across languages** too — if `lager` is a
`de` term of one concept and an `en` term of another, that is still ambiguous.

klix enforces this on load and on merge:

```python
from klix import Glossary, GlossaryConflict

try:
    Glossary.from_file("mine.json").assert_valid()
except GlossaryConflict as exc:
    print(exc)          # names the conflicting terms
```

`Glossary.merge(other)` refuses conflicting merges by default. Pass
`strict=False` only when you have inspected the conflicts and accept first-wins.

There is **no separate "domain conflict" level.** Earlier drafts had one; it was
removed deliberately. Domains are ordinary tags now, so adding a fourth, fifth or
entirely external glossary goes through exactly the same single rule as the
built-ins — no special case to fall through.

## Tags, not domains

`tags` is free-form metadata. The engine never reads it: tags do not affect
merging, validation or routing. They exist so a caller can select a subset:

```python
from klix import Glossary

g = Glossary.from_file("mine.json")
it_terms = {k: v for k, v in g.mapping.items() if "it" in g.tags.get(k, [])}
```

If you used an earlier version where the domain lived in a separate
`curated_domains.json`, that data is still available but is now carried by the
schema itself — see `klix.glossaries.curated_it()` for the filtering helper.

## Provenance: per glossary, not per concept

Provenance is recorded once per glossary. With 124 concepts you write one
`source`/`license` pair, not 124 — and mixing sources inside a single glossary is
a deliberately unsolved problem (it would need per-term provenance; see
`DATA_SOURCES.md`).

```python
from klix import GlossaryRegistry

reg = GlossaryRegistry(fallback="curated")
reg.register("acme", "acme_terms.json", source="Acme GmbH", license="CC-BY-4.0")
print(reg.provenance_report())
```

## Using your own glossary

Three equivalent routes — none requires editing klix:

```python
from klix import DecisionEngine, Glossary, GlossaryRegistry

# 1. directly, by path
eng = DecisionEngine(glossary="acme_terms.json")

# 2. explicitly, when you want the object
g = Glossary.from_file("acme_terms.json")

# 3. via a registry, for several named glossaries
reg = GlossaryRegistry(fallback="curated")
reg.register("acme", "acme_terms.json", source="Acme GmbH", license="CC-BY-4.0")
eng = DecisionEngine(glossary=reg.get("acme"))
merged = reg.merge("acme", "curated")      # refuses ambiguities
```

`Choice(glossary=...)` and `DecisionEngine(glossary=...)` each accept `None`, a
dict, a file path, or a `Glossary`.

## Validation summary

| Check | Enforced where | On failure |
|---|---|---|
| Shape (`language -> list[str]`) | `Glossary.load` | `ValueError`, nothing loaded |
| `schema_version` newer than known | `Glossary.load` | `ValueError`, refused |
| Term under two concepts | `assert_valid`, `merge` | `GlossaryConflict` |
| Term twice in one concept | `assert_valid`, `merge` | `GlossaryConflict` |
| Term in two languages, two concepts | `assert_valid`, `merge` | `GlossaryConflict` |
| Term collides with an anchor/criterion | `validate(anchors=...)` | reported, not raised |

`Glossary.validate()` returns findings instead of raising, for reporting.
`Glossary.conflicts()` returns just the ambiguous-mapping subset.
