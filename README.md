# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![CI](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml/badge.svg)](https://github.com/Max-Christoph/klix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Sort text into categories, get yes/no flags, score on an axis — by writing
example sentences instead of training a model.**

A **high-throughput semantic decision engine**: dense sentence embeddings (ONNX
Runtime, CPU-only) feed decoupled decision heads, each reasoning in its own
mathematical space. ~46 ms per single call, ~13.9 ms per document in bulk
(72 docs/s → 100k documents in ~24 minutes).

```python
from klix import DecisionEngine, Choice, Score, Flag

engine = DecisionEngine()
engine.add_head(Choice(name="queue", options={
    "it_ops":   ["vpn down", "server unreachable", "laptop won't boot"],
    "ot_plant": ["robot cell stopped", "PLC fault", "cycle time deviation"],
    "facility": ["oil spill in hall 2", "heating broken"],
}))
engine.compile()

res = engine.decide("plc-34 reports a fault, conveyor belt stopped")
res.queue        # 'ot_plant'
```

No training run, no labelled dataset, no GPU, no API keys. The schema is a
declarative list of example sentences that lives in your repo, reads like
documentation, and changes in milliseconds — running fully offline on CPU.

**The mechanism is semantic, not lexical.** Every decision is made by vector
geometry: one dense sentence embedding per text, scored by cosine similarity
against anchor vectors (or their per-label centroids). No regex, no keyword
list, no rule pattern decides anything by itself. `rules` and `glossaries` are
**optional, deterministic guardrails layered on top** of that semantic core — a
`Rule` can force or boost a keyword hit, a `Glossary` can bridge synonyms across
languages — and a schema that uses neither behaves identically to one that has
them registered but never triggered.

**Honest framing:** klix is *packaging*, not a novel algorithm — the core is
embedding + nearest-anchor / logistic probe, standard since 2019. What it adds
is the schema abstraction, uncertainty handling, and explainability around that
core. Read [Why klix — and when it isn't the right
tool](#why-klix--and-when-it-isnt-the-right-tool) before adopting it; that
section states plainly where it wins, where a trained classifier wins, and
where a copy-pasted 30-line snippet is the better choice.

## Installation

```bash
uv add klix-engine
# or
pip install klix-engine
```

On first use, FastEmbed downloads the `paraphrase-multilingual-MiniLM-L12-v2` model
(**240 MB on disk**: 224 MB ONNX + 16 MB tokenizer, one-time, then cached locally;
~0.3 GB resident once loaded). Everything runs offline afterwards.

## What you get

Three head types, each in its own mathematical space. Register as many as you
need — each text is embedded exactly once, so head count barely affects latency.

```text
Text ──► HybridBackbone (FastEmbed dense + TF-IDF sparse, once, tens of ms)
              │
              ├──► Choice   (routing/classification: max-similarity + keyword boost)
              ├──► Score    (continuous axis: low/high anchors + sigmoid)
              ├──► Flag     (boolean: 2/3-class softmax with temperature)
              └──► custom heads (subclass BaseHead)
```

- **Shared backbone:** Each text is embedded and TF-IDF-transformed exactly once.
  Whether you register 3 heads or 50, the extraction cost stays the same.
- **Decoupled heads:** Adding options to one `Choice` never affects `Score` or `Flag`
  results. Every head encapsulates its own logic.
- **Declarative:** Define schemas with example sentences, call `compile()`, done.
- **Honest uncertainty:** Optional reject poles (`Choice(reject_anchors=...)`,
  `Flag(neutral_anchors=...)`) return `None` instead of guessing; the `Score` head
  reports a `coverage` signal so you know when a score is noise.
- **Language-agnostic anchors:** The backbone model is multilingual, so anchor
  sentences in any language work — German, English, mixed, whatever fits your domain.

## Quickstart

```python
from klix import DecisionEngine, Choice, Score, Flag, MultiLabel

engine = DecisionEngine()

engine.add_head(
    Choice(
        name="target",
        options={
            "it_ops": ["VPN down", "server unreachable", "laptop won't boot"],
            "ot_plant": ["robot cell stopped", "PLC fault", "plc-34 error", "cycle time deviation"],
            "finance": ["cost center over budget", "approve invoice"],
            "facility": ["oil spill in hall 2", "heating broken"],
        },
        # Optional: texts resembling these get value=None instead of a forced guess.
        reject_anchors=["casual office chat", "birthday wishes", "off topic request"],
    )
)

engine.add_head(
    MultiLabel(
        name="tags",
        options={
            "hardware": ["laptop won't boot", "broken screen", "PLC hardware failure"],
            "network": ["VPN disconnected", "wifi unreachable", "DNS issue"],
            "critical": ["production halted", "urgent line stop", "immediate danger"],
        },
        threshold=0.5,
    )
)

engine.add_head(
    Score(
        name="urgency",
        low_anchors=["routine maintenance", "casual question"],
        high_anchors=["emergency right now", "production line down", "acute danger"],
        min_val=0.0,
        max_val=3.0,
        # "topk" pools the best 2 anchors per pole (robust against a single
        # noisy anchor). Every result carries "coverage": if it is low (< ~0.3),
        # the text matched neither pole and the score is mostly noise.
        aggregation="topk",
    )
)

engine.add_head(
    Flag(
        name="is_security",
        true_anchors=["hacker attack", "ransomware infection", "data exfiltration"],
        false_anchors=["hardware broken", "ordinary IT problem", "network outage"],
        # Optional third pole: when "neutral" wins, value=None instead of True/False.
        neutral_anchors=["routine request", "general question", "other topic"],
        threshold=0.5,
    )
)

engine.compile()

res = engine.decide("plc-34 reports a fault, conveyor belt stopped immediately!")

print(res)                                   # e.g. <DecisionResult (61 ms): target=ot_plant, tags=['critical'], urgency=2.4, is_security=False>
print(res.target)                            # 'ot_plant'
print(res.tags)                              # ['critical'] (multi-label list of active categories >= threshold)
print(res.details("tags")["scores"])         # {'critical': 0.88, 'hardware': 0.42, 'network': 0.12} (continuous [0, 1])
print(res.urgency)                           # continuous score between 0.0 and 3.0
print(res.is_security)                       # True / False / None (neutral won)
print(res.details("is_security"))            # full dict: value, probability, probabilities
```

Exact score values depend on your anchors — always treat the outputs as calibrated
signals, not ground truth, and tune the anchors to your domain.

## Configuration

Every head and the engine expose meaningful knobs:

| Knob | Where | Effect |
|------|-------|--------|
| `options`, anchors | all heads | The schema itself — more/better example sentences are the main quality lever |
| `classifier` | `Choice` | `"nearest"` (default), `"linear"`, `"centroid"`, or `"auto"`. `"auto"` is a **deprecated alias for `"centroid"`** (v0.9.0): it used to guess anchor languages to pick nearest-vs-linear, and that heuristic is gone. **`"centroid"` scores against the mean anchor vector per label** — no training, fully deterministic, and measured to reach the trained probe's accuracy: cross-domain 71.4 % → 84.3 % (+12.9 pt, bootstrap CI [+2.9, +22.9]) and 273-case corpus 93.0 % → 96.7 % (+3.7 pt, CI [+1.5, +6.2]); neutral on the bilingual set |
| `classifier_C` | `Choice` | Regularization strength for the linear probe (lower = more regularization, use with few anchors) |
| `translate_fn` | `Choice` | Optional `(text, target_lang) -> str` hook: mirrors each anchor into the missing language at compile time, closing the cross-lingual gap without writing anchors twice. The target language comes from `translate_target=` (default `None` = the callback decides) — v0.9.0 removed the automatic de/en guess. **Only active on the `classifier="linear"` / `"hybrid"` path** — it augments the probe's training matrix, which `nearest` does not have; on `nearest` the hook is silently unused. A `translate_fn` that raises is reported once per compile via `UserWarning` (it never breaks `compile()`) |
| `translate_target` | `Choice` | Language handed to `translate_fn`, e.g. `"en"` for a German-anchor schema. `None` (default) passes `None` — the callback decides. Explicit on purpose: no language guessing anywhere in the engine |
| `glossary` | `Choice`, `DecisionEngine` | Optional **flat, language-agnostic** term map whose synonyms are bridged on anchors **and** queries, so the *keyword* channel can match cross-lingually. Accepts a `dict`, a **JSON file path**, a `klix.Glossary` or `None` (all equivalent). Set it on the engine to cover every head at once. No model, stdlib only, works on `nearest`, `centroid` and `linear`. Any number of ISO-639-1 keys per concept: `{"CONCEPT": {"de": [...], "en": [...], "fr": [...]}}` |
| `matched_concepts` | `Choice` result | The glossary concepts a query matched — the stable explanation of *what a text meant in glossary terms*. Populated even when no bridge was needed. `matched_terms` lists the cross-language terms that had to be added |
| `per_concept_topk` | `Glossary` | Max synonyms bridged per matched concept (default `6`), allocated round-robin over the concept's languages so no language crowds out another. Plus a hard `max_added` overall. This is what keeps a 20k-concept glossary safe |
| `glossary_weight` | `Choice` | Weight of the glossary channel (default `0.5`). Exact tokens keep weight 1.0; glossary terms enter as `alpha · v_glossary` before normalization — on queries *and* on anchor rows. `0.0` leaves the sparse vectors identical to the no-glossary baseline. Compose glossaries via `Glossary.load(...)` / `glossary.merge(other)` |
| `sparse_fastpath` | `DecisionEngine` | Opt-in early exit: answer from the keyword channel alone *before* the dense embedding pass. Three conservative gates (sparse score, margin relative **and** absolute, reject-pole check) and all-or-nothing across heads. Counters via `engine.fastpath_stats()`; per-result provenance in `details(head)["engine"]` (`"sparse_fastpath"` / `"dense_hybrid"`). **v0.9.0: a miss costs nothing relative to the baseline.** The query is vectorized once in `decide()` and the same `SparseQuery` is handed to the gate and to `evaluate()`, so a miss vectorizes exactly as often as a schema *without* the fast path — the v0.8.8 duplicate is gone. Measured cost of that eliminated work: **~0.03 ms median** per sparse-state build (`evals/fastpath_overhead.py`, n=300; load-dependent, 0.027–0.045 ms across runs). A wall-clock miss-vs-baseline comparison is *not* usable as evidence here — the dense pass dominates and its variance exceeds the effect, so the sign flips between runs |
| `reject_anchors` | `Choice` | Texts matching these return `value=None` (don't-know instead of guess); with `classifier="linear"` they are learned as their own class |
| `keyword_boost` | `Choice` | Weight of exact keyword hits (asset IDs like `plc-34`) vs. semantic similarity |
| `aggregation` | `Score` | `"max"` (default) or `"topk"` — topk averages the best-k anchors per pole, robust against a single noisy anchor |
| `coverage` | `Score` result | Pooled similarity to the better pole; low (< ~0.3) means the score is noise |
| `min_val` / `max_val` / `sharpness` | `Score` | Output range and sigmoid steepness |
| `neutral_anchors` | `Flag` | Third pole for out-of-domain: returns `value=None` when it wins |
| `aggregation` | `Flag` | `"max"` (default) or `"topk"` — topk averages the best-k anchors per pole, robust against a single noisy anchor |
| `threshold` / `temp` | `Flag` | Decision cutoff and softmax temperature (lower = sharper) |
| `options`, `threshold` | `MultiLabel` | Multi-label classification: returns all categories with calibrated score ≥ `threshold` |
| `sharpness` / `center` | `MultiLabel` | Sigmoid steepness and cosine similarity midpoint for continuous $[0.0, 1.0]$ scores |
| `classifier` | `MultiLabel` | `"centroid"` (default, vectorized BLAS product), `"max"`, or `"topk"` anchor pooling |
| `calibration` | `MultiLabel` | `"sigmoid"` (default), `"linear"`, or `"cosine"` continuous mapping |
| `model_name` | `DecisionEngine` | Any FastEmbed-compatible embedding model |
| `truncate_dim` | `DecisionEngine` | Opt-in: slice every dense vector to N dims + re-normalize (MRL-style, no training, deterministic). Lowers cosine cost proportionally. **Measured on the repo's own corpora it improves `nearest` accuracy consistently** (60 cases 68.3 % → 76.7 %, 70 cases 71.4 % → 77.1 %, 273 cases 93.0 % → 94.9 % at 64 dims) — but the curve is **not monotone** on the hard sets (96 dims dips below 128 dims) and the corpus is small, so it stays opt-in. Tested with `classifier="nearest"` only; not recommended together with the `linear` probe. See `evals/backbone_compare.py` |
| `stop_words` | `DecisionEngine` | Custom stopword list for the TF-IDF index (default: extended EN+DE list filtering grammatical fillers; pass `[]` to disable filtering) |
| `evaluate(encoded)` | `BaseHead` subclass | Add entirely custom head types (regex, business rules, ...) |
| `rules` | `Choice` | **Optional deterministic guardrail** (not the decision mechanism): hard keyword/regex `Rule`s (force/boost) layered *over* the semantic result. Leave empty for a purely vector-geometric schema |
| `engine.calibrate(head, samples)` | `DecisionEngine` | Learn Flag threshold / Score sharpness+remap / Choice reject_threshold from labeled samples; k-fold CV for n ≥ 6, stability reported via `spread` |
| `res.explain(head)` | `DecisionResult` | Token-level attribution: which keywords and which anchor drove the decision |
| `engine.decide_batch(texts)` | `DecisionEngine` | Bulk mode: one embedding pass for the whole list — per-item overhead drops sharply for large volumes. Measured **~13.9 ms/item at n=500 (72 sentences/s)** on the dev CPU, i.e. ~24 min for 100,000 documents; see [Throughput](#throughput-and-cost-per-document) |
| `engine.validate_anchors()` | `DecisionEngine` | Read-only anchor-quality report: overlapping classes (centroid cosine), shared confuser terms, sharpening hints, misplaced and duplicate anchors. `validate_anchors_report()` returns a formatted string |
| `glossary.validate(anchors=None)` | `Glossary` | Read-only structural report: duplicate terms, circular/ambiguous mappings (one term under two concepts), homograph conflicts (same word, different languages, different concepts) and collisions between glossary tokens and anchor text. Also `engine.validate_glossary()` |
| `klix.langid.detect(text)` | `klix.langid` | Opt-in language identification across 10 languages (`de, en, fr, es, it, pt, nl, pl, sv, da`). Trigram rank distance + function words. Pure stdlib, ~33 µs median, budget 0.2 ms. Never called in `decide()` (verified by test) |
| `klix.multilingual_glossary()` / `workflow_glossary()` / `basic_manufacturing()` / `manufacturing()` / `default_glossary()` / `empty_glossary()` / `merge_all(...)` | `klix.glossaries` | Domain packs, so the engine carries no vocabulary. `multilingual()` (aliased by `workflow()`) = 25 concepts across 10 languages, `basic_manufacturing()` = original 16 terms, `manufacturing()` = 44 curated concepts, `default()` = broad bundled DE↔EN vocabulary (~10k concepts), `empty()` = default (pure dense+TF-IDF) |

## Cross-lingual routing with a glossary

The sparse channel builds its vocabulary from the anchors alone, so a German
query term can never match an English anchor — the dense channel carries the
whole load. `translate_fn` does not fix this (it only runs on the `linear` /
`hybrid` path). A **glossary** does, deterministically and without a model:

```python
from klix import DecisionEngine, Choice
from klix.glossaries import manufacturing, workflow, merge_all

engine = DecisionEngine()
engine.add_head(Choice(
    name="target",
    options={"ot_plant": ["conveyor belt stopped", "cycle time doubled"],
             "maintenance": ["spare part missing", "sensor calibration overdue"]},
))
engine.compile()

# one glossary for the whole schema, or per head — both work
engine = DecisionEngine(glossary=merge_all(manufacturing(), workflow()))
```

Pass a **dict**, a **JSON file path**, a `Glossary` or `None` — all equivalent:

```json
{
  "conveyor":   {"de": ["foerderband", "transportband"], "en": ["conveyor belt"]},
  "cycle_time": {"de": ["taktzeit", "zykluszeit"],       "en": ["cycle time"]},
  "urgent":     {"de": ["dringend", "eilig"],            "en": ["urgent", "asap"]}
}
```

```python
engine = DecisionEngine(glossary="my_glossary.json")   # path
engine = DecisionEngine(glossary={"conveyor": {...}})  # dict
```

The lookup is **flat and language-agnostic**: `word -> concept -> synonyms`. The
engine never guesses which language a text is in (v0.9.0 removed the last de/en
heuristics). Its one language-dependent decision is made on evidence instead —
a synonym the anchors already use is not bridged, because it bridges nothing:

```python
glossary.expand_terms("conveyor belt stopped", vocab=head._core_vocab)
# -> [] : the anchors already contain "conveyor belt"; nothing to bridge
```

Expansion is bounded (at most `per_concept_topk` per matched concept, allocated
round-robin across languages, plus a global `max_added`), sorted and repeatable —
the glossary is part of `schema_hash()`, so logged decisions reproduce exactly.

**Honest cost:** on monolingual text the hook is neutral (v0.8.7 diluted the
sparse vector; v0.8.8's damping plus the vocabulary-aware skip fixed that). The
measured gain applies to cross-lingual schemas — see the `glossary` row in the
configuration table.

**How much is that gain worth?** Measured, because it decides how much glossary
work is worth doing (`evals/glossary_vs_bilingual_anchors.py`, full write-up in
`docs/glossary-vs-anchors.md`). Same schema, same 20 test cases, four cells,
bootstrap CI:

| | centroid | linear |
|---|---|---|
| (a) EN anchors, no glossary | 15/20 | 16/20 |
| (b) EN anchors + glossary | 16/20 | 16/20 |
| (c) EN+DE anchors, no glossary | 15/20 | 15/20 |
| (d) EN+DE anchors + glossary | 14/20 | 15/20 |

**The honest reading: at n=20 the difference between (b) and (c) is not
resolvable.** The 95 % CI for (b) − (c) is −10…+20 points and includes zero. That
is an absence of evidence in either direction, **not** a demonstrated equivalence —
so do not read this as "bilingual anchors and a glossary are interchangeable".
(d) − (c) is −5 points under `centroid` and 0 under `linear`.

**Read the `linear` column carefully: it is not a result about the glossary.** Under
`classifier="linear"` the probe predicts from the dense vector alone and returns
before the glossary expansion runs (`src/klix/heads.py:1170-1171`, return at
`:1197`), so the sparse channel never reaches the decision. Verified mechanically:
varying `glossary_weight` from 0 to 50 produces exactly one answer pattern under
`linear`, but 2–3 patterns under `nearest`/`centroid`/`hybrid`. The (a)=(b) and
(c)=(d) equality in that column is what the code path predicts — a useful check on
the analysis, not a measurement of the glossary.

**Practical consequence that *does* follow** (an architecture finding, not an
anchor-count one): if you want the glossary to influence routing, use `nearest`,
`centroid` or `hybrid`. With `linear` it is inert for the decision. Where the
channel is active, the glossary fires — it expands 6 of 10 German queries — and it
moved one query to the right answer (`elternzeit`: wrong → `hr`). The remaining
errors are mutually ambiguous *anchors* (`bildschirm`, `kaffeemaschine`,
`erstattung` all landing on `facility`), which no glossary can repair.

Caveats that bound all of the above: n=20 test cases, one domain, CI up to ±20
points, anchor *count* coupled to language coverage by design (2 vs 4 per class),
and one of the two classifiers structurally inert. A test that could actually settle
the question needs n≥60, a sparse-active configuration, and equal anchor counts
across cells — the design and its missing data are specified in proposal P9
(`docs/proposals.md`).

### Limits of `multilingual()` / `workflow()` — read before adopting 10 languages

The 25x10 pack (`multilingual()`, aliased by `workflow()`) covers 10 languages (`de, en, fr, es, it, pt, nl, pl, sv, da`). It is hand-written, single-author, MIT.

A spot-check of 96 terms across the 8 non-native languages against independent Wikidata item labels (`evals/multilingual_spotecheck.py`, seed 20260929, 12 terms per language) confirmed 37 terms directly, while the 54 mismatches were all read manually and classified as legitimate synonyms (~32, whose sibling in the same concept matched the label, e.g. French `panne` vs. `erreur`) or register/sense nuances (~22, e.g. vocational training vs. sports), with **0 confirmed wrong mappings in the sample**.

**The honest limits (Principle 13):**
* **10 of 25 concepts are NOT externally verified:** For 10 concepts (`urgent`, `cancel`, `help`, `approve`, `reject`, `access`, `status`, `escalate`, `cost`, `delivery`), no clean, single unambiguous Wikidata item exists (homonyms dominate search hits, e.g. songs or broad concepts). Their correctness rests solely on author curation — an honest unverified gap, not a pass. Full sense disambiguation would need SPARQL over senses, which was in a hard outage during this check.
* **Label matching proves naming, not exclusivity:** A label match confirms that the term names the target concept; it cannot prove that the word does not also carry polysemous meanings in that language.
* **Sampled, not audited:** 96 of 250 language-term entries were checked (38.4%), not the entire pack.

### Limits of the bundled presets — read before trusting `curated()`

What has been **measured**: internal ambiguity 0 (build-enforced on the shipped
file), and correctness confirmed against an independent Wikidata-item ground
truth for 40 % of a random sample. What has **not**: for the other **60 % no
independent ground truth exists, so their correctness is unknown** — concentrated
in the office/IT vocabulary but reaching manufacturing too (8 of 16 sampled
manufacturing terms). Nothing here is known to be wrong; it is simply unverified.
A real error rate needs a labelling pass by someone other than the author.

Two consequences you can hit in practice:

* **Flat lookup, no context — this is structural, not a fixable list of typos.**
  A German word with more than one meaning resolves to whichever concept claimed
  it, and the format cannot express otherwise: one term maps to exactly one
  concept, `merge()` refuses the second assignment with `GlossaryConflict`, and
  `validate()` reports it as an ambiguous mapping. Measured
  (`evals/polysemy_review.py`): **62 of 768 glossary terms (8.1 %) carry more than
  one Wikidata sense**, so this is a property of the vocabulary, not of a single
  entry. Reproduced end-to-end: `das lager der welle ist verschlissen` (*shaft
  bearing*) resolves to `warehouse`, and `der leiter ist kaputt` (*ladder*)
  resolves to `supervisor`.
* **Deleting the synonym is not a fix.** Measured: dropping `leiter` from
  `supervisor` leaves the ladder query on the office concept anyway, because the
  English `supervisor` anchors are semantically close to it. The failure is the
  absence of context, not the presence of a word.
* **In a single-domain schema the exposure is small.** Of the 11 polysemous
  manufacturing terms, every one resolves to a manufacturing concept — none leaks
  into another domain, and for 10 of them no competing concept exists in the
  glossary at all. The one exception is `lager` (bearing *and* stock): it resolves
  to `warehouse`, and no part-level concept exists for the other reading. If your
  schema is manufacturing-only, that is the single case worth knowing about.

Real mitigations, none of them in this release: per-domain glossaries (the engine
already supports this via `curated_where(tag)` or one glossary per head), a
context-bearing synonym such as `kugellager`/`wellenlager` for the part reading, or
context disambiguation (a scope change — the engine is deliberately stateless).

If your schema mixes domains, prefer your **own** glossary over the bundled presets
— the presets are a starting point and a demonstration, not a calibrated artifact
for your vocabulary. Numbers, method and adjudication: `docs/curated-correctness.md`
and `DATA_SOURCES.md`.

## Custom Heads

Subclass `BaseHead` and implement `evaluate(encoded)`:

```python
import re
from klix import BaseHead

class RegExExtractionHead(BaseHead):
    def __init__(self, name: str, pattern: str):
        super().__init__(name)
        self.re = re.compile(pattern)

    def get_reference_texts(self) -> list[str]:
        return []  # no reference texts needed

    def fit(self, backbone) -> None:
        pass

    def evaluate(self, encoded) -> dict:
        match = self.re.search(encoded.text)
        return {"value": match.group(0) if match else None}
```

## Production Pattern

`examples/production_pattern.py` shows the complete production-ready pattern in a
single file: Choice/Score/Flag with `classifier="auto"` and `translate_fn`, an explicit
catch-all class (`not_relevant`), the two-stage action pattern (security flag as
veto, confidence gate, auto-close), a custom head via `BaseHead` subclassing,
and the interpretation of all confidence signals.

```bash
uv run python examples/production_pattern.py
```

## Why klix — and when it isn't the right tool

Klix is a **packaging** decision, not an algorithm decision. The core
(embedding + KNN / logistic probe) is standard practice since
sentence-transformers popularized it in 2019; you could write an equivalent
~30-line snippet with `sentence-transformers` + `sklearn`. What klix adds is
the product around that core:

- a **declarative, reviewable schema** (`Choice` / `Score` / `Flag` heads)
  that lives in the repo, doubles as documentation, and updates in
  milliseconds — no training step, no model artifact per schema;
- **honest uncertainty handling** out of the box (reject poles, `coverage`
  signals, calibrated thresholds) instead of a forced guess;
- **explainability** (token-level attribution) and **no ML infrastructure**:
  no GPU, no API keys, fully offline after a one-time model cache.

**Where it genuinely helps:** rapid prototyping of text-routing logic
without labeled data and without an ML pipeline — e.g. coarsely sorting
incoming tickets or fault reports into categories when you have neither the
time nor the data volume for a trained model, and plain keyword matching is
too brittle. Small volumes, clearly separable categories, no
training-data-pipeline required.

**Where it does not:** with a few hundred labeled examples per class, a
trained classifier will beat it; and for a single throwaway routing problem,
copy-pasting the 30-line KNN/LogReg snippet is simpler than adopting a
library. Measured default mode is on par with a trivial dense Embed-KNN
(72 % vs 78 % in the cross-domain benchmark) — the accuracy edge appears
only in specific modes (`classifier="linear"` on single-language schemas:
84 %). See the [Benchmarks](#benchmarks) section for the honest numbers.

**One property to know before you design a schema:** not every question is a
text question. Urgency, business impact, SLA-breach risk and customer tier are
decided by *context* the message does not carry — four measured mechanisms
against an urgency target all returned the same number. Use klix heads for what
the wording *says* (category, topic, tone, whether something looks like an
incident) and do the valuation in your application from real data. Details and
the measured evidence: `Score` docstring, "WHEN NOT TO USE THIS HEAD".

## Benchmarks

Every number below is **reproducible from this repo** — the scripts live in
`evals/`, and within a benchmark all methods see the *same* labeled data. Where a
figure comes from an external harness or a vendor, the table says so.

### At a glance

| Benchmark / domain | Classes | Test cases | klix | Baseline to compare against | Throughput (CPU) |
|---|:---:|:---:|:---:|:---:|:---:|
| **BANKing77** *(bank intents)* | 77 | 3,080 | **61.4 %** *(k=3 anchors)* | 53.3 % *(bare label as anchor)* | 72 docs/s |
| **MASSIVE** *(voice assistant, EN)* | 60 | 2,974 | **43.8 %** *(k=3 anchors)* | **47.9 %** *(bare label as anchor)* | 72 docs/s |
| **MASSIVE** *(voice assistant, DE)* | 60 | 2,974 | **36.2 %** *(k=3 anchors)* | 40.5 % *(Qwen2B LLM, 500-case subset)* | 72 docs/s |
| **Cross-domain support** | 6 | 70 | **84 %** *(linear / centroid)* | 78 % *(dense Embed-KNN)* — 67 % *(Laya, other harness)* | 72 docs/s |
| **Few-shot vs. training** | 5 | 60 | **85 %** *(linear, no training)* | 85 % *(SetFit, trained, same CPU)* | 72 docs/s |
| Bilingual EN/DE *(small)* | 5 | 20 | 16/20 *(nearest + topk2)* | 16/20 *(dense Embed-KNN)* | 72 docs/s |

**The honest readings — including the ones that don't flatter klix:**

- **Three real examples per class (k=3) beat a bare label on two of three sets,
  and lose on the third.** BANKing77 +8.2 pt, MASSIVE-de +6.1 pt — but on
  MASSIVE-en the bare label *wins* by 4.1 pt (47.9 % vs 43.8 %). The reason is not
  established; the MASSIVE-en labels are short and near-verbatim English
  (`alarm_set` → `alarm set`), which is a plausible contributor but not something
  we measured in isolation. Anyone citing "few-shot anchors always help" should
  read that row first.
- **A 2B LLM on the same CPU buys +4.3 pt for ~650× the latency** on MASSIVE-de
  (40.5 % vs 36.2 %), and it is a 500-case subsample against klix's full split,
  with all 60 labels in its prompt while klix sees 3 anchors per class. Not a
  like-for-like comparison — see the section below.
- **MASSIVE's own data caps the German number.** 3.9 % of German test texts appear
  verbatim in `train`, 8 of them with a contradictory label. Measured, documented,
  and asserted by tests — it is a property of the dataset, not of klix.
- **At n=20 and n=70, small differences are noise.** The bilingual table is 20
  cases (±20 pt CI); in the cross-domain table only the `linear`/`centroid` row
  separates clearly from the rest (±9 pt CI at n=70). Treat both as directional.
- **Everything here runs on CPU, offline**, no GPU and no API calls. `72 docs/s`
  is the sustained `decide_batch()` rate, not a peak.

### Throughput and cost per document

Single-decision latency is dominated by one MiniLM forward pass and is
**~46 ms p50 / ~77 ms p95** on the dev CPU (`evals/measure_footprint.py`, 60
warm calls, measured in a clean process — never measure this while anything
else holds the CPU).

Bulk mode is what matters for volumes. `decide_batch()` embeds the whole list in
one pass (`evals/bench_batch.py`, n=500, one `Choice` + one `Score` head):

| n | batch ms/item | throughput |
|---|---|---|
| 25 | 8.7 ms | 116 sentences/s |
| 100 | 13.3 ms | 75 sentences/s |
| **500** | **13.9 ms** | **72 sentences/s** |

**≈ 72 sentences/s sustained → 100,000 documents in ~24 minutes** on one CPU
core, no GPU, no API calls, no per-document cost. The single-call figure
(~46 ms) and the bulk figure (~14 ms) differ because the forward pass amortizes
across the batch. The ~46 ms is a *latency* figure (one text, waiting for the
answer); the ~13.9 ms is a *throughput* figure (large list, total time divided by
items). Quoting one as the other is the most common way to misread this table.

Footprint, measured (not estimated): the package wheel is **289 KB**; the
multilingual embedding model is **240 MB on disk** (224 MB ONNX + 16 MB
tokenizer) and ~0.3 GB resident once loaded. Note: the `118 MB` figure that
circulated in earlier README versions was a hardcoded label in
`evals/backbone_shootout.py`, never a measurement — the numbers above are;
`~120 MB` elsewhere in this file is corrected to the same 240 MB.

### On large public datasets (MASSIVE, BANKing77)

Both are **repurposing** runs, not the task klix is built for: klix needs anchor
sentences per label and these datasets ship only `text` + `label`. Anchors are
therefore constructed by `evals/bespoke_anchors.py` — `few_shot_k3` draws 3 real
sentences per class from the dataset's own `train` split (seeded, no test
overlap), and `label_string` uses the bare label as the honest lower bound.
Reproduce with `evals/run_bespoke.py`; full provenance, licences and known data
defects in `evals/data/bespoke/PROVENANCE.md`.

| Dataset | classes | test cases | anchors | accuracy | 95 % CI |
|---|---|---|---|---|---|
| **BANKing77** | 77 | 3,080 | few_shot_k3 | **61.4 %** | [59.8, 63.2] |
| BANKing77 | 77 | 3,080 | label_string | 53.3 % | [51.5, 55.1] |
| **MASSIVE** (en) | 60 | 2,974 | few_shot_k3 | **43.8 %** | [41.9, 45.7] |
| MASSIVE (en) | 60 | 2,974 | label_string | **47.9 %** | [46.1, 49.7] |
| **MASSIVE** (de) | 60 | 2,974 | few_shot_k3 | **36.2 %** | [34.5, 38.0] |
| MASSIVE (de) | 60 | 2,974 | label_string | 30.1 % | [28.5, 31.8] |

The k=3 anchors beat the label-string baseline on BANKing77 (+8.2 pt) and
MASSIVE-de (+6.1 pt) — but **lose on MASSIVE-en** (43.8 % vs 47.9 %, −4.1 pt).
All three CIs are non-overlapping, so this is a real effect rather than noise.
The likely reason is that MASSIVE-en's labels are short and near-verbatim
English (`alarm_set` → `alarm set`, `weather_query` → `weather query`), which
makes the bare label a strong anchor; we did not isolate that cause, so treat it
as a hypothesis. The practical reading: **few-shot anchors are not universally
better — test both on your own schema.** Coverage is 100 % throughout — klix
answers every case rather than abstaining.

**Known defect in the source data, measured:** MASSIVE's `train`/`test` splits
are not sentence-disjoint. In German, 115/2974 (3.9 %) of test texts appear
verbatim in `train`, **8 of them with a contradictory label** (identical
sentence, two different intents); English 21/2974 (0.7 %), 2 contradictory. This
caps achievable accuracy on the German set and is the main reason MASSIVE-de
scores below MASSIVE-en. Details and the assertions that pin it:
`evals/data/bespoke/PROVENANCE.md` §1.1.

### Accuracy scales with anchor density (BANKing77, 77 classes)

The runs above used minimal few-shot anchoring ($k=3$). On large, fine-grained schemas (like BANKing77 with 77 distinct banking intents, random chance = 1.3 %), accuracy scales directly with anchor density and classifier choice on the exact same CPU without increasing model footprint:

| Classifier | Anchors / class ($k$) | Compile time | Accuracy | Latency (bulk) |
|---|---|---|---|---|
| `nearest` | $k=3$ | 4.7 s | 60.2 % | 29.7 ms/doc |
| `centroid` | $k=3$ | 5.3 s | **64.8 %** (+4.6 pt) | 26.6 ms/doc |
| `linear` | $k=3$ | 10.5 s | **67.8 %** (+7.6 pt) | 26.1 ms/doc |
| `centroid` | $k=10$ | 22.4 s | **76.6 %** (+16.4 pt) | 31.3 ms/doc |
| `linear` | $k=10$ | 85.0 s | **79.8 %** (+19.6 pt) | 28.1 ms/doc |
| `centroid` | $k=20$ | 39.6 s | **79.6 %** (+19.4 pt) | 26.7 ms/doc |
| `linear` | **$k=20$** | 132 s | **85.2 %** (+25.0 pt) | 41.9 ms/doc |

*Measured on 500 subsampled test cases from BANKing77.* Note that `centroid` delivers massive zero-training gains (64.8 % → 79.6 %) with purely an embedding pass at compile time, while `linear` reaches **85.2 %** when provided with 20 examples per class — remaining orders of magnitude faster and lighter than 9B LLMs (like Nimble).

### vs. a local 2B LLM (same 60-class task, same CPU)

Same 500 MASSIVE-de cases, same machine, no GPU: `qwen3.5:2b` via Ollama,
prompted with all 60 labels and told to answer with one label name.

| | klix (few_shot_k3) | qwen3.5:2b (Ollama) |
|---|---|---|
| accuracy (answered) | 36.2 % (full 2,974 cases) | 40.5 % (500-case subsample) |
| model footprint | 240 MB | 2.7 GB |
| sustained latency | **13.9 ms/item** (bulk) | **~9,000 ms/call** |
| throughput | 72 docs/s | ~0.11 docs/s |
| coverage | 100 % | 99.2 % (4 unparseable) |

**Read this carefully — it is not a like-for-like comparison.** The LLM prompt
contains all 60 labels; klix sees 3 anchors per class. The LLM run is a
500-case subsample of the same 2,974-case split; klix's figure is the full
split. Both systems are scored on identical case texts, and the LLM figure is
the *sustained* rate, not an average that includes model load.

What it does show is the cost/benefit shape: the LLM's **+4.3 pt** costs
**~650× the latency**, **11× the footprint**, and ~5 GB of downloads to run at
all. Published vendor figures for Tev1 (Together AI) and Nimble (Bespoke Labs)
are **not** reproduced here; if you quote them, quote them as vendor numbers —
nothing in this repository measures them.

### Bilingual routing (EN/DE, 5 classes, n=20) — small, directional

`evals/benchmark_bilingual.py` — 10 English + 10 German test cases with parallel
meaning, anchors mixed EN/DE. **n=20, so the 95 % CI is roughly ±20 pt: read this
as directional, not as a ranking.** `uv run python -m evals.benchmark_bilingual`

| Method | EN | DE | Combined |
|---|---|---|---|
| TF-IDF + LogReg | 8/10 | 6/10 | 14/20 |
| Embed-KNN (dense) | 9/10 | 7/10 | **16/20** |
| klix nearest | 8/10 | 7/10 | 15/20 |
| klix nearest + topk2 | 9/10 | 7/10 | 16/20 |
| klix linear | 9/10 | 6/10 | 15/20 |

**Why this small set still earns a place here:** it is the one benchmark that
isolates the *mixed-language few-anchor* case, and there the `linear` probe
overfits to English (90 % EN / 60 % DE) while dense Embed-KNN stays the most
language-robust. That is the measurement behind rule 2 of [Choosing a
variant](#choosing-a-variant--three-rules): mixed-language anchors → `centroid` or
`nearest`, not `linear`. Latency: TF-IDF+LogReg ≈ 1–5 ms (no embeddings),
embedding-based methods ≈ 46–80 ms single-call (see [Throughput](#throughput-and-cost-per-document)).

**Reading this honestly:** on this *mixed-language, few-anchor* schema the
`linear` probe overfits to English (90 % EN / 60 % DE). The dense Embed-KNN is
the most language-robust. Recommendation: with few mixed-language anchors, use
`classifier="nearest"`; the `linear` probe pays off on *single-language* schemas
with several anchors per class (see the cross-domain result below).

### Cross-domain routing (6 domains, 70 cases, mostly EN)

`evals/benchmark.py` — HR, Finance, Image-captions, Tasks, Shop, and the
LLM-guardrail scenario (n=70; note: five of the six domains are the same
labeled sets used in the SetFit comparison below, plus the GUARD set):

| Method | avg accuracy | median latency |
|---|---|---|
| TF-IDF + LogReg | 51 % | ~1–5 ms |
| Embed-KNN (dense) | 78 % | ~58 ms |
| klix nearest | 72 % | ~71 ms |
| **klix linear** | **84 %** | ~80 ms |

Here `klix linear` is the accuracy winner (+12 pts over the nearest-anchor
ceiling). Latency is dominated by the embedding forward pass and scales
with hardware (measured on the dev workstation, 2026-09-24).

**Read the anchor count with these numbers — they are not comparable
without it.** This benchmark uses **3 anchors per class** (the eval sets
were kept deliberately small so the comparison against TF-IDF/Embed-KNN is
honest on identical data). With 3 anchors per class, 72–84 % is the
expected range, not a ceiling. The dominant quality lever is anchor
*coverage of the class*, not the algorithm:

| Anchors per class (6 classes, same engine) | Category accuracy |
|---|---|
| 3 — instance-style ("plc-34 meldet fehler") | 57 % |
| 3 — phenomenon-style ("sps fehlermeldung an der anlage") | 87 % |
| 9 — phenomenon-style, no rules | 93 % |
| 9 — phenomenon-style + 2 `Rule`s for asset tags | **97 %** |

So a reader who sees "3 anchors → 72 %" and concludes "weak framework" is
measuring the anchor set, not the engine. Two separate levers, both free:
phrase anchors as *descriptions of the phenomenon* rather than instances,
and give the schema enough coverage (roughly 6–9 per class for a
production schema). The benchmark's low anchor count is a property of the
benchmark, not a recommendation.

The 9-anchor rows come from a self-measuring reference application
(6 classes, 29 labelled messages disjoint from the anchors, `klix_demo.py`
in the sibling `klix-demo` project). It re-computes its own accuracy on
every run, so the figures are reproducible rather than transcribed —
the pattern worth copying when you build your own schema.

**Statistical honesty:** with n=70, differences of 1–2 points between
embedding-based rows are within the 95 % CI (roughly ±9 pts at n=70); the
`klix linear` lead is the only row pair that separates clearly. Treat the
table as directional, not as a ranking with that precision.

**Version note (0.8.1):** the numbers above were re-measured 2026-09-24
with the current FastEmbed release (0.8.x), which computes mean-pooled
MiniLM embeddings (previously CLS pooling). On the fixed 60-case 5-domain
set the klix accuracy is unchanged (nearest 41/60 = 68 %, linear 51/60 =
85 %, verified via `evals/linear_sweep.py`); the `benchmark.py` row for
`klix nearest` moved 76 % → 72 % and `klix linear` 86 % → 84 %, because
that harness includes the GUARD set. Latency was re-measured as well:
the ~10 ms claims of earlier README versions predate the current FastEmbed
release. **Current measurement (2026-09-30): ~46 ms p50 / ~77 ms p95 single-call
and ~13.9 ms/item in bulk** — see
[Throughput](#throughput-and-cost-per-document). Still CPU-only and offline, but
expect tens of milliseconds per query on similar hardware.

### vs. SetFit (few-shot training, same examples)

`evals/setfit_baseline.py` — SetFit trains a contrastive few-shot classifier
on the same anchor texts (identical example budget, `num_epochs=1`, same
MiniLM backbone, CPU):

| Dataset | klix (nearest) | klix (linear) | SetFit |
|---|---|---|---|
| HR | 8/12 | 9/12 | 9/12 |
| FIN | 10/12 | 12/12 | 11/12 |
| IMAGE | 9/12 | 11/12 | 11/12 |
| TASK | 7/12 | 11/12 | 10/12 |
| SHOP | 7/12 | 8/12 | 10/12 |
| **total (n=60)** | 41/60 = 68 % | 51/60 = 85 % | **51/60 = 85 %** |

**The honest verdict:** trained few-shot classification (SetFit) *matches*
the klix linear probe (85 %) but buys nothing beyond it — on this benchmark,
training reaches exactly what klix already achieves without any training
step. The klix linear probe and bm25+topk3+coverage configs reach the same
accuracy in sub-100 ms at compile time, with no model artifact, no extra
dependency stack, and full keyword explainability via the sparse channel.
Klix's pitch is therefore NOT "as accurate as training at zero cost" — it is:
equivalent accuracy to few-shot training on these sets, but instant schema
updates (no training step), no saved model per schema, and hard-negative
mining as the training-free accuracy lever (+7 pts measured). SetFit is the
right choice when anchor sets are tiny per class (its contrastive pairing
extracts more from 3-4 anchors); klix is the right choice when schemas
change with the business, the process is iterative, or the deployment must
stay tiny and offline.

### vs. Laya (`convaiinnovations/laya`)

Klix's original inspiration is the trained zero-shot engine
[`laya`](https://pypi.org/project/laya/) (Torch/ModernBERT-large, GPU-oriented).
Measured on the same 70 cases on **CPU**:

| | klix-linear | Laya (zero-shot) |
|---|---|---|
| accuracy | **86 %** | 67 % |
| latency (CPU) | **~13 ms** | ~1.3–1.5 s |
| model size | 240 MB | ~2 GB |
| setup | anchors (few-shot) | instructions + criteria (zero-shot) |

*Note on the `~13 ms` klix figure: it comes from the Laya comparison harness, which
measures a warm query against a small schema — it is not comparable to the ~46 ms
p50 in [Throughput](#throughput-and-cost-per-document), which is a 60-call
distribution on a different schema. Treat the ratio (≈100×) as the meaningful part,
not the absolute milliseconds.*

**The honest trade-off:** Laya needs *no* examples and natively routes 100+
languages with automatic script detection — a real advantage for low-resource
scripts on GPU. Klix is the opposite design point: a tiny model, few-shot anchors
you fully control, and 100× lower CPU latency. They are complements, not
substitutes.

### Choosing a variant — three rules

1. **`classifier="centroid"` — start here for most schemas.** Cosine to the
   per-label mean anchor vector. Deterministic, no training, robust to a single
   odd anchor, and it **scales independently of anchor count** at volume (one
   vector comparison per label, however many anchors feed it). It reaches the
   trained probe's accuracy on the repo's own corpora: cross-domain 71.4 % → 84.3 %
   (+12.9 pt, CI [+2.9, +22.9]), 273-case corpus 93.0 % → 96.7 % (+3.7 pt,
   CI [+1.5, +6.2]). **Caveat measured on the bilingual set: it is neutral there,
   not better — so it is not a drop-in replacement for `nearest` on mixed-language
   few-anchor schemas.** The `"auto"` value is a deprecated alias for `"centroid"`.
2. **`classifier="linear"` — when you have 5+ anchors per class and one language.**
   Learns a logistic separation plane; the accuracy ceiling on focused,
   single-language schemas. Measured +12 pt over the nearest-anchor ceiling on the
   cross-domain set (84 % vs 72 % at 3 anchors/class). It **overfits on
   mixed-language few-shot data** (90 % EN / 60 % DE measured) — if your anchors
   are mixed-language, use `centroid` instead.
3. **`classifier="hybrid"` — for asset IDs, SKUs, error numbers.** Fuses dense
   semantics with exact BM25 keyword matching, so an identifier is matched
   literally rather than approximately. It trails `linear` on plain
   natural-language sets (81.7 % vs 85.0 % at n=60), so do not reach for it unless
   your texts actually contain keyword-like tokens.

Two more knobs worth knowing:

- **`sparse_metric="bm25"`** — BM25 instead of TF-IDF cosine; better when anchor
  lengths vary a lot (+3 pts at n=60 on the nearest path). Combined with
  `topk3 + coverage` it matches `linear` accuracy *without any probe training*
  (85 % at n=60).
- **`label_aggregation="topk"`** — damps single-anchor noise when you have 4+
  anchors per label.
- **Hard negatives** — for production schemas, collect low-confidence live
  decisions (`HardNegativeStore`), review them by hand, attach them as
  counterexamples and recompile. **Honest measured effect (holdout eval,
  `evals/hard_negative_e2e.py`):** on cases the mining step never saw, the
  gain is ≈0 (11/20 → 10/20 on n=20 holdout); an earlier reported +7 pts
  was dominated by memorization of the mining set itself (corrected in
  0.8.1). The workflow is still valuable as a *diagnosis* loop (it surfaces
  which label pairs the schema confuses — fix those by adding/sharpening
  anchors), not as an automatic accuracy lever.

## Evaluation scripts (`evals/`)

`evals/` is a **catalog**, not a test suite (functional tests live in
`tests/`). Every script is runnable from the repo root via
`uv run python -m evals.<name>` (module form, so package imports
`from evals.X import ...` work).
The table distinguishes *live results* (maintained, referenced from this
README/CHANGELOG) from *historical* experiments (kept for reproducibility,
results are snapshots in time — re-running may show different numbers).

| Script | Purpose | Status |
|---|---|---|
| `run_bespoke.py` | klix vs. Ollama on MASSIVE/BANKing77/PAWS — the README table | live |
| `bespoke_loader.py` | Fetch + normalise those datasets (parquet, no `datasets` dependency) | live (data source) |
| `bespoke_anchors.py` | Anchors for 60/77-class sets (`few_shot_k3`, `label_string`) + leakage check | live (data source) |
| `bespoke_probe.py` | Feasibility/latency probe before a long Ollama sweep | live |
| `measure_footprint.py` | Isolated single-call latency + memory + on-disk model size | live |
| `multilingual_spotecheck.py` | Wikidata item-label spot check of the 10-language core (96 terms) | live |
| `langid_bench.py` | Accuracy, per-language breakdown, tuning calibration for `klix.langid` | live |
| `langid_experiment.py`, `langid_hybrid.py` | Language ID ablation studies (n-gram order, channel weights, profiles) | live |
| `benchmark.py` | Cross-domain routing (6 domains, 70 cases) — the README table | live |
| `benchmark_bilingual.py` | Bilingual routing (EN/DE, 20 cases) — the README table | live |
| `setfit_baseline.py` | SetFit few-shot comparison (n=60, leakage-verified) | live |
| `hard_negative_e2e.py` | Holdout-verified hard-negative mining effect | live |
| `eval_domains.py` | Labeled 5-domain corpus (60 cases) shared by several evals | live (data source) |
| `variant_sweep.py` | Nearest/topk/coverage sweeps (pre-0.2 history) | historical |
| `corpus_expansion.py` | 273-case paraphrase corpus + bootstrap CIs | historical |
| `expanded_benchmark.py` | The expanded corpus benchmark | historical |
| `linear_sweep.py`, `knob_sweep.py`, `bm25_sweep.py`, `hybrid_sweep.py`, `hybrid_bm25_sweep.py` | Probe / sparse-channel sweeps behind 0.2.x–0.8.0 | historical |
| `charngram_experiment.py` | Did char-n-grams help the probe? (measured: no) | historical |
| `verify_crosslingual.py`, `whatif.py`, `whatif_anchors.py`, `instrument.py` | Feature verification / what-if simulations | historical |
| `eval_baseline.py`, `eval_after.py` | Pre/post head-feature comparisons | historical |
| `bench_batch.py` | Batch-vs-serial latency/throughput — the README throughput table | live |
| `bug_hunt.py` | Edge-case hunting script (not a test file) | historical |
| `export_datasets.py` | Export the labeled datasets to JSONL | historical |

**Conventions:** scripts share the labeled datasets via package imports
(`from evals.linear_sweep import ...`, `from evals.eval_domains import
...`). Run them as modules from the repo root (`uv run python -m
evals.<name>`). If you change a shared dataset, re-run the evals that
depend on it.

## Development

```bash
uv sync          # install dependencies
uv run pytest -q # functional test suite (fully offline, no timing gates)
uv run pytest -q -m benchmark  # optional: latency/microbenchmark gates
uv run python examples/demo.py
```

**Test-suite structure:** functional tests (`tests/`) are the CI gate and
never assert wall-clock numbers. Timing-sensitive cases — head evaluation
latency and batch throughput — are marked `@pytest.mark.benchmark` and
excluded from the CI gate, because timing assertions under shared CPU load
are inherently flaky and would mask real failures. Run them explicitly
when you care about latency regressions locally.

**Release chain (automated, no token):** bump the version in `pyproject.toml`
and `__init__.py`, update `CHANGELOG.md`, commit, tag, push — GitHub Actions
builds and publishes to PyPI automatically via **Trusted Publishing (OIDC)**
(workflow `publish.yml`, environment `pypi`; configured once in the PyPI
project settings — no API token is stored in the repo):

```bash
uv build && uv run --with twine python -m twine check dist/*   # pre-tag check
git tag vX.Y.Z
git push origin main vX.Y.Z
```

## Offline / air-gapped deployment

Klix needs no API keys, but the first run downloads the ONNX embedding model
(240 MB) into the fastembed cache. To prepare an air-gapped machine, cache
the model on a connected machine and transfer it:

```bash
# 1. On a machine with internet: warm the cache once.
uv run python -c "from klix import DecisionEngine; e=DecisionEngine(); e.add_head(__import__('klix').Choice(name='x', options={'a':['alpha']})); e.compile()"

# 2. Find the cache dir (fastembed uses the HF-style local cache):
uv run python -c "from fastembed import TextEmbedding; print(TextEmbedding.list_supported_models()[0]['sources'])"  # model id reference
# cache location (default): ~/.cache/fastembed  (respects XDG_CACHE_HOME / LOCALAPPDATA)

# 3. Copy the cache directory to the air-gapped machine, same path, then
#    set the cache env var if the path differs:
#    XDG_CACHE_HOME=/data/cache   (Linux)
#    LOCALAPPDATA=%CUSTOM_PATH%   (Windows)
```

After that, klix runs fully offline — no network access is ever attempted
at inference time.

## License

MIT