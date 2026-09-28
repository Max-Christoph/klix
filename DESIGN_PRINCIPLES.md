# Design principles

These are not aspirations. Each one exists because it was violated in this
codebase and the violation caused a wrong result, a wrong number, or a wrong
claim. The incidents are named so the reasoning survives.

**This document is a gate, not documentation.** Before any non-trivial change,
work the [pre-flight checklist](#pre-flight-checklist) at the bottom. If every
line passes, proceed without asking. Come back with a question only when a
principle is genuinely ambiguous for the task at hand, or when a principle
collides with a concrete instruction — and say which two things collide.

Language note: the repository's user-facing docs are English, so this file is
too. That is a convention, not a judgement about the reader.

---

## 1. Fail explicitly instead of being silently wrong

**Rule.** An ambiguity or conflict aborts with a message that names the
conflicting items and the way out. It is never resolved on the caller's behalf:
no renaming, no silent overwrite, no first-match-wins on an ambiguous lookup.

**Why.** The glossary index is flat — one term resolves to one concept. The
v0.9.0 build script resolved a concept-key collision by silently renaming
(`f"{domain}_{concept}"`) while aborting on term collisions. Two aspects of the
same conflict class, handled differently, and the silent branch only fired for
the exact configuration that existed at the time. A fourth glossary would have
hit neither branch and lost data. Separately, Wikidata sense resolution picked
the first linked sense, which is how `mutter` (the fastener) was mapped to
`mam` — a wrong answer delivered confidently instead of an error.

**How to comply.**
* Ambiguity → raise. Use `GlossaryConflict`, `ValueError`, whatever fits, but
  raise.
* The message names the offending items and the documented escape
  (`merge(..., strict=False)`).
* If a permissive path is needed, it is an explicit parameter that the caller
  passes deliberately, never a default.
* One conflict class gets one handling, everywhere (see 14).

**Anti-pattern.** "Take the first match", "if already present, skip", "rename to
avoid the clash", `dict.setdefault` used to paper over a duplicate.

---

## 2. Extensibility is a public API, not an internal script

**Rule.** Any capability used inside the project — merging glossaries, filtering
by category, adding a language, loading data — must be reachable from outside
without editing the source. If only the project's own scripts can do it, it is
not a framework feature.

**Why.** `merged()` lived in `scripts/curated_glossary_data.py` and read a
module-level `DOMAINS` constant. The three built-in sets went through it; a
fourth, or a user's own glossary, could not. `Glossary.merge()` existed but
skipped the conflict check that the build script performed — so the *public* API
was less safe than the private one. And `from_file()` did not exist at all,
despite `Glossary.load` accepting paths internally.

**How to comply.**
* Put the capability in the library (`src/klix/`), not in `scripts/` or `evals/`.
  Those two directories exist to *use* the API, not to hold it.
* Expose it on a documented class: `Glossary.from_file/from_dict`,
  `GlossaryRegistry.register/get/merge/provenance_report`.
* Test it from the *installed wheel* as a stranger would, not only from the repo
  (see 12 and the checklist). The extension contract in
  `docs/glossary-format.md` exists for exactly this.
* The public path must be at least as safe as the internal one. If a script
  checks something, the library checks it.

**Anti-pattern.** A feature that only works because `sys.path` includes the repo
root. A "user extensibility" story that begins with "fork and edit".

---

## 3. No language, domain or category hard-coded when it is a data question

**Rule.** `"de"`, `"en"`, `"manufacturing"`, `"it"` are values, not control-flow
constants. They may appear in data files, presets and tests. They may not appear
in an `if`, a fixed bucket literal, or a validator branch.

**Why.** The engine was already language-agnostic (it iterates whatever keys it
finds), but the build path was not: `bucket = out.setdefault(key, {"de": [],
"en": []})` meant a French-only concept silently gained two empty buckets.
And "domain" had a privileged role in validation — a second conflict level that
existed only to serve the three domains that happened to exist, and would have
failed to fire for a fourth.

**How to comply.**
* Buckets are created on demand: `bucket.setdefault(lang, [])`, never a literal
  pair.
* Categories become free-floating metadata (`tags`). Filtering is a function over
  that metadata (`curated_where(tag)`) that works for any tag, including ones
  added later. No list of categories exists in the engine.
* If a language-specific behaviour is genuinely required, it belongs in the
  generator (which is allowed to be DE↔EN by design, because that is what it is
  *for*), not in the engine.
* Test with a value that does not exist yet (a language with no other data, a
  tag invented in the test) so the general path is exercised.

**Anti-pattern.** `if lang == "de"`. A validator branch named after a category.
A constant that must be updated when a new category is added.

---

## 4. Provenance and licence are required fields, not retroactive documentation

**Rule.** Every new data source is licence-checked at the source *before*
integration, and carries its origin in a metadata header. "It should be fine" is
not a licence review. An unverifiable source is rejected.

**Why.** The requested multi-source glossaries were all unusable, and only
checking revealed it: dbnary/Wiktionary is CC BY-SA 4.0, FreeDict `deu-eng` is
GPLv2+/AGPLv3, Apertium is GPL-3.0, PanLex was unreachable. All incompatible
with an MIT package. Checking first saved building an integration that would have
had to be thrown away — and the finding *changed the plan* (the curated layer had
to be written rather than derived). Had this been checked late, the licence
problem would have surfaced after the content existed.

**How to comply.**
* Fetch the licence from the source and record how it was verified: the
  `COPYING` file, the TEI header, a vendor API field, a rights API. Put it in
  `DATA_SOURCES.md` with the method, not just the conclusion.
* Provenance goes in the artifact's metadata header (`source`, `license`), so the
  shipped file is self-describing.
* Track provenance **per artifact/glossary**, never per entry — that was an
  explicit decision, and per-entry provenance is a separate unsolved problem.
  Do not pre-build it.
* A source that cannot be verified is out, even if the data is good. Record it as
  rejected with the reason.

**Anti-pattern.** Adding a dataset because it is convenient. Inferring a licence
from the platform that hosts it. Recording provenance only in the changelog.

---

## 5. Every number is reproducible from a script in the repository

**Rule.** A claim that will be reported — a hit rate, a latency, an error rate, a
file size — is produced by an experiment script committed alongside it. A number
that exists only in a chat message does not count.

**Why.** Most quantified claims in this project's history were revised after
being re-measured, and several were wrong the first time:

| Claim | What re-measurement showed |
|---|---|
| "fast-path miss costs 1–2 ms" | the eliminated duplicate vectorisation costs **~0.03 ms** median (n=300, load-dependent: 0.027–0.045 ms across runs) |
| "miss is 18.4 ms faster than baseline" | the sign flipped between runs (+7.9 ms) — not resolvable on this host |
| "20 % glossary error rate" | came from a failure-enriched probe; not the file's rate |
| "38.8 % error rate" | scored by key, counting correct mappings as errors → corrected to 8.4 % |
| "6.6 % wrong-mapping rate" | true of a 310-term denominator; the same measurement over 768 curated terms gives **8.4 %** |
| "glossary build is deterministic/byte-identical" | a re-run produced 10,049 concepts instead of 10,134 |
| "all 40 probe terms lack a Wikidata sense" | the query was broken; ~11 had a technical sense |
| "5.3 % of concepts are proper names" | produced by an ad-hoc command, not a script; a written criterion gives **8.2 %** (`evals/glossary_noise.py`) |

Every one of those was found by a script, not by argument. The scripts are
`evals/glossary_error_rate.py`, `evals/fastpath_overhead.py`,
`evals/centroid_regression_diag.py`, `evals/wikidata_sense_audit.py`,
`evals/glossary_footprint.py`.

**How to comply.**
* Write the measurement as `evals/<name>.py` with a module docstring stating the
  question, and commit it in the same change as the claim.
* The script prints the raw numbers, the sample size, and the method — so a
  reader can disagree with the method, not just the conclusion.
* Record the measurement in the docstring of the code it justifies, so the reason
  for a constant's value travels with the constant.
* Never restate a number from memory; re-run or cite the script.

**Anti-pattern.** A latency figure in a commit message with no script behind it.
Quoting a number from a previous session without re-measuring.

---

## 6. No feature without an explicit order or an explicitly documented proposal

**Rule.** Scope is what was asked for. Anything else is a proposal in a file, not
code. Extending scope because it seems obviously needed is a violation.

**Why.** During the glossary work it was tempting to add sense disambiguation,
retry logic for failed query classes, a place-name guard and a CI threshold —
four plausible improvements. They were written up as proposals instead, and the
write-up is what revealed that they would move a 20 % error rate to ~12 % while
the real problem was the data basis. Building them first would have consumed the
effort and hidden the finding.

**How to comply.**
* Do what was asked, to the standard required.
* Everything else goes to `docs/proposals.md` (or the response, clearly labelled
  as a proposal) with: what it would change, the measured motivation, the cost.
* If a task cannot be completed correctly without a change that was not
  requested, say so and stop — do not make the change quietly.
* A proposal is not a commitment, and silence is not consent.

**Anti-pattern.** "While I was in there, I also…" Refactoring adjacent code
because it looked wrong. Adding configuration nobody asked for.

---

## 7. Rejected approaches are documented

**Rule.** A path that was tried and abandoned, or considered and rejected, is
recorded with the reason in `docs/rejected-approaches.md`. A rejection that lives
only in a transcript will be repeated.

**Why.** NLTK/OMW was the obvious bilingual source and was abandoned for three
separate reasons (no `deu/` in `omw-1.4`; OMW sets are CC BY-SA; the model needs
a licence-compatible corpus). Re-deriving that costs an afternoon each time. The
same applies to the discarded dense-similarity metric — it is a genuinely
plausible quality metric that does not work, and nothing about reading it
suggests why.

**How to comply.**
* Append to `docs/rejected-approaches.md` when you abandon an approach, reject a
  source, or delete a mechanism — including mechanisms that were deleted as part
  of a release (`_guess_lang`, `_cross_lingual_mixup`).
* The entry states: what it was, why it looked right, what the disproof was
  (measured, with the script), and what replaced it.
* Do not delete the reasoning when the code goes. The code is recoverable from
  git; the reasoning is not.

**Anti-pattern.** A commit message that says "removed X" without saying why X
existed. A rejected source with no licence note.

---

## 8. Measure the mechanism, not a proxy — and measure before fixing

**Rule.** When a defect is described in terms of an end-to-end effect, measure
the mechanism that causes it. Report the resolution limit when the effect is
below the noise floor. And establish the size of the problem before designing the
fix.

**Why.** The fast-path miss was described as a 1–2 ms penalty. Measuring the
mechanism (how many times a query is vectorised) showed the real figure: ~0.03 ms.
The fix was correct but several orders of magnitude smaller than assumed. In
parallel, the end-to-end wall-clock comparison that "proved" the fix flipped sign
between runs (+7.9 ms vs −18.4 ms) because the dense embedding pass dominates and
outweighs variance.
A claim built on that comparison was not merely imprecise — it was unsupportable,
and it was stated as fact twice before being retracted.

**How to comply.**
* Prefer a deterministic count (invocations, entries, bytes) over a duration.
* When timing, report median **and** p95 **and** n, interleave the variants, and
  state the noise floor.
* If an effect cannot be resolved on the machine, say that instead of picking the
  favourable run.
* Before fixing, quantify the defect. The bug report is not a measurement.

**Anti-pattern.** A single timed run as proof. Quoting p95 from one variant and
median from another. Reporting a difference smaller than the run-to-run spread
without saying so.

---

## 9. Every number carries its denominator and its sampling method

**Rule.** State what was measured, over what population, and whether the sample
was randomly or deliberately chosen. A rate without a denominator is not a rate.

**Why.** The "20 % error rate" was true of a 40-term probe selected as *what a
manufacturing glossary must cover* — a deliberately demanding, failure-enriched
list. It was reported and re-reported as if it were the file's error rate. The
honest figure for the whole file was 6.6 %, and the honest statement for the
probe was "20 % on domain-critical terms". Two different questions, one number
carrying both.

**How to comply.**
* Name the population: "of 310 curated German terms", "of 362 concepts, all of
  them, not a sample", "of a random 300-concept sample with seed …".
* Say whether the sample is random or selected, and if selected, on what
  criterion.
* Distinguish *prevalence* (how often something occurs) from *severity* (how bad
  it is when it occurs) — they need different denominators.
* Report missing data separately from wrong data. They have different causes and
  different fixes.

**Anti-pattern.** "The error rate is X %" with no population. Reusing a
deliberately chosen probe as if it were a random sample.

---

## 10. Verify on one known case before running on many; check the data's real shape

**Rule.** Before a query, a script or a metric runs over N items, run it on one
item whose answer you know. And confirm the shape of the data you actually
receive — language, encoding, orthography — rather than the shape you assume.

**Why.** A single audit reported "40/40 terms have no Wikidata sense" — a
completely false result. Three independent causes, all of them assumptions:

1. `wikibase:lemma "Mutter"` matches nothing: lexeme lemmas are language-tagged
   (`"Mutter"@de`) and an untagged literal does not compare equal to them in
   SPARQL. The right form is `FILTER(STR(?lemma) = "Mutter")`.
2. The German probe used ASCII forms (`pruefung`) while the data holds `Prüfung`.
3. The "is this technical?" regex was English while `itemLabel`/`itemDescription`
   come back German, so almost nothing matched.

Each would have been caught by running the pipeline on one case with a known
answer. All three were reported as a data finding first.

**How to comply.**
* Pick one item with a known expected result; assert it before the loop.
* Inspect a raw response, not just the parsed one, when a result looks uniform.
* Never assume the language, casing or normalisation of incoming data — read a
  sample.
* In a test, use the value that does not exist yet (a fourth language, a new
  tag), so the general path is exercised rather than the configured one.

**Anti-pattern.** A batch job whose output is "everything failed". A regex
written in the language you think the data is in.

---

## 11. When something fails, decide explicitly whether the test or the artifact is wrong

**Rule.** A failing test is a question, not a verdict. Determine which side is
wrong, state which, and only then change something. Never adjust the assertion to
make it pass without naming what was wrong and why.

**Why.** Three times this session the test was wrong, not the code: a concept
with one term in one concept is not ambiguous (so a merge that succeeded was
correct); a nested provenance object was asserted as a string; a glossary clean
on its own was expected to be refused at registration. Conversely the two
`merge_all` failures were real and valuable — they exposed that `workflow()`
genuinely contradicts the curated layer on 7 terms.

Telling those apart matters: in the first case the fix is the test; in the second
it is a finding about the data.

**How to comply.**
* Read the failure and reproduce it outside the test harness before editing
  anything.
* Say plainly: "the test was wrong, here is why" or "the code was wrong, here is
  the defect". Both are acceptable; silently loosening an assertion is not.
* When loosening a guard, pin what actually matters instead. Replacing
  `centroid >= linear` with `centroid >= baseline AND centroid >= linear - 1`
  keeps both claims checkable; a blanket tolerance would have lost the signal.
* A failure that reveals real ambiguity in the data is a *finding* — record it.

**Anti-pattern.** Changing an expected value until the suite is green. Deleting
an inconvenient test. `pytest.mark.skip` without a reason.

---

## 12. Determinism and reproducibility claims require an actual repeat run

**Rule.** Never claim a build, pipeline or function is deterministic without
running it twice and comparing the output. State which parts are reproducible and
which are not.

**Why.** The glossary build was described as deterministic and byte-identical.
Running it again produced 10,049 concepts instead of 10,134. The cause was real
and structural: deep class subtrees intermittently time out on the query service,
failed classes are skipped by design, and the builder stops once it hits its
target — so a run with one additional lost class lands on a different, smaller
concept set. Query *ordering* was reproducible; the composed *artifact* was not.

**How to comply.**
* Run it twice, diff the result, before making the claim.
* Distinguish levels: query ordering reproducible / cache replay reproducible /
  composed artifact **not** reproducible.
* If an artifact is not reproducible, say so in the docs and treat the file as
  an artifact to be reviewed, with a provenance record (which inputs
  contributed).

**Anti-pattern.** "Should be deterministic" as a justification for a committed
generated file with no review.

---

## 13. Retract visibly, everywhere the claim appears — including shipped artifacts

**Rule.** When a claim is found to be wrong, correct it at every location it was
made: code comments, docstrings, changelog, documentation, tests, and the files
that ship to users. Record what the old number was and why it was wrong.

**Why.** The "1–2 ms miss penalty" was retracted in the changelog and in the
benchmark's own text — but not in `README.md`. The README is what the build
backend embeds into the wheel's `METADATA`, so the shipped artifact continued
advertising the number that had just been disproven. It was found only while
investigating an unrelated report, by an explicit grep. Meanwhile the commit
message of the tagged release still contains the retracted claim and cannot be
changed — which is why the correction had to be a commit of its own.

**How to comply.**
* After any retraction, grep the whole repo for the old figure — including
  `README.md`, and including generated files.
* Keep a retraction note rather than silently replacing the number; see the
  "Note on measurement validity" section in `DATA_SOURCES.md`.
* Remember that annotated tags and commit messages are immutable: the
  correction has to live in a later commit, and the changelog should say so.

**Anti-pattern.** Fixing the number in the place you noticed it. Assuming a doc
change does not reach users because "it is only documentation".

---

## 14. One conflict class, one handling — no special cases for the current shape

**Rule.** If two things are the same kind of conflict, they get the same
treatment. A rule that only exists for the configuration that currently happens
to exist is a latent failure.

**Why.** Term collisions aborted; concept-key collisions silently renamed. Both
are "two registrations competing for one lookup", so the asymmetry had no
justification beyond history. It also meant the guard would not fire at all for a
fourth glossary — the exact case the guard was supposed to protect. The fix was
to remove the special case, not to add a second one.

**How to comply.**
* Ask of every branch: "for what input does this NOT apply?" If the answer is
  "the ones that do not exist yet", the branch is wrong.
* Prefer one general mechanism over two specific ones.
* When removing a special case, delete its vocabulary too — `domain` as a
  validation concept is gone, not renamed.

**Anti-pattern.** `if key in reserved_names:`. A guard whose condition can only
be true for the inputs you tested.

---

## 15. Additive parameters must not break existing subclasses; new checks declare their dependencies

**Rule.** A new optional parameter on a library hook must not break callers that
do not know about it. New verification adds its dependency to the correct group
(runtime vs test).

**Why.** Passing a new `sparse=` keyword through to every head broke any user
subclass of `BaseHead` that did not accept it — a backward-compatibility break in
an opt-in feature, which is the worst place for one. Separately, adding
JSON-Schema validation of the glossary documents required `jsonschema`; putting
it in the runtime dependency list would have made an MIT package with three
runtime deps grow a fourth for a test's sake.

**How to comply.**
* Inspect the signature before passing a new keyword to an overridable hook; skip
  it for implementations that do not accept it (the project does this via
  `_accepts_keyword`).
* Keep the runtime dependency list as small as possible; test-only tools go to
  the dev group.
* A new optional feature must be a no-op when unused.

**Anti-pattern.** A new parameter threaded through every implementation. A test
helper added to the runtime dependencies.

---

## 16. The deliverable is the program's output, not the file's content

**Rule.** A change is verified by running the thing and inspecting what it
produces — not by reading the diff. Docstrings and comments are invisible to the
user of a library.

**Why.** The glossary's purpose is the terms the engine resolves at runtime. It
is entirely possible for the code to be correct, the tests to pass, and the
output to still contain the wrong vocabulary. Equivalently, an explanation in a
docstring does not change what `decide()` returns. The check is: run the
pipeline, grep the *output* for the new terms, and confirm the numbers the docs
promise are the numbers the code produces.

**How to comply.**
* After implementing, exercise the artifact and inspect its output.
* For data changes: confirm the new entries are actually present in the produced
  file and resolve as intended.
* For API changes: call it from outside, at least once, the way a user would.

**Anti-pattern.** "Implemented and tested" based on a diff that only adds an
explanation. Verifying a data pipeline by reading the pipeline.

---

## 17. No third-party names in artifacts; neutral vocabulary

**Rule.** Project artifacts — code, glossaries, documentation, examples, commit
messages — carry no employer, customer or vendor names as content or provenance.
Vocabulary profiles are described generically.

**Why.** An explicit instruction, and the right default for a public library:
baked-in organisational names make an artifact unusable elsewhere and leak
context that has no business being in a released package. Example vocabulary here
is described by function — support, everyday, mail, IT — never by whose workplace
it came from.

**How to comply.**
* Generic domain names in data and docs.
* If a specific organisation genuinely needs its own vocabulary, that is its own
  glossary file, registered from the outside (see 2) — not part of the default.
* Grep for names before committing, and include the generated data files in that
  grep, not just the source.

**Anti-pattern.** "Based on requirements from <organisation>". Example data
containing real internal terminology.

---

## 18. When a report contradicts expectation, inspect the reported object

**Rule.** A report that something is wrong, unchanged or missing is a claim about
an object. Inspect that object. Never dismiss a report as stale, or explain it
away with the command that produced it.

**Why.** Background jobs reported completion with failures that appeared to
concern already-fixed code. The tempting explanation was "stale run" — which is
both unfalsifiable and unhelpful. Investigating the reported object instead
found a real defect: a claim retracted in the changelog was still present in the
README, and therefore in the shipped wheel metadata (see 13). On another
occasion a reported "file not modified" was accurate and revealed that the file
had never existed in the first place — a different, simpler answer than any
theory about race conditions.

**How to comply.**
* Check the object the report names, in its current state.
* Prefer a fact that distinguishes the hypotheses (does the file exist? what does
  the artifact contain?) over reasoning about how the report came to be.
* Say what you found, including "the report was correct and here is why".

**Anti-pattern.** "That is from an older run." Explaining a report instead of
checking it.

---

## Pre-flight checklist

Work this before starting any non-trivial change. It is short on purpose.

- [ ] **Scope (6).** Is everything I am about to build explicitly requested? What
      is extra, and is it in `docs/proposals.md` rather than in code?
- [ ] **Data vs code (3).** Do I introduce any language, category or domain name
      into control flow? Could it be a value instead?
- [ ] **Failure mode (1, 14).** What happens on a conflict, ambiguity or missing
      key? Does it fail loudly, with the offending item and the way out? Does the
      same conflict class get the same treatment everywhere?
- [ ] **API (2, 15).** Can this be used from outside without editing the source?
      Does the public path check at least as much as the internal one? Do new
      parameters leave existing implementations working?
- [ ] **Licence (4).** Any new data or dependency: is the licence verified at the
      source, recorded with the method, and compatible with MIT? Runtime vs test
      dependency?
- [ ] **Evidence (5, 8, 9, 12).** For every number I will report: which script
      produces it, on what population, and does the claim require a repeat run?
- [ ] **Verification (10, 11, 16).** Have I run it on one known case first? Will I
      verify by exercising the output rather than reading the diff? If a test
      fails, am I able to say which side is wrong?
- [ ] **Rejections (7).** Did I abandon an approach? Is the reason recorded with
      its disproof?
- [ ] **Retractions (13).** Does this change supersede a claim made elsewhere —
      including in `README.md` and generated files? Grep for it.
- [ ] **Measurement matches the question (8, 9).** Does the number I am about to
      report actually answer what was asked? A coverage rate ("is this
      independently checkable") is not an error rate ("is this right"), and a
      metric whose reference is the thing under test cannot find its errors.
- [ ] **Is the recommendation measured, or argued? (8).** Before telling a user
      which configuration to choose, run the ablation — the plausible-sounding
      option is not always the better one, and the measurement may narrow a
      recommendation the docs already give. `docs/glossary-vs-anchors.md` is the
      worked example: the argument was right about (b) vs (c) and wrong about the
      value of *more* glossary work.
- [ ] **Neutrality (17).** Any third-party names in what I am committing?

## Related documents

| Document | Holds |
|---|---|
| `docs/glossary-format.md` | The data format and its stability guarantees (public API) |
| `docs/curated-correctness.md` | What can and cannot be measured about the curated list's correctness, with the figures and their limits |
| `docs/glossary-vs-anchors.md` | The ablation: does a glossary replace bilingual anchors or add to them — with the caveats that bound the answer |
| `docs/rejected-approaches.md` | Paths abandoned, with the disproof |
| `docs/proposals.md` | Suggested changes, not built (principle 6) |
| `DATA_SOURCES.md` | Licence chain per shipped file, with verification method |
| `evals/` | One script per reported number (principle 5) |
