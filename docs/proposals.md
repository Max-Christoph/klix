# Proposals

Changes that look worth making but have **not** been built, because they were
not requested. Principle 6: scope is what was asked for; everything else lives
here until someone decides.

Each entry states the measured motivation, what it would change, the cost, and
what is uncertain. Entries are not commitments. Silence is not approval.

---

## P1 — Finish the Wikidata sense filter (raise `broad()` from 6.6 % to ~4 %)

**Motivation, measured.** `broad()` has a 6.6 % wrong-mapping rate
(`evals/glossary_error_rate.py`). `evals/wikidata_sense_check.py` shows the
cause splits cleanly in two:

* **~40 % of the wrong cases have a correct technical item linked in Wikidata
  already** — `Mutter` → `Q190977` ("Gegenstück einer Schraube, mit Innengewinde"),
  `Verschleiß` → `Q617224`, `Leistung` → `Q25342`. The builder reads `P5137` but
  ignores that a polyseme has several senses and takes whichever comes first.
  Preferring a sense whose item sits in the manufacturing/technical class would
  fix these.
* **~60 % have no usable sense at all** — `Freigabe`, `Ausfall`, `Getriebe`,
  `Relais`, `Ersatzteil`, `Taktzeit`, `Stillstand` have no P5137 link, and
  `Störung` / `Ausschuss` / `Dichtung` link only to a different domain (ecology,
  committee, art). Not fixable by querying.

**What it would change.** Maybe 6.6 % → ~4 % on `broad()`. The curated default is
unaffected — it has no measured errors.

**Cost.** Moderate: sense-selection logic in `scripts/build_default_glossary.py`,
plus a rebuild (~7 min from cache) and a re-measure.

**Uncertain.** The sense-preference heuristic needs a technical-class marker, and
that marker is itself a judgement call. Worth doing only if `broad()` gains users.

---

## P2 — Retry and re-fetch the 7 class downloads that fail intermittently

**Motivation, measured.** `src/klix/data/build_meta.json` records 7 classes that
were skipped, including `processes` (`Q1914636`) and `qualities` (`Q1207505`) —
precisely the process/property vocabulary a manufacturing glossary wants. Skipped
classes are also why a rebuild is not byte-reproducible.

**What it would change.** Better coverage in `broad()` and a more reproducible
build. Would not improve correctness (see P1) — more words, not better ones.

**Cost.** Low: smaller page sizes and a backoff retry in the builder.

**Note.** Explicitly *not* the fix for the error rate. This was one of four
"obvious" fixes proposed earlier; the measurement showed it addresses coverage,
not correctness.

---

## P3 — Place-name guard for the generated vocabulary

**Motivation, measured** (`evals/glossary_noise.py`). 834 of 10,134 concepts
(8.2 %) are name-like, in two forms:

* place/institution-qualified: 241 (2.4 %) — `altar_from_medias`, `windmill_in_auma`
* place + common noun: 593 (5.9 %) — `achel_brewery`, `abbensen_windmill`,
  `adolf_von_baeyer_gold_medal`

The existing `NAME_LIKE_RE` guard only catches German/English common nouns, so
`<place> <common noun>` slips through because the proper name comes first.

An earlier "5.3 % / 533" came from an ad-hoc command and used an undocumented
criterion; it is superseded. Taxonomic Latin is also real noise in this
vocabulary but is **not** counted — a regex cannot separate it from ordinary
two-word compounds (see the note in the script).

**What it would change.** Noise 8.2 % → ~2–3 % on `broad()`. Curated default
unaffected.

**Cost.** Low.

**Uncertain.** A stricter guard risks dropping legitimate compounds.

---

## P4 — Put the error-rate measurement in CI as a threshold

**Motivation.** The wrong-mapping rate is currently a measurement you have to
remember to run. A regression that adds ambiguous vocabulary would not be caught.

**What it would change.** `evals/glossary_error_rate.py` becomes a gate: fail if
the rate rises above a documented ceiling.

**Cost.** Low, but it needs a deterministic sampling seed and a stable
denominator — and the curated list is the ground truth, so the check only covers
terms the curated list happens to contain (39 % coverage of it).

**Uncertain.** A CI gate on a heuristic metric can produce false alarms and
invite loosening, which principle 11 warns about. Would need care.

---

## P5 — Grow the curated glossary to 300-500 concepts

**Motivation, measured.** The curated layer is the default and has no measured
errors, but it was narrow: 124 concepts at the time this proposal was written
(now 362 after the P5 work below), covering 39 % of the curated German terms
in the broader Wikidata vocabulary. Coverage by domain was uneven — IT 26 %,
everyday 38 %, manufacturing 48 %.

**Requested distribution** (explicitly *not* an even split): manufacturing may
stay small, since it already holds the central terms; IT and especially everyday
language should gain the most volume.

**What it would change.** Better default coverage in the two weak domains.

**Cost.** Writing time at the same quality bar — every concept through the
validator, collisions aborting, and the error rate re-measured on a **new**
random sample (not the 310 known cases, which are no longer blind).

**Status: DONE.** 362 concepts (44 manufacturing, 186 IT, 132 everyday/support/
mail), 1495 terms. Manufacturing deliberately unchanged. Four gates passed:
`merge_sources` (0 collisions), `validate()` (0 findings), `assert_valid()`
(0 ambiguous mappings), JSON Schema (0 errors). New-sample measurement
(`evals/curated_error_rate_new.py`, seed 20260928, n=160): internal ambiguity 0,
external attestation 28.7 % overall (manufacturing 44 %, IT 26 %, everyday 28 %).
Two defects were surfaced by the measurement and fixed: `merged()` read the
`DOMAINS` constant and silently missed the second content module, and
`Glossary({...})` did not strip per-concept `tags`, so tag values were read as a
language called "tags" and produced 359 bogus ambiguous mappings.

---

## P6 — Per-term provenance

**Motivation.** Provenance is currently per glossary, so a single file cannot mix
sources with different licences. If a future glossary wants to blend curated and
generated concepts with distinct licence obligations, one header is insufficient.

**Status.** Deliberately **not** pre-built. Explicit user decision: per-glossary
provenance now, per-term provenance only if the need actually arises.

---

## P7 — Split the generated DE↔EN builder from a language-general one

**Motivation.** `scripts/build_default_glossary.py` hard-wires DE (`Q188`) and EN
(`Q1860`) in its SPARQL queries. That is intentional — building a DE↔EN
dictionary is what it is *for* — but it means the generator cannot produce, say,
a FR↔EN vocabulary.

**What it would change.** A parameterised builder, so a new language pair is
configuration rather than a second script.

**Cost.** Moderate; the query construction and the label/lemma handling both
assume the pair.

**Uncertain.** Whether a second language pair will ever be needed. The engine side
is already language-agnostic (principle 3), so this is the only remaining
language-specific component — and it is a script, not the library.

---

## P8 — Per-domain presets as the recommended default, and a part-level concept

**Motivation, measured.** The curated glossary is one merged tri-domain list, and
the lookup is flat and context-free. Measured (`evals/polysemy_review.py`): **62 of
768 terms (8.1 %) carry more than one Wikidata sense**, and one term can map to only
one concept — the format refuses a second assignment, so the ambiguity is
structural. On a mixed-domain schema the other reading can therefore be bridged to
the wrong concept.

The manufacturing exposure is contained today (all 11 polysemous manufacturing
terms resolve inside manufacturing; 10 have no competing concept at all), with one
real exception: `lager` means *bearing* as well as *stock*, resolves to
`warehouse`, and **no part-level concept exists** for the other reading — so it
could not resolve correctly even in principle.

**What it would change.** Two independent things, either of which is useful on its
own:

1. A part-level concept (`bearing` / `kugellager` / `wellenlager`) closes the
   coverage half of `lager`, and gives the bearing reading an unambiguous surface
   form of its own.
2. Documenting `curated_where(tag)` as the recommended entry point for a
   single-domain schema — the engine already supports per-domain glossaries
   (one glossary per head, or a tag filter), but the README leads with the merged
   preset, which is the configuration that carries the ambiguity.

**Cost.** (1) is a data edit plus a rebuild and re-measure — small. (2) is
documentation only.

**Uncertain.** Whether (1) changes anything measurable: the bare `lager` query would
still resolve to `warehouse`, because adding a concept does not add context. It
would only help queries that name the part explicitly. That is worth measuring
before claiming a benefit — the honest expectation is that it converts one silent
wrong bridge into one silent miss.

**Not a fix for the general case.** Removing synonyms does not help (measured: the
ladder query stays on the office concept regardless), and context disambiguation is
out of scope because the engine is deliberately stateless.

---

## P9 — A test that can actually settle the glossary-vs-bilingual-anchors question

**Motivation.** `evals/glossary_vs_bilingual_anchors.py` ran the four-cell ablation
and could not answer it: the 95 % CI for (b) − (c) spans −10…+20 points at n=20, and
one of the two classifiers turned out to be structurally inert for the glossary
(`classifier="linear"` predicts from the dense vector and returns before the
expansion runs). Result: an absence of evidence that was first written up as
equivalence — corrected in `README.md` and `docs/glossary-vs-anchors.md`. Doing it
properly needs the following.

**1. n ≥ 60 test cases per cell, paired.** At n=20 the CI on a paired difference of
proportions is ~±20 points, which is wider than any effect worth detecting. For a
10-point effect at 80 % power and α=0.05, a paired design needs on the order of
60–100 *discordant* pairs, i.e. several hundred cases in total, not 60. The
estimate must be done before the run, not after.

**2. A configuration in which the sparse channel decides.** `classifier="centroid"`
or `"hybrid"` — not `"linear"`. Verified: `glossary_weight` 0→50 yields 1 answer
pattern under `linear`, 2–3 under the others. For `hybrid`, the query's sparse part
only addresses columns inside the head's anchor-derived vocabulary, so the glossary's
influence there depends on those terms being in that vocabulary — worth confirming
per schema.

**3. Equal anchor COUNTS across cells.** The current design couples anchor count to
language coverage (2 EN vs 4 EN+DE per class), so a (b)/(c) difference is not
attributable to language coverage alone. Needed: three cells with the *same* number
of anchors per class, e.g. 4 anchors as (i) 4 EN, (ii) 2 EN + 2 DE, (iii) 2 EN +
2 machine-translated DE. (iii) is the important one — it separates "two languages"
from "two *native* languages", which is the difference the glossary is supposed to
buy.

**4. Cases where the dense model alone fails.** The multilingual backbone already
routes many German queries against English anchors correctly, so the glossary has
little room to show an effect. The test population must be *filtered* to queries the
dense path gets wrong without the glossary — otherwise it is structurally
underpowered no matter how large n is.

**What test data is missing.** None of it exists in the repo today:

* **A large parallel EN/DE case set with sentence-length phrasing.** The existing
  bilingual set is 20 curated pairs of short phrases; `evals/benchmark.py` and
  `evals/expanded_benchmark.py` carry EN-heavy domain cases, not parallel pairs.
* **Native DE authoring, independent of the author.** Cases written by translating
  the EN side inherit the translator's phrasing, which is exactly what makes them
  easy for the dense model. A second native speaker must write the DE side.
* **A machine-translated DE variant** for the (iii) cell above — produced by a
  pinned, documented translation step so the cell is reproducible.
* **A density-failure subset**: the cases where `nearest`/`centroid` without a
  glossary get the German query wrong. Requires running the baseline first and
  selecting on it, which introduces its own selection bias — must be split into a
  selection half and an evaluation half.

**Cost.** Data collection dominates: several hundred native-checked parallel cases,
plus a filtering pass. The harness itself is a small extension of the existing
script.

**Uncertain.** Whether the effect is large enough to be worth the data collection.
The point estimate was +5 points for (b) over (c) — if that is the true size, ~100
pairs per cell are needed to see it. If the true size is 0, no amount of data will
show a difference, and the honest answer is then the one already in the docs: not
resolvable at the tested scale, and the glossary's measurable value must be argued
per schema rather than in general.

**STATUS: ZURÜCKGESTELLT — Aufwand ohne erwartbaren Nutzen.**

Die Power-Rechnung ist der Grund, und sie steht gegen die Durchführung:

* Beobachteter Punktschätzer für (b) − (c): **+5 Punkte**.
* Für einen Effekt dieser Größe braucht ein gepaartes Design bei 80 % Power und
  α = 0,05 rund **60–100 diskordante Paare**, praktisch mehrere hundert Fälle —
  also die vollständige Neuerhebung eines parallelen EN/DE-Satzes.
* Der Aufwand dafür liegt bei **mehreren hundert muttersprachlich geprüften
  Testfällen** plus Vorfilterung und Selektionsbias-Trennung.
* Der Nutzen wäre eine Aussage über *Anker-Konfiguration* in **einem** Domänenschema.
  Die Architektur-Aussage, die tatsächlich trägt (welcher Klassifikator den
  Sparse-Kanal überhaupt sieht), ist bereits belegt und braucht keine neuen Daten.
* Bei einer wahren Effektgröße von 0 zeigt kein Datenvolumen einen Unterschied — der
  Aufwand wäre dann vollständig verloren, und das ist nicht ausschließbar.

Damit ist der erwartete Nutzen nicht nur klein, sondern von der Größe abhängig, die
der Test erst bestimmen soll. Das ist die Definition eines schlechten Geschäfts.

---

## P10 — `classifier="linear"` + Glossar: warnen, oder ist es ein Fehler?

**Motivation, gemessen.** Unter `classifier="linear"` läuft die Vorhersage über
`self._probe.predict_proba(encoded.dense_vec)` und **kehrt zurück**
(`src/klix/heads.py:1170-1171`, Return `:1197`), bevor die Glossar-Expansion bei
`:1210` erreicht wird. Ein gesetztes Glossar ist auf diesem Pfad **wirkungslos** —
und zwar **still**: gemessen löst `glossary_weight` 0,0 vs. 0,5 unter `linear`
*keine* Änderung aus und erzeugt *keine* Warnung, während dieselbe Konfiguration
unter `centroid` die Antwort kippt (`security` → `hr` bei `elternzeit`).

**Warum das ein Problem ist.** Ein Nutzer, dem Cross-Lingual-Routing wichtig ist,
setzt `glossary=` und `classifier="linear"` — zwei für sich vernünftige
Entscheidungen — und bekommt ein Schema, das die eine Hälfte seiner Absicht
stillschweigend nicht ausführt. Genau die Fehlerklasse, die Prinzip 1 verbietet
(„fail explicitly instead of being silently wrong").

**Der `auto`-Fall ist nicht betroffen, und das ist zu prüfen gewesen.**
`classifier="auto"` löst seit v0.9.0 **immer** zu `"centroid"` auf (`:665-667`); der
frühere „linear bei einsprachigen Ankern"-Zweig wurde mit der Sprachheuristik
entfernt. Gemessen: `auto` → `_effective_classifier == "centroid"`,
`_probe is None`, `_centroid_matrix` gesetzt — das Glossar **wirkt** (Antwort kippt
von `security` auf `hr`). Eine Warnung für `auto` wäre also gegenstandslos. Offen
ist nur, dass `auto` als veralteter Alias **ohne DeprecationWarning** parst
(gemessen: keine Warnung) — das ist ein separater, kleinerer Punkt.

**Offener zweiter Fall.** `hybrid` erreicht den Sparse-Kanal, aber nur über Spalten
innerhalb der anker-abgeleiteten Vokabel des Heads (`:1163`). Gemessen auf diesem
Schema liegen nur **30 von 1424** Glossar-Termen in dieser Vokabel (2 %). Ein
`hybrid`-Schema mit Glossar kann also *teilweise* wirken — schwerer zu beurteilen
als `linear`, aber zur Compile-Zeit berechenbar.

**Drei Optionen, mit Begründung — Entscheidung liegt beim Nutzer.**

1. **`UserWarning` bei Compile (empfohlen).** Erkennungsbedingung ist statisch und
   eindeutig: `glossary is not None` **und** `glossary_weight > 0` **und**
   `effective_classifier == "linear"`. Kein Verhaltensbruch, kein Test bricht, kein
   bestehendes Schema hört auf zu funktionieren — und der Nutzer erfährt in dem
   Moment davon, in dem er das Glossar verdrahtet. Für `hybrid` dieselbe Warnung,
   wenn der Overlap zwischen Glossar-Termen und `head._vocab` **0** ist (dann ist
   die Wirkung garantiert null); bei kleinen Overlaps nur ein Hinweis.
2. **`ValueError` bei Compile.** Härter, aber verhaltensbrechend: jedes bestehende
   Schema mit dieser Kombination würde beim `compile()` sterben — auch wenn das
   Glossar dort nur „für später" gesetzt war. Das verstößt gegen Prinzip 15
   (additive Parameter dürfen Bestehendes nicht brechen).
3. **Nichts tun, nur dokumentieren.** Billigste Variante, aber sie lässt genau die
   stille Fehlannahme stehen, die den Befund ausgelöst hat — und die Doku ist der
   Ort, den ein Nutzer mit „warum routet das nicht?" am seltensten liest.

**Empfehlung.** Option 1, dazu ein `DeprecationWarning` für `classifier="auto"`
(separater kleiner Punkt). Nicht implementiert — wartet auf Entscheidung.

**Cost.** Klein: eine Prüfung pro Head in `compile()`, keine Laufzeitkosten, keine
neuen Abhängigkeiten. Bestehende Tests, die `linear` + Glossar kombinieren, müssten
die Warnung ggf. filtern — betrifft nur eigene Tests, nicht Nutzer.
