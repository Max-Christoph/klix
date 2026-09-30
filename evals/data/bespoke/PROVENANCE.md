# Bespoke-Eval-Daten — Provenienz, Lizenzen, Hashes

Diese Datei ist die **einzige** Quelle der Wahrheit über Herkunft und Lizenz der
Datensätze in `evals/data/bespoke/`. Die JSONL-Dateien selbst sind **nicht** im Repo
(Entscheidung §7.4): sie werden zur Laufzeit von `evals/bespoke_loader.py` geladen.
Wer die Zahlen aus `evals/run_bespoke.py` zitiert, zitiert die Hashes hier mit.

---

## 1. Datensätze

Abruf: **2026-09-30**, via `evals/bespoke_loader.py` (Parquet-Zweig der HF-Repos).

| Datei (lokal) | Quelle | Split | Zeilen | SHA256 |
|---|---|---|---|---|
| `massive_intent.de.jsonl` | `mteb/amazon_massive_intent`, config `de` | `test` | 2974 | `a4ebd46b159b63b7224f1941eb1df5c0397585d6214bebfb21884db41375d25f` |
| `massive_intent.de.train.jsonl` | ebd. | `train` | 11514 | `3c873e686c8e5d3ecdc4f3933e934c2ecece0ef5aad290561846e63375e9f9fb` |
| `massive_intent.en.jsonl` | ebd., config `en` | `test` | 2974 | `c27f8cf81641c8768637cd0a586061b1ead0bbc547d3af9a9110c2018c660426` |
| `massive_intent.en.train.jsonl` | ebd. | `train` | 11514 | `8ebade15f754c8c6e33f2a6bb0af8a19621f8dbc6adb3b1682fde6449af1c8b3` |
| `paws_adversarial.jsonl` | `paws`, config `labeled_final` | `test` | 8000 | `c1e4c07b7d34a6c02d2d50652a5634c0625fb2974d30629104a08c53c39abe5f` |
| `banking77.jsonl` | `banking77` | `test` | 3080 | `97cb2fbbed2adb7d4f25a7d77446f4aab087012cb2e08351fd2eb2fe4b63899a` |
| `banking77.train.jsonl` | ebd. | `train` | 10003 | `47fd17f80ef91c832c63b27d41e8aa37ba306bdb9eb274c86cf22bc6f364e081` |

**Lizenzen:** MASSIVE — Apache-2.0 laut HF-Karte *(MASSIVE-Standard: CC BY 4.0)*;
banking77 — **CC-BY-4.0, Attribution Pflicht** (Casanueva et al. 2020);
PAWS — Google Research, *„may be freely used for any purpose"*, Attribution erbeten
(Zhang et al. 2019). Nachweise in §2.

**Gesamtgröße:** 40 MB (7 Dateien). Deshalb **nicht** im Repo (§7.4, `.gitignore`).

### 1.1 Bekannte Verzerrung — bitte vor dem Zitieren lesen

MASSIVE (die MTEB-Republikation des Datensatzes) trennt `train` und `test`
**nicht auf Satzebene**. Gemessen am 2026-09-30:

| Sprache | Testtexte, die wortgleich auch in `train` stehen | davon mit **abweichendem** Label |
|---|---|---|
| de | 115 / 2974 = **3,9 %** | **8** |
| en | 21 / 2974 = **0,7 %** | 2 |

**Beispiel für Label-Inkonsistenz:** „welches datum haben wir" steht in `test` als
`calendar_query` und in `train` als `datetime_query`.

Konsequenzen, ausdrücklich festgehalten:

1. Die erreichbare Accuracy ist **gedeckelt** — der Anteil ist klein (≤ 3,9 %), aber
   er ist nicht 0. Ein Teil der Fehler ist nicht dem Modell anzulasten.
2. Beim Few-Shot-Anker-Verfahren (§7.2, `k=3` aus `train`) sind **7 von 180 deutschen
   Ankern** identisch mit einem Testtext (en: 0, banking77: 0). Diese 7 sind in allen
   beobachteten Fällen **label-konsistent** — sie erzeugen also kein falsches Training,
   aber sie sind trivial beantwortbare Fälle. `assert_no_leakage()` weist den Wert aus;
   `evals/run_bespoke.py` berichtet ihn im Ergebnis-JSON (`overlap_anchors`).
3. Wer den Datensatz weiterverwendet, sollte diesen Punkt kennen — er ist eine
   Eigenschaft der Quelle, nicht unseres Loaders.

---

## 1.2 Lizenz-Nachweise

- **MASSIVE** — HF-Karte von `mteb/amazon_massive_intent` nennt `license: apache-2.0`;
  das ursprüngliche MASSIVE (Amazon) steht unter CC BY 4.0. Für die Nutzung hier ist
  die Angabe der HF-Karte maßgeblich. Attribution: Amazon MASSIVE / MTEB.
- **banking77** — HF-Karte: `license: cc-by-4.0`. **Attribution ist Pflicht.**
  Quelle: Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020).
- **PAWS** — die HF-Karte führt `license: other`, was allein unbrauchbar ist. Die
  **Original-Lizenz** von Google Research (github.com/google-research-datasets/paws,
  `LICENSE`) sagt ausdrücklich:
  > *„The dataset may be freely used for any purpose, although acknowledgement of
  > Google LLC ('Google') as the data source would be appreciated."*

  Die Nutzung ist damit zulässig; die HF-Karte ist für diesen Datensatz irreführend.
  Attribution: PAWS, Zhang et al. (2019), Google Research.

**Kein Vendoring:** Es wird nichts in dieses Repository kopiert. Laden bedeutet hier
Abrufen zur Laufzeit. Ein Fehlschlag führt zum **Abbruch**, nie zu einem stillen
Rückgriff auf einen Cache — ein veralteter Cache würde unbemerkt verändern, was
„8000 Fälle" bedeutet (dieselbe Regel wie in `evals/run_jevbench.py`).

---

## 2. Fehlende / abweichende Task-Angaben

Beim Verifizieren am 2026-09-30 wichen vier Angaben der ursprünglichen Aufgabe von der
Realität ab. Sie sind hier festgehalten, damit sie nicht später erneut auftauchen:

| Angabe | Befund |
|---|---|
| `qanastek/MASSIVE`, Configs `de-DE`/`en-US` | Lädt nicht über `datasets`: das Repo wird von `datasets-server` nicht bedient (HTTP 501). Verwendet wird `mteb/amazon_massive_intent` mit den Configs **`de`** und **`en`**. |
| „je ~2.000 Beispiele" | Der `test`-Split hat **2974** Zeilen je Sprache (nicht ~2000). |
| `paws` Test „ca. 1.000" | `labeled_final/test` hat **8000** Zeilen (nicht ~1000). PAWS ist außerdem ein **Satzpaar**-Task, kein Ein-Text-Task — siehe Join-Strategie unten. |
| `banking77` 3.080 / 77 Klassen | **Bestätigt**, exakt 3080 Zeilen und 77 Klassen. |
| `ollama pull bespoke-minilm` | **Existiert nicht** (HTTP 404 auf ollama.com). Entfällt ersatzlos. |

**Herkunfts-Korrektur zu den Vergleichsmodellen:** `tev1` stammt von **Together AI**,
nicht von Bespoke Labs; `nimble` ist von **Bespoke Labs**. Die Formulierung „Bespoke Labs /
JevBench Baselines (Tev1 4B, Nimble 9B)" mischt zwei verschiedene Anbieter.

---

## 3. Repurposing-Caveat

⚠️ **Diese Daten dienen einem Repurposing-Vergleich, nicht einem Baseline-Vergleich.**

klix ist ein Support-/Prozess-Textklassifikator. MASSIVE (Intents), banking77
(Bank-Intents) und PAWS (Paraphrase-Erkennung) sind fremde Aufgabenstellungen. Der
Anker-Mechanismus von klix' `Choice`-Head verlangt Anker-Texte, die diese Datensätze
**nicht mitliefern** — sie werden nach einer dokumentierten Methode erzeugt
(`evals/bespoke_anchors.py`, Entscheidung §7.2).

**Die Zahlen sind deshalb nicht vergleichbar** mit den öffentlich genannten
Tev1-/Nimble-/JevBench-Werten. Es gibt keinen gemeinsam genutzten System-One-Benchmark;
die JevBench-Quelle sagt das selbst. Jedes Ergebnis-JSON trägt aus diesem Grund ein
`caveat`-Feld — analog zu `evals/jevbench_result.json` (Entscheidung §7.1, Option a).

---

## 4. Anker-Methoden (§7.2)

Für MASSIVE (60 Klassen) und banking77 (77 Klassen) werden zwei Anker-Varianten
gemessen, damit die Zahl nicht allein von erfundenen Formulierungen abhängt:

| Variante | Methode | Aussage |
|---|---|---|
| `few_shot_k3` | **k=3 Trainingsbeispiele pro Klasse** aus dem offiziellen `train`-Split, deterministischer Seed, **kein** Overlap mit `test` | Hauptmessung: klix im echten Few-Shot-Szenario |
| `label_string` | reiner Label-String (`_` → Leerzeichen) | **Baseline-Untergrenze**, ausdrücklich die schwächste Variante |

Der Train-/Test-Overlap wird zur Laufzeit **geprüft und behauptet** (nicht angenommen):
kein Test-Text darf als Anker auftreten.

---

## 5. PAWS-Join-Strategie (§7.3)

PAWS liefert Satzpaare. Gewählte Serialisierung (Variante a):

```
text    = sentence1 + " [SEP] " + sentence2
label   = "paraphrase" | "not_paraphrase"
options = ["paraphrase", "not_paraphrase"]
```

Verlustfrei: der Adversarial-Charakter von PAWS (gleiche Wortwahl, vertauschte
Satzstruktur) bleibt im gemeinsamen `text` erhalten. Die Label-Verteilung wird beim
Laden ausgegeben — PAWS `labeled_final` ist **nicht** 50/50.

---

## 6. Vergleichsmodelle (Ollama)

*(wird in Task 7 gefüllt — Modelldigest, Größe, Abrufdatum)*

| Modell | Anbieter | Größe | Digest | Zweck |
|---|---|---|---|---|
| `tev1:4b` | Together AI | 4.5 GB | *(Task 7)* | Baseline laut Aufgabe |
| `nimble:9b` | Bespoke Labs | 9.5 GB | *(Task 7)* | Baseline laut Aufgabe |
| `qwen3.5:2b` | lokal vorhanden | 2.7 GB | *(Task 7)* | Nullkosten-Untergrenze |
| `tev1:0.8b` | Together AI | 812 MB | *(Task 7)* | Fallback bei CPU-Unverträglichkeit |

**Hardware-Kontext:** Die Zielmaschine hat **keine NVIDIA-GPU** (Intel iGPU, 32 GB RAM,
12 Kerne). Die 4B/9B-Modelle laufen auf CPU. Deshalb schaltet Task 6 eine
Feasibility-Probe vor jeden Vollauf; bei einer Hochrechnung > 30 min wird auf 500 Fälle
subsampled und die lokale Baseline verwendet (Entscheidung §7.5).

---

## 7. Hashes erfassen

Nach dem Laden (Tasks 2/4/5) und vor dem ersten Bericht:

```bash
cd evals/data/bespoke && sha256sum *.jsonl
```

Die Ausgabe wird in die Tabelle in §1 eingetragen. Ein Bericht ohne eingetragene Hashes
gilt als unvollständig.
