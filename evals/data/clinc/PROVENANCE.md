# CLINC150 (oos-eval) — Provenienz, Lizenz, Hashes

Diese Datei dokumentiert die Herkunft, Lizenz und Prüfsummen des CLINC150 Out-of-Scope Datensatzes in `evals/data/clinc/`. Die Datendatei `data_full.json` selbst ist via `.gitignore` von der Versionsverwaltung ausgeschlossen, um das Repository leichtgewichtig zu halten.

---

## 1. Datensatz-Spezifikation

* **Projekt:** CLINC150 / `oos-eval`
* **Veröffentlichung:** Larson et al., EMNLP 2019: *"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction"*
* **Quelle:** [https://github.com/clinc/oos-eval](https://github.com/clinc/oos-eval)
* **Datei (lokal):** `evals/data/clinc/data_full.json`
* **Dateigröße:** 2.495.390 Bytes (~2,38 MB)
* **SHA256:** `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

### Split-Struktur

| Split Key | Beschreibung | Anzahl Samples | Klassen |
|---|---|:---:|:---:|
| `train` | In-Scope Training (100 Samples / Klasse) | 15.000 | 150 Intents |
| `val` | In-Scope Validierung (20 Samples / Klasse) | 3.000 | 150 Intents |
| `test` | In-Scope Test (30 Samples / Klasse) | 4.500 | 150 Intents |
| `oos_train` | Out-of-Scope Trainingsbeispiele | 100 | OOS |
| `oos_val` | Out-of-Scope Validierungsbeispiele | 100 | OOS |
| `oos_test` | Out-of-Scope Testevaluierung | 1.000 | OOS |

---

## 2. Lizenz & Attribution

* **Lizenz:** Creative Commons Attribution 3.0 Unported (CC BY 3.0)
* **Attribution Pflicht:**
  > Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, Jason Mars.  
  > *"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction"*, EMNLP 2019.

---

## 3. Reproduzierbarkeit

Die Datei kann automatisiert über das Skript `evals/clinc_fetcher.py` mit SHA256-Validierung bezogen werden:

```bash
uv run python evals/clinc_fetcher.py
```
