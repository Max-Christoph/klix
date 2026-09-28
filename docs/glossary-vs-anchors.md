# Glossar vs. zweisprachige Anker: eine Ablation — und was sie *nicht* belegt

## Die Frage

Die Behauptung: *ein Glossar schenkt dir beide Sprachen, während du einsprachig
schreibst*. Die Alternative wäre, die Anker gleich zweisprachig zu schreiben und
gar kein Glossar zu brauchen. Gemessen auf demselben Schema und denselben 20
Testfällen (10 EN + 10 DE), beide Klassifikatoren, Bootstrap-CI (2000 Resamples,
Seed 20260928). Skript: `evals/glossary_vs_bilingual_anchors.py`.

| Zelle | Anker | Glossar |
|---|---|---|
| (a) | nur EN (2/Klasse) | nein |
| (b) | nur EN | ja |
| (c) | EN+DE (4/Klasse) | nein |
| (d) | EN+DE | ja |

## Zuerst: auf welchem Pfad wirkt das Glossar überhaupt?

Bevor Zahlen gedeutet werden, muss klar sein, **ob der Sparse-Kanal die
Entscheidung überhaupt erreicht**. Eine Null-Differenz kann „kein Effekt" heißen
oder „das Feature wurde nie befragt". Die Codestellen, alle in
`src/klix/heads.py`:

| classifier | Sparse-Kanal in der Entscheidung? | Wo |
|---|---|---|
| `nearest` | **ja** — `hybrid_sims = dense_sims + boost * sparse_sims`, dann Max über die Anker | `:1234`, Pooling `:1256-1261` |
| `centroid` | **ja** — Sparse pro Label gemittelt und mit demselben `boost` addiert | `:1237-1247` |
| **`linear`** | **nein** — der Probe sagt aus `encoded.dense_vec` vorher und **kehrt zurück**; die Glossar-Expansion bei `:1210` ist unerreichbar | `:1170-1171`, Rücksprung `:1197` |
| `hybrid` | **ja** — Query-Zeile ist `[dense \| tfidf]`, aus `sparse` gebaut | `:1158-1169` |

Mechanisch geprüft statt aus dem Code geschlossen — `glossary_weight` von 0 bis 50
variiert, gezählt wie viele verschiedene Antwortmuster entstehen:

```
nearest    2 Muster   -> Kanal wirkt
centroid   3 Muster   -> Kanal wirkt
linear     1 Muster   -> Kanal wirkungslos, selbst bei 50-fachem Gewicht
hybrid     2 Muster   -> Kanal wirkt
```

Zusatz bei `hybrid`: der Sparse-Teil der Query-Zeile nimmt nur Spalten
`< sparse_part.shape[0]` auf, also nur Terme, die in der **anker-abgeleiteten
Vokabel** des Heads stehen. Ein Glossar-Term außerhalb dieser Vokabel trägt dort
nichts bei.

**Konsequenz für die Ablation: die `linear`-Zeile hat das Glossar nie befragt.**
Sie ist kein Messergebnis über das Glossar, sondern eine Aussage über den Code-Pfad.
Die frühere Formulierung „unter `linear` verschiebt das Glossar keine Antwort" war
als Befund dargestellt — das war falsch.

## Ergebnis

### centroid (Kanal ist aktiv — die Zeile ist aussagekräftig)

| Zelle | EN | DE | gesamt | Median-Latenz |
|---|---|---|---|---|
| (a) EN, kein Glossar | 9/10 | 6/10 | 15/20 (75 %) | 21,2 ms |
| (b) EN + Glossar | 9/10 | 7/10 | 16/20 (80 %) | 11,0 ms |
| (c) EN+DE, kein Glossar | 9/10 | 6/10 | 15/20 (75 %) | 10,3 ms |
| (d) EN+DE + Glossar | 9/10 | 5/10 | 14/20 (70 %) | 10,2 ms |

**(b) − (c) = +5,0 %, 95 %-CI [−10 %, +20 %]**
**(d) − (c) = −5,0 %, 95 %-CI [−15 %, +0 %]**

### linear (Kanal ist inaktiv — nicht aussagekräftig)

| Zelle | EN | DE | gesamt |
|---|---|---|---|
| (a) | 10/10 | 6/10 | 16/20 |
| (b) | 10/10 | 6/10 | 16/20 |
| (c) | 10/10 | 5/10 | 15/20 |
| (d) | 10/10 | 5/10 | 15/20 |

(a)=(b) und (c)=(d) **exakt** — das ist exakt, was der Code-Pfad vorhersagt. Diese
Zeile gehört als Beleg für die Pfad-Analyse gelesen, nicht als Beleg über das
Glossar.

## Was der Fallmechanismus zeigt (centroid, wo der Kanal greift)

* Das Glossar **feuert**: es expandiert **6 von 10** deutschen Anfragen. Es ist
  also nicht wirkungslos.
* Es ändert die Antwort in **genau einem** Fall zum Richtigen: `elternzeit`
  (`security` → `hr`).
* Die übrigen Fehler liegen **innerhalb der Domäne fest**, weil die *Anker*
  untereinander mehrdeutig sind: `erstattung`, `bildschirm`, `kaffeemaschine`
  landen alle auf `facility`. Ein Glossar kann mehrdeutige Anker nicht reparieren.
* Bei `glossary_weight = 5` kippt zusätzlich `wo bleibt die erstattung …` nach
  `billing` (richtig), bei 50 wieder zurück — die Wirkung ist nicht monoton.

## Antwort auf die gestellte Frage — mit der Einschränkung davor

**Diese Messung entscheidet die Frage nicht.** Der ehrliche Satz lautet:

> Auf n=20 ist der Unterschied zwischen (b) und (c) **nicht auflösbar**. Das CI
> schließt die Null ein. Das ist kein Nachweis der Gleichwertigkeit, sondern das
> Fehlen eines Nachweises in beide Richtungen.

Was daraus *nicht* folgt — und was in einer früheren Fassung dieses Dokuments zu
stark behauptet wurde:

* **Nicht** „zweisprachige Anker sind der einfachere und ebenso gute Weg." Bei
  n=20 und CI ±20 Punkten ist „ebenso gut" nicht belegt.
* **Nicht** „weitere Glossararbeit lohnt sich nicht." Dafür ist der Test zu klein
  und der Kanal nur in einer von zwei Konfigurationen überhaupt aktiv.
* **Nicht** „das Glossar bringt nichts." Es bringt in dieser Messung einen Fall
  (`elternzeit`) — bei n=20 ist das nicht von Rauschen zu trennen.

Der belastbare Teil des Ergebnisses ist die **Pfad-Analyse**: unter `linear` ist
der Kanal baulich inaktiv, unter `centroid`/`nearest`/`hybrid` aktiv. Wer
Cross-Lingual-Verhalten mit einem Glossar steuern will, muss einen dieser drei
Klassifikatoren wählen — mit `linear` ist das Glossar für die Entscheidung
wirkungslos. Das ist ein Befund über die Architektur, nicht über Ankerzahlen.

## Grenzen, und was ein aussagekräftiger Test braucht

Die Schwächen dieses Aufbaus, benannt:

1. **n = 20, ein Domänenschema, 5 Klassen.** Das gepaarte CI ist bis ±20 Punkte weit.
2. **Ankerzahl und Sprachabdeckung sind gekoppelt.** (a)/(b) haben 2 Anker/Klasse,
   (c)/(d) haben 4. Ein (b)/(c)-Unterschied ist damit nicht allein der
   Sprachabdeckung zuzuschreiben.
3. **Ein Klassifikator war inaktiv.** Die halbe Messung ist verschenkt.
4. **Der multilinguale Backbone erledigt einen Teil der Arbeit selbst.** Deutsche
   Anfragen gegen englische Anker laufen schon über das Dense-Modell recht gut —
   das Glossar kann nur dort etwas bewirken, wo Dense versagt. Ohne Vorfilterung
   auf genau solche Fälle ist der Test strukturell unterpowert.

Der notwendige Aufbau steht als **Vorschlag P9** in `docs/proposals.md`, inklusive
Power-Rechnung und der Liste, welche Testdaten dafür fehlen.
