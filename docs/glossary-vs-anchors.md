# Glossar vs. zweisprachige Anker: eine Ablation

## Die Frage

Im Chat stand die Behauptung: *ein Glossar schenkt dir beide Sprachen, während du
einsprachig schreibst* — plausibel, aber nie gemessen. Die Alternative ist
einfacher: Anker gleich in beiden Sprachen schreiben, das braucht gar kein Glossar.

Gemessen auf **demselben Schema** und **denselben 20 Testfällen** (10 EN + 10 DE),
für **beide Klassifikatoren**, mit Bootstrap-CI (2000 Resamples, Seed 20260928,
gepaart über Testfälle):

| Zelle | Anker | Glossar |
|---|---|---|
| (a) | nur EN (2/Klasse, 10 gesamt) | nein |
| (b) | nur EN | ja |
| (c) | EN+DE (4/Klasse, 20 gesamt) | nein |
| (d) | EN+DE | ja |

Reproduzierbar über einen zweiten Lauf (identische Zählwerte).
Skript: `evals/glossary_vs_bilingual_anchors.py`.

## Ergebnis

### centroid

| Zelle | EN | DE | gesamt | Median-Latenz |
|---|---|---|---|---|
| (a) EN, kein Glossar | 9/10 | 6/10 | 15/20 (75 %) | 21,2 ms |
| (b) EN + Glossar | 9/10 | 7/10 | 16/20 (80 %) | 11,0 ms |
| (c) EN+DE, kein Glossar | 9/10 | 6/10 | 15/20 (75 %) | 10,3 ms |
| (d) EN+DE + Glossar | 9/10 | 5/10 | **14/20 (70 %)** | 10,2 ms |

### linear

| Zelle | EN | DE | gesamt | Median-Latenz |
|---|---|---|---|---|
| (a) EN, kein Glossar | 10/10 | 6/10 | 16/20 (80 %) | 17,4 ms |
| (b) EN + Glossar | 10/10 | 6/10 | 16/20 (80 %) | 17,2 ms |
| (c) EN+DE, kein Glossar | 10/10 | 5/10 | 15/20 (75 %) | 21,1 ms |
| (d) EN+DE + Glossar | 10/10 | 5/10 | 15/20 (75 %) | 26,0 ms |

## Die entscheidenden Vergleiche

**(b) − (c)**: +5,0 % bei beiden Klassifikatoren, 95 %-CI [−10 %, +20 %] respektive
[−10 %, +20 %] → **nicht unterscheidbar**. Zweisprachige Anker sind also *nicht*
besser als einsprachige Anker plus Glossar. Die Vermutung hält der Messung stand.

**(d) − (c)**: centroid −5,0 % (CI [−15 %, +0 %]), linear ±0 %
→ das Glossar **addiert sich nicht** auf zweisprachige Anker. Es trägt nichts
Zusätzliches bei, sobald die Anker beide Sprachen abdecken.

## Der Mechanismus, fallweise

Das Erklärungsstück fehlte in der Zählung, deshalb hier ausgeschrieben — wie viele
der 10 deutschen Anfragen vom Glossar überhaupt berührt werden, und ob sich die
Antwort ändert:

* **Das Glossar expandiert 6 von 10 deutschen Anfragen.** Es ist also nicht
  wirkungslos. Die vier übrigen (`kreditkarte`, `tuerknauf`, `loesegeld`,
  `verschluesselt` …) haben keinen Eintrag im Preset.
* **Es ändert die Antwort nur bei `centroid` — ein einziger Fall:** `elternzeit`
  (`security` → `hr`, korrekt). Bei `linear` bleibt die Antwort in **allen** 10
  Fällen identisch, obwohl in denselben 6 Fällen Text ergänzt wird.
* Alle übrigen Fehler liegen **innerhalb der Domäne fest**, weil die Anker selbst
  mehrdeutig sind: `erstattung`/`kein Geld zurueck` → `facility`,
  `bildschirm`/`wlan` → `facility`, `kaffeemaschine` → `facility`. Das Glossar
  verschiebt diese Anfragen nur innerhalb eines Clusters bereits verwandter
  Anker.

**Kontraintuitiv und deshalb ausdrücklich:** in Zelle (d) verschlechtert sich
`auf meiner abrechnung stehen null stunden` (erwartet `hr`) von `hr` (c) auf
`billing` (d). Das Hinzufügen von Glossar-Text zu einem bereits zweisprachigen
Schema schadet hier — vermutlich, weil die Expansion die Abfrage näher an die
`billing`-Anker zieht (die deutsche `rechnung wurde doppelt abgebucht` /
`gutschrift` enthalten), während `abrechnung` selbst schon auf `hr` zeigte. Bei
n=20 ist ein einzelner Fall kein Beweis, aber es ist die Richtung, in der das
Glossar *nicht* hilft.

## Antwort auf die gestellte Frage

1. **Ersetzt (b) den Aufwand von (c)?** Gemessen: **ja, gleichwertig** (+5 %,
   nicht unterscheidbar). Einsprachig schreiben und das Glossar die Brücke
   schlagen lassen kostet nichts an Qualität — auf diesem Set.
2. **Addieren sich die Effekte in (d)?** **Nein.** Auf zweisprachigen Ankern
   bringt das Glossar keinen messbaren Zusatz und kostet bei centroid einen Fall.
3. **Wie viel weitere Glossararbeit lohnt sich?** Diese Messung stützt **keine**
   Ausweitung. Das Preset wird von 6/10 Anfragen getroffen, aber der Nutzen
   verschwindet, sobald die Anker zweisprachig sind — und das ist in einem
   deutschen Umfeld die naheliegende Konfiguration. Der belastbare Nutzen liegt
   in einem Fall (`elternzeit`) auf diesem Set; die Latenz-Unterschiede sind
   Rauschen, keine Aussage.

**Die praktische Empfehlung kippt damit gegenüber der bisherigen Doku:** für ein
zweisprachiges Schema sind **zweisprachige Anker der einfachere und ebenso gute
Weg**, und zusätzliche Glossararbeit ist nur dort sinnvoll, wo die Anker
*tatsächlich* einsprachig bleiben müssen (z. B. Schema aus einer fremden Quelle,
in der die zweite Sprache nicht gepflegt werden kann).

## Grenzen dieser Messung

Sie ist klein, und das steht hier, damit die Zahlen nicht überdehnt werden:

* **n = 20 Testfälle, 20 bzw. 10 Anker.** Das 95 %-CI ist entsprechend weit
  (bis ±20 Prozentpunkte). Die Aussage „nicht unterscheidbar" ist echt — aber sie
  heißt „diese Größe lässt sich hier nicht auflösen", nicht „die Effekte sind
  identisch".
* **Ein Domänenschema** (Support-Tickets, 5 Klassen). Nicht übertragbar auf
  Fertigungsrouting, andere Klassenzahl oder Satz-Anker statt Kurzphrasen.
* **Anzahl und Sprache sind im Design gekoppelt.** (a)/(b) haben 2 Anker pro
  Klasse, (c)/(d) haben 4 — die Verdopplung ist gewollt (das *ist* die
  Zweisprachigkeit), aber ein Unterschied zwischen (b) und (c) ist deshalb nicht
  allein der Sprachabdeckung zuzuschreiben.
* **Ein Glossar-Preset**, nicht ein maßgeschneidertes. Es deckt 12 von 20
  Domänen-Proben ab. Ein vollständigeres Glossar könnte (b)/(d) ändern — dann
  wäre die Messung zu wiederholen.
