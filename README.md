# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Entkoppelte Entscheidungs-Köpfe auf einem geteilten semantischen Backbone.
Ein Text geht **genau einmal** durch das Embedding-Modell (Dense + Sparse), beliebig viele
Köpfe (`Choice`, `Score`, `Flag`) arbeiten anschließend auf den vorberechneten Vektoren –
jeder in seinem eigenen mathematischen Raum. Kein Modelltraining, keine Slot-Limits,
vollständig offline und CPU-only.

## Architektur

```text
Text ──► HybridBackbone (FastEmbed-Dense + TF-IDF-Sparse, einmalig ~10 ms)
              │
              ├──► Choice   (Routing/Klassifikation: Max-Similarity + Keyword-Boost)
              ├──► Score    (kontinuierliche Achse: Low/High-Anker + Sigmoid)
              ├──► Flag     (Boolesch: 2/3-Klassen-Softmax mit Temperatur)
              └──► eigene Köpfe (von BaseHead erben)
```

- **Shared Backbone:** Der Text wird einmal embedding + einmal TF-IDF transformiert.
  3 Köpfe oder 50 Köpfe – die Extraktionskosten bleiben gleich.
- **Entkoppelte Köpfe:** Neue Optionen in einem `Choice` beeinflussen weder `Score`- noch
  `Flag`-Ergebnisse. Jeder Kopf kapselt seine eigene Logik.
- **Deklarativ:** Nur Schemata mit Beispielsätzen definieren, `compile()`, fertig.

## Installation

```bash
uv add klix-engine
# oder
pip install klix-engine
```

Beim ersten Aufruf lädt FastEmbed das Modell `paraphrase-multilingual-MiniLM-L12-v2`
(~120 MB, einmalig, danach lokal gecacht). Danach läuft alles offline.

## Schnellstart

```python
from klix import DecisionEngine, Choice, Score, Flag

engine = DecisionEngine()

engine.add_head(
    Choice(
        name="target",
        options={
            "it_ops": ["VPN abgerissen", "Server down", "Rechner bootet nicht"],
            "ot_plant": ["Roboterzelle steht", "SPS Fehler", "Taktzeit deviation"],
            "finance": ["KST 4210 über Budget", "Rechnung freigeben"],
            "facility": ["Öllache Halle 2", "Heizung defekt"],
        },
    )
)

engine.add_head(
    Score(
        name="urgency",
        low_anchors=["Routine-Wartung", "Informelle Frage"],
        high_anchors=["Notfall sofort", "Produktionsstillstand", "Akute Gefahr"],
        min_val=0.0,
        max_val=3.0,
    )
)

engine.add_head(
    Flag(
        name="is_security",
        true_anchors=["Hackerangriff", "Ransomware Befall", "Datenabfluss"],
        false_anchors=["Hardware kaputt", "Netzwerkstörung", "Alltägliche Anfrage"],
        threshold=0.5,
    )
)

engine.compile()

res = engine.decide("plc-34 meldet fehler, förderband steht sofort!")

print(res)                                   # <DecisionResult (11 ms): target=ot_plant, urgency=2.9, is_security=False>
print(res.target)                            # 'ot_plant'
print(res.urgency)                           # 2.87
print(res.is_security)                       # False
print(res.details("is_security"))            # {'value': False, 'probability': 0.03}
```

## Eigene Köpfe

Von `BaseHead` erben und `evaluate(encoded)` implementieren:

```python
import re
from klix import BaseHead

class RegExExtractionHead(BaseHead):
    def __init__(self, name: str, pattern: str):
        super().__init__(name)
        self.re = re.compile(pattern)

    def get_reference_texts(self) -> list[str]:
        return []  # keine Referenztexte nötig

    def fit(self, backbone) -> None:
        pass

    def evaluate(self, encoded) -> dict:
        match = self.re.search(encoded.text)
        return {"value": match.group(0) if match else None}
```

## Entwicklung

```bash
uv sync          # Abhängigkeiten installieren
uv run pytest    # Tests
uv run python examples/demo.py
uv build         # PyPI-Artefakte (wheel + sdist) nach dist/
uv publish       # Hochladen (erfordert Token/Account)
```

## Lizenz

MIT