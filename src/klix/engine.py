"""Die Orchestrierungs-Engine: bindet Backbone und Köpfe zusammen."""

import time

from klix.backbone import HybridBackbone
from klix.heads import BaseHead


class DecisionResult:
    """Ergebnis eines `decide()`-Aufrufs; Kopf-Werte als Attribute abrufbar."""

    def __init__(self, text: str, latency_ms: float, head_data: dict):
        self.text = text
        self.latency_ms = latency_ms
        self.data = head_data

    def __getattr__(self, name: str):
        if name in self.data:
            return self.data[name]["value"]
        raise AttributeError(f"Kopf '{name}' existiert nicht im Ergebnis.")

    def details(self, name: str) -> dict:
        """Vollständiges Ergebnis-Dict eines einzelnen Kopfs."""
        return self.data.get(name, {})

    def __repr__(self):
        items = [f"{key}={value['value']}" for key, value in self.data.items()]
        return f"<DecisionResult ({self.latency_ms:.1f}ms): {', '.join(items)}>"


class DecisionEngine:
    """Shared-Backbone-Engine mit beliebig vielen entkoppelten Köpfen.

    Der Text wird pro `decide()` genau einmal encoded; alle Köpfe evaluieren
    danach parallel auf denselben Vektoren.
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.backbone = HybridBackbone(model_name=model_name)
        self.heads: list[BaseHead] = []
        self._compiled = False

    def add_head(self, head: BaseHead) -> "DecisionEngine":
        """Registriert einen Kopf (fluent, verkettbar)."""
        self.heads.append(head)
        self._compiled = False
        return self

    def compile(self) -> None:
        """Sammelt alle Texte aller Köpfe ein und initialisiert die Indizes."""
        all_texts = []
        for head in self.heads:
            all_texts.extend(head.get_reference_texts())

        if not all_texts:
            raise ValueError("Keine Referenztexte vorhanden: erst Köpfe via add_head() registrieren.")

        self.backbone.build_vocabulary(all_texts)

        for head in self.heads:
            head.fit(self.backbone)

        self._compiled = True

    def decide(self, text: str) -> DecisionResult:
        """Encoded den Text einmal und evaluiert alle Köpfe auf den Vektoren."""
        if not self.heads:
            raise ValueError("Keine Köpfe registriert: erst add_head() aufrufen.")
        if not self._compiled:
            self.compile()

        start = time.perf_counter()

        # 1. Einmalige Vektorisierung (~10-12 ms).
        encoded = self.backbone.encode(text)

        # 2. Evaluation aller Köpfe (< 0.2 ms insgesamt).
        results: dict[str, dict] = {}
        for head in self.heads:
            results[head.name] = head.evaluate(encoded)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return DecisionResult(text=text, latency_ms=elapsed_ms, head_data=results)