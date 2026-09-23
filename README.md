# Klix

[![PyPI version](https://img.shields.io/pypi/v/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![Python](https://img.shields.io/pypi/pyversions/klix-engine.svg)](https://pypi.org/project/klix-engine/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Decoupled decision heads on a shared semantic backbone.
A text passes through the embedding model **exactly once** (dense + sparse), after which any number
of heads (`Choice`, `Score`, `Flag`) operate on the precomputed vectors — each in its own
mathematical space. No model training, no slot limits, fully offline and CPU-only.

## Architecture

```text
Text ──► HybridBackbone (FastEmbed dense + TF-IDF sparse, once, ~10 ms)
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

## Installation

```bash
uv add klix-engine
# or
pip install klix-engine
```

On first use, FastEmbed downloads the `paraphrase-multilingual-MiniLM-L12-v2` model
(~120 MB, one-time, then cached locally). Everything runs offline afterwards.

## Quickstart

```python
from klix import DecisionEngine, Choice, Score, Flag

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

print(res)                                   # e.g. <DecisionResult (11 ms): target=ot_plant, urgency=2.4, is_security=False>
print(res.target)                            # 'ot_plant'
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
| `reject_anchors` | `Choice` | Texts matching these return `value=None` (don't-know instead of guess) |
| `keyword_boost` | `Choice` | Weight of exact keyword hits (asset IDs like `plc-34`) vs. semantic similarity |
| `aggregation` | `Score` | `"max"` (default) or `"topk"` — topk averages the best-k anchors per pole, robust against a single noisy anchor |
| `coverage` | `Score` result | Pooled similarity to the better pole; low (< ~0.3) means the score is noise |
| `min_val` / `max_val` / `sharpness` | `Score` | Output range and sigmoid steepness |
| `neutral_anchors` | `Flag` | Third pole for out-of-domain: returns `value=None` when it wins |
| `threshold` / `temp` | `Flag` | Decision cutoff and softmax temperature (lower = sharper) |
| `model_name` | `DecisionEngine` | Any FastEmbed-compatible embedding model |
| `stop_words` | `DecisionEngine` | Custom stopword list for the TF-IDF index (default: small German list) |
| `evaluate(encoded)` | `BaseHead` subclass | Add entirely custom head types (regex, business rules, ...) |

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

## Development

```bash
uv sync          # install dependencies
uv run pytest    # run tests (30 tests, fully offline)
uv run python examples/demo.py
```

The `evals/` directory contains a labeled evaluation harness (routing accuracy,
score bands, flag behavior, out-of-domain rejection) — use it to measure changes
to your anchor schemas.

**Release chain (automated):** bump the version in `pyproject.toml` and `__init__.py`,
commit, tag, push — GitHub Actions builds and publishes to PyPI automatically
(workflow `publish.yml`, secret `PYPI_TOKEN`):

```bash
git tag vX.Y.Z
git push origin main vX.Y.Z
```

## License

MIT