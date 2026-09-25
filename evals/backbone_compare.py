"""Backbone & embedding-transform shootout on the repo's own corpora.

Three questions, one script (all deterministic, no training beyond the
existing compile-time linear probe, no external data):

  H1  Does truncating MiniLM embeddings to N dims really help, or was the
      earlier +5.7pt on n=70 a small-sample artifact?  -> measured on the
      expanded 273-case corpus.
  H2  Is truncation secretly a crude whitening?  Whitening (ZCA from the
      head's own anchors) is the principled version -> measured on full 384d,
      combined with truncation, and against the linear probe.
  H3  Was the Granite linear-probe loss a model problem or just un-tuned
      regularization (classifier_C)?  -> swept.

HEADLINE RESULTS (2026-09-25, Windows CPU, dev workstation)
----------------------------------------------------------
Backbone swap (60 frozen cases):
    MiniLM  nearest 41/60 = 68.3%   linear 51/60 = 85.0%
    Granite nearest 42/60 = 70.0%   linear 47/60 = 78.3%
  Granite is NOT an upgrade. Its nearest gain (+1.7pt) is inside the 95% CI
  (+-9pt at n=70) = noise; its linear loss (-6.7pt) is stable across every
  domain. H3 confirms it is not a tuning artifact: classifier_C from 0.5 to
  200 keeps Granite at 78-80% while MiniLM stays at 85%. Rejection final.

Truncation (MiniLM, nearest):
     60 cases   70 cases   273 cases
 full 68.3%     71.4%      93.0%
 256d 71.7%     74.3%      93.8%
 128d 75.0%     75.7%      94.5%
  64d 76.7%     77.1%      94.9%
  The gain is consistent across all three corpora but small, and NOT monotone
  on the frozen sets (96d dips) -- keep it opt-in, not default.

Whitening: DOES NOT WORK at this anchor scale, and the reason is structural.
  With 15 anchors in 384 dims the anchor covariance has RANK 14 (370 dims have
  zero anchor variance). Consequences measured directly:
    - nearest: accuracy identical for every dims setting (254/273 five times),
      because ZCA on the anchor span collapses all pairwise cosine
      similarities to one value (off-diagonal std = 0.0000) -> the ranking
      becomes a tie and any argmax is arbitrary.
    - linear: 96.7% -> 40.3%, because the 370 zero-variance dims get amplified
      by 1/sqrt(1e-6) = 1000x, injecting pure query noise.
  ZCA needs n_anchors >> dim (rule of thumb >=10x). Here it is 15 vs 384.
  This is a limit of the method at few-shot scale, not an implementation bug.

EmbeddingGemma-300M: dropped. Gated model (access request required) and its
ONNX export needs a separate 309 MB external-data file that FastEmbed's
model_file parameter cannot fetch (reproducible ONNXRuntimeError).

nomic-embed-text-v2-moe: not testable. Not in FastEmbed (only v1/v1.5 are),
the sole ONNX repo contains a README but no model files, and the model card
requires task prefixes ("the text prompt MUST include a task instruction
prefix"). MRL truncation needs a custom ONNX export + sentence-transformers.

Qwen3-Embedding-0.6B: not practical. Quantized ONNX is 613 MB (vs 98 MB
Granite), needs last_token pooling (FastEmbed offers CLS/MEAN/DISABLED only)
and instruction prefixes.

Run: uv run python -m evals.backbone_compare
     uv run python -m evals.whitening_vs_truncation
"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import time  # noqa: E402

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from fastembed.common.model_description import ModelSource, PoolingType  # noqa: E402

from klix import Choice, DecisionEngine  # noqa: E402
from evals.corpus_expansion import build_expanded_corpus  # noqa: E402
from evals.eval_domains import IMG_CASES, IMG_OPTIONS, SHOP_CASES, SHOP_OPTIONS, TASK_CASES, TASK_OPTIONS  # noqa: E402
from evals.variant_sweep import FIN_OPTIONS, FIN_TESTS, HR_OPTIONS, HR_TESTS  # noqa: E402

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GRANITE = "granite-97m-r2-onnx"

SETS = [
    ("HR", HR_OPTIONS, HR_TESTS),
    ("FIN", FIN_OPTIONS, FIN_TESTS),
    ("IMAGE", IMG_OPTIONS, [(q, e) for q, e in IMG_CASES if e]),
    ("TASK", TASK_OPTIONS, [(q, e) for q, e in TASK_CASES if e]),
    ("SHOP", SHOP_OPTIONS, [(q, e) for q, e in SHOP_CASES if e]),
]
CORPUS = build_expanded_corpus(SETS)


def _register_granite() -> None:
    try:
        TextEmbedding.add_custom_model(
            model=GRANITE, pooling=PoolingType.MEAN, normalization=True,
            sources=ModelSource(hf="philipp-zettl/granite-embedding-97m-multilingual-r2-ONNX"),
            dim=384, model_file="onnx/model_quantized.onnx", description=GRANITE,
            license="apache-2.0", size_in_gb=0.10,
        )
    except Exception as ex:  # noqa: BLE001
        print(f"[warn] {GRANITE}: {type(ex).__name__}: {ex}")


def _transform(anchor_vecs: np.ndarray, mode: str, dim: int | None):
    """Returns an apply(V)->V function, fit from this head's own anchors."""
    if mode == "none":
        return None

    def norm(out):
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        return out / np.where(n == 0, 1.0, n)

    if mode == "truncate":
        return lambda Z: norm(np.asarray(Z, dtype=np.float32)[:, :dim])

    X = np.asarray(anchor_vecs, dtype=np.float32)
    sl = slice(0, dim) if mode == "whiten_trunc" else slice(None)
    Xs = X[:, sl]
    mean = Xs.mean(axis=0, keepdims=True)
    cov = np.cov(Xs - mean, rowvar=False)
    d, V = np.linalg.eigh(cov)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(d, 1e-6))) @ V.T
    return lambda Z: norm((np.asarray(Z, dtype=np.float32)[:, sl] - mean) @ W)


def run(model: str, mode: str = "none", dim: int | None = None, classifier: str = "nearest",
        classifier_C: float | None = None, corpus: str = "expanded"):
    ok = tot = 0
    for dname, options, _t in SETS:
        tests = CORPUS["domains"][dname]["cases" if corpus == "expanded" else "base"]
        eng = DecisionEngine(model_name=model)
        kw = {"classifier": classifier}
        if classifier_C is not None:
            kw["classifier_C"] = classifier_C
        head = Choice(name="h", options=options, **kw)
        if mode != "none":
            raw = np.array(list(eng.backbone.embed_model.embed(head.get_reference_texts())), dtype=np.float32)
            apply_fn = _transform(raw, mode, dim)
            orig = eng.backbone.embed_model.embed

            def embed(texts, _o=orig, _f=apply_fn):
                return _f(np.array(list(_o(texts)), dtype=np.float32))
            eng.backbone.embed_model.embed = embed
        eng.add_head(head)
        eng.compile()
        ok += sum(1 for t, e in tests if eng.decide(t).h == e)
        tot += len(tests)
    return ok, tot


def show(label: str, **kw) -> None:
    kw.setdefault("model", MINILM)
    try:
        ok, tot = run(**kw)
        print(f"  {label:34s} {ok:3d}/{tot} = {ok/tot:6.1%}")
    except Exception as ex:  # noqa: BLE001
        print(f"  {label:34s} FAILED: {type(ex).__name__}: {str(ex)[:50]}")


def main() -> None:
    _register_granite()
    t0 = time.perf_counter()

    print(f"\nEXPANDED CORPUS: {CORPUS['total']} cases "
          "(60 frozen + generated anchor paraphrases). Variants are templates of")
    print("the anchors themselves, so absolute numbers are optimistic by design --")
    print("compare ROWS against each other, never against the 60/70 numbers.\n")

    print("=========== H1: truncation (MiniLM, nearest) ===========")
    for d in (None, 256, 192, 128, 96, 64):
        show(f"truncate {d or 'full'}", mode="truncate" if d else "none", dim=d)
    print("\n=========== H1b: same on the FROZEN 60 (the hard set) ===========")
    for d in (None, 256, 128, 96, 64):
        show(f"truncate {d or 'full'}", mode="truncate" if d else "none", dim=d, corpus="base")

    print("\n=========== H2: whitening vs truncation (expanded) ===========")
    show("none (baseline)")
    show("whiten (full 384d)", mode="whiten")
    show("whiten + 128d", mode="whiten_trunc", dim=128)
    show("truncate 128 (for compare)", mode="truncate", dim=128)

    print("\n=========== H2b: whitening on the FROZEN 60 ===========")
    show("none (baseline)", corpus="base")
    show("whiten (full 384d)", mode="whiten", corpus="base")
    show("truncate 128", mode="truncate", dim=128, corpus="base")

    print("\n=========== H2c: whitening with the LINEAR probe ===========")
    show("linear none (baseline)", classifier="linear")
    show("linear + whiten", mode="whiten", classifier="linear")
    print("  (whitening collapses the linear probe: rank-14 anchor span + 370")
    print("   zero-variance dims amplified 1000x = pure query noise)")

    print("\n=========== H3: Granite fairness -- classifier_C sweep (60 cases) ===========")
    show("MiniLM linear C=10 (default)", model=MINILM, classifier="linear", corpus="base")
    for c in (0.5, 2.0, 10.0, 50.0, 200.0):
        show(f"Granite linear C={c}", model=GRANITE, classifier="linear", classifier_C=c, corpus="base")

    print(f"\n[{time.perf_counter() - t0:.0f}s]")


if __name__ == "__main__":
    main()
