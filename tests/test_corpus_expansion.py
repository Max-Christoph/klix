"""E.1 corpus regression: expanded corpus is deterministic and labels valid."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.corpus_expansion import generate_variants, build_expanded_corpus  # noqa: E402
from evals.variant_sweep import HR_OPTIONS, HR_TESTS  # noqa: E402


class TestCorpusExpansion:
    def test_deterministic(self):
        a = generate_variants(HR_OPTIONS, per_anchor=3, seed=42)
        b = generate_variants(HR_OPTIONS, per_anchor=3, seed=42)
        assert a == b

    def test_labels_match_schema(self):
        for text, label in generate_variants(HR_OPTIONS, per_anchor=3):
            assert label in HR_OPTIONS
            assert isinstance(text, str) and text.strip()

    def test_variants_added_not_replaced(self):
        corpus = build_expanded_corpus([("HR", HR_OPTIONS, HR_TESTS)])
        d = corpus["domains"]["HR"]
        assert d["base"] == list(HR_TESTS)  # frozen cases untouched
        assert len(d["cases"]) > len(d["base"])

    def test_no_duplicate_texts(self):
        corpus = build_expanded_corpus([("HR", HR_OPTIONS, HR_TESTS)])
        texts = [t for t, _ in corpus["domains"]["HR"]["cases"]]
        assert len(texts) == len(set(texts))

    def test_corpus_size_grows_substantially(self):
        corpus = build_expanded_corpus([("HR", HR_OPTIONS, HR_TESTS)])
        assert len(corpus["domains"]["HR"]["cases"]) >= 45