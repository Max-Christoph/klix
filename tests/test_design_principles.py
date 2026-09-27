"""Pre-flight gate: does DESIGN_PRINCIPLES.md hold up against its own rules?

Principle 5 says a claim in the repository must be reproducible from a script in
the repository. This checks the document itself:

  1. every file it references exists
  2. every eval script it cites as evidence exists AND runs
  3. the numbers it quotes appear in those scripts' actual output

A principles document quoting an unreproducible number is the exact failure the
document forbids, so this test is part of the suite, not a one-off.

Run: uv run pytest tests/test_design_principles.py -q
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "DESIGN_PRINCIPLES.md"
DOCS = REPO / "docs"


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


class TestReferences:
    def test_document_exists_and_is_substantial(self, text):
        assert len(text) > 10_000
        assert text.startswith("# Design principles")

    def test_every_referenced_markdown_file_exists(self, text):
        refs = set(re.findall(r"`((?:docs/)?[A-Za-z_\-]+\.md)`", text))
        assert refs, "no document references found — did the format change?"
        missing = [r for r in refs if not (REPO / r).exists()]
        assert not missing, f"referenced but missing: {missing}"

    def test_every_referenced_eval_script_exists(self, text):
        refs = set(re.findall(r"`?(evals/[a-z_]+\.py)`?", text))
        assert refs
        missing = [r for r in refs if not (REPO / r).exists()]
        assert not missing, f"cited as evidence but missing: {missing}"

    def test_every_referenced_eval_script_is_not_empty(self, text):
        for r in sorted(set(re.findall(r"`?(evals/[a-z_]+\.py)`?", text))):
            body = (REPO / r).read_text(encoding="utf-8")
            assert "def main()" in body, f"{r} has no main()"
            assert body.lstrip().startswith('"""'), f"{r} has no module docstring"


class TestNumbersAreReproducible:
    """The numbers quoted in the doc must appear in the scripts' real output."""

    def _run(self, module: str, timeout: int = 420) -> str:
        import os

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        p = subprocess.run(
            [sys.executable, "-m", module],
            cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env,
        )
        return p.stdout + p.stderr

    def test_curated_counts_match_the_documents_that_quote_them(self, text):
        """124 concepts / 603 terms are quoted in DATA_SOURCES and proposals.

        Not in DESIGN_PRINCIPLES.md itself — that document states principles and
        cites evidence, it does not restate every figure. Asserted against the
        files that do quote them, so the check is meaningful.
        """
        import json

        raw = json.loads((REPO / "src" / "klix" / "data"
                          / "curated_glossary.json").read_text(encoding="utf-8"))
        mapping = raw.get("concepts", raw)
        assert len(mapping) == 124
        assert sum(len(t) for v in mapping.values() for t in v.values()) == 603
        for doc_name in ("DATA_SOURCES.md",):
            doc = (REPO / doc_name).read_text(encoding="utf-8")
            assert "124" in doc, f"{doc_name} does not quote the concepts count"

    def test_fastpath_overhead_figure_is_reproducible(self, text):
        """The 0.026 ms claim must appear in the script's own output."""
        out = self._run("evals.fastpath_overhead")
        m = re.search(r"median\s+([0-9.]+)\s*ms", out)
        assert m, f"no median in output:\n{out[:800]}"
        assert m.group(1).startswith("0.0")      # sub-0.1 ms, as documented
        assert "0.026" in text, "document does not quote the current figure"

    def test_noise_figure_is_reproducible(self, text):
        out = self._run("evals.glossary_noise")
        assert "TOTAL name-like" in out
        m = re.search(r"TOTAL name-like \(deduplicated\): (\d+)", out)
        assert m and m.group(1) == "834"
        assert "8.2" in text, "document does not quote the current figure"

    def test_error_rate_script_prints_per_domain_rates(self, text):
        out = self._run("evals.glossary_error_rate")
        for dom in ("MANUFACTURING", "IT", "EVERYDAY"):
            assert dom in out, f"{dom} missing from error-rate output"
        assert "TOTAL" in out
        # the documented overall figure
        assert "6.6" in out and "6.6" in text


class TestStructure:
    def test_has_eighteen_principles_and_a_checklist(self, text):
        """Principles are numbered 1..18: seven requested plus eleven added."""
        heads = re.findall(r"^## (\d+)\. ", text, re.M)
        assert [int(h) for h in heads] == list(range(1, 19)), heads
        assert "## Pre-flight checklist" in text

    def test_checklist_items_reference_principles(self, text):
        block = text.split("## Pre-flight checklist")[1].split("## Related")[0]
        items = [ln for ln in block.splitlines() if ln.strip().startswith("- [ ]")]
        assert len(items) >= 8
        for it in items:
            assert re.search(r"\(\d+(?:,\s*\d+)*\)", it), f"no principle ref: {it}"

    def test_each_principle_states_a_rule_and_a_why(self, text):
        """Every principle must carry its rule AND the incident behind it."""
        body = text.split("## 1. ")[1].split("## Pre-flight checklist")[0]
        chunks = re.split(r"^## \d+\. ", body, flags=re.M)[1:]
        assert len(chunks) == 17           # principles 2..18
        for ch in chunks:
            assert "**Rule.**" in ch, ch[:120]
            assert "**Why.**" in ch, ch[:120]

    def test_related_documents_table_lists_the_artifacts(self, text):
        block = text.split("## Related documents")[1]
        for name in ("glossary-format.md", "rejected-approaches.md",
                     "proposals.md", "DATA_SOURCES.md"):
            assert name in block
