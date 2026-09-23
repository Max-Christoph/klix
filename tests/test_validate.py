"""Tests for validate_anchors(): read-only class-overlap reporting."""

import pytest

from klix import Choice, DecisionEngine


def build_overlapping():
    """Two deliberately overlapping classes (near-duplicate anchors with
    only the decisive noun swapped) + one clearly distinct class."""
    eng = DecisionEngine()
    eng.add_head(Choice(
        name="c",
        options={
            "billing": [
                "please approve this invoice from the supplier",
                "the invoice needs approval before payment",
                "approve invoice number 42",
            ],
            "finance": [
                "please approve this invoice from the supplier",
                "approve the invoice for accounting",
                "invoice approval for the finance team",
            ],
            "technical": ["server down", "vpn broken", "laptop won't boot", "wifi outage"],
        },
    ))
    eng.compile()
    return eng


class TestValidateAnchors:
    def test_detects_billing_finance_overlap(self):
        eng = build_overlapping()
        findings = eng.validate_anchors()
        overlaps = [f for f in findings if f["kind"] == "overlap"]
        pair = next((f for f in overlaps if {f["a"], f["b"]} == {"billing", "finance"}), None)
        assert pair is not None, f"billing/finance overlap not detected: {overlaps}"
        assert pair["centroid_cos"] >= 0.75
        assert pair["severity"] in {"high", "medium"}

    def test_shared_terms_are_confusers(self):
        eng = build_overlapping()
        findings = eng.validate_anchors()
        pair = next(f for f in findings if f["kind"] == "overlap" and {f["a"], f["b"]} == {"billing", "finance"})
        # "approve" is the classic confuser and must appear among shared terms
        assert any("approve" in t for t in pair["shared_terms"]), pair["shared_terms"]

    def test_sharpening_hints_are_exclusive(self):
        eng = build_overlapping()
        findings = eng.validate_anchors()
        pair = next(f for f in findings if f["kind"] == "overlap" and {f["a"], f["b"]} == {"billing", "finance"})
        # exclusives must not overlap each other
        assert not (set(pair["a_exclusive"]) & set(pair["b_exclusive"]))

    def test_single_anchor_warning(self):
        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["only one"], "b": ["beta one", "beta two"]}))
        eng.compile()
        findings = eng.validate_anchors()
        struct = [f for f in findings if f["kind"] == "structure"]
        assert any("only 1 anchor" in f.get("message", "") for f in struct)

    def test_duplicate_anchor_warning(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={"a": ["alpha one", "alpha one"], "b": ["beta one", "beta two"]},
        ))
        eng.compile()
        findings = eng.validate_anchors()
        struct = [f for f in findings if f["kind"] == "structure"]
        assert any("duplicate" in f.get("message", "") for f in struct)

    def test_well_separated_gives_no_overlap_finding(self):
        eng = DecisionEngine()
        eng.add_head(Choice(
            name="c",
            options={
                "coffee": ["kaffeeemaschine reinigen", "neue kaffeebohnen besorgen"],
                "network": ["server down", "vpn verbindung abgebrochen"],
            },
        ))
        eng.compile()
        findings = eng.validate_anchors()
        overlaps = [f for f in findings if f["kind"] == "overlap"]
        # 'coffee' vs 'network' are semantically far apart
        assert all(f["centroid_cos"] < 0.75 for f in overlaps)

    def test_report_formatter(self):
        eng = build_overlapping()
        text = eng.validate_anchors_report()
        assert isinstance(text, str)
        assert ("billing" in text and "finance" in text) or "well separated" in text

    def test_never_mutates_anchors(self):
        eng = build_overlapping()
        before = dict(eng.heads[0].options)
        eng.validate_anchors()
        assert eng.heads[0].options == before  # identical, untouched

    def test_non_choice_heads_ignored(self):
        from klix import BaseHead

        class PlainHead(BaseHead):
            def __init__(self):
                super().__init__("p")

            def get_reference_texts(self):
                return []

            def fit(self, backbone):
                pass

            def evaluate(self, encoded):
                return {"value": 1}

        eng = DecisionEngine()
        eng.add_head(Choice(name="c", options={"a": ["alpha one"], "b": ["beta one"]}))
        eng.add_head(PlainHead())
        eng.compile()
        findings = eng.validate_anchors()  # must not raise on PlainHead
        assert all(f["head"] == "c" for f in findings)