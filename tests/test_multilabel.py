"""Tests for the MultiLabel decision head (multi-label classification with continuous scores)."""
import pytest
from klix import DecisionEngine, MultiLabel


@pytest.fixture
def multilabel_schema():
    return {
        "security": ["hacker attack", "ransomware infection", "data breach", "unauthorized access"],
        "performance": ["system latency", "slow query response", "server lag", "high CPU usage"],
        "billing": ["invoice inquiry", "credit card payment failed", "subscription charge", "refund request"],
    }


def test_multilabel_validation_checks():
    with pytest.raises(ValueError, match="requires at least one category"):
        MultiLabel(name="empty", options={})

    with pytest.raises(ValueError, match="has no anchor texts"):
        MultiLabel(name="bad", options={"cat": []})


def test_multilabel_reference_texts(multilabel_schema):
    head = MultiLabel(name="topics", options=multilabel_schema)
    refs = head.get_reference_texts()
    assert len(refs) == 12
    assert "hacker attack" in refs
    assert "invoice inquiry" in refs


def test_multilabel_centroid_single_and_multi_hits(multilabel_schema):
    eng = DecisionEngine()
    eng.add_head(MultiLabel(name="topics", options=multilabel_schema, threshold=0.5, classifier="centroid"))
    eng.compile()

    # Case 1: Pure single topic
    res1 = eng.decide("A hacker gained unauthorized access to our customer database!")
    assert "security" in res1.topics
    assert "billing" not in res1.topics
    details1 = res1.details("topics")
    assert details1["scores"]["security"] > 0.6
    assert details1["scores"]["billing"] < 0.4

    # Case 2: Multi-topic overlap (security + performance)
    res2 = eng.decide("Ransomware attack caused massive server lag and high CPU usage across nodes")
    assert "security" in res2.topics
    assert "performance" in res2.topics
    assert "billing" not in res2.topics

    # Case 3: Completely unrelated text
    res3 = eng.decide("What is the weather like in Paris today?")
    assert res3.topics == []
    for score in res3.details("topics")["scores"].values():
        assert score < 0.5


def test_multilabel_aggregation_modes(multilabel_schema):
    for clf in ("centroid", "max", "topk"):
        eng = DecisionEngine()
        eng.add_head(MultiLabel(name="topics", options=multilabel_schema, classifier=clf, topk=2))
        eng.compile()
        res = eng.decide("server response is terribly slow")
        assert "performance" in res.topics
        scores = res.details("topics")["scores"]
        assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_multilabel_calibration_modes(multilabel_schema):
    for calib in ("sigmoid", "linear", "cosine"):
        eng = DecisionEngine()
        eng.add_head(MultiLabel(name="topics", options=multilabel_schema, calibration=calib))
        eng.compile()
        res = eng.decide("Payment failed for my monthly subscription")
        assert "billing" in res.topics
        scores = res.details("topics")["scores"]
        assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_multilabel_decide_batch(multilabel_schema):
    eng = DecisionEngine()
    eng.add_head(MultiLabel(name="topics", options=multilabel_schema, threshold=0.5))
    eng.compile()

    texts = [
        "Unauthorized login detected from foreign IP",
        "The web application has high latency and takes 10s to load",
        "Can I get a refund on invoice 1234?",
    ]
    results = eng.decide_batch(texts)
    assert len(results) == 3
    assert "security" in results[0].topics
    assert "performance" in results[1].topics
    assert "billing" in results[2].topics


def test_multilabel_explain_and_schema_hash(multilabel_schema):
    eng = DecisionEngine()
    head = MultiLabel(name="topics", options=multilabel_schema, threshold=0.5)
    eng.add_head(head)
    eng.compile()

    res = eng.decide("urgent data breach alert")
    explanation = res.explain("topics")
    assert "value" in explanation
    assert "scores" in explanation
    assert "MultiLabel detected" in explanation["interpretation"]

    h1 = eng.schema_hash()
    assert isinstance(h1, str) and len(h1) == 16

    # Schema hash changes if threshold changes
    head.threshold = 0.8
    h2 = eng.schema_hash()
    assert h1 != h2
