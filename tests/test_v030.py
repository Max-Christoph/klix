"""Tests for v0.3.0 features: classifier="auto", translate_fn, bilingual
stopwords, and honest Flag calibration signals (margin/coverage)."""

from klix import Choice, DecisionEngine, Flag
from klix.backbone import _DEFAULT_STOPWORDS


class TestBilingualStopwords:
    def test_default_includes_english_and_german(self):
        assert "the" in _DEFAULT_STOPWORDS  # English
        assert "die" in _DEFAULT_STOPWORDS  # German

    def test_empty_list_disables_filtering(self):
        eng = DecisionEngine(stop_words=[])
        eng.add_head(Choice(name="c", options={"a": ["the alpha"], "b": ["the beta"]}))
        eng.compile()
        # "the" is now a vocabulary term (no stopword filtering)
        assert "the" in eng.backbone.tfidf_vec.vocabulary_

    def test_custom_list_replaces_default(self):
        eng = DecisionEngine(stop_words=["customword"])
        eng.add_head(Choice(name="c", options={"a": ["alpha customword"], "b": ["beta"]}))
        eng.compile()
        assert "customword" not in eng.backbone.tfidf_vec.vocabulary_
        # default stopword "the" is now kept (custom list replaced default)
        eng2 = DecisionEngine(stop_words=["customword"])
        eng2.add_head(Choice(name="c", options={"a": ["the alpha"], "b": ["beta"]}))
        eng2.compile()
        assert "the" in eng2.backbone.tfidf_vec.vocabulary_


class TestClassifierAuto:
    def test_auto_picks_nearest_for_mixed_language(self):
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={
                    "a": ["the server is down", "der laptop startet nicht"],
                    "b": ["refund my order", "gutschrift fehlt auf dem konto"],
                },
                classifier="auto",
            )
        )
        eng.compile()
        assert eng.heads[0]._effective_classifier == "nearest"

    def test_auto_picks_linear_for_single_language(self):
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={
                    "a": ["the server is down", "network outage reported"],
                    "b": ["refund my order", "charge was wrong"],
                },
                classifier="auto",
            )
        )
        eng.compile()
        assert eng.heads[0]._effective_classifier == "linear"


class TestTranslateFn:
    def test_translate_fn_augments_and_improves(self):
        # A tiny dictionary translator: German anchor -> English mirror.
        DICT = {
            "die rechnung wurde doppelt abgebucht": "the invoice was charged twice",
            "gutschrift fehlt auf dem konto": "credit note is missing",
        }

        def fake_translate(text, target):
            return DICT.get(text.lower())

        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={
                    "billing": [
                        "die rechnung wurde doppelt abgebucht",
                        "gutschrift fehlt auf dem konto",
                        "the invoice amount is wrong",
                    ],
                    "technical": ["the server keeps crashing", "vpn keeps dropping"],
                },
                classifier="linear",
                translate_fn=fake_translate,
            )
        )
        eng.compile()
        head = eng.heads[0]
        # German query should now route to billing thanks to the mirror anchors.
        assert eng.decide("die rechnung ist falsch").c == "billing"

    def test_translate_fn_none_is_ok(self):
        eng = DecisionEngine()
        eng.add_head(
            Choice(
                name="c",
                options={"a": ["alpha"], "b": ["beta"]},
                classifier="linear",
                translate_fn=None,
            )
        )
        eng.compile()
        assert eng.decide("alpha").c == "a"


class TestFlagCalibration:
    def test_flag_returns_margin_and_coverage(self):
        eng = DecisionEngine()
        eng.add_head(
            Flag(
                name="f",
                true_anchors=["hacker attack", "ransomware infection"],
                false_anchors=["hardware broken", "printer jam"],
            )
        )
        eng.compile()
        d = eng.decide("ransomware encrypted files").details("f")
        assert "margin" in d
        assert "coverage" in d
        assert 0.0 <= d["margin"] <= 1.0
        assert 0.0 <= d["coverage"] <= 1.0

    def test_flag_clear_case_has_high_margin(self):
        eng = DecisionEngine()
        eng.add_head(
            Flag(
                name="f",
                true_anchors=["hacker attack", "ransomware infection", "data breach"],
                false_anchors=["hardware broken", "printer jam", "network down"],
            )
        )
        eng.compile()
        d_clear = eng.decide("ransomware encrypted our files").details("f")
        # A clear case should have a large true/false margin.
        assert d_clear["margin"] > 0.5
