"""Unit tests for the bespoke eval loaders.

Network is NOT touched here. Every test either exercises a pure normalisation
function or reads an already-downloaded JSONL and skips when it is absent — a
test suite that silently depends on huggingface.co being up fails for reasons
that have nothing to do with the code (the same rule tests/test_langid.py
follows for its figures).

The one property worth asserting about the module itself: it must import
WITHOUT pyarrow installed. pyarrow is an eval-only concern, and the loader is
run as `uv run --with pyarrow python -m evals.bespoke_loader`. If a top-level
`import pyarrow` ever creeps in, the module becomes unimportable for anyone who
has not installed it — so that is a test, not a comment.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.bespoke_loader import (  # noqa: E402
    DATA_DIR,
    intents_from_rows,
    normalize_massive,
    normalize_paws_pair,
    resolve_banking77_label,
)

MASSIVE_ROW = {
    "id": "1",
    "label": "alarm_set",
    "label_text": "alarm_set",
    "text": "weck mich am freitag um neun uhr auf",
    "lang": "de",
}


class TestBanking77LabelResolution:
    """banking77 ships integer label indices; the names live in parquet metadata.

    This is the bug the first version of the loader had: it wrote `label: 11` while
    `options` held strings, every prediction was then compared against an int, and
    the run reported `accuracy=None` over 100 cases. The old test asserted
    `label in options`, which is vacuously true for `11 in [0..76]` — so the test
    suite was green while the pipeline was broken. These tests use names.
    """

    NAMES = ["activate_my_card", "age_limit", "apple_pay_or_google_pay"]

    def test_index_becomes_the_class_name(self):
        assert resolve_banking77_label(1, self.NAMES) == "age_limit"
        assert resolve_banking77_label("2", self.NAMES) == "apple_pay_or_google_pay"

    def test_index_zero_is_not_confused_with_missing(self):
        # `if not value` would swallow index 0 and silently mislabel the first class.
        assert resolve_banking77_label(0, self.NAMES) == "activate_my_card"

    def test_numeric_strings_are_indices_not_labels(self):
        # A republication may hand the column over as strings; "1" is still index 1.
        assert resolve_banking77_label("1", self.NAMES) == "age_limit"

    def test_real_label_strings_pass_through(self):
        assert resolve_banking77_label("age_limit", self.NAMES) == "age_limit"

    def test_missing_class_names_refuses_instead_of_guessing(self):
        with pytest.raises(ValueError, match="cannot resolve the label"):
            resolve_banking77_label(1, None)

    def test_out_of_range_index_is_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            resolve_banking77_label(99, self.NAMES)


class TestMassiveNormalisation:
    def test_produces_the_required_schema(self):
        rec = normalize_massive(MASSIVE_ROW, options=["alarm_set", "alarm_query"])
        assert set(rec) == {"id", "text", "label", "options"}

    def test_keeps_the_text_and_label_verbatim(self):
        rec = normalize_massive(MASSIVE_ROW, options=["alarm_set"])
        assert rec["text"] == MASSIVE_ROW["text"]
        assert rec["label"] == "alarm_set"
        # id is coerced to str because the parquet column type differs between
        # the de and en configs; downstream code keys on it.
        assert rec["id"] == "1"

    def test_options_is_a_list_not_a_tuple(self):
        # A tuple would round-trip to a JSON array anyway, but a list keeps the
        # in-memory shape equal to the on-disk shape, so `json.loads(line) == rec`.
        rec = normalize_massive(MASSIVE_ROW, options=("alarm_set", "alarm_query"))
        assert isinstance(rec["options"], list)
        assert json.loads(json.dumps(rec)) == rec

    def test_label_must_be_among_the_options(self):
        # Otherwise the record is unscoreable: the model can never be right.
        with pytest.raises(ValueError, match="not among options"):
            normalize_massive(MASSIVE_ROW, options=["something_else"])


class TestIntentList:
    def test_dedupes_and_sorts(self):
        rows = [
            {"label": "alarm_set"},
            {"label": "alarm_query"},
            {"label": "alarm_set"},
        ]
        assert intents_from_rows(rows) == ["alarm_query", "alarm_set"]

    def test_is_not_typed_by_hand(self):
        # The 60 intents must come from the data. This guards against someone
        # later "fixing" the list by pasting a literal in: the function below
        # is the only intended source.
        rows = [{"label": f"intent_{i:02d}"} for i in range(60)]
        assert len(intents_from_rows(rows)) == 60


class TestPawsJoin:
    def test_joins_with_the_sep_marker(self):
        rec = normalize_paws_pair(
            {"id": 7, "sentence1": "A cat sat.", "sentence2": "A cat sat down.", "label": 1}
        )
        assert rec["text"] == "A cat sat. [SEP] A cat sat down."
        assert rec["options"] == ["paraphrase", "not_paraphrase"]

    def test_maps_the_integer_label_to_a_name(self):
        # PAWS ships 0/1 as ints; a raw int in a "label" field would break every
        # downstream comparison against the string options.
        assert normalize_paws_pair(
            {"id": 1, "sentence1": "a", "sentence2": "b", "label": 0}
        )["label"] == "not_paraphrase"
        assert normalize_paws_pair(
            {"id": 1, "sentence1": "a", "sentence2": "b", "label": 1}
        )["label"] == "paraphrase"

    def test_rejects_an_unknown_label(self):
        with pytest.raises(ValueError, match="unexpected PAWS label"):
            normalize_paws_pair(
                {"id": 1, "sentence1": "a", "sentence2": "b", "label": 2}
            )


class TestModuleImportsWithoutPyarrow:
    def test_no_top_level_pyarrow_import(self):
        """The loader must stay importable where pyarrow is not installed."""
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.modules['pyarrow'] = None;"
             "sys.path.insert(0, 'src'); sys.path.insert(0, '.');"
             "import evals.bespoke_loader as m; print('ok')"],
            capture_output=True, text=True, timeout=120,
        )
        assert "ok" in probe.stdout, probe.stderr


class TestDownloadedFilesAreConsistent:
    """Skipped until the corresponding download task has run."""

    @pytest.mark.parametrize("name,expect_rows,expect_options", [
        ("massive_intent.de.jsonl", 2974, 60),
        ("massive_intent.en.jsonl", 2974, 60),
        ("banking77.jsonl", 3080, 77),
        ("paws_adversarial.jsonl", 8000, 2),
    ])
    def test_file_shape(self, name, expect_rows, expect_options):
        path = Path(DATA_DIR) / name
        if not path.exists():
            pytest.skip(f"{name} not downloaded yet")
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == expect_rows, f"{name}: {len(lines)} != {expect_rows}"
        assert all(set(r) == {"id", "text", "label", "options"} for r in lines)
        assert all(len(r["options"]) == expect_options for r in lines)
        assert all(r["label"] in r["options"] for r in lines)

    @pytest.mark.parametrize("name", [
        "massive_intent.de.jsonl", "massive_intent.en.jsonl", "banking77.jsonl",
        "paws_adversarial.jsonl",
    ])
    def test_labels_are_names_not_integer_indices(self, name):
        """Every `label` must be a class NAME and a real member of `options`.

        This is the assertion the banking77 bug slipped through. The old check was
        `label in options`, which is vacuously true when both sides hold integers —
        a file full of `label: 11` / `options: [0..76]` passed it. Requiring the
        label to be non-numeric, to be a member of options AND (where the options
        are names) to look like a name makes the integer case impossible to miss.
        """
        path = Path(DATA_DIR) / name
        if not path.exists():
            pytest.skip(f"{name} not downloaded yet")
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for rec in lines[:500]:
            label, options = rec["label"], rec["options"]
            assert isinstance(label, str), f"{name}: label {label!r} is not a string"
            assert not label.isdigit(), (
                f"{name}: label {label!r} is an integer index, not a class name — "
                f"the loader skipped the metadata lookup"
            )
            assert label in options, f"{name}: {label!r} not among options"
            assert any(not o.isdigit() for o in options), (
                f"{name}: options are all integers, expected class names"
            )

    @pytest.mark.parametrize("lang", ["de", "en"])
    def test_massive_has_no_more_than_the_measured_train_test_overlap(self, lang):
        """MASSIVE's splits are not sentence-disjoint — bound the known overlap.

        Measured 2026-09-30: de 115/2974 (3.9%), en 21/2974 (0.7%), of which 8 resp.
        2 are label-INCONSISTENT (same sentence, different intent on the two sides).

        This is a property of the MTEB republication, not of the loader. It is
        asserted rather than merely documented because it caps achievable accuracy:
        a later run that reports 0% overlap would mean the data changed, and a run
        far above these bounds would mean the loader mixed up the splits. Both are
        worth failing on. See PROVENANCE.md §1.1.
        """
        train_p = Path(DATA_DIR) / f"massive_intent.{lang}.train.jsonl"
        test_p = Path(DATA_DIR) / f"massive_intent.{lang}.jsonl"
        if not (train_p.exists() and test_p.exists()):
            pytest.skip(f"massive {lang} not downloaded yet")

        train = [json.loads(l) for l in train_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        test = [json.loads(l) for l in test_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        train_label = {r["text"]: r["label"] for r in train}
        dup = [r for r in test if r["text"] in train_label]
        inconsistent = [r for r in dup if train_label[r["text"]] != r["label"]]

        rate = len(dup) / len(test)
        bound = 0.05 if lang == "de" else 0.02
        assert rate <= bound, (
            f"{lang}: {len(dup)}/{len(test)} = {rate:.1%} test texts also in train "
            f"(measured 3.9%/0.7% on 2026-09-30, bound {bound:.0%})"
        )
        assert len(inconsistent) == (8 if lang == "de" else 2), (
            f"{lang}: {len(inconsistent)} label-inconsistent duplicates, expected "
            f"{8 if lang == 'de' else 2} — the source data changed"
        )

    def test_anchors_are_drawn_from_train_with_bounded_leakage(self):
        """The k=3 anchors must come from `train` and overlap `test` only rarely.

        Measured: 7 of 180 German anchors, 0 English, 0 banking77. Reported, not
        forbidden — but if this ever jumps, the anchors are being drawn from the
        wrong file, which would make every few-shot number meaningless.
        """
        from evals.bespoke_anchors import assert_no_leakage, few_shot_anchors

        train_p = Path(DATA_DIR) / "massive_intent.de.train.jsonl"
        test_p = Path(DATA_DIR) / "massive_intent.de.jsonl"
        if not (train_p.exists() and test_p.exists()):
            pytest.skip("massive de not downloaded yet")

        train = [json.loads(l) for l in train_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        test = [json.loads(l) for l in test_p.read_text(encoding="utf-8").splitlines() if l.strip()]

        anchors = few_shot_anchors(train, k=3, seed=20260930)
        assert len(anchors) == 60
        assert all(len(v) == 3 for v in anchors.values())
        leaked = assert_no_leakage(anchors, {r["text"] for r in test})
        assert len(leaked) == 7, f"expected 7 known overlaps, got {len(leaked)}: {leaked[:5]}"
