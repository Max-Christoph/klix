"""Loader and split validation for CLINC150 (oos-eval).

Dataset provenance:
    Larson et al., EMNLP 2019: 'An Evaluation Dataset for Intent Classification
    and Out-of-Scope Prediction'.
    Source: https://github.com/clinc/oos-eval
    File: evals/data/clinc/data_full.json
"""

import json
from dataclasses import dataclass
from pathlib import Path

CLINC_DIR = Path(__file__).parent / "data" / "clinc"
DATA_FULL_PATH = CLINC_DIR / "data_full.json"


@dataclass(frozen=True)
class ClincSplit:
    train: list[tuple[str, str]]       # (text, intent)
    val: list[tuple[str, str]]         # (text, intent)
    test: list[tuple[str, str]]        # (text, intent)
    oos_train: list[str]               # out-of-scope train texts
    oos_val: list[str]                 # out-of-scope val texts
    oos_test: list[str]                # out-of-scope test texts (1,000 cases)
    classes: list[str]                 # 150 unique in-scope intent names


def load_clinc_dataset(path: Path = DATA_FULL_PATH) -> ClincSplit:
    """Loads and validates the CLINC150 dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"CLINC dataset not found at {path}. "
            f"Please run the download script or have Hermes fetch data_full.json into {CLINC_DIR}."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    train = [(item[0], item[1]) for item in raw.get("train", [])]
    val = [(item[0], item[1]) for item in raw.get("val", [])]
    test = [(item[0], item[1]) for item in raw.get("test", [])]

    oos_train = [item[0] for item in raw.get("oos_train", [])]
    oos_val = [item[0] for item in raw.get("oos_val", [])]
    oos_test = [item[0] for item in raw.get("oos_test", [])]

    # Collect unique classes
    classes = sorted(list({label for _, label in train}))

    # Sanity checks
    assert len(classes) == 150, f"Expected 150 in-scope classes, got {len(classes)}"
    assert len(oos_test) == 1000, f"Expected 1,000 oos_test cases, got {len(oos_test)}"
    assert len(test) == 4500, f"Expected 4,500 test cases, got {len(test)}"

    return ClincSplit(
        train=train,
        val=val,
        test=test,
        oos_train=oos_train,
        oos_val=oos_val,
        oos_test=oos_test,
        classes=classes,
    )


def extract_clinc_anchors(
    split: ClincSplit,
    k_per_class: int = 3,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Extracts k_per_class anchor sentences deterministically per class from train."""
    import random

    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {c: [] for c in split.classes}
    for text, label in split.train:
        by_class[label].append(text)

    anchors: dict[str, list[str]] = {}
    for c in split.classes:
        candidates = list(by_class[c])
        rng.shuffle(candidates)
        anchors[c] = candidates[:k_per_class]

    # Verify no test leakage
    test_texts = {text for text, _ in split.test} | set(split.oos_test)
    for c, anchor_list in anchors.items():
        for a in anchor_list:
            assert a not in test_texts, f"Anchor leak detected in class {c}: {a}"

    return anchors
