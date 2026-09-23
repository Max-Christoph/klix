"""Experiment: do char-n-gram features help the linear probe catch typos?

The word-level TF-IDF and the dense embedding both miss near-miss spellings
(swich/switch, stoerung/störung, abrechung/abrechnung). Char-n-grams capture
the shared substring structure. This measures whether adding them to the probe's
feature vector actually improves routing on the typo-heavy cases.
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from scipy.sparse import hstack  # noqa: E402

from klix import DecisionEngine, Choice  # noqa: E402

OPTIONS = {
    "support": ["the printer does not print", "vpn connection keeps dropping", "software update fails"],
    "billing": ["refund for my order is missing", "the invoice amount is wrong", "card was charged twice"],
    "facility": ["the elevator is stuck", "heating in the office is broken", "water leaks from the ceiling"],
}

# Typo-heavy test cases (deliberately misspelled / umlaut-stripped)
TESTS = [
    ("the printr does not print", "support"),          # "printr"
    ("vpn conecton keeps droppng", "support"),         # "conecton", "droppng"
    ("sofware upate fails again", "support"),           # "sofware", "upate"
    ("refund for my oder is missng", "billing"),       # "oder", "missng"
    ("the invoic amout is wrng", "billing"),           # "invoic", "amout", "wrng"
    ("card was chared twice", "billing"),              # "chared"
    ("the elevatr is stck", "facility"),               # "elevatr", "stck"
    ("heting in the ofice is brokn", "facility"),      # "heting", "ofice", "brokn"
    ("water leks from the ceilin", "facility"),        # "leks", "ceilin"
    ("die heizung ist kaput", "facility"),             # German, correct spelling
]


def build_features(texts, char_ngrams, min_df=1):
    """Dense (not used here) is simulated via word TF-IDF + optional char n-grams."""
    vecs = []
    word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
    word_vec.fit(texts)
    vecs.append(word_vec.transform(texts))
    if char_ngrams:
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        char_vec.fit(texts)
        vecs.append(char_vec.transform(texts))
    return hstack(vecs), word_vec, (char_vec if char_ngrams else None)


def measure(char_ngrams: bool) -> tuple[int, int]:
    # Build reference texts from options
    flat, labels = [], []
    for lab, exs in OPTIONS.items():
        for ex in exs:
            flat.append(ex)
            labels.append(lab)

    # Fit vectorizers on reference texts only
    all_texts = flat  # note: for a fair test, vocab from anchors only
    if char_ngrams:
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
        char_vec.fit(all_texts)
        word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
        word_vec.fit(all_texts)
        X_ref = hstack([word_vec.transform(all_texts), char_vec.transform(all_texts)])
    else:
        word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
        word_vec.fit(all_texts)
        X_ref = word_vec.transform(all_texts)

    clf = LogisticRegression(C=10.0, max_iter=1000)
    clf.fit(X_ref, labels)

    ok = 0
    for text, expected in TESTS:
        if char_ngrams:
            q = hstack([word_vec.transform([text]), char_vec.transform([text])])
        else:
            q = word_vec.transform([text])
        pred = clf.predict(q)[0]
        if pred == expected:
            ok += 1
    return ok, len(TESTS)


if __name__ == "__main__":
    # Baseline: dense-only probe (the actual engine, nearest classifier)
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options=OPTIONS))
    eng.compile()
    nearest_ok = sum(1 for t, e in TESTS if eng.decide(t).c == e)
    print(f"Engine nearest (status quo):          {nearest_ok}/{len(TESTS)}")

    # Engine linear probe (dense only)
    eng2 = DecisionEngine()
    eng2.add_head(Choice(name="c", options=OPTIONS, classifier="linear"))
    eng2.compile()
    lin_ok = sum(1 for t, e in TESTS if eng2.decide(t).c == e)
    print(f"Engine linear (dense only):           {lin_ok}/{len(TESTS)}")

    # Standalone: word TF-IDF only (sparse logistic)
    w_ok, n = measure(char_ngrams=False)
    print(f"Sparse logistic, word-only:           {w_ok}/{n}")

    # Standalone: word + char n-grams
    c_ok, n = measure(char_ngrams=True)
    print(f"Sparse logistic, word + char n-grams: {c_ok}/{n}")
