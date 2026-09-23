"""Bilingual (EN/DE) benchmark: klix variants + baselines, accuracy split by language.

Self-contained and reproducible. One run prints a markdown-ready table plus a
per-language breakdown. Runs klix variants and the sklearn/embedding baselines
on IDENTICAL anchors, so the comparison is honest.

Run:
    uv run python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'src'); from evals import benchmark_bilingual; benchmark_bilingual.main()"
"""

import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from klix import DecisionEngine, Choice  # noqa: E402

# ---------------------------------------------------------------------------
# Anchors (reference texts). Mixed EN/DE, real support-ticket phrasing.
# ---------------------------------------------------------------------------
OPTIONS = {
    "billing": [
        "refund for my order is missing",
        "die rechnung wurde doppelt abgebucht",
        "the invoice amount is wrong",
        "gutschrift fehlt auf dem konto",
    ],
    "technical": [
        "the server keeps crashing",
        "vpn verbindung bricht staendig ab",
        "software update fails with an error",
        "der laptop startet nicht mehr",
    ],
    "facility": [
        "the elevator is stuck between floors",
        "die heizung im buero ist kaputt",
        "water leaks from the ceiling",
        "parkplatz licht ist ausgefallen",
    ],
    "hr": [
        "how do i apply for parental leave",
        "gehaltsabrechnung stimmt nicht",
        "i need a copy of my employment contract",
        "urlaub beantragen fuer august",
    ],
    "security": [
        "suspicious login from another country",
        "ransomware hat den fileserver verschluesselt",
        "someone tried to hack our admin account",
        "phishing mail an alle mitarbeiter",
    ],
}

# ---------------------------------------------------------------------------
# Test cases: parallel EN/DE, SAME meaning, none are verbatim anchors.
# ---------------------------------------------------------------------------
TEST_EN = [
    ("you charged my card twice this month", "billing"),
    ("where is the refund for the returned item", "billing"),
    ("my monitor stays black after boot", "technical"),
    ("the wifi never connects on my laptop", "technical"),
    ("the coffee machine in the kitchen leaks", "facility"),
    ("someone broke the glass door handle", "facility"),
    ("i want to extend my parental leave", "hr"),
    ("my payslip shows zero hours", "hr"),
    ("someone logged into my email from abroad", "security"),
    ("all our files are encrypted, ransom demanded", "security"),
]

TEST_DE = [
    ("meine kreditkarte wurde doppelt belastet", "billing"),
    ("wo bleibt die erstattung fuer die stornierte bestellung", "billing"),
    ("der bildschirm bleibt nach dem start schwarz", "technical"),
    ("wlan verbindet sich nicht auf meinem laptop", "technical"),
    ("die kaffeemaschine in der kueche tropft", "facility"),
    ("jemand hat den tuerknauf aus glas zerbrochen", "facility"),
    ("ich moechte meine elternzeit verlaengern", "hr"),
    ("auf meiner abrechnung stehen null stunden", "hr"),
    ("unbekannte haben sich in mein postfach eingeloggt", "security"),
    ("alle unsere dateien sind verschluesselt, loesegeld gefordert", "security"),
]

# OOD / reject cases (should NOT be force-routed; best effort = reject/None)
OOD_EN = ["happy birthday to the whole team", "nice weather this weekend"]
OOD_DE = ["alles gute zum geburtstag euch allen", "schoenes wetter am wochenende"]


def _flatten(options):
    texts, labels = [], []
    for label, examples in options.items():
        for ex in examples:
            texts.append(ex)
            labels.append(label)
    return texts, labels


# --- Baselines -------------------------------------------------------------
def run_tfidf_lr(options, tests, _embed):
    texts, labels = _flatten(options)
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
    vec.fit(texts)
    clf = LogisticRegression(C=10.0, max_iter=1000, class_weight="balanced").fit(vec.transform(texts), labels)
    ok = 0
    for text, expected in tests:
        if clf.predict(vec.transform([text]))[0] == expected:
            ok += 1
    return ok, len(tests)


def run_embed_knn(options, tests, embed):
    texts, labels = _flatten(options)
    ref = np.array(list(embed.embed(texts)))
    ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    ok = 0
    for text, expected in tests:
        q = np.array(list(embed.embed([text]))[0])
        q = q / (np.linalg.norm(q) or 1.0)
        if labels[int(np.argmax(ref @ q))] == expected:
            ok += 1
    return ok, len(tests)


def run_klix(options, tests, classifier, **kw):
    eng = DecisionEngine()
    eng.add_head(Choice(name="c", options=options, classifier=classifier, **kw))
    eng.compile()
    ok = 0
    for text, expected in tests:
        if eng.decide(text).c == expected:
            ok += 1
    return ok, len(tests)


def main():
    embed = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Methods: (name, kind, kwargs)
    methods = [
        ("TF-IDF + LogReg", "tfidf", {}),
        ("Embed-KNN (dense)", "knn", {}),
        ("klix nearest", "klix", {"classifier": "nearest"}),
        ("klix nearest + topk2", "klix", {"classifier": "nearest", "label_aggregation": "topk", "label_topk": 2}),
        ("klix linear", "klix", {"classifier": "linear"}),
    ]

    # Headline table: EN+DE combined
    combined = TEST_EN + TEST_DE
    print("# Bilingual benchmark (support-ticket routing, 5 classes)")
    print()
    print("| Method | EN | DE | Combined |")
    print("|---|---|---|---|")
    rows = {}
    for name, kind, kw in methods:
        en_ok, en_n = _dispatch(kind, options=OPTIONS, tests=TEST_EN, embed=embed, kw=kw)
        de_ok, de_n = _dispatch(kind, options=OPTIONS, tests=TEST_DE, embed=embed, kw=kw)
        comb = en_ok + de_ok
        rows[name] = (en_ok, de_ok, comb)
        print(f"| {name} | {en_ok}/{en_n} | {de_ok}/{de_n} | {comb}/{en_n + de_n} |")

    # Per-language detail
    print()
    print("## Per-language accuracy")
    print()
    print("| Method | EN % | DE % |")
    print("|---|---|---|")
    for name, (en_ok, de_ok, _) in rows.items():
        print(f"| {name} | {en_ok/len(TEST_EN):.0%} | {de_ok/len(TEST_DE):.0%} |")

    # Latency
    print()
    print("## Latency per decision (CPU, includes embedding forward pass where applicable)")
    print()
    print("| Method | latency |")
    print("|---|---|")
    for name, kind, kw in methods:
        if kind == "klix":
            eng = DecisionEngine()
            eng.add_head(Choice(name="c", options=OPTIONS, **kw))
            eng.compile()
            # warmup
            eng.decide("warmup query")
            t = []
            for _ in range(20):
                t0 = time.perf_counter()
                eng.decide("the server keeps crashing")
                t.append((time.perf_counter() - t0) * 1000)
            lat = sorted(t)[len(t) // 2]
        elif kind == "knn":
            # Precompute reference embeddings ONCE (no per-query refit).
            texts, labels = _flatten(OPTIONS)
            ref = np.array(list(embed.embed(texts)))
            ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
            for _ in range(3):
                q = np.array(list(embed.embed(["warmup"]))[0])
            t = []
            for _ in range(50):
                t0 = time.perf_counter()
                q = np.array(list(embed.embed(["the server keeps crashing"]))[0])
                q = q / (np.linalg.norm(q) or 1.0)
                ref @ q
                t.append((time.perf_counter() - t0) * 1000)
            lat = sorted(t)[len(t) // 2]
        else:
            vec = TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b[\w-]+\b", lowercase=True)
            texts, labels = _flatten(OPTIONS)
            vec.fit(texts)
            clf = LogisticRegression(max_iter=1000).fit(vec.transform(texts), labels)
            t = []
            for _ in range(200):
                t0 = time.perf_counter()
                clf.predict(vec.transform(["the server keeps crashing"]))
                t.append((time.perf_counter() - t0) * 1000)
            lat = sorted(t)[len(t) // 2]
        print(f"| {name} | {lat:.1f} ms |")


def _dispatch(kind, options, tests, embed, kw):
    if kind == "tfidf":
        return run_tfidf_lr(options, tests, embed)
    if kind == "knn":
        return run_embed_knn(options, tests, embed)
    return run_klix(options, tests, kw.get("classifier", "nearest"), **{k: v for k, v in kw.items() if k != "classifier"})


if __name__ == "__main__":
    main()
