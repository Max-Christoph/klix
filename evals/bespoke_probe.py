"""Feasibility probe for the Ollama comparison models — run BEFORE any full sweep.

WHY THIS EXISTS
---------------
The target machine has **no NVIDIA GPU** (Intel iGPU, 32 GB RAM, 12 cores). A 4B or
9B model answering ~9 000 classification queries on CPU is not obviously possible,
and the cost of finding out the hard way is a wasted overnight run. So this script
spends 20 queries per model to produce the one number that decides: the projected
wall-clock time of a full sweep.

DECISION RULE (agreed, §7.5)
----------------------------
    projected > 30 min for the full split  ->  subsample to 500 cases and use
    `qwen3.5:2b` / `tev1:0.8b` as the local baseline.

The projection is printed, never acted on silently: a run that subsamples says so
in its own output and in the result file.

WHAT IT MEASURES
----------------
  * p50 / p95 latency per query (generation AND total, separately)
  * parse rate — how often the label could be recovered from the answer. A model
    that answers in prose is not wrong, it is unusable, and those are different
    problems: `unparseable` is reported separately and never folded into accuracy.
  * projected full-sweep time for the given case count

WHEN THE PROJECTION IS WRONG
----------------------------
Single-sample projections on this machine proved unreliable, and the reason is not
established — so this records the measurements rather than a theory:

  idle (first probe, 20 cases):        p50 9.0 s/case, p95 10.4 s/case
  steady state, test suite alongside:  4.8 s/case (125 -> 150 cases in 120 s)
  first hour, averaged:                28 s/case (125 cases in ~59 min)

Three rates spanning a factor of six for the same model, dataset and prompt. The
slow first hour is the part that matters operationally: dividing elapsed time by
completed cases early in a run overestimates the remaining cost badly, and an idle
probe does not predict the loaded rate either. The only number worth scheduling on
is the steady-state rate measured over a short window *while the real workload
runs* — which is what pointed at ~28 remaining minutes here, against a "4.7 hours"
extrapolation from the first hour.

It does NOT measure accuracy. 20 cases cannot, and a 20-case accuracy has misled
this project before.

Run (server must be up):
    python -m evals.bespoke_probe --model qwen3.5:2b --n 20
    python -m evals.bespoke_probe --model tev1:4b --n 20
    python -m evals.bespoke_probe --all --n 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.bespoke_loader import DATA_DIR  # noqa: E402

OLLAMA = "http://localhost:11434"
FULL_SPLIT_CASES = 2974          # MASSIVE de/en test size — the default sweep target
SUBSAMPLE_THRESHOLD_MIN = 30.0   # §7.5
LOCAL_BASELINES = ["qwen3.5:2b", "tev1:0.8b"]

PROMPT = """Classify the user request into exactly one of these intents.

Intents:
{options}

Request: {text}

Answer with the intent name only, nothing else."""


def server_ready(timeout: int = 10) -> tuple[bool, list[str]]:
    """Health check instead of a blind sleep. Returns (up, model names)."""
    try:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return True, [m["name"] for m in data.get("models", [])]
    except (urllib.error.URLError, TimeoutError, OSError):
        return False, []


def model_capabilities(model: str) -> dict:
    """Ask Ollama what the model can do (capabilities, context length)."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/show", data=json.dumps({"model": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        return {
            "capabilities": data.get("capabilities", []),
            "context_length": (data.get("model_info", {}).get("general.context_length")
                               or data.get("model_info", {}).get("llama.context_length")),
        }
    except Exception:  # noqa: BLE001 — capabilities are best-effort metadata
        return {"capabilities": [], "context_length": None}


def parse_label(answer: str, options: list[str]) -> str | None:
    """Recover the chosen label, or None if the answer is unusable.

    Deliberately strict: an exact (case/whitespace-insensitive) match against an
    option, or the option appearing as a whole token. Anything else is
    `unparseable` — guessing which of 60 labels a sentence meant would make the
    parse rate unfalsifiable.
    """
    a = answer.strip().strip(".,;:!?\"'`*")
    low = a.lower().replace("-", "_").replace(" ", "_")
    for opt in options:
        if low == opt.lower():
            return opt
    for opt in options:
        o = opt.lower()
        if low.startswith(o) or f" {o} " in f" {low} ":
            return opt
    return None


def query(model: str, prompt: str, timeout: int = 600,
          think: bool | None = False, num_predict: int = 64,
          num_ctx: int = 4096) -> dict:
    """One generation. Returns the full Ollama response (caller reads timings).

    `think=False` is passed as a **top-level** request field, not inside
    `options`. This matters more than it looks: `qwen3.5:2b` advertises the
    `thinking` capability, and without the flag it spends the entire `num_predict`
    budget inside the thinking channel and returns an EMPTY `response` at ~9-20 s
    per call. The first probe run measured exactly that (parse_rate 0%, p95 19.6 s)
    and the conclusion "the model cannot handle this" would have been wrong — the
    model answers in ~0.7 s with the flag set. The capability is detected and the
    setting recorded in the result so the number cannot be misread later.
    """
    body: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if think is not None:
        body["think"] = think
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    data["_wall_ms"] = (time.perf_counter() - t0) * 1000
    return data


def load_cases(n: int) -> list[dict]:
    path = Path(DATA_DIR) / "massive_intent.de.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} missing — run: uv run --with pyarrow python -m evals.bespoke_loader "
            f"--dataset massive --lang de"
        )
    # Stride sampling instead of the first n: the test split is ordered by class,
    # so a head slice would measure one intent 20 times.
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    stride = max(1, len(lines) // n)
    return lines[::stride][:n]


def probe(model: str, n: int, full_cases: int) -> dict:
    up, models = server_ready()
    if not up:
        raise SystemExit(f"Ollama not reachable at {OLLAMA}. Start it, then re-run.")
    have = {m.split(":")[0] for m in models}
    if model.split(":")[0] not in have and model not in models:
        print(f"  NOTE: {model!r} is not in the local model list {models}")
        print(f"        pull it first:  ollama pull {model}")

    caps = model_capabilities(model)
    uses_thinking = "thinking" in caps["capabilities"]
    print(f"\n=== {model} — {len(cases := load_cases(n))} cases ===")
    print(f"  capabilities={caps['capabilities']} context={caps['context_length']}")
    if uses_thinking:
        print("  thinking-capable -> sending think=false (see query() docstring)")

    lat, parsed, empty = [], [], 0
    for i, case in enumerate(cases, 1):
        prompt = PROMPT.format(options="\n".join(case["options"]), text=case["text"])
        try:
            resp = query(model, prompt)
        except Exception as exc:  # noqa: BLE001 — a probe reports, it does not raise
            print(f"  [{i}/{len(cases)}] FAILED: {type(exc).__name__}: {exc}")
            continue
        answer = resp.get("response", "")
        if not answer.strip():
            empty += 1
        got = parse_label(answer, case["options"])
        lat.append(resp["_wall_ms"])
        parsed.append(got is not None)
        print(f"  [{i:2}/{len(cases)}] {resp['_wall_ms']:7.0f} ms  parsed={got is not None!s:5} "
              f"want={case['label']:22} answer={answer.strip()[:40]!r}")

    if not lat:
        return {"model": model, "n": 0, "error": "all queries failed"}

    lat_sorted = sorted(lat)
    p50 = statistics.median(lat)
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]
    parse_rate = sum(parsed) / len(parsed)
    projected_min = p95 * full_cases / 1000 / 60

    verdict = ("SUBSAMPLE to 500 cases, use a local baseline"
               if projected_min > SUBSAMPLE_THRESHOLD_MIN
               else "full split is feasible")
    print(f"  p50={p50:.0f} ms  p95={p95:.0f} ms  parse_rate={parse_rate:.0%}  "
          f"empty={empty}/{len(lat)}")
    print(f"  projected full sweep ({full_cases} cases) = {projected_min:.1f} min -> {verdict}")

    return {
        "model": model,
        "n": len(lat),
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "parse_rate": round(parse_rate, 4),
        "empty_responses": empty,
        "projected_full_sweep_min": round(projected_min, 1),
        "verdict": verdict,
        "full_cases_assumed": full_cases,
        "capabilities": caps["capabilities"],
        "thinking_disabled": uses_thinking,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--full-cases", type=int, default=FULL_SPLIT_CASES)
    args = ap.parse_args()

    up, models = server_ready()
    print(f"ollama reachable: {up}; local models: {models}")
    if not up:
        return 1

    targets = ([args.model] if args.model else
               (LOCAL_BASELINES + ["tev1:4b", "nimble:9b"] if args.all else []))
    if not targets:
        ap.error("pass --model NAME or --all")

    results = [probe(m, args.n, args.full_cases) for m in targets]
    out = Path("evals/bespoke_probe_result.json")
    out.write_text(json.dumps({"probe_n": args.n, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
