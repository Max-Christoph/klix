"""Export the benchmark datasets (options + labeled choice test cases) as JSON,
so the Laya benchmark (separate venv) reads the exact same cases."""

import sys
import json

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from evals.eval_domains import (  # noqa: E402
    IMG_OPTIONS, TASK_OPTIONS, SHOP_OPTIONS, IMG_CASES, TASK_CASES, SHOP_CASES,
)
from evals.variant_sweep import HR_OPTIONS, FIN_OPTIONS, HR_TESTS, FIN_TESTS  # noqa: E402
from evals.eval_guardrail import OPTIONS as GUARD_OPTIONS, TESTS as GUARD_TESTS  # noqa: E402

datasets = {
    "HR": {"options": HR_OPTIONS, "tests": [[q, e] for q, e in HR_TESTS]},
    "FIN": {"options": FIN_OPTIONS, "tests": [[q, e] for q, e in FIN_TESTS]},
    "IMAGE": {"options": IMG_OPTIONS, "tests": [[q, e] for q, e in IMG_CASES if e]},
    "TASK": {"options": TASK_OPTIONS, "tests": [[q, e] for q, e in TASK_CASES if e]},
    "SHOP": {"options": SHOP_OPTIONS, "tests": [[q, e] for q, e in SHOP_CASES if e]},
    "GUARD": {"options": GUARD_OPTIONS, "tests": [[q, e] for q, e in GUARD_TESTS if e]},
}

out = r"C:\Users\z0041ufz\Desktop\laya_bench\datasets.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(datasets, f, ensure_ascii=False, indent=1)
print("exported to", out)
print("datasets:", list(datasets.keys()))
for k, v in datasets.items():
    print(f"  {k}: {len(v['options'])} Klassen, {len(v['tests'])} Testfälle")
