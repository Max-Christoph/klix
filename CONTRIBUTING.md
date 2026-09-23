# Contributing to klix

Thanks for your interest in improving klix! This document explains how to set
up, where to discuss changes first, and what a good pull request looks like.

## Development setup

```bash
git clone https://github.com/Max-Christoph/klix.git
cd klix
uv sync          # install dependencies (Python 3.10–3.12)
uv run pytest    # run the test suite (must be fully offline)
```

## Ground rules

1. **Anchors and labeled eval cases are frozen.** The 5 labeled sets in
   `evals/` (60 cases) are the shared measurement basis. Framework changes
   must never tune anchors or test cases to look better — measure on them
   unchanged, and report honest numbers (including negative results; several
   shipped features exist as documented negative results).
2. **Backward compatibility by default.** New behavior is opt-in via new
   parameters; defaults never change silently. If a default must change,
   discuss it in an issue first and document the migration.
3. **CPU latency is a hard constraint.** Inference is ~10 ms per decide()
   (encoding dominates; head math must stay sub-millisecond). Cross-encoder
   models are out of scope for the core heads.
4. **Honest statistics.** Small-sample claims need caveats; calibrate()
   warns below n=20; benchmark tables carry sample sizes. No point estimates
   without an n.
5. **Everything English** (code, docs, comments) for international users.

## Workflow

1. Open an issue first for anything that changes behavior or public API —
   schema semantics are a design surface, not just code.
2. Fork / branch (`feat/...` or `fix/...`).
3. Add tests for your change. Run `uv run pytest` — CI runs pytest on Python
   3.10–3.12 and must stay green.
4. If you touched accuracy-relevant code, run the eval harness and include the
   numbers in your PR description:

   ```bash
   uv run python evals/linear_sweep.py
   uv run python evals/bm25_sweep.py
   uv run python evals/hard_negative_e2e.py
   ```

5. Bump the version and update `CHANGELOG.md` (Keep-a-Changelog style).
6. Open the PR. Releases happen via tag push (`git tag vX.Y.Z && git push
   origin vX.Y.Z`) with PyPI Trusted Publishing — maintainers only.

## Reporting bugs

Open an issue with: klix version, Python version, a minimal reproducible
schema (anchors + query + unexpected output), and what you expected instead.
Security-sensitive behavior (rules, reject poles) deserves an explicit test —
include one if you can.

## License

By contributing you agree that your contributions are licensed under the MIT
license of this project.