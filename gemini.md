# Engineering Standards

**Read this before writing any code in this repo.** It applies to all engineers and AI agents — human or otherwise. These standards are not aspirational; they are enforced by the test suite and the runbook gates.

---

## Two non-negotiable principles

### 1. Documentation is paramount

Every change to behavior, architecture, or workflow must be reflected in the docs in the same commit:

- **`README.md`** — user-facing how-to: install, run, troubleshoot
- **`ARCHITECTURE.md`** — design rationale, phase status, data contracts, look-ahead controls
- **`AGENT_RUNBOOK.md`** — autonomous-execution playbook with hard pass/fail gates
- **`tradingview/README_TRADINGVIEW.md`** — TradingView load procedure for end users

Stale docs are worse than missing docs. If you change a CLI flag, an env var, an API endpoint, a file path, or a phase status — update every doc that references it. Search the tree before declaring done.

When in doubt, link rather than duplicate. Specific runbook details belong in `AGENT_RUNBOOK.md`; the README links to it.

### 2. High-speed, high-quality test coverage is paramount

The repo ships with **108 offline tests that run in ~3 seconds in fast mode**. Every change must keep that property intact:

- **No new code without tests.** Math primitives, signal generators, classifiers, API routes, and storage helpers all have direct unit coverage. Match that bar.
- **No PR with red tests.** Run `./run_tests.sh fast` before every commit. Run `./run_tests.sh` (full, ~2 min) before every PR.
- **Refactor freely to make code testable.** If logic is hard to test, the design is wrong — extract pure functions, inject dependencies, mock at boundaries. **Refactoring for testability is always in scope, never a "stretch goal."**
- **Tests must be deterministic and fast.** Seed all random generators. Mock all network and slow I/O. The whole suite (full mode) must stay under a few minutes.
- **Bugs found in production code while writing tests must be fixed**, not papered over with tests that match the broken behavior.

Coverage targets per module: signal generators, classifiers, math primitives, and storage helpers should be ≥ 80%. Live-only modules (ingestors, orchestrators, real Chronos load) are intentionally uncovered offline — that's documented in `ARCHITECTURE.md` and is fine.

---

## Code standards

### Python

- **Python 3.11+**, type hints on every function signature, no exceptions.
- **Loguru** for all logging — never `print()` in library code.
- **No look-ahead bias** in any feature, signal, or backtest calculation. Use `.shift(1)`, `center=False`, and `rolling().apply(raw=True)` defensively. The test suite enforces this for ATR, Bollinger, and Hurst — extend it for anything new.
- **No Python loops over DataFrame rows** in hot paths. Use vectorized pandas/numpy.
- **No magic numbers** — every threshold is either a function parameter or a CLI/input flag.
- **Single responsibility per file.** Ingestors don't compute features; feature engineers don't write files; classifiers don't call APIs.
- **Pure functions wherever possible** — they are trivially testable and reusable.

### Pine Script

| Code type | Version | Why |
|---|---|---|
| New strategy and library scripts (`tradingview/`) | **v6** (`//@version=6`) | Better strategy framework, namespaced functions (`ta.*`, `str.*`, `math.*`), type annotations |
| Legacy unmodified indicators (`ia_mean_reversion.pine`) | v5 (`//@version=5`) | Don't churn working code — modernize only when adding features |

- Use `ta.*` namespace functions (`ta.sma`, `ta.atr`, `ta.stdev`) — never deprecated globals.
- Use `str.tostring()` not `tostring()`. Use `str.format_time()` not legacy formatters.
- Type-annotate function signatures (`series float`, `simple int`).
- Group inputs logically with `group=`. Place all `input.*` calls at the top, immediately after the `library`/`indicator`/`strategy` declaration.
- Self-documenting names: `bb_pct_entry`, `atr_trail_multiplier` — not `bbpe`, `atm`.
- Comments where they add genuine value (algorithm rationale, non-obvious math). Not narration of `// set foo to 5`.

### Commits and PRs

- One logical change per commit. The commit message explains *why*, not *what* (the diff shows what).
- PR descriptions follow the template in existing PRs: Summary, Test plan checklist, honest list of what's not covered.
- Never commit `.env`, real credentials, large data files, or `__pycache__/`. The `.gitignore` is the safety net, not an excuse to be careless.

---

## Workflow expectations

When picking up a task:

1. **Read the docs first**: `README.md` → `ARCHITECTURE.md` → `AGENT_RUNBOOK.md` → relevant phase code.
2. **Run `./run_tests.sh fast`** to confirm a clean baseline before changing anything.
3. **Write the test first** when adding new behavior. Watch it fail. Make it pass. Refactor.
4. **Update all relevant docs in the same commit** as the code change.
5. **Run `./run_tests.sh` (full)** before opening a PR.

When something is hard to test:

1. Don't skip the test — refactor until it's easy to test.
2. If a dependency is the obstacle (network, real model, file system), mock it via the boundary it crosses.
3. If you genuinely cannot test a behavior offline, document it explicitly in `ARCHITECTURE.md` under "Testing strategy" with the reason.

---

## Pointer for AI agents

If you are an AI agent operating in this repo, read [`AGENT_RUNBOOK.md`](AGENT_RUNBOOK.md) for the full execution playbook with pass/fail gates. The runbook starts with a Phase 0 self-test that runs this same suite — if it fails, stop and report rather than proceeding into the production phases.
