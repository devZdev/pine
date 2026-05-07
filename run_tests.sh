#!/usr/bin/env bash
# run_tests.sh — offline test runner for the Glass Box CSP stack.
#
# Usage:
#   ./run_tests.sh             # all tests, summary output
#   ./run_tests.sh -v          # verbose
#   ./run_tests.sh fast        # skip slow tests (Hurst on long series, smoke test)
#   ./run_tests.sh phase1      # only Phase 1 tests
#   ./run_tests.sh phase2      # only Phase 2 tests
#   ./run_tests.sh phase3      # only Phase 3 tests
#   ./run_tests.sh phase4      # only Phase 4 tests
#   ./run_tests.sh smoke       # only the integration smoke test
#   ./run_tests.sh coverage    # generate coverage report (HTML in htmlcov/)

set +u  # do not fail on unbound array expansion across bash versions

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prefer the project venv if present
if [[ -x "$REPO_ROOT/.venv/bin/pytest" ]]; then
    PYTEST="$REPO_ROOT/.venv/bin/pytest"
else
    PYTEST="$(command -v pytest || true)"
    if [[ -z "$PYTEST" ]]; then
        echo "pytest not found.  Install with: pip install -r requirements.txt" >&2
        exit 2
    fi
fi

mode="${1:-all}"
shift || true

case "$mode" in
    -v|--verbose|all)
        if [[ "$mode" == "-v" || "$mode" == "--verbose" ]]; then
            ARGS=(-v ${@:+"$@"})
        else
            ARGS=(${@:+"$@"})
        fi
        ;;
    fast)
        ARGS=(-m "not slow" -x "$@")
        ;;
    phase1)
        ARGS=(-m "phase1" -v "$@")
        ;;
    phase2)
        ARGS=(-m "phase2" -v "$@")
        ;;
    phase3)
        ARGS=(-m "phase3" -v "$@")
        ;;
    phase4)
        ARGS=(-m "phase4" -v "$@")
        ;;
    smoke)
        ARGS=(-m "smoke" -v "$@")
        ;;
    coverage)
        ARGS=(--cov=pipeline --cov=backtest --cov=regime --cov-report=term --cov-report=html "$@")
        ;;
    -h|--help|help)
        sed -n '2,15p' "${BASH_SOURCE[0]}"
        exit 0
        ;;
    *)
        # Treat as raw pytest args
        ARGS=("$mode" "$@")
        ;;
esac

cd "$REPO_ROOT"
echo "[run_tests] mode=$mode pytest=$PYTEST"
echo "[run_tests] args: ${ARGS[*]}"
"$PYTEST" "${ARGS[@]}"
status=$?

echo
if [[ $status -eq 0 ]]; then
    echo "[run_tests] ✅ all selected tests passed"
else
    echo "[run_tests] ❌ failures detected (exit $status)"
fi
exit $status
