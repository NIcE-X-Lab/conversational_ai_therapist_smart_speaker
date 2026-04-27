#!/bin/bash
# Run all smoke tests in sequence. Non-zero exit on any failure.
#
# Usage:
#   dev/smoke_tests/run_all.sh
#
# These tests exercise the full pipeline WITHOUT Gemma or hardware, so
# they're safe to run on a dev machine. They verify:
#   1. Static invariants (config, flags, wordings, prompts)
#   2. RL math (Q-table init, choose_action, get_env_feedback, Q update)
#   3. Questioner pipeline (Yes/No/Stop paths, RV, G4 regression)
#   4. CBT flow (Stage 0 wording, recap, 3 stages, closing, G7 off)
#   5. End-to-end demo replay (greeting → DLA → RV → CBT → warm close)
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
    PY="python3"
fi

echo "============================================"
echo "  CaiTI smoke-test suite"
echo "  python: $($PY --version 2>&1)"
echo "  cwd:    $PROJECT_ROOT"
echo "============================================"
echo ""

tests=(
    "dev/smoke_tests/test_1_static.py"
    "dev/smoke_tests/test_2_rl_math.py"
    "dev/smoke_tests/test_3_questioner.py"
    "dev/smoke_tests/test_4_cbt.py"
    "dev/smoke_tests/test_5_e2e_replay.py"
)

passed=0
failed=0
for t in "${tests[@]}"; do
    echo ">>> $t"
    if timeout 120 "$PY" "$t" > /tmp/caiti_smoke_out 2>&1; then
        ok=$(grep -c "\[OK\]" /tmp/caiti_smoke_out || true)
        fl=$(grep -c "\[FAIL\]" /tmp/caiti_smoke_out || true)
        echo "    PASS  (ok=$ok fail=$fl)"
        passed=$((passed + 1))
    else
        echo "    FAIL"
        grep -E "\[FAIL\]|FAILED|Traceback" /tmp/caiti_smoke_out | head -10
        failed=$((failed + 1))
    fi
done

echo ""
echo "============================================"
echo "  Smoke-test summary: $passed passed, $failed failed"
echo "============================================"
exit $failed
