#!/bin/bash
# Phase C: Prepare the Jetson for a new participant.
#
# Policy (per operator):
#   - therapist.db is PRESERVED (no truncation).
#   - data/sessions/, data/clinical/, data/logs/ are PRESERVED.
#   - The per-subject Q-table cache at data/q_tables/item_qtable_{SUBJECT_ID}.csv
#     is RESET so the next subject starts from the therapist-authored empirical
#     priors in config.yaml → rl.item_importance (not from an inherited
#     longitudinal state belonging to a prior participant).
#
# Usage:
#   ./scripts/new_participant_init.sh [NEW_SUBJECT_ID]
#
# If NEW_SUBJECT_ID is omitted, the current SUBJECT_ID from .env is used
# (i.e., just reset the existing subject's Q-table without changing identity).
#
# After running this script:
#   1. The next `main.py` boot for that subject rebuilds the Q-table from
#      ITEM_IMPORTANCE on first `setup()` call.
#   2. Historical DB rows remain intact for longitudinal analysis.

set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v realpath >/dev/null 2>&1; then
  SCRIPT_PATH="$(realpath "$SCRIPT_PATH")"
elif command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
fi
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$PROJECT_ROOT"

NEW_ID="${1:-}"

if [[ ! -f config.yaml ]]; then
  echo "[ERROR] config.yaml not found in $PROJECT_ROOT" >&2
  exit 1
fi

CURRENT_ID=$(awk '/^app:/{flag=1;next} flag && /subject_id:/{gsub(/"/,"");print $2; exit}' config.yaml | tr -d '[:space:]')
if [[ -z "$CURRENT_ID" ]]; then
  echo "[ERROR] Could not parse app.subject_id from config.yaml" >&2
  exit 1
fi
echo "[Phase C] Current subject_id: $CURRENT_ID"

if [[ -n "$NEW_ID" && "$NEW_ID" != "$CURRENT_ID" ]]; then
  echo "[Phase C] Switching to new subject_id: $NEW_ID"
  # Rewrite only the subject_id line under the app: block.
  python3 - <<PY
import re, pathlib
p = pathlib.Path("config.yaml")
text = p.read_text()
# Replace only inside the app: block, on the first subject_id line.
new_text, count = re.subn(
    r'(app:\s*\n(?:.*\n)*?\s*subject_id:\s*)"[^"]*"',
    r'\1"${NEW_ID}"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("[ERROR] Could not locate app.subject_id for rewrite")
p.write_text(new_text)
print("[OK] config.yaml updated.")
PY
  ACTIVE_ID="$NEW_ID"
else
  ACTIVE_ID="$CURRENT_ID"
fi

QFILE="data/q_tables/item_qtable_${ACTIVE_ID}.csv"
if [[ -f "$QFILE" ]]; then
  BACKUP="${QFILE}.pre_reset_$(date +%Y%m%d_%H%M%S)"
  mv "$QFILE" "$BACKUP"
  echo "[OK] Archived stale Q-table: $BACKUP"
else
  echo "[INFO] No prior Q-table for subject '$ACTIVE_ID'; nothing to archive."
fi

echo ""
echo "[Phase C] Participant init complete."
echo "  Active subject_id: $ACTIVE_ID"
echo "  Next main.py boot will rebuild the Q-table from config.yaml"
echo "  rl.item_importance (the therapist-authored empirical priors)."
echo ""
echo "[INFO] Per policy, therapist.db / sessions / clinical reports / logs"
echo "       are PRESERVED. Only the per-subject Q-table cache was reset."
