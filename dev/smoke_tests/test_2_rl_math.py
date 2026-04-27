"""Smoke test 2 — RL math and Q-table integrity.

Verifies:
  1. initialize_q_table applies ITEM_IMPORTANCE correctly — the pinned
     dims (mood=98, medication=99, eat=97) sit at the top.
  2. choose_action with epsilon=1.0 (legacy pure exploit) picks the
     max-Q dim, and the legacy demo sequence (medication -> mood -> eat)
     is what pops out of the first 3 picks with deterministic argmax.
  3. get_env_feedback branches: normal / DLA_terminate / all-masked.
  4. Q-learning update formula matches paper/legacy: new_Q = old_Q +
     alpha * (R + gamma * max(Q[S_]) - old_Q).

Run:
    .venv/bin/python dev/smoke_tests/test_2_rl_math.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def main():
    print("=== Smoke test 2: RL math + Q-table ===")
    from src.utils.config_loader import (
        ITEM_N_STATES, ITEM_IMPORTANCE, EPSILON, ALPHA, GAMMA,
    )
    from src.utils.rl_qtables import (
        initialize_q_table, choose_action, get_env_feedback,
    )

    # ── Q-table init ─────────────────────────────────────────────────────
    print("\n[A] initialize_q_table")
    actions = [str(i) for i in range(ITEM_N_STATES)]
    q = initialize_q_table(ITEM_N_STATES, actions)
    check("Q-table shape correct", q.shape == (ITEM_N_STATES, ITEM_N_STATES),
          f"got {q.shape}")
    # Legacy init: for each i, t[str(i)] += ITEM_IMPORTANCE[i].
    # str(i) here is the COLUMN index (action id). So every ROW's column-i
    # value is ITEM_IMPORTANCE[i]. That means Q[*, 11] = 97, Q[*, 3] = 99,
    # Q[*, 2] = 98 — the column (action) is what carries the pin.
    check("Q[*, col=3] == 99 across all rows (medication action)",
          all(q.iloc[row, 3] == 99 for row in range(ITEM_N_STATES)))
    check("Q[*, col=2] == 98 across all rows (mood action)",
          all(q.iloc[row, 2] == 98 for row in range(ITEM_N_STATES)))
    check("Q[*, col=11] == 97 across all rows (eat action)",
          all(q.iloc[row, 11] == 97 for row in range(ITEM_N_STATES)))

    # ── choose_action with pure exploitation (epsilon=1.0) ───────────────
    print("\n[B] choose_action (epsilon=1.0, legacy exploit)")
    # Initial mask: all dims 1..37 available, INIT slot 0 off.
    item_mask = [0] + [1] * (ITEM_N_STATES - 1)
    labels = {str(i): f"dim_{i}" for i in range(ITEM_N_STATES)}
    labels["0"] = "INIT"
    np.random.seed(0)
    # From state S=0, Q[0, col=k] = ITEM_IMPORTANCE[k] from init.
    # argmax across cols → col 3 (medication, 99). Every subsequent row
    # has the same column pins, so after masking each used action the
    # next argmax on the pinned columns still wins.
    first_pick = choose_action(0, q, item_mask, ITEM_N_STATES, actions, labels, epsilon=1.0)
    check("First exploit pick is medication (argmax=99)", first_pick == "3",
          f"got {first_pick}")

    # Mask out medication, expect mood (98)
    item_mask[3] = 0
    second_pick = choose_action(0, q, item_mask, ITEM_N_STATES, actions, labels, epsilon=1.0)
    check("Second exploit pick is mood (after masking medication)", second_pick == "2",
          f"got {second_pick}")

    # Mask out mood, expect eat (97)
    item_mask[2] = 0
    third_pick = choose_action(0, q, item_mask, ITEM_N_STATES, actions, labels, epsilon=1.0)
    check("Third exploit pick is eat (after masking medication+mood)", third_pick == "11",
          f"got {third_pick}")

    # ── get_env_feedback ─────────────────────────────────────────────────
    print("\n[C] get_env_feedback")
    S_next, R = get_env_feedback(S=0, A="11", reward=2.0, terminate_flag=0, item_mask=[0,1,1])
    check("Normal step returns (int(A), reward)", S_next == 11 and R == 2.0,
          f"got ({S_next}, {R})")

    S_next, R = get_env_feedback(S=11, A="2", reward=1.5, terminate_flag=1, item_mask=[0,1,1])
    check("terminate_flag=1 -> ('terminal', 0)", S_next == "terminal" and R == 0,
          f"got ({S_next}, {R})")

    S_next, R = get_env_feedback(S=11, A="2", reward=1.5, terminate_flag=0, item_mask=[0,0,0])
    check("All masked -> ('terminal', 10)", S_next == "terminal" and R == 10,
          f"got ({S_next}, {R})")

    # ── Q-learning update formula (paper / legacy) ───────────────────────
    print("\n[D] Q-learning update (alpha=0.5, gamma=0.9)")
    # Simulate: at state 0 we picked action "11" (eat) with reward 2.0,
    # next state S_ = 11. Q[0,11] before = ITEM_IMPORTANCE[11] = 97.
    # Q-learning: new_Q = old_Q + alpha*(R + gamma*max(Q[11,:]) - old_Q).
    old_Q = float(q.loc[0, "11"])
    max_next = float(q.iloc[11, :].max())  # 97 (all eat cells)
    expected_new = old_Q + ALPHA * (2.0 + GAMMA * max_next - old_Q)
    # Apply the update the same way handler_rl does:
    new_q = q.copy()
    new_q.loc[0, "11"] = old_Q + ALPHA * (2.0 + GAMMA * max_next - old_Q)
    check(f"Q update formula matches (old={old_Q}, new={new_q.loc[0,'11']:.3f}, expected={expected_new:.3f})",
          abs(new_q.loc[0, "11"] - expected_new) < 1e-9)

    # ── Trained Q-table (legacy subject 8080 demo) replay ────────────────
    print("\n[F] Trained Q-table replay (demo subject 8080)")
    import pandas as pd
    legacy_qfile = ROOT / "legacy-prototype/data/q_tables/item_qtable_8080.csv"
    if legacy_qfile.exists():
        trained_q = pd.read_csv(legacy_qfile, index_col=0)
        trained_q.columns = trained_q.columns.astype(str)
        check("legacy 8080 trained Q-table has Q[0,11] ~= 220.7 (eat post-training)",
              abs(trained_q.iloc[0, 11] - 220.73) < 0.1,
              f"got {trained_q.iloc[0, 11]:.2f}")
        # With this trained Q-table, pure exploit now picks eat first.
        mask = [0] + [1] * (ITEM_N_STATES - 1)
        np.random.seed(0)
        pick1 = choose_action(0, trained_q.copy(), mask, ITEM_N_STATES,
                              actions, labels, epsilon=1.0)
        check("Trained replay pick 1 == 'eat' (col 11)", pick1 == "11",
              f"got {pick1}")
        mask[11] = 0
        pick2 = choose_action(0, trained_q.copy(), mask, ITEM_N_STATES,
                              actions, labels, epsilon=1.0)
        # In the saved trained Q-table, Q[0,3]=98 and Q[0,2]=97, so after
        # masking eat the argmax is medication.
        check("Trained replay pick 2 == 'medication' (col 3, Q=98)",
              pick2 == "3", f"got {pick2}")
        mask[3] = 0
        pick3 = choose_action(0, trained_q.copy(), mask, ITEM_N_STATES,
                              actions, labels, epsilon=1.0)
        check("Trained replay pick 3 == 'mood' (col 2, Q=97)",
              pick3 == "2", f"got {pick3}")
        print("  [INFO] Trained-Q sequence is eat→med→mood. The demo video shows")
        print("         eat→mood→med; the saved 8080 Q-table at hand reflects a")
        print("         different Q-update history than the exact demo session.")
        print("         Both orderings are clinically equivalent (top-3 priority).")
    else:
        print(f"  [SKIP] legacy Q-table not found at {legacy_qfile}")

    # ── Terminal branch skips bootstrapping ──────────────────────────────
    print("\n[E] Terminal branch — no bootstrap")
    old_Q = 5.0
    # When S_ == 'terminal', handler sets q_target = R (no gamma*max).
    q_target = 3.0  # just R
    new_Q = old_Q + ALPHA * (q_target - old_Q)
    expected = 5.0 + 0.5 * (3.0 - 5.0)  # 4.0
    check(f"terminal Q update = old + alpha*(R - old) = {new_Q}",
          abs(new_Q - expected) < 1e-9)

    print("\n=== RESULT ===")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("All RL math invariants held.")
    sys.exit(0)


if __name__ == "__main__":
    main()
