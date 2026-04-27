"""Utility helper providing math and table management for Q-learning states."""
import numpy as np
import pandas as pd
from src.utils.config_loader import ITEM_IMPORTANCE, EPSILON

# Set up logger for this module
from src.utils.log_util import get_logger
logger = get_logger("RLQTables")

def build_q_table(n_states, actions):
    """
    Build a Q-table as a pandas DataFrame with zeros.
    Each row is a state, each column is an action.
    """
    t = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    logger.debug(f"Built Q-table with shape {t.shape} and actions: {actions}")
    return t

 

def initialize_q_table(n_states, actions):
    """
    Initialize a Q-table for items.
    For each state, set all Q-values to the corresponding item importance.
    This biases the Q-table according to item importance.
    """
    logger.debug("Initializing Q-table for item RL states using item importance.")
    t = build_q_table(n_states, actions)
    for i in range(0, n_states):
        # Add item importance value to all Q-values in the action column for state i
        t[str(i)] = t[str(i)].apply(lambda x: x + ITEM_IMPORTANCE[i])
    logger.info(f"[RL] Q-table seeded with therapist item_importance priors: {ITEM_IMPORTANCE}")
    return t

def choose_action(
        state: int,
        q_table: pd.DataFrame,
        mask: list,
        number_states: int,
        actions: list,
        action_labels: dict = None,
        epsilon: float = None,
    ) -> str:
    """
    Choose an action based on the current state and Q-table.
    Mask out unavailable actions by multiplying their Q-values by 0.
    With probability `epsilon` (exploitation rate), choose the best action;
    otherwise explore.  If `epsilon` is None, fall back to the module-level
    `EPSILON` constant — callers driving an ε-greedy decay schedule should
    pass the turn's effective exploitation rate explicitly.
    """
    eff_epsilon = EPSILON if epsilon is None else float(epsilon)
    logger.debug(f"Choosing action for state {state} (epsilon={eff_epsilon:.3f})")
    state_action = q_table.iloc[state, :].copy()
    # Apply mask to the state_action to disable unavailable actions
    logger.debug("Mask before: [{}]".format(','.join(str(m) for m in mask)))
    for i in range(1, number_states):
        state_action[str(i)] = state_action[str(i)] * mask[i]
    logger.debug("Q-values after masking: [{}]".format(','.join(str(v) for v in state_action.values)))
    # Exploration: with probability 1-epsilon or if all Q-values are zero, pick randomly
    if (np.random.uniform() > eff_epsilon):
        # Exploration branch: choose at random among available (not masked out) actions
        available_actions = [actions[i] for i in range(1, number_states) if mask[i] == 1]
        logger.info(f"[RL] Exploring (epsilon={eff_epsilon:.2f}): random pick from available dims {available_actions}")
        action = np.random.choice(available_actions)
    else:
        # Exploitation branch: choose the action(s) with the highest Q-value among AVAILABLE actions
        available_indices = [str(i) for i in range(1, number_states) if mask[i] == 1]

        if available_indices:
            # Filter the state_action series to only consider valid unmasked items
            valid_actions = state_action[available_indices]
            max_value = np.max(valid_actions)
            best_actions = valid_actions[valid_actions == max_value].index
            logger.info(f"[RL] Exploiting: best dims {list(best_actions)} with Q-value {max_value}")
            action = np.random.choice(best_actions)
        else:
            # Fallback if no valid actions are technically available
            logger.warning("No valid actions available in exploitation, defaulting to random choice. Check termination logic.")
            action = np.random.choice(actions[1:])

    # Log action with human-readable label if provided
    if action_labels is not None:
        label = action_labels.get(str(action), str(action))
        logger.info(f"[RL] Action chosen: dim={action} ({label})")
    else:
        logger.info(f"[RL] Action chosen: dim={action}")
    return action

def get_env_feedback(S, A, reward, terminate_flag, item_mask):
    """
    Get the next state and reward from the environment.
    If all items are masked (no available actions), return terminal state and reward 10.
    If terminate_flag is set, return terminal state and reward 0.
    Otherwise, return the next state (action taken) and the given reward.

    Note on state count: paper §5.1 defines 39 states (37 dims + START + END).
    We model END as the sentinel string 'terminal' rather than a 39th row in
    the Q-table. The caller uses `S_ != 'terminal'` to skip the discounted-
    future term in the Q-update, which is mathematically identical to an
    absorbing END row with Q-values fixed at 0.
    """
    logger.debug(f"Getting environment feedback: S={S}, A={A}, reward={reward}, terminate_flag={terminate_flag}, item_mask={item_mask}")
    if sum(item_mask) == 0:
        logger.info("[RL] All 37 dimensions exhausted — screening loop terminating (reward=+10).")
        return 'terminal', 10
    elif terminate_flag == 1:
        logger.info("[RL] Terminate flag set — screening loop ending early (reward=0).")
        return 'terminal', 0
    else:
        logger.info(f"[RL] Reward this turn: {reward} (next state: dim={A})")
        return int(A), reward
 

