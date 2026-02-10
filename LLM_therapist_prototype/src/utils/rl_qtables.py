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
    logger.info("Initializing Q-table for item RL states using item importance.")
    t = build_q_table(n_states, actions)
    for i in range(0, n_states):
        # Add item importance value to all Q-values in the action column for state i
        t[str(i)] = t[str(i)].apply(lambda x: x + ITEM_IMPORTANCE[i])
        # logger.debug(f"Set Q-values for state {i} to importance {ITEM_IMPORTANCE[i]}")
    logger.info(f"Initialization of Q-table complete based on {ITEM_IMPORTANCE}.")
    return t

def choose_action(
        state: int,
        q_table: pd.DataFrame,
        mask: list,
        number_states: int,
        actions: list,
        action_labels: dict = None
    ) -> str:
    """
    Choose an action based on the current state and Q-table.
    Mask out unavailable actions by multiplying their Q-values by 0.
    With probability EPSILON, choose the best action; otherwise, explore.
    """
    logger.info(f"Choosing action for state {state}")
    state_action = q_table.iloc[state, :]
    # Apply mask to Q-table to disable unavailable actions
    logger.debug("Mask before: [{}]".format(','.join(str(m) for m in mask)))
    original_q_table = q_table.copy()  # For debugging/more traceable logs
    for i in range(1, number_states):
        q_table[str(i)] = q_table[str(i)].apply(lambda x: x * mask[i])
    logger.debug("Q-table after masking (row {}): [{}]".format(state, ','.join(str(v) for v in q_table.iloc[state, :].values)))
    # Exploration: with probability 1-EPSILON or if all Q-values are zero, pick randomly
    if (np.random.uniform() > EPSILON):
        # Exploration branch: choose at random among available (not masked out) actions
        available_actions = [actions[i] for i in range(1, number_states) if mask[i] == 1]
        logger.info(f"Exploring: choosing randomly among available actions {available_actions}")
        action = np.random.choice(available_actions)
    else:
        # Exploitation branch: choose the action(s) with the highest Q-value
        max_value = np.max(state_action)
        best_actions = state_action[state_action == max_value].index
        logger.info(f"Exploiting: choosing among best actions {list(best_actions)} with Q-value {max_value}")
        action = np.random.choice(best_actions)
    # Log action with human-readable label if provided
    if action_labels is not None:
        label = action_labels.get(str(action), str(action))
        logger.info(f"Action chosen: {action}_{label}")
    else:
        logger.info(f"Action chosen: {action}")
    return action

def get_env_feedback(S, A, reward, terminate_flag, item_mask):
    """
    Get the next state and reward from the environment.
    If all items are masked (no available actions), return terminal state and reward 10.
    If terminate_flag is set, return terminal state and reward 0.
    Otherwise, return the next state (action taken) and the given reward.
    """
    logger.debug(f"Getting environment feedback: S={S}, A={A}, reward={reward}, terminate_flag={terminate_flag}, item_mask={item_mask}")
    if sum(item_mask) == 0:
        logger.info("All items exhausted (no actions available). Transitioning to terminal state with reward 10.")
        return 'terminal', 10
    elif terminate_flag == 1:
        logger.info("Terminate flag set. Transitioning to terminal state with reward 0.")
        return 'terminal', 0
    else:
        logger.info(f"Proceeding to next state: {A} with reward {reward}")
        return int(A), reward
 

