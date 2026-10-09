"""
TASK 4: How V(s) and Q(s, a) relate
===================================

State value:         V_pi(s)    = E[G_t | S_t = s]                 "how good is this tile?"
State-action value:  Q_pi(s, a) = E[G_t | S_t = s, A_t = a]        "how good is this move from this tile?"

They're two views of the same thing:

  V from Q (average over what the policy would do):
      V(s) = sum_a pi(a | s) * Q(s, a)

  Q from V (look one step ahead, which requires knowing the environment's dynamics):
      Q(s, a) = sum_{s', r} p(s', r | s, a) * [ r + gamma * V(s') ]
      where V(s') counts as 0 if s' is terminal (nothing comes after the cat).

The second one explains why control methods learn Q instead of V. Picking the best
action from V needs the model p(s', r | s, a), which MC and TD don't have.
With Q you just take argmax_a Q(s, a).

env.model(state, action) returns the dynamics as a list of
(probability, next_state, reward, terminated) tuples.

Check:  python check.py 4
"""

import numpy as np  # noqa: F401

from mouse_env import N_ACTIONS, N_STATES, TERMINAL_STATES  # noqa: F401


def epsilon_greedy_probs(Q: np.ndarray, state: int, epsilon: float) -> np.ndarray:
    """
    Return pi(. | state), an array of N_ACTIONS probabilities for the epsilon-greedy policy.

    Each action gets epsilon / N_ACTIONS. The remaining (1 - epsilon) is split
    EVENLY among all tied best actions. (Same behavior as Task 3, written as probabilities.)
    """
    # TODO
    raise NotImplementedError("Task 4: epsilon_greedy_probs")


def v_from_q(Q: np.ndarray, epsilon: float) -> np.ndarray:
    """Return V (shape (N_STATES,)) with V[s] = sum_a pi(a|s) Q[s, a] under epsilon-greedy."""
    # TODO
    raise NotImplementedError("Task 4: v_from_q")


def q_from_v(env, V: np.ndarray, gamma: float) -> np.ndarray:
    """
    Return Q (shape (N_STATES, N_ACTIONS)) from a one-step lookahead on V with env.model.
    Leave rows of terminal states at 0.
    """
    # TODO
    raise NotImplementedError("Task 4: q_from_v")
