"""
TASK 3: Policies
================

A policy pi(a | s) says which action to take in each state. Here a policy is a
function `policy(state, rng) -> action`.

Q is a numpy array of shape (N_STATES, N_ACTIONS). Q[s, a] estimates how good
it is to take action a in state s and then keep following the policy.

- random policy:     every action has probability 1/4
- greedy:            pick argmax_a Q[s, a]  (EXPLOIT what you know)
- epsilon-greedy:    with probability epsilon act randomly (EXPLORE),
                     otherwise act greedily

Break ties RANDOMLY in greedy_action. At the start Q is all zeros, and
np.argmax would always pick action 0 (Up), so the mouse would never try the others.

Actions are ints: UP=0, DOWN=1, LEFT=2, RIGHT=3 (see mouse_env.py).

Check:  python check.py 3
"""

import numpy as np  # noqa: F401

from mouse_env import N_ACTIONS  # noqa: F401


def random_policy(state: int, rng: np.random.Generator) -> int:
    """Pick one of the N_ACTIONS actions uniformly at random. Hint: rng.integers."""
    # TODO
    raise NotImplementedError("Task 3: random_policy")


def greedy_action(Q: np.ndarray, state: int, rng: np.random.Generator) -> int:
    """Return an action with the highest Q[state, a], breaking ties uniformly at random.

    Hint: np.flatnonzero(q == q.max()) gives all tied best actions; rng.choice picks one.
    """
    # TODO
    raise NotImplementedError("Task 3: greedy_action")


def epsilon_greedy_action(Q: np.ndarray, state: int, epsilon: float, rng: np.random.Generator) -> int:
    """With probability epsilon return a random action, otherwise the greedy one."""
    # TODO
    raise NotImplementedError("Task 3: epsilon_greedy_action")
