"""
TASK 9: Q-learning (TD control, off-policy)
===========================================

The algorithm the course builds next. Same as SARSA except for the target:

    SARSA:       target = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})     the action you'll take
    Q-learning:  target = R_{t+1} + gamma * max_a Q(S_{t+1}, a)     the BEST action

Q-learning is OFF-POLICY: it behaves epsilon-greedily (to explore) but learns
the value of the GREEDY policy (the one it would use with exploration turned off).

If terminated, target = R_{t+1}. If truncated, still bootstrap.

Check:  python check.py 9
Watch:  python play.py control qlearning   /   python play.py control all
"""

import numpy as np  # noqa: F401

from mouse_env import N_ACTIONS, N_STATES  # noqa: F401
from tasks.t3_policies import epsilon_greedy_action  # noqa: F401


def q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma) -> float:
    """Apply one Q-learning update to Q[state, action] IN PLACE and return the new value."""
    # TODO
    raise NotImplementedError("Task 9: q_learning_update")


def q_learning(env, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, rng=None, callback=None):
    """Return (Q, episode_returns) like Task 7. Call callback(Q) after each episode if given."""
    # TODO
    raise NotImplementedError("Task 9: q_learning")
