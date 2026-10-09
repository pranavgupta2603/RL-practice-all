"""
TASK 8: SARSA (TD control, on-policy)
=====================================

TD control: the TD(0) idea from Task 6 applied to Q. The name comes from the 5 things one update uses:

    S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}

    target = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1})      (A_{t+1} = the action you WILL actually take)
    Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [ target - Q(S_t, A_t) ]

If terminated, target = R_{t+1} (no bootstrap). If truncated, still bootstrap.

Loop structure (note the next action is chosen BEFORE the update):
    s = env.reset(); a = epsilon_greedy(s)
    while not done:
        s2, r, terminated, truncated = env.step(a)
        a2 = epsilon_greedy(s2)
        update Q(s, a) using (r, s2, a2)
        s, a = s2, a2

SARSA is ON-POLICY: it learns the value of the epsilon-greedy policy it's
actually following, exploration mistakes included.

Check:  python check.py 8
Watch:  python play.py control sarsa
"""

import numpy as np  # noqa: F401

from mouse_env import N_ACTIONS, N_STATES  # noqa: F401
from tasks.t3_policies import epsilon_greedy_action  # noqa: F401


def sarsa_update(Q, state, action, reward, next_state, next_action, terminated, alpha, gamma) -> float:
    """Apply one SARSA update to Q[state, action] IN PLACE and return the new value."""
    # TODO
    raise NotImplementedError("Task 8: sarsa_update")


def sarsa(env, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, rng=None, callback=None):
    """Return (Q, episode_returns) like Task 7. Call callback(Q) after each episode if given."""
    # TODO
    raise NotImplementedError("Task 8: sarsa")
