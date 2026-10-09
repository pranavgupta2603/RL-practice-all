"""
TASK 7: Monte Carlo control (learn a good policy)
=================================================

Prediction (Tasks 5 and 6) evaluates a FIXED policy. Control IMPROVES the policy too.

Generalized policy iteration:
    evaluate:  learn Q(s, a) for the current policy
    improve:   act epsilon-greedily w.r.t. Q   (greedy = better, epsilon = keep exploring)
    repeat; both happen a little every episode.

We learn Q instead of V because choosing an action from V needs the environment
model (see Task 4); argmax_a Q(s, a) doesn't.

On-policy first-visit MC control, every episode:
    1. play an episode with epsilon-greedy(Q)
    2. compute G_t for every t
    3. backwards over t, for the FIRST visit of each (state, action) pair:
           Q(s, a) <- Q(s, a) + alpha * [ G_t - Q(s, a) ]

Trick: write the policy as a closure over Q, e.g.
    policy = lambda s, rng: epsilon_greedy_action(Q, s, epsilon, rng)
Q is updated in place, so the policy improves automatically.

Why gamma = 0.9 here: the best behavior is to bounce between the two cheese tiles
forever. With gamma = 1 and bootstrapping, values of loops can grow without bound.

Check:  python check.py 7
Watch:  python play.py control mc
"""

import numpy as np  # noqa: F401

from mouse_env import N_ACTIONS, N_STATES  # noqa: F401
from tasks.t1_episode import run_episode  # noqa: F401
from tasks.t2_returns import compute_all_returns  # noqa: F401
from tasks.t3_policies import epsilon_greedy_action  # noqa: F401


def mc_control(env, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, rng=None, callback=None):
    """
    Return (Q, episode_returns):
      Q:               array (N_STATES, N_ACTIONS)
      episode_returns: list with the UNDISCOUNTED total reward of each episode (for learning curves)
    After each episode, call callback(Q) if callback is not None.
    """
    # TODO
    raise NotImplementedError("Task 7: mc_control")
