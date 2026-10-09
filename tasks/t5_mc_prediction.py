"""
TASK 5: Monte Carlo prediction (learn V_pi for a fixed policy)
==============================================================

Monte Carlo waits until the episode is OVER, computes the real return G_t, and
moves V(S_t) toward it:

    V(S_t) <- V(S_t) + alpha * [ G_t - V(S_t) ]
                                 \___________/
                                   the error

Course example (alpha = 0.1, gamma = 1, V starts at 0, G_0 = 3):
    V(S_0) = 0 + 0.1 * [3 - 0] = 0.3

FIRST-VISIT vs EVERY-VISIT: a tile can appear several times in one episode
(e.g. the mouse bumps into a wall and stays put).
  - first-visit MC updates V(s) only with the return from the FIRST time s appears
  - every-visit MC updates V(s) once for EVERY time s appears

alpha=None means SAMPLE AVERAGE: use step size 1/N(s), where N(s) counts how many
updates s has received. Then V(s) is exactly the mean of all returns seen from s.

Check:  python check.py 5
Watch:  python play.py walkthrough   /   python play.py predict
"""

import numpy as np  # noqa: F401

from mouse_env import N_STATES  # noqa: F401
from tasks.t1_episode import run_episode  # noqa: F401
from tasks.t2_returns import compute_all_returns  # noqa: F401


def mc_update(V: np.ndarray, state: int, G: float, alpha: float) -> float:
    """Apply one MC update to V[state] IN PLACE and return the new V[state]."""
    # TODO
    raise NotImplementedError("Task 5: mc_update")


def mc_episode_update(V, episode, gamma, alpha, first_visit=True, counts=None):
    """
    Update V in place from one finished episode (a list of Transitions) and return V.

    Loop BACKWARDS over time, t = T-1, ..., 0 (as in Sutton & Barto), using G_t from
    compute_all_returns. With first_visit=True, skip time t if episode[t].state also
    appears earlier in the episode (at some index < t).

    If alpha is None, use step size 1 / counts[s]: increment counts[s] BEFORE each
    update. (counts is an int array of shape (N_STATES,), owned by the caller.)
    """
    # TODO
    raise NotImplementedError("Task 5: mc_episode_update")


def mc_prediction(env, policy, n_episodes, gamma=1.0, alpha=0.1, first_visit=True, rng=None, callback=None):
    """
    Estimate V_pi for `policy` from n_episodes episodes. Return V, shape (N_STATES,).

    1. V = zeros, counts = zeros (ints)
    2. repeat n_episodes times: play an episode (Task 1), then mc_episode_update
    3. after each episode, if callback is not None, call callback(V); the simulator uses this to plot learning curves
    """
    # TODO
    raise NotImplementedError("Task 5: mc_prediction")
