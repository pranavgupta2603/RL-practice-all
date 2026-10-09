"""
TASK 6: TD(0) prediction (learn V_pi, one step at a time)
==========================================================

TD doesn't wait for the episode to end. After ONE step (S_t, A_t, R_{t+1}, S_{t+1})
it estimates G_t with the "TD target" and updates right away:

    TD target:  R_{t+1} + gamma * V(S_{t+1})         <- uses its OWN estimate (bootstrapping)
    TD error:   delta = target - V(S_t)
    update:     V(S_t) <- V(S_t) + alpha * delta

Course example (alpha = 0.1, gamma = 1, V all 0, first move Left onto cheese, reward +1):
    V(S_0) = 0 + 0.1 * [1 + 1 * 0 - 0] = 0.1

TERMINATED vs TRUNCATED (very important, and a classic bug):
  - terminated (the cat): there is no future, so the target is just R_{t+1}. Do NOT bootstrap.
  - truncated (time limit): the world didn't end; we just stopped watching.
    The next state still has a future, so you DO bootstrap from V(S_{t+1}).
That's why td_update takes `terminated`, not "done".

Check:  python check.py 6
Watch:  python play.py walkthrough   /   python play.py manual   /   python play.py predict
"""

import numpy as np  # noqa: F401

from mouse_env import N_STATES  # noqa: F401


def td_update(V, state, reward, next_state, terminated, alpha, gamma) -> float:
    """Apply one TD(0) update to V[state] IN PLACE and return the new V[state]."""
    # TODO
    raise NotImplementedError("Task 6: td_update")


def td_prediction(env, policy, n_episodes, gamma=1.0, alpha=0.1, rng=None, callback=None):
    """
    Estimate V_pi with TD(0). Return V, shape (N_STATES,).

    Write the env loop yourself here (don't use run_episode). The point of TD is that
    you update INSIDE the loop, after every single env.step().
    After each episode, call callback(V) if callback is not None.
    """
    # TODO
    raise NotImplementedError("Task 6: td_prediction")
