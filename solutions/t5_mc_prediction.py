import numpy as np

from mouse_env import N_STATES
from solutions.t1_episode import run_episode
from solutions.t2_returns import compute_all_returns


def mc_update(V, state, G, alpha):
    V[state] += alpha * (G - V[state])
    return V[state]


def mc_episode_update(V, episode, gamma, alpha, first_visit=True, counts=None):
    states = [tr.state for tr in episode]
    returns = compute_all_returns([tr.reward for tr in episode], gamma)
    for t in reversed(range(len(episode))):
        s = states[t]
        if first_visit and s in states[:t]:
            continue
        if alpha is None:
            counts[s] += 1
            step_size = 1.0 / counts[s]
        else:
            step_size = alpha
        mc_update(V, s, returns[t], step_size)
    return V


def mc_prediction(env, policy, n_episodes, gamma=1.0, alpha=0.1, first_visit=True, rng=None, callback=None):
    rng = rng if rng is not None else np.random.default_rng()
    V = np.zeros(N_STATES)
    counts = np.zeros(N_STATES, dtype=int)
    for _ in range(n_episodes):
        episode = run_episode(env, policy, rng)
        mc_episode_update(V, episode, gamma, alpha, first_visit, counts)
        if callback is not None:
            callback(V)
    return V
