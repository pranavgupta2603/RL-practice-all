import numpy as np

from mouse_env import N_ACTIONS


def random_policy(state, rng):
    return int(rng.integers(N_ACTIONS))


def greedy_action(Q, state, rng):
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))


def epsilon_greedy_action(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return random_policy(state, rng)
    return greedy_action(Q, state, rng)
