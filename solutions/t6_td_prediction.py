import numpy as np

from mouse_env import N_STATES


def td_update(V, state, reward, next_state, terminated, alpha, gamma):
    target = reward + (0.0 if terminated else gamma * V[next_state])
    V[state] += alpha * (target - V[state])
    return V[state]


def td_prediction(env, policy, n_episodes, gamma=1.0, alpha=0.1, rng=None, callback=None):
    rng = rng if rng is not None else np.random.default_rng()
    V = np.zeros(N_STATES)
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy(state, rng)
            next_state, reward, terminated, truncated = env.step(action)
            td_update(V, state, reward, next_state, terminated, alpha, gamma)
            state = next_state
            done = terminated or truncated
        if callback is not None:
            callback(V)
    return V
