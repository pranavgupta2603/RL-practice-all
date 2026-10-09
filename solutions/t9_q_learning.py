import numpy as np

from mouse_env import N_ACTIONS, N_STATES
from solutions.t3_policies import epsilon_greedy_action


def q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma):
    target = reward + (0.0 if terminated else gamma * Q[next_state].max())
    Q[state, action] += alpha * (target - Q[state, action])
    return Q[state, action]


def q_learning(env, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, rng=None, callback=None):
    rng = rng if rng is not None else np.random.default_rng()
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = []
    for _ in range(n_episodes):
        state = env.reset()
        total, done = 0.0, False
        while not done:
            action = epsilon_greedy_action(Q, state, epsilon, rng)
            next_state, reward, terminated, truncated = env.step(action)
            total += reward
            q_learning_update(Q, state, action, reward, next_state, terminated, alpha, gamma)
            state = next_state
            done = terminated or truncated
        episode_returns.append(total)
        if callback is not None:
            callback(Q)
    return Q, episode_returns
