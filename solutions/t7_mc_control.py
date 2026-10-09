import numpy as np

from mouse_env import N_ACTIONS, N_STATES
from solutions.t1_episode import run_episode
from solutions.t2_returns import compute_all_returns
from solutions.t3_policies import epsilon_greedy_action


def mc_control(env, n_episodes, gamma=0.9, alpha=0.1, epsilon=0.1, rng=None, callback=None):
    rng = rng if rng is not None else np.random.default_rng()
    Q = np.zeros((N_STATES, N_ACTIONS))
    episode_returns = []

    def policy(state, rng):
        return epsilon_greedy_action(Q, state, epsilon, rng)

    for _ in range(n_episodes):
        episode = run_episode(env, policy, rng)
        pairs = [(tr.state, tr.action) for tr in episode]
        returns = compute_all_returns([tr.reward for tr in episode], gamma)
        for t in reversed(range(len(episode))):
            s, a = pairs[t]
            if (s, a) in pairs[:t]:
                continue
            Q[s, a] += alpha * (returns[t] - Q[s, a])
        episode_returns.append(sum(tr.reward for tr in episode))
        if callback is not None:
            callback(Q)
    return Q, episode_returns
