import numpy as np

from mouse_env import N_ACTIONS, N_STATES, TERMINAL_STATES


def epsilon_greedy_probs(Q, state, epsilon):
    q = Q[state]
    best = np.flatnonzero(q == q.max())
    probs = np.full(N_ACTIONS, epsilon / N_ACTIONS)
    probs[best] += (1.0 - epsilon) / len(best)
    return probs


def v_from_q(Q, epsilon):
    V = np.zeros(N_STATES)
    for s in range(N_STATES):
        V[s] = epsilon_greedy_probs(Q, s, epsilon) @ Q[s]
    return V


def q_from_v(env, V, gamma):
    Q = np.zeros((N_STATES, N_ACTIONS))
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            continue
        for a in range(N_ACTIONS):
            for p, next_state, reward, terminated in env.model(s, a):
                Q[s, a] += p * (reward + (0.0 if terminated else gamma * V[next_state]))
    return Q
