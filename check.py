#!/usr/bin/env python3
"""
Check your task implementations.

    python check.py            # all tasks
    python check.py 2 5        # only tasks 2 and 5
    python check.py --solutions  # run the checks against the reference solutions
"""

from __future__ import annotations

import argparse
import importlib
import traceback

import numpy as np

from mouse_env import (
    CAT,
    DOWN,
    LEFT,
    N_ACTIONS,
    N_STATES,
    RIGHT,
    START,
    UP,
    MouseEnv,
    Transition,
    expected_return,
    to_state,
)

TOP2, BOTTOM2, BOTTOM1, TOP3 = to_state(0, 1), to_state(1, 1), to_state(1, 0), to_state(0, 2)
COURSE_ACTIONS = [LEFT, LEFT, DOWN, DOWN, LEFT, DOWN, RIGHT, UP, RIGHT, DOWN]
COURSE_REWARDS = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0]
# Deterministic policy: 3 bottom -Left-> 2 bottom (+1) -Up-> 2 top (+1) -Left-> cat (-10)
SUICIDE_POLICY = {START: LEFT, BOTTOM2: UP, TOP2: LEFT}

TASKS: dict[int, tuple[str, str, callable]] = {}


class Fail(Exception):
    pass


def task(number: int, module: str, title: str):
    def register(fn):
        TASKS[number] = (module, title, fn)
        return fn

    return register


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise Fail(message)


def close(a, b, tol=1e-6) -> bool:
    return bool(np.allclose(a, b, atol=tol))


def scripted(actions):
    it = iter(actions)
    return lambda state, rng: next(it)


def by_state(table):
    return lambda state, rng: table[state]


def course_episode():
    env, episode, state = MouseEnv(), [], None
    state = env.reset()
    for a in COURSE_ACTIONS:
        ns, r, term, trunc = env.step(a)
        episode.append(Transition(state, a, r, ns, term, trunc))
        state = ns
    return episode


def greedy_return(Q, max_steps=10) -> float:
    env = MouseEnv(max_steps=max_steps)
    state, total, done = env.reset(), 0.0, False
    while not done:
        state, r, term, trunc = env.step(int(np.argmax(Q[state])))
        total += r
        done = term or trunc
    return total


# ----------------------------------------------------------------------------- Task 1


@task(1, "t1_episode", "run_episode: the agent-environment loop")
def check_1(m):
    ep = m.run_episode(MouseEnv(), scripted(COURSE_ACTIONS), np.random.default_rng(0))
    expect(isinstance(ep, list) and all(isinstance(t, Transition) for t in ep), "should return a list of Transition objects")
    expect(len(ep) == 10, f"course episode should last 10 steps (time limit), got {len(ep)}")
    expect([t.reward for t in ep] == COURSE_REWARDS, f"rewards should be {COURSE_REWARDS}, got {[t.reward for t in ep]}")
    expect(ep[0].state == START, "first transition should start from START (tile 3 bottom)")
    expect(all(a.next_state == b.state for a, b in zip(ep, ep[1:])), "each transition's next_state must be the next one's state")
    expect([t.action for t in ep] == COURSE_ACTIONS, "actions recorded don't match the actions the policy chose")
    expect(ep[-1].truncated and not ep[-1].terminated, "last step of the course episode should be truncated, not terminated")
    expect(not any(t.truncated or t.terminated for t in ep[:-1]), "only the last transition should end the episode")

    ep = m.run_episode(MouseEnv(), by_state(SUICIDE_POLICY), np.random.default_rng(0))
    expect(len(ep) == 3 and ep[-1].terminated and ep[-1].next_state == CAT,
           "Left, Up, Left from the start should end in 3 steps with the cat (terminated)")


# ----------------------------------------------------------------------------- Task 2


@task(2, "t2_returns", "compute_return / compute_all_returns")
def check_2(m):
    expect(close(m.compute_return(COURSE_REWARDS, 1.0), 3.0), "course example: G_0 should be 3")
    expect(close(m.compute_return([], 0.9), 0.0), "empty episode should have return 0")
    expect(close(m.compute_return([1, 2, 3], 0.5), 1 + 0.5 * 2 + 0.25 * 3), "gamma=0.5, rewards [1,2,3] -> 2.75 (first reward is NOT discounted)")
    got = m.compute_all_returns(COURSE_REWARDS, 1.0)
    expect(close(got, [3, 2, 2, 2, 2, 2, 2, 1, 0, 0]), f"course returns per step should be [3,2,2,2,2,2,2,1,0,0], got {got}")
    got = m.compute_all_returns([1, -10], 0.9)
    expect(close(got, [1 - 9, -10]), f"[1, -10] with gamma 0.9 -> [-8, -10], got {got}")
    expect(list(m.compute_all_returns([], 0.9)) == [], "empty episode -> []")


# ----------------------------------------------------------------------------- Task 3


@task(3, "t3_policies", "random / greedy / epsilon-greedy policies")
def check_3(m):
    rng = np.random.default_rng(0)
    counts = np.bincount([m.random_policy(START, rng) for _ in range(8000)], minlength=N_ACTIONS)
    expect(len(counts) == N_ACTIONS and np.all(np.abs(counts / 8000 - 0.25) < 0.03), f"random_policy should be uniform, got counts {counts}")

    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[START] = [0.0, 3.0, -1.0, 0.0]
    expect(all(m.greedy_action(Q, START, rng) == DOWN for _ in range(50)), "greedy_action should pick argmax")
    Q[START] = [1.0, 5.0, 5.0, 0.0]
    picks = {m.greedy_action(Q, START, rng) for _ in range(200)}
    expect(picks == {DOWN, LEFT}, f"ties must be broken randomly between the best actions only; picked {picks}")

    Q[START] = [0.0, 0.0, 9.0, 0.0]
    expect(all(m.epsilon_greedy_action(Q, START, 0.0, rng) == LEFT for _ in range(50)), "epsilon=0 should be fully greedy")
    counts = np.bincount([m.epsilon_greedy_action(Q, START, 1.0, rng) for _ in range(8000)], minlength=N_ACTIONS)
    expect(np.all(np.abs(counts / 8000 - 0.25) < 0.03), f"epsilon=1 should be uniform, got {counts}")
    counts = np.bincount([m.epsilon_greedy_action(Q, START, 0.2, rng) for _ in range(10000)], minlength=N_ACTIONS)
    expect(abs(counts[LEFT] / 10000 - 0.85) < 0.02,
           f"epsilon=0.2: best action prob should be 0.8 + 0.2/4 = 0.85, got {counts[LEFT] / 10000:.3f}")


# ----------------------------------------------------------------------------- Task 4


@task(4, "t4_value_relations", "V <-> Q relations")
def check_4(m):
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[START] = [0.0, 2.0, 2.0, -4.0]
    p = m.epsilon_greedy_probs(Q, START, 0.2)
    expect(close(p, [0.05, 0.45, 0.45, 0.05]), f"probs with a 2-way tie at eps=0.2 should be [.05,.45,.45,.05], got {p}")
    expect(close(np.sum(m.epsilon_greedy_probs(Q, TOP2, 0.3)), 1.0), "probabilities must sum to 1")

    V = m.v_from_q(Q, 0.2)
    expect(np.shape(V) == (N_STATES,), "v_from_q should return shape (N_STATES,)")
    expect(close(V[START], 0.05 * 0 + 0.45 * 2 + 0.45 * 2 + 0.05 * -4), f"V[start] should be 1.6, got {V[START]}")
    expect(close(m.v_from_q(Q, 0.0)[START], 2.0), "greedy (eps=0): V(s) = max_a Q(s, a)")

    env = MouseEnv()
    Qv = m.q_from_v(env, np.zeros(N_STATES), 1.0)
    expect(np.shape(Qv) == (N_STATES, N_ACTIONS), "q_from_v should return shape (N_STATES, N_ACTIONS)")
    expect(close(Qv[START], [0, 0, 1, 0]), f"with V=0, from 3 bottom only Left (cheese) is worth +1; got {Qv[START]}")
    V = np.arange(N_STATES, dtype=float)
    V[CAT] = 100.0
    Qv = m.q_from_v(env, V, 0.5)
    expect(close(Qv[TOP2, LEFT], -10.0), "moving onto the cat is terminal: Q = -10, ignore V[cat]")
    expect(close(Qv[TOP2, DOWN], 1 + 0.5 * V[BOTTOM2]), "Q(2 top, Down) = 1 + gamma * V(2 bottom)")
    expect(close(Qv[CAT], 0.0), "terminal state rows should stay 0")
    Qs = m.q_from_v(MouseEnv(slip=0.4), np.zeros(N_STATES), 1.0)
    expect(close(Qs[START, LEFT], 0.6 * 1 + 0.4 * 0.25 * 1), "with slip=0.4, Q(start, Left) = 0.6*1 + 0.1*1 = 0.7 (use env.model probabilities)")


# ----------------------------------------------------------------------------- Task 5


@task(5, "t5_mc_prediction", "Monte Carlo prediction")
def check_5(m):
    V = np.zeros(N_STATES)
    new = m.mc_update(V, START, 3.0, 0.1)
    expect(close(V[START], 0.3) and close(new, 0.3), "course example: V(S0) = 0 + 0.1 * [3 - 0] = 0.3 (update V in place AND return it)")

    ep = course_episode()
    V = m.mc_episode_update(np.zeros(N_STATES), ep, 1.0, 0.1, first_visit=True)
    want = np.zeros(N_STATES)
    want[[START, BOTTOM2, BOTTOM1]] = [0.3, 0.2, 0.2]
    expect(close(V, want), f"first-visit on the course episode should give {want}, got {V}")

    V = m.mc_episode_update(np.zeros(N_STATES), ep, 1.0, 0.1, first_visit=False)
    want[[START, BOTTOM2, BOTTOM1]] = [0.3, 0.29, 0.81902]
    expect(close(V, want), f"every-visit (backwards in time) should give {want}, got {V}. "
                           "Did you loop t = T-1..0 and update every occurrence?")

    counts = np.zeros(N_STATES, dtype=int)
    V = m.mc_episode_update(np.zeros(N_STATES), ep, 1.0, None, first_visit=False, counts=counts)
    expect(close(V[[BOTTOM1, BOTTOM2]], [2.0, 1.5]) and counts[BOTTOM1] == 5,
           "alpha=None, every-visit: V(1 bottom) = mean of its five returns (all 2) = 2, V(2 bottom) = mean(2, 1) = 1.5")

    rng = np.random.default_rng(1)
    V = m.mc_prediction(MouseEnv(), by_state(SUICIDE_POLICY), 1, gamma=0.9, alpha=None, rng=rng)
    expect(close(V[[START, BOTTOM2, TOP2]], [-6.2, -8.0, -10.0]),
           f"deterministic Left-Up-Left policy, gamma 0.9: V = [-6.2, -8, -10] after one episode, got {V[[START, BOTTOM2, TOP2]]}")

    seen = []
    pi = lambda s, rng: int(rng.integers(N_ACTIONS))
    V = m.mc_prediction(MouseEnv(), pi, 4000, gamma=1.0, alpha=None, rng=np.random.default_rng(2), callback=lambda V: seen.append(1))
    exact = expected_return(MouseEnv(), np.full((N_STATES, N_ACTIONS), 0.25), 1.0)[START]
    expect(len(seen) == 4000, "callback should be called once per episode")
    expect(abs(V[START] - exact) < 0.35, f"random policy: V(start) should approach the true E[G_0] = {exact:.3f}, got {V[START]:.3f}")


# ----------------------------------------------------------------------------- Task 6


@task(6, "t6_td_prediction", "TD(0) prediction")
def check_6(m):
    V = np.zeros(N_STATES)
    new = m.td_update(V, START, 1.0, BOTTOM2, False, 0.1, 1.0)
    expect(close(V[START], 0.1) and close(new, 0.1), "course example: V(S0) = 0 + 0.1 * [1 + 1*0 - 0] = 0.1 (in place AND returned)")

    V = np.zeros(N_STATES)
    V[BOTTOM2] = 5.0
    m.td_update(V, START, 1.0, BOTTOM2, False, 0.5, 0.9)
    expect(close(V[START], 0.5 * (1 + 0.9 * 5)), "target should be r + gamma * V(next)")
    V = np.zeros(N_STATES)
    V[CAT] = 50.0
    m.td_update(V, TOP2, -10.0, CAT, True, 1.0, 0.9)
    expect(close(V[TOP2], -10.0), "terminated: target = reward only. Do NOT bootstrap from V(terminal)")

    rng = np.random.default_rng(0)
    V = m.td_prediction(MouseEnv(), by_state(SUICIDE_POLICY), 1, gamma=0.9, alpha=1.0, rng=rng)
    expect(close(V[[START, BOTTOM2, TOP2]], [1.0, 1.0, -10.0]),
           "one episode, alpha=1: TD updates step by step DURING the episode, so V(start) = 1 + 0.9*V(2 bottom)=1 "
           f"(V(2 bottom) was still 0). Got {V[[START, BOTTOM2, TOP2]]}")
    seen = []
    V = m.td_prediction(MouseEnv(), by_state(SUICIDE_POLICY), 500, gamma=0.9, alpha=0.3, rng=rng, callback=lambda V: seen.append(1))
    expect(len(seen) == 500, "callback should be called once per episode")
    expect(close(V[[START, BOTTOM2, TOP2]], [-6.2, -8.0, -10.0], 1e-3), f"should converge to [-6.2, -8, -10], got {V[[START, BOTTOM2, TOP2]]}")


# ----------------------------------------------------------------------------- Tasks 7-9


def check_control(fn, name, seeds=5):
    greedy = []
    for seed in range(seeds):
        seen = []
        Q, returns = fn(MouseEnv(), 3000, gamma=0.9, alpha=0.1, epsilon=0.05, rng=np.random.default_rng(seed),
                        callback=lambda Q: seen.append(1))
        expect(np.shape(Q) == (N_STATES, N_ACTIONS), "Q should have shape (N_STATES, N_ACTIONS)")
        expect(len(returns) == 3000, "episode_returns should have one entry per episode")
        expect(len(seen) == 3000, "callback should be called once per episode")
        expect(close(Q[CAT], 0.0), "Q of the terminal (cat) state should stay 0")
        avg = np.mean(returns[-500:])
        expect(avg > 4, f"{name} (seed {seed}): average return over the last 500 episodes should be > 4, got {avg:.2f}")
        greedy.append(greedy_return(Q))
    # Settling for 5 (bouncing 2 bottom <-> 3 bottom) is a legit exploration trap, not a bug.
    expect(min(greedy) >= 5, f"greedy policy returns over {seeds} seeds were {greedy}; each should be >= 5")
    expect(max(greedy) == 10, f"greedy policy returns over {seeds} seeds were {greedy}; at least one should find the optimal 10")


@task(7, "t7_mc_control", "Monte Carlo control")
def check_7(m):
    check_control(m.mc_control, "MC control")


@task(8, "t8_sarsa", "SARSA")
def check_8(m):
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[BOTTOM2] = [4.0, 0.0, 0.0, 9.0]
    new = m.sarsa_update(Q, START, LEFT, 1.0, BOTTOM2, UP, False, 0.5, 0.9)
    expect(close(new, 0.5 * (1 + 0.9 * 4.0)) and close(Q[START, LEFT], new), "SARSA target uses Q(s', a') of the NEXT ACTION, not the max")
    Q[CAT] = 7.0
    m.sarsa_update(Q, TOP2, LEFT, -10.0, CAT, UP, True, 1.0, 0.9)
    expect(close(Q[TOP2, LEFT], -10.0), "terminated: no bootstrap")
    check_control(m.sarsa, "SARSA")


@task(9, "t9_q_learning", "Q-learning")
def check_9(m):
    Q = np.zeros((N_STATES, N_ACTIONS))
    Q[BOTTOM2] = [4.0, 0.0, 0.0, 9.0]
    new = m.q_learning_update(Q, START, LEFT, 1.0, BOTTOM2, False, 0.5, 0.9)
    expect(close(new, 0.5 * (1 + 0.9 * 9.0)) and close(Q[START, LEFT], new), "Q-learning target uses max_a Q(s', a)")
    Q[CAT] = 7.0
    m.q_learning_update(Q, TOP2, LEFT, -10.0, CAT, True, 1.0, 0.9)
    expect(close(Q[TOP2, LEFT], -10.0), "terminated: no bootstrap")
    check_control(m.q_learning, "Q-learning")


# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasks", nargs="*", type=int, help="task numbers to check (default: all)")
    parser.add_argument("--solutions", action="store_true", help="check the reference solutions instead of tasks/")
    args = parser.parse_args()
    package = "solutions" if args.solutions else "tasks"

    results = []
    for n in args.tasks or sorted(TASKS):
        module, title, fn = TASKS[n]
        label = f"Task {n}: {title}"
        try:
            fn(importlib.import_module(f"{package}.{module}"))
            status, detail = "PASS", ""
        except NotImplementedError as e:
            status, detail = "TODO", str(e) or "not implemented yet"
        except Fail as e:
            status, detail = "FAIL", str(e)
        except Exception:
            status, detail = "ERROR", traceback.format_exc(limit=-3).strip()
        results.append(status)
        icon = {"PASS": "✅", "TODO": "⬜", "FAIL": "❌", "ERROR": "💥"}[status]
        print(f"{icon} {status:<5} {label}")
        if detail and status != "PASS":
            print("        " + detail.replace("\n", "\n        "))
    print(f"\n{results.count('PASS')}/{len(results)} passing")


if __name__ == "__main__":
    main()
