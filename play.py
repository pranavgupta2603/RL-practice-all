#!/usr/bin/env python3
"""
Mouse & Cheese simulator for learning Monte Carlo and Temporal Difference.

  python play.py walkthrough          the course's MC and TD examples, step by step
  python play.py manual               drive the mouse yourself (w/a/s/d) and watch V update
  python play.py random               watch random-policy episodes
  python play.py predict              MC vs TD estimating V under the random policy
  python play.py control qlearning    learn Q, then watch the greedy mouse (mc | sarsa | qlearning | all)

Every command runs YOUR code in tasks/. Add --solutions to use the reference code.
Run `python play.py <command> -h` for options.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np

import mouse_env as me
import visualize as viz
from mouse_env import ACTION_NAMES, DOWN, LEFT, N_ACTIONS, N_STATES, RIGHT, START, UP, MouseEnv, Transition, state_name

COURSE_ACTIONS = [LEFT, LEFT, DOWN, DOWN, LEFT, DOWN, RIGHT, UP, RIGHT, DOWN]
CONTROL = {
    "mc": ("t7_mc_control", "mc_control", "Monte Carlo control"),
    "sarsa": ("t8_sarsa", "sarsa", "SARSA"),
    "qlearning": ("t9_q_learning", "q_learning", "Q-learning"),
}


class NotReady(Exception):
    pass


def load(module: str, args):
    return importlib.import_module(f"{'solutions' if args.solutions else 'tasks'}.{module}")


def call(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except NotImplementedError as e:
        raise NotReady(str(e)) from None


def rollout(env, policy, rng=None):
    episode, state, done = [], env.reset(), False
    while not done:
        action = policy(state, rng)
        ns, r, term, trunc = env.step(action)
        episode.append(Transition(state, action, r, ns, term, trunc))
        state, done = ns, term or trunc
    return episode


def scripted(actions):
    it = iter(actions)
    return lambda state, rng: next(it)


def uniform_random(state, rng):
    return int(rng.integers(N_ACTIONS))


def greedy(Q):
    return lambda state, rng: int(np.argmax(Q[state]))


def pause(args, msg="\nPress Enter to continue... "):
    if not args.no_pause and sys.stdin.isatty():
        input(msg)


def make_env(args) -> MouseEnv:
    return MouseEnv(max_steps=args.max_steps, slip=args.slip, seed=args.seed)


def td_line(state, reward, next_state, terminated, old, nxt, new, alpha, gamma) -> str:
    boot = "0 (terminal)" if terminated else f"{gamma:g} * {nxt:.3f}"
    return f"V({state_name(state)}) = {old:.3f} + {alpha:g} * [{reward:g} + {boot} - {old:.3f}] = {new:.3f}"


# ----------------------------------------------------------------------------- commands


def cmd_walkthrough(args):
    t2, t5, t6 = load("t2_returns", args), load("t5_mc_prediction", args), load("t6_td_prediction", args)
    lr, gamma = 0.1, 1.0
    episode = rollout(MouseEnv(), scripted(COURSE_ACTIONS))

    viz.animate_episode(episode, args.delay, "PART 1: Monte Carlo. Play the WHOLE episode before learning anything.")
    print("\nThe episode as [state, action, reward, next state] tuples:")
    for tr in episode:
        print(f"  [{state_name(tr.state)}, Go {ACTION_NAMES[tr.action]}, {tr.reward:+g}, {state_name(tr.next_state)}]")
    rewards = [tr.reward for tr in episode]
    G0 = call(t2.compute_return, rewards, gamma)
    print(f"\nG_0 = R_1 + R_2 + ... + R_10 = {' + '.join(f'{r:g}' for r in rewards)} = {G0:g}    <- your compute_return")
    V = np.zeros(N_STATES)
    new = call(t5.mc_update, V, START, G0, lr)
    print(f"V(S_0) = V(S_0) + lr * [G_0 - V(S_0)] = 0 + {lr} * [{G0:g} - 0] = {new:g}    <- your mc_update")
    V = call(t5.mc_episode_update, np.zeros(N_STATES), episode, gamma, lr, True, None)
    print("\nMC also updates every OTHER tile the mouse visited, each with its own G_t (first-visit):")
    print(viz.format_values(V, "V after one MC episode"))
    pause(args)

    V = np.zeros(N_STATES)
    log = []
    for t, tr in enumerate(episode):
        old, nxt = V[tr.state], V[tr.next_state]
        new = call(t6.td_update, V, tr.state, tr.reward, tr.next_state, tr.terminated, lr, gamma)
        log.append(f"  t={t}: " + td_line(tr.state, tr.reward, tr.next_state, tr.terminated, old, nxt, new, lr, gamma))
        viz.clear()
        print("PART 2: TD(0). Learn after EVERY step from (S_t, A_t, R_t+1, S_t+1).\n")
        print(viz.render(tr.next_state))
        print(f"step {t + 1}/{len(episode)}   (your td_update)\n")
        print("\n".join(log))
        if t == 0:
            print("\n  ^ the course's example: New V(S_0) = 0 + 0.1 * [1 + 1 * 0 - 0] = 0.1")
            pause(args, "\nPress Enter to step through the rest... ")
        elif args.delay > 0:
            import time

            time.sleep(args.delay)
    print()
    print(viz.side_by_side(viz.format_values(V, "V after one TD episode"),
                           viz.format_values(call(t5.mc_episode_update, np.zeros(N_STATES), episode, gamma, lr, True, None),
                                             "V after one MC episode")))
    print("\nQuestion: why do TD and MC disagree after the same episode? (Hint: what did TD use for V(S_t+1)?)")


def cmd_manual(args):
    t5, t6 = load("t5_mc_prediction", args), load("t6_td_prediction", args)
    env = make_env(args)
    keys = {"w": UP, "s": DOWN, "a": LEFT, "d": RIGHT}
    alpha, gamma = args.alpha, args.gamma
    V_td, V_mc = np.zeros(N_STATES), np.zeros(N_STATES)

    def tables():
        return viz.side_by_side(viz.format_values(V_td, "V from TD (updates every step)"),
                                viz.format_values(V_mc, "V from MC (updates at episode end)"))

    episode_no = 0
    while True:
        episode_no += 1
        state, episode, log, done = env.reset(), [], [], False
        while not done:
            viz.clear()
            print(f"Episode {episode_no}   step {len(episode)}/{env.max_steps}   alpha={alpha:g}  gamma={gamma:g}  slip={env.slip:g}\n")
            print(viz.render(state))
            print("\n".join(log[-5:]) + "\n")
            print(tables())
            key = input("\nMove with w/a/s/d then Enter (q quits): ").strip().lower()
            if key == "q":
                return
            if key not in keys:
                continue
            action = keys[key]
            ns, r, term, trunc = env.step(action)
            episode.append(Transition(state, action, r, ns, term, trunc))
            old, nxt = V_td[state], V_td[ns]
            new = call(t6.td_update, V_td, state, r, ns, term, alpha, gamma)
            log.append(f"  {ACTION_NAMES[action]:>5}: reward {r:+g}   TD: " + td_line(state, r, ns, term, old, nxt, new, alpha, gamma))
            state, done = ns, term or trunc
        call(t5.mc_episode_update, V_mc, episode, gamma, alpha, True, None)
        G0 = sum(gamma**k * tr.reward for k, tr in enumerate(episode))
        viz.clear()
        print(viz.render(state))
        print("\n".join(log[-5:]))
        why = "The cat got you (terminated)." if episode[-1].terminated else "Time's up (truncated)."
        print(f"\n{why}  G_0 = {G0:+g}.  MC has now updated every tile you visited.\n")
        print(tables())
        if input("\nEnter for another episode, q to quit: ").strip().lower() == "q":
            return


def cmd_random(args):
    t3 = load("t3_policies", args)
    rng = np.random.default_rng(args.seed)
    env = make_env(args)
    policy = t3.random_policy
    try:
        policy(START, rng)
    except NotImplementedError:
        print("(Task 3 random_policy not done yet, so using the simulator's own random policy)")
        policy = uniform_random
    for i in range(args.episodes):
        episode = rollout(env, policy, rng)
        viz.animate_episode(episode, args.delay, f"Random policy, episode {i + 1}/{args.episodes}")
        print(f"\nG_0 (gamma=1) = {sum(tr.reward for tr in episode):+g}")
        if i + 1 < args.episodes:
            pause(args)


def cmd_predict(args):
    t3, t5, t6 = load("t3_policies", args), load("t5_mc_prediction", args), load("t6_td_prediction", args)
    env = make_env(args)
    exact = me.expected_return(env, np.full((N_STATES, N_ACTIONS), 1 / N_ACTIONS), args.gamma)
    alpha = None if args.alpha <= 0 else args.alpha
    mc_curve, td_curve = [], []
    print(f"Learning V for the RANDOM policy over {args.episodes} episodes (alpha={alpha}, gamma={args.gamma:g})...")
    V_mc = call(t5.mc_prediction, env, t3.random_policy, args.episodes, args.gamma, alpha, True,
                np.random.default_rng(args.seed), lambda V: mc_curve.append(V[START]))
    V_td = call(t6.td_prediction, env, t3.random_policy, args.episodes, args.gamma, alpha or 0.05,
                np.random.default_rng(args.seed), lambda V: td_curve.append(V[START]))
    no_limit = me.expected_return(env, np.full((N_STATES, N_ACTIONS), 1 / N_ACTIONS), args.gamma, horizon=5000)
    print()
    print(viz.side_by_side(viz.format_values(V_mc, "Monte Carlo"),
                           viz.format_values(exact, f"True: {env.max_steps}-step episode from here")))
    print()
    print(viz.side_by_side(viz.format_values(V_td, "TD(0)"),
                           viz.format_values(no_limit, "True: NO time limit")))
    print(f"\nV(start):  MC {V_mc[START]:+.3f} vs true-with-limit {exact[START]:+.3f}"
          f"   |   TD {V_td[START]:+.3f} vs true-without-limit {no_limit[START]:+.3f}")
    print(f"""
Why they disagree: at the {env.max_steps}-step time limit the episode is TRUNCATED, not terminated, so TD
bootstraps from V(next tile) as if the mouse kept walking. Its values describe a world with NO time
limit, where the random mouse eventually walks into the cat. MC only averages returns it actually saw,
so it learns the value under the limit. (Only the start tile has a clean with-limit truth; the
other tiles get reached at different times, and the time remaining matters.)
Try: --max-steps 200 (the two truths converge), or --gamma 0.5 (the limit barely matters).""")
    if not args.no_plot:
        p1 = viz.plot_curves({"Monte Carlo": mc_curve, "TD(0)": td_curve}, "Estimate of V(start) while learning",
                             "V(start)", "predict_curves.png",
                             hlines=[(exact[START], "true, with time limit"), (no_limit[START], "true, no time limit")])
        p2 = viz.plot_values({"Monte Carlo": V_mc, "TD(0)": V_td}, "predict_values.png")
        print(f"\nSaved {p1}\nSaved {p2}")
        viz.show()


def cmd_control(args):
    names = list(CONTROL) if args.algo == "all" else [args.algo]
    results = {}
    for name in names:
        module, fn_name, label = CONTROL[name]
        fn = getattr(load(module, args), fn_name)
        print(f"Training {label} for {args.episodes} episodes "
              f"(alpha={args.alpha:g}, gamma={args.gamma:g}, epsilon={args.epsilon:g})...")
        Q, returns = call(fn, make_env(args), args.episodes, args.gamma, args.alpha, args.epsilon, np.random.default_rng(args.seed))
        results[label] = (Q, returns)
        print(viz.format_q(Q, f"\n{label}: Q(s, a)"))
        print(viz.format_policy(Q, f"{label}: greedy policy argmax_a Q(s, a)"))
        print(f"average return, last 100 episodes: {np.mean(returns[-100:]):+.2f}\n")

    if not args.no_plot:
        paths = [viz.plot_curves({k: r for k, (_, r) in results.items()}, "Return per episode (epsilon-greedy, while learning)",
                                 "return", "control_curves.png", smooth=max(1, args.episodes // 50))]
        paths += [viz.plot_q(Q, label, f"q_{name}.png") for name, (label, (Q, _)) in zip(names, results.items())]
        print("Saved " + "\nSaved ".join(paths))

    for label, (Q, _) in results.items():
        pause(args, f"\nPress Enter to watch the GREEDY {label} mouse... ")
        episode = rollout(make_env(args), greedy(Q))
        viz.animate_episode(episode, args.delay, f"{label}: greedy policy (no exploration)")
        print(f"\nTotal reward: {sum(tr.reward for tr in episode):+g}   (best possible: +{args.max_steps})")
    if not args.no_plot:
        viz.show()


# -----------------------------------------------------------------------------


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--solutions", action="store_true", help="use the reference solutions instead of tasks/")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--slip", type=float, default=0.0, help="chance the mouse slips into a random action")
    common.add_argument("--max-steps", type=int, default=10, help="episode time limit")
    common.add_argument("--delay", type=float, default=0.5, help="seconds between animation frames")
    common.add_argument("--no-pause", action="store_true", help="don't wait for Enter")
    common.add_argument("--no-plot", action="store_true", help="skip matplotlib plots")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("walkthrough", parents=[common], help="the course's MC and TD examples").set_defaults(fn=cmd_walkthrough)

    p = sub.add_parser("manual", parents=[common], help="control the mouse with w/a/s/d")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=1.0)
    p.set_defaults(fn=cmd_manual)

    p = sub.add_parser("random", parents=[common], help="watch random episodes")
    p.add_argument("--episodes", type=int, default=3)
    p.set_defaults(fn=cmd_random)

    p = sub.add_parser("predict", parents=[common], help="MC vs TD prediction of V for the random policy")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05, help="step size; 0 = sample average for MC (TD then uses 0.05)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("control", parents=[common], help="learn Q with MC / SARSA / Q-learning")
    p.add_argument("algo", choices=[*CONTROL, "all"])
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.set_defaults(fn=cmd_control)

    args = parser.parse_args()
    try:
        args.fn(args)
    except NotReady as e:
        print(f"\nThis command needs code you haven't written yet: {e}.\n"
              f"Implement it in tasks/ (check with `python check.py`), or add --solutions to use the reference code.")
        sys.exit(1)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
