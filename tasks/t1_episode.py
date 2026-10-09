"""
TASK 1: The agent-environment loop
==================================

Before learning anything, the agent has to generate experience. One episode is:

    S_0 --A_0--> R_1, S_1 --A_1--> R_2, S_2 ... until the episode ends

At each step:
    1. the policy picks an action from the current state
    2. the environment returns (next_state, reward, terminated, truncated)
    3. you record the step as a Transition(state, action, reward, next_state, terminated, truncated)
    4. stop if terminated (the cat got the mouse) or truncated (10-step time limit)

Monte Carlo methods need this whole list before they can learn anything.

Check:  python check.py 1
"""

from mouse_env import MouseEnv, Transition  # noqa: F401  (Transition is what you'll build)


def run_episode(env: MouseEnv, policy, rng) -> list[Transition]:
    """
    Play one full episode and return the list of Transitions in time order.

    env:    a MouseEnv. Call env.reset() to get the start state, then env.step(action).
    policy: a function policy(state, rng) -> action
    rng:    a numpy Generator; just pass it through to the policy.

    Hint: a `while not done:` loop. Remember to move to the next state at the end of each step.
    """
    # TODO
    raise NotImplementedError("Task 1: run_episode")
