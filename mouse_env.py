"""
Mouse & Cheese gridworld: the example from the Hugging Face Deep RL course, Unit 2
https://huggingface.co/learn/deep-rl-course/en/unit2/mc-vs-td

          tile 1     tile 2     tile 3
        +----------+----------+----------+
  top   |   cat    |  cheese  |          |
        +----------+----------+----------+
 bottom |          |  cheese  |  mouse   |
        +----------+----------+----------+

Rules
- The mouse always starts on tile 3 bottom.
- Actions: Up, Down, Left, Right. Walking into a wall leaves the mouse where it is.
- Moving onto a cheese tile gives +1. Cheese is unlimited, so the state is just the tile.
- Moving onto the cat gives -10 and the episode TERMINATES (a real end state, worth 0 afterwards).
- After `max_steps` (10) steps the episode is TRUNCATED (a time limit, not a real end state).
- Every other move gives 0.

This file is complete. You don't need to edit it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_ROWS, N_COLS = 2, 3
N_STATES = N_ROWS * N_COLS
N_ACTIONS = 4

UP, DOWN, LEFT, RIGHT = range(N_ACTIONS)
ACTION_NAMES = ("Up", "Down", "Left", "Right")
ACTION_ARROWS = ("↑", "↓", "←", "→")
_MOVES = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def to_state(row: int, col: int) -> int:
    return row * N_COLS + col


def to_pos(state: int) -> tuple[int, int]:
    return divmod(state, N_COLS)


def state_name(state: int) -> str:
    row, col = to_pos(state)
    return f"tile {col + 1} {'top' if row == 0 else 'bottom'}"


START = to_state(1, 2)
CAT = to_state(0, 0)
CHEESE = frozenset({to_state(0, 1), to_state(1, 1)})
TERMINAL_STATES = frozenset({CAT})

CHEESE_REWARD = 1.0
CAT_REWARD = -10.0


@dataclass(frozen=True)
class Transition:
    """One step of experience (S_t, A_t, R_{t+1}, S_{t+1}) plus how the episode ended."""

    state: int
    action: int
    reward: float
    next_state: int
    terminated: bool
    truncated: bool


def _move(state: int, action: int) -> int:
    row, col = to_pos(state)
    dr, dc = _MOVES[action]
    r, c = row + dr, col + dc
    if 0 <= r < N_ROWS and 0 <= c < N_COLS:
        return to_state(r, c)
    return state


def _outcome(state: int, next_state: int) -> tuple[float, bool]:
    if next_state == CAT:
        return CAT_REWARD, True
    if next_state != state and next_state in CHEESE:
        return CHEESE_REWARD, False
    return 0.0, False


class MouseEnv:
    """
    Gym-style API:
        state = env.reset()
        next_state, reward, terminated, truncated = env.step(action)

    slip: probability that the chosen action is swapped for a random one, which
          makes the world stochastic.
    """

    def __init__(self, max_steps: int = 10, slip: float = 0.0, seed: int | None = None):
        self.max_steps = max_steps
        self.slip = slip
        self.rng = np.random.default_rng(seed)
        self.state = START
        self.steps = 0
        self.done = True

    def reset(self) -> int:
        self.state = START
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, bool]:
        if self.done:
            raise RuntimeError("The episode is over. Call env.reset() first.")
        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be 0..{N_ACTIONS - 1}, got {action!r}")
        if self.slip > 0 and self.rng.random() < self.slip:
            action = int(self.rng.integers(N_ACTIONS))
        next_state = _move(self.state, action)
        reward, terminated = _outcome(self.state, next_state)
        self.steps += 1
        truncated = not terminated and self.steps >= self.max_steps
        self.state = next_state
        self.done = terminated or truncated
        return next_state, reward, terminated, truncated

    def model(self, state: int, action: int) -> list[tuple[float, int, float, bool]]:
        """
        The true dynamics as (probability, next_state, reward, terminated) tuples.
        MC and TD never need this; it's only here for planning exercises and for
        computing exact answers to compare against.
        """
        probs = np.full(N_ACTIONS, self.slip / N_ACTIONS)
        probs[action] += 1.0 - self.slip
        outcomes: dict[tuple[int, float, bool], float] = {}
        for a, p in enumerate(probs):
            if p == 0:
                continue
            next_state = _move(state, a)
            reward, terminated = _outcome(state, next_state)
            key = (next_state, reward, terminated)
            outcomes[key] = outcomes.get(key, 0.0) + p
        return [(p, ns, r, term) for (ns, r, term), p in outcomes.items()]


def expected_return(env: MouseEnv, policy_probs: np.ndarray, gamma: float, horizon: int | None = None) -> np.ndarray:
    """
    Exact E[G_0] for an episode of `horizon` steps (default env.max_steps) that starts
    on each tile, under policy_probs[state, action]. Uses finite-horizon dynamic programming.
    A large horizon gives the value with no time limit.
    """
    V = np.zeros(N_STATES)
    for _ in range(env.max_steps if horizon is None else horizon):
        new_V = np.zeros(N_STATES)
        for s in range(N_STATES):
            if s in TERMINAL_STATES:
                continue
            for a in range(N_ACTIONS):
                for p, ns, r, term in env.model(s, a):
                    new_V[s] += policy_probs[s, a] * p * (r + (0.0 if term else gamma * V[ns]))
        V = new_V
    return V
