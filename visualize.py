"""Terminal rendering and matplotlib plots for the Mouse & Cheese world. Complete; no edits needed."""

from __future__ import annotations

import os
import time

import numpy as np

from mouse_env import (
    ACTION_ARROWS,
    ACTION_NAMES,
    CAT,
    CHEESE,
    DOWN,
    LEFT,
    N_ACTIONS,
    N_COLS,
    N_ROWS,
    N_STATES,
    RIGHT,
    START,
    TERMINAL_STATES,
    UP,
    state_name,
    to_pos,
    to_state,
)

ICON_MOUSE, ICON_CHEESE, ICON_CAT = "🐭", "🧀", "🐱"
CELL_W = 14
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def _width(text: str) -> int:
    return sum(2 if ord(ch) > 0x2FFF else 1 for ch in text)


def _center(text: str, width: int = CELL_W) -> str:
    pad = max(width - _width(text), 0)
    left = pad // 2
    return " " * left + text + " " * (pad - left)


def _label(state: int) -> str:
    row, col = to_pos(state)
    return f"{col + 1} {'top' if row == 0 else 'bottom'}"


def _grid(cells: list[list[str]]) -> str:
    sep = "+" + "+".join(["-" * CELL_W] * N_COLS) + "+"
    out = [sep]
    for row in range(N_ROWS):
        for i in range(len(cells[0])):
            out.append("|" + "|".join(_center(cells[to_state(row, col)][i]) for col in range(N_COLS)) + "|")
        out.append(sep)
    return "\n".join(out)


def render(mouse_state: int | None = None) -> str:
    cells = []
    for s in range(N_STATES):
        icons = []
        if s == CAT:
            icons.append(ICON_CAT)
        if s in CHEESE:
            icons.append(ICON_CHEESE)
        if s == mouse_state:
            icons.append(ICON_MOUSE)
        cells.append([" ".join(icons), _label(s)])
    return _grid(cells)


def format_values(V: np.ndarray, title: str = "V(s)") -> str:
    cells = []
    for s in range(N_STATES):
        value = "terminal" if s in TERMINAL_STATES else f"{V[s]:+.3f}"
        cells.append([value, _label(s)])
    return f"{title}\n{_grid(cells)}"


def _best_actions(q: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isclose(q, q.max()))


def format_policy(Q: np.ndarray, title: str = "Greedy policy") -> str:
    cells = []
    for s in range(N_STATES):
        arrows = "terminal" if s in TERMINAL_STATES else " ".join(ACTION_ARROWS[a] for a in _best_actions(Q[s]))
        cells.append([arrows, _label(s)])
    return f"{title}\n{_grid(cells)}"


def format_q(Q: np.ndarray, title: str = "Q(s, a)") -> str:
    lines = [title, f"  {'state':<15}" + "".join(f"{n:>11}" for n in ACTION_NAMES) + "    best"]
    for s in range(N_STATES):
        if s in TERMINAL_STATES:
            lines.append(f"  {state_name(s):<15}{'(terminal)':>12}")
            continue
        best = " ".join(ACTION_ARROWS[a] for a in _best_actions(Q[s]))
        lines.append(f"  {state_name(s):<15}" + "".join(f"{q:>+11.3f}" for q in Q[s]) + f"    {best}")
    return "\n".join(lines)


def side_by_side(*blocks: str, gap: int = 4) -> str:
    split = [b.split("\n") for b in blocks]
    widths = [max(_width(line) for line in lines) for lines in split]
    height = max(len(lines) for lines in split)
    rows = []
    for i in range(height):
        parts = []
        for lines, w in zip(split, widths):
            line = lines[i] if i < len(lines) else ""
            parts.append(line + " " * (w - _width(line)))
        rows.append((" " * gap).join(parts).rstrip())
    return "\n".join(rows)


def clear() -> None:
    print("\033[2J\033[H", end="")


def describe(t: int, tr) -> str:
    end = "  [terminated: the cat got the mouse]" if tr.terminated else ("  [truncated: time limit]" if tr.truncated else "")
    return (
        f"  t={t}: {state_name(tr.state):<14} --{ACTION_NAMES[tr.action]:>5}--> "
        f"{state_name(tr.next_state):<14} reward {tr.reward:+g}{end}"
    )


def animate_episode(episode, delay: float = 0.5, title: str = "") -> None:
    total = 0.0
    log: list[str] = []
    frames = [(START, "")] + [(tr.next_state, "") for tr in episode]
    for i, (state, _) in enumerate(frames):
        if i > 0:
            tr = episode[i - 1]
            total += tr.reward
            log.append(describe(i - 1, tr))
        clear()
        if title:
            print(title + "\n")
        print(render(state))
        print(f"step {i}/{len(episode)}   total reward so far: {total:+g}\n")
        print("\n".join(log))
        if delay > 0:
            time.sleep(delay)


# ----------------------------------------------------------------------------- plots


def _plt():
    import matplotlib.pyplot as plt

    os.makedirs(PLOT_DIR, exist_ok=True)
    return plt


def _save(fig, filename: str) -> str:
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path


def plot_values(values: dict[str, np.ndarray], filename: str) -> str:
    plt = _plt()
    fig, axes = plt.subplots(1, len(values), figsize=(4.2 * len(values), 3.2), squeeze=False)
    finite = [V[s] for V in values.values() for s in range(N_STATES) if s not in TERMINAL_STATES]
    vmax = max(abs(min(finite)), abs(max(finite)), 1e-6)
    for ax, (title, V) in zip(axes[0], values.items()):
        grid = V.reshape(N_ROWS, N_COLS).astype(float).copy()
        mask = np.zeros_like(grid, dtype=bool)
        for s in TERMINAL_STATES:
            mask[to_pos(s)] = True
        ax.imshow(np.ma.masked_array(grid, mask), cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        for s in range(N_STATES):
            r, c = to_pos(s)
            text = "cat\n(terminal)" if s in TERMINAL_STATES else f"{V[s]:+.2f}\n{_label(s)}"
            ax.text(c, r, text, ha="center", va="center", fontsize=10)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("State values V(s)")
    return _save(fig, filename)


def plot_q(Q: np.ndarray, title: str, filename: str) -> str:
    plt = _plt()
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(7, 4.6))
    live = [Q[s, a] for s in range(N_STATES) if s not in TERMINAL_STATES for a in range(N_ACTIONS)]
    vmax = max(abs(min(live)), abs(max(live)), 1e-6)
    cmap = plt.get_cmap("RdYlGn")
    for s in range(N_STATES):
        row, col = to_pos(s)
        y = N_ROWS - 1 - row
        cx, cy = col + 0.5, y + 0.5
        if s in TERMINAL_STATES:
            ax.add_patch(Polygon([(col, y), (col + 1, y), (col + 1, y + 1), (col, y + 1)], facecolor="#bbbbbb", edgecolor="k"))
            ax.text(cx, cy, "cat\n(terminal)", ha="center", va="center")
            continue
        tl, tr_, bl, br = (col, y + 1), (col + 1, y + 1), (col, y), (col + 1, y)
        tris = {UP: [tl, tr_, (cx, cy)], DOWN: [bl, br, (cx, cy)], LEFT: [tl, bl, (cx, cy)], RIGHT: [tr_, br, (cx, cy)]}
        offsets = {UP: (0, 0.32), DOWN: (0, -0.32), LEFT: (-0.3, 0), RIGHT: (0.3, 0)}
        best = set(_best_actions(Q[s]))
        for a, pts in tris.items():
            color = cmap(0.5 + 0.5 * Q[s, a] / vmax)
            ax.add_patch(Polygon(pts, facecolor=color, edgecolor="k", linewidth=0.6))
            dx, dy = offsets[a]
            ax.text(cx + dx, cy + dy, f"{ACTION_ARROWS[a]}{Q[s, a]:+.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold" if a in best else "normal")
        ax.text(cx, y + 0.06, _label(s), ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xlim(0, N_COLS)
    ax.set_ylim(0, N_ROWS)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}: Q(s, a) (bold = greedy action)")
    return _save(fig, filename)


def plot_curves(curves: dict[str, list[float]], title: str, ylabel: str, filename: str,
                smooth: int = 1, hlines: list[tuple[float, str]] = ()) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, ys in curves.items():
        ys = np.asarray(ys, dtype=float)
        if smooth > 1 and len(ys) >= smooth:
            ys = np.convolve(ys, np.ones(smooth) / smooth, mode="valid")
        ax.plot(ys, label=name, linewidth=1.2)
    for (value, label), style in zip(hlines, ["--", ":", "-."]):
        ax.axhline(value, color="k", linestyle=style, linewidth=1, label=label)
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def show() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() != "agg":
        plt.show()
