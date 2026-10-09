# Mouse & Cheese: Monte Carlo vs Temporal Difference, from scratch

A hands-on companion to the Hugging Face Deep RL course page
[Monte Carlo vs Temporal Difference Learning](https://huggingface.co/learn/deep-rl-course/en/unit2/mc-vs-td).
You get the course's mouse world as a simulator, and you write every RL piece yourself:
returns, policies, V and Q, MC, TD, SARSA, and Q-learning.

Only `numpy` and `matplotlib` are needed.

```bash
python check.py                 # grades your tasks (all ⬜ TODO at the start)
python play.py walkthrough      # the course's worked examples, run with YOUR code
python play.py -h               # all simulator commands
```

## Files

| File | What it is |
|---|---|
| `tasks/t1_*.py ... t9_*.py` | **You edit these.** Each one explains a concept, then has `TODO`s. |
| `check.py` | Grades the tasks: `python check.py 5` checks only Task 5. |
| `play.py` | The simulator. Every command runs your code from `tasks/`. |
| `mouse_env.py` | The environment (complete). |
| `visualize.py` | Terminal rendering and plots (complete). |
| `solutions/` | Reference answers. Try not to peek; `--solutions` on any command uses them. |

## The world

```
          tile 1     tile 2     tile 3
        +----------+----------+----------+
  top   |   🐱     |   🧀     |          |
        +----------+----------+----------+
 bottom |          |   🧀     |   🐭     |   <- start
        +----------+----------+----------+
```

- Actions: Up / Down / Left / Right (0 / 1 / 2 / 3). Walking into a wall = stay put.
- Step onto cheese: **+1** (cheese never runs out, so the state is just the tile).
- Step onto the cat: **-10**, and the episode **terminates**.
- After **10 steps** the episode is **truncated** (a time limit).
- `--slip 0.2` makes the world stochastic: 20% of moves go in a random direction.

The best possible episode is +10: Left, then bounce Up/Down between the two cheese tiles.

## Cheat sheet

### Returns and values

**Return** (total discounted reward from time $t$):

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

**Recursive form** (used for the backward pass in Task 2, and the seed of TD):

$$
G_t = R_{t+1} + \gamma\, G_{t+1}
$$

**State value** and **state-action value** under a policy $\pi$:

$$
V_\pi(s) = \mathbb{E}_\pi\left[\, G_t \mid S_t = s \,\right]
\qquad\qquad
Q_\pi(s,a) = \mathbb{E}_\pi\left[\, G_t \mid S_t = s,\ A_t = a \,\right]
$$

### How V and Q relate (Task 4)

**V from Q** (average over what the policy would do):

$$
V(s) = \sum_{a} \pi(a \mid s)\, Q(s,a)
$$

**Q from V** (one-step lookahead; needs the model $p$):

$$
Q(s,a) = \sum_{s',\,r} p(s', r \mid s, a)\, \big[\, r + \gamma\, V(s') \,\big]
$$

### Prediction: learning $V$ (Tasks 5 and 6)

**Monte Carlo**: wait for the episode to end, then move toward the real return:

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[\, G_t - V(S_t) \,\big]
$$

**TD(0)**: after one step, move toward an estimated return:

$$
V(S_t) \leftarrow V(S_t) + \alpha \big[\, \underbrace{R_{t+1} + \gamma\, V(S_{t+1})}_{\text{TD target}} - V(S_t) \,\big]
$$

$$
\delta_t = R_{t+1} + \gamma\, V(S_{t+1}) - V(S_t) \qquad \text{(TD error)}
$$

### Control: learning $Q$ (Tasks 7 to 9)

**Monte Carlo control**:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[\, G_t - Q(S_t, A_t) \,\big]
$$

**SARSA** (on-policy: uses the next action $A_{t+1}$ you actually take):

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[\, R_{t+1} + \gamma\, Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \,\big]
$$

**Q-learning** (off-policy: uses the best next action):

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \big[\, R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \,\big]
$$

**Epsilon-greedy policy** with $|\mathcal{A}| = 4$ actions:

$$
\pi(a \mid s) =
\begin{cases}
1 - \varepsilon + \dfrac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\[6pt]
\dfrac{\varepsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}
$$

(With ties, the $1 - \varepsilon$ share is split evenly among the tied best actions.)

### Terminated vs truncated

If the step **terminated** (the cat), there is no future, so drop the bootstrap term:

$$
\text{target} = R_{t+1}
$$

If it was only **truncated** (time limit), the next state still has a future, so you still bootstrap with $\gamma V(S_{t+1})$ or $\gamma Q(S_{t+1}, \cdot)$.

## Tasks

Do them in order; later tasks import your earlier code. After each one, run `python check.py N`, then the "see it" command.

| # | File | You implement | You learn | See it |
|---|---|---|---|---|
| 1 | `t1_episode.py` | `run_episode` | the agent-environment loop, Transitions, terminated vs truncated | `play.py random` |
| 2 | `t2_returns.py` | `compute_return`, `compute_all_returns` | returns, discounting, the backward recursion | `play.py walkthrough` |
| 3 | `t3_policies.py` | random, greedy, epsilon-greedy | exploration vs exploitation, tie-breaking | `play.py random` |
| 4 | `t4_value_relations.py` | `epsilon_greedy_probs`, `v_from_q`, `q_from_v` | how V and Q relate; why control learns Q | |
| 5 | `t5_mc_prediction.py` | `mc_update`, `mc_episode_update`, `mc_prediction` | MC, first- vs every-visit, constant alpha vs sample average | `play.py walkthrough`, `play.py manual` |
| 6 | `t6_td_prediction.py` | `td_update`, `td_prediction` | bootstrapping, TD target / TD error, online updates | `play.py manual`, `play.py predict` |
| 7 | `t7_mc_control.py` | `mc_control` | policy improvement with Q, on-policy MC control | `play.py control mc` |
| 8 | `t8_sarsa.py` | `sarsa_update`, `sarsa` | TD control, on-policy | `play.py control sarsa` |
| 9 | `t9_q_learning.py` | `q_learning_update`, `q_learning` | off-policy TD control (the course's next unit) | `play.py control all` |

The checker reproduces the course's numbers: $G_0 = 3$, MC gives $V(S_0) = 0.3$, TD gives $V(S_0) = 0.1$.

## The simulator

```bash
python play.py walkthrough            # course example: MC part, then TD step by step
python play.py manual                 # you are the policy: w/a/s/d. TD updates every step, MC at the end
python play.py random --episodes 5    # watch a random mouse
python play.py predict                # MC vs TD learning V(random policy), with exact answers + plots
python play.py control all            # MC control vs SARSA vs Q-learning, then watch each greedy mouse
```

Useful flags: `--slip 0.2`, `--alpha`, `--gamma`, `--epsilon`, `--episodes`, `--max-steps`,
`--delay 0` (no animation), `--no-pause`, `--no-plot`, `--seed`, `--solutions`.
Plots are saved to `plots/`.

## Experiments (do these after the tasks)

1. **Bias vs variance.** `python play.py predict` and open `plots/predict_curves.png`. MC's estimate
   is noisy but centered on the truth; TD's is smooth. Why does MC have higher variance? (A return
   adds up 10 random rewards; a TD target has only one.)
2. **TD learns a different number here.** In `predict`, TD converges to the value *with no time limit*,
   MC to the value *with* the 10-step limit. That's because TD bootstraps through truncation.
   Rerun with `--max-steps 200` and with `--gamma 0.5`. When do they agree? What would you add to
   the state to make the time limit "Markov"? (Hint: steps remaining.)
3. **The terminal-state bug.** In your `td_update`, temporarily bootstrap even when `terminated`,
   and set `V[CAT] = 5` before learning. What happens to the tiles next to the cat?
4. **Step size.** `predict --alpha 0.5` vs `--alpha 0.01` vs `--alpha 0` (MC sample average).
   Speed vs noise.
5. **SARSA plays it safe.** `python play.py control all --epsilon 0.3`. SARSA learns the value of the
   *exploring* policy, so being next to the cat looks dangerous; Q-learning learns the *greedy* value
   and ignores that risk. Look at tile 2 bottom: MC and SARSA stop going Up (next to the cat) and settle
   for the safer +5 loop, while Q-learning still goes Up. (This is the "cliff walking" lesson.)
6. **Exploration traps.** Try `control mc --seed 2`. Sometimes the greedy mouse settles for +5
   (bouncing between tile 2 bottom and tile 3 bottom) because it rarely tried "Up" from tile 2 bottom.
   Fix it with a larger epsilon, an epsilon that decays over episodes, or optimistic initial Q-values.
7. **Stochastic world.** Add `--slip 0.2` to `predict` and `control`. Which methods cope, and how do
   the learned values change?
8. **gamma = 1 with loops.** `control qlearning --gamma 1.0`. Watch the Q-values: why do they keep
   growing? (An endless cheese loop plus bootstrapping through truncation.)

## Bonus tasks

- **n-step TD**: target $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$. $n = 1$ is TD(0); $n = \infty$ is MC.
- **Expected SARSA**: target $R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1})\, Q(S_{t+1}, a)$ (your Task 4 `epsilon_greedy_probs`).
- **Time-aware state**: make the state `(tile, steps_left)` and redo experiment 2.
- **Consumable cheese**: cheese disappears once eaten. The state must now include which cheese remains.
- **Iterative policy evaluation**: use `env.model` to compute V exactly with dynamic programming, and compare with MC/TD.
