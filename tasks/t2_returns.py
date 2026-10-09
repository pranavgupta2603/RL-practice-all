"""
TASK 2: The return G_t
======================

The return is the total (discounted) reward from time t until the end of the episode:

    G_t = R_{t+1} + gamma * R_{t+2} + gamma^2 * R_{t+3} + ...

`rewards` is the list [R_1, R_2, ..., R_T], so rewards[0] is R_1, the reward
received after the FIRST action.

Course example: rewards = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0], gamma = 1  ->  G_0 = 3

Returns satisfy a recursion, which is the key idea behind TD learning later:

    G_t = R_{t+1} + gamma * G_{t+1}        (with G_T = 0 after the last step)

Check:  python check.py 2
"""


def compute_return(rewards: list[float], gamma: float) -> float:
    """Return G_0 for the whole reward list. An empty list has return 0."""
    # TODO
    raise NotImplementedError("Task 2: compute_return")


def compute_all_returns(rewards: list[float], gamma: float) -> list[float]:
    """
    Return [G_0, G_1, ..., G_{T-1}], one return for every time step.

    Do it in ONE backward pass using G_t = R_{t+1} + gamma * G_{t+1}.
    Don't call compute_return for every t; that's O(T^2).
    """
    # TODO
    raise NotImplementedError("Task 2: compute_all_returns")
