def compute_return(rewards, gamma):
    G = 0.0
    for k, reward in enumerate(rewards):
        G += gamma**k * reward
    return G


def compute_all_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    G = 0.0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns
