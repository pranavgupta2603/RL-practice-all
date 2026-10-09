from mouse_env import Transition


def run_episode(env, policy, rng):
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = policy(state, rng)
        next_state, reward, terminated, truncated = env.step(action)
        episode.append(Transition(state, action, reward, next_state, terminated, truncated))
        state = next_state
        done = terminated or truncated
    return episode
