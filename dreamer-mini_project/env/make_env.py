import gymnasium as gym

def make_env():
    env = gym.make("CartPole-v1")
    return env