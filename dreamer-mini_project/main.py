import torch

from env.make_env import make_env
from models.encoder import Encoder
from models.actor import Actor
from training.evaluate import evaluate_agent
from utils.plot import plot_rewards

# create environment
env = make_env()

# model dimensions
obs_dim = 4
latent_dim = 32
action_dim = 1

encoder = Encoder(obs_dim, latent_dim)
actor = Actor(latent_dim, action_dim)

reward_history = []

for episode in range(10):

    avg_reward = evaluate_agent(env, encoder, actor)

    reward_history.append(avg_reward)

    print("Episode:", episode, "Reward:", avg_reward)

plot_rewards(reward_history)