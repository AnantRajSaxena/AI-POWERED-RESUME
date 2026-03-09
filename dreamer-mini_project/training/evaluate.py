def evaluate_agent(env, encoder, actor, episodes=5):

    total_rewards = []

    for episode in range(episodes):

        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:

            import torch

            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            z = encoder(obs_tensor)

            action = actor(z)

            action = action.detach().numpy()[0]

            # convert continuous output to discrete action
            action = 1 if action > 0 else 0

            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            episode_reward += reward

            obs = next_obs

        total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)

    return avg_reward