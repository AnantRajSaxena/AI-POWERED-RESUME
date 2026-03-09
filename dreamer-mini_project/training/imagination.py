import torch


def imagine_trajectory(actor, world_model, start_state, horizon=5):

    state = start_state

    hidden = torch.zeros(1, 1, 128)

    imagined_states = []
    imagined_rewards = []

    for t in range(horizon):

        # actor chooses action
        action = actor(state)

        # world model predicts next state
        next_state, reward, hidden = world_model(state, action, hidden)

        imagined_states.append(next_state)
        imagined_rewards.append(reward)

        state = next_state

    return imagined_states, imagined_rewards