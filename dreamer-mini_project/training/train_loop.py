import torch


def train_step(encoder, actor, critic, world_model, optimizer, obs):

    # encode observation
    z = encoder(obs)

    # actor chooses action
    action = actor(z)

    # critic evaluates state
    value = critic(z)

    # simple dummy loss (for now)
    loss = value.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()