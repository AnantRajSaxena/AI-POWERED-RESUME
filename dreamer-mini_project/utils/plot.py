import matplotlib.pyplot as plt


def plot_rewards(rewards):

    plt.plot(rewards)

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.title("Learning Curve")

    plt.show()