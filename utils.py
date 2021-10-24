import matplotlib.pyplot as plt

# Add in graph
def plot_reward(episodes, avg_score, min_score, max_score, game_name):
  if len(episodes) == len(avg_score):
    plt.plot(episodes, avg_score, label='Average')
    plt.plot(episodes, min_score, label='Minimum')
    plt.plot(episodes, max_score, label='Maximum')
    plt.legend(loc=2)
    plt.xlabel('Training Episodes')
    plt.ylabel('Reward per Episode')
    plt.title('Reward on '+game_name)
    plt.show()