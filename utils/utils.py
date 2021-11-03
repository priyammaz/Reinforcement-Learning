import matplotlib.pyplot as plt
import numpy as np

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


from collections import deque


class TimeCapsule():
    def __init__(self, max_mems):
        self.max_mems = max_mems  # Total number of memories we will store, more mems is more ram
        self.stored_mems = 0  # To iterate and keep track of number of memories

        """
        Look up the deque data structure, it is basically a double linked list. The reason we will use this is
        we can set a max size of the list. Lets say we set the max to 3. As we append memories on we will fill the list
        like expected, but once we append the 4 item on, the 1st item gets removed from the beginning, thus always keeping
        3 in the list. The actual value we choose will be the size of the total number of memories we want to store. 
        """
        self.curr_observation = deque(maxlen=self.max_mems)
        self.action = deque(maxlen=self.max_mems)
        self.reward = deque(maxlen=self.max_mems)
        self.new_observation = deque(maxlen=self.max_mems)
        self.end_game = deque(maxlen=self.max_mems)

    def memorize(self, observation, action, reward, done, new_observation):
        """
        After every iteration, we want to store the memory of what happened, so we can
        then randomly sample it later
        """
        self.curr_observation.append(observation)
        self.action.append(action)
        self.reward.append(reward)
        self.new_observation.append(new_observation)
        self.end_game.append(done)
        self.stored_mems += 1


    def recall(self, batch_size):
        if self.stored_mems >=  self.max_mems:
          total_available =  self.max_mems
        else:
          total_available = self.stored_mems

        random_chosen = np.random.choice(total_available, batch_size, replace=False)

        curr_observations = np.array(self.curr_observation)[random_chosen].astype(np.float32)
        actions = np.array(self.action)[random_chosen].astype(np.int64)
        rewards = np.array(self.reward)[random_chosen].astype(np.float32)
        new_observations = np.array(self.new_observation)[random_chosen].astype(np.float32)
        end_games = np.array(self.end_game)[random_chosen].astype(np.bool)

        return curr_observations, actions, rewards, new_observations, end_games



