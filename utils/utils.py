import matplotlib.pyplot as plt
from collections import deque
import cv2
import gym
import numpy as np

def plot_reward(episodes, avg_scores, epsilon, game_name):
  fig, ax1 = plt.subplots()
  if len(episodes) == len(avg_scores):
    color = 'tab:blue'
    ax1.set_xlabel(' Training Episode')
    ax1.set_ylabel('Reward Score per Episode', color=color)
    ax1.plot(episodes, avg_scores,color=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(episodes, epsilon, color=color)
    
    plt.title('Reward on '+game_name)
    fig.tight_layout() 
    plt.show()

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

class TimeCapsule2():
    def __init__(self, max_mems, input_shape):
        self.max_mems = max_mems  # Total number of memories we will store, more mems is more ram
        self.stored_mems = 0  # To iterate and keep track of number of memories

        """
        Look up the deque data structure, it is basically a double linked list. The reason we will use this is
        we can set a max size of the list. Lets say we set the max to 3. As we append memories on we will fill the list
        like expected, but once we append the 4 item on, the 1st item gets removed from the beginning, thus always keeping
        3 in the list. The actual value we choose will be the size of the total number of memories we want to store. 
        """
        self.curr_observation = np.zeros((self.max_mems, *input_shape), dtype=np.float32)
        self.action = np.zeros(self.max_mems, dtype=np.int64)
        self.reward = np.zeros(self.max_mems, dtype=np.float32)
        self.new_observation = np.zeros((self.max_mems, *input_shape), dtype=np.float32)
        self.end_game = np.zeros(self.max_mems, dtype=np.bool)

    def memorize(self, observation, action, reward, done, new_observation):
        """
        After every iteration, we want to store the memory of what happened, so we can
        then randomly sample it later
        """
        index = self.stored_mems % self.max_mems
        self.curr_observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.new_observation[index] = new_observation
        self.end_game[index] = done
        self.stored_mems += 1

    def recall(self, batch_size):
        if self.stored_mems >=  self.max_mems:
          total_available =  self.max_mems
        else:
          total_available = self.stored_mems

        random_chosen = np.random.choice(total_available, batch_size, replace=False)

        curr_observations = self.curr_observation[random_chosen]
        actions = self.action[random_chosen]
        rewards = self.reward[random_chosen]
        new_observations = self.new_observation[random_chosen]
        end_games = self.end_game[random_chosen]

        return curr_observations, actions, rewards, new_observations, end_games

class FlickerReductionAndRepeat(gym.Wrapper):
    """
    The purpose of this class is two-fold:

    Frame Skipping:
    The atari games have a  built in stochastic nature. If given an action, the environment can randomly
    repeat that action 2-4 times. We will remove this by using the "noframeskip" in the namespace of the game and we will
    have a parameter called repeat that will determine the number of times we repeat an action. The reason we do this is,
    frame to frame, there is very little difference in the images. If we repeat an action we can get more change in the
    observations which will allow the network to learn a sense of position and direction.

    FlickerReduction:
    Some games are half-rendered, where every other frame, only half the objects in the game will be visible. To the human
    eye, the frames are so quick that it is indistinguishable, but the network will quickly get confused about the location of
    objects. To fix this we will take two consecutive frames and take the max of them as a single observation.
    """
    def __init__(self, repeat=4, env=None):
        super(FlickerReductionAndRepeat, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.high.shape # Gives us the initial array shape
        self.flicker_buffer = deque(maxlen=2)

    def step(self, action):
        """
        We will update the step to take a repeat amount of steps and append the last two frames available for maxing
        """
        total_rewards = 0.0
        done = False
        for _ in range(self.repeat):
            observation, reward, done, info = self.env.step(action)
            total_rewards += reward
            self.flicker_buffer.append(observation)
            if done:
                break

        maxxed_frame = np.maximum(np.array(self.flicker_buffer[0]), np.array(self.flicker_buffer[1])) # Take the max of the last two consective frames
        return maxxed_frame, total_rewards, done, info

    def reset(self):
        """
        This function will reset the environment before the start of every new game. This means clearing the flicker_buffer
        and appending on the new observation retrieved from reset
        """
        observation = self.env.reset()
        self.flicker_buffer.clear()
        self.flicker_buffer.append(observation)

        return observation

class PreprocessAndStackFrames(gym.ObservationWrapper):
    """
    This class will convert our image to black and white as well as resize it to the wanted shape

    Once we have reshaped our image, we will add it to a buffer to stack multiple images together. By doing so,
    the model will be able to understand directionality of the moving objects in the game.

    Number of images we will stack is denoted by the variable phi
    """
    def __init__(self, shape, phi, env=None):
        super(PreprocessAndStackFrames, self).__init__(env)

        ### Setup the Observation Space ###
        self.shape = (shape[2], shape[0], shape[1])  # Channels first
        highest_frame = np.ones(self.shape)  # highest possible value will be 1 for pixel value
        lowest_frame = np.zeros(self.shape)  # lowest possible value will be 0 for pixel value

        self.observation_space = gym.spaces.Box(low=lowest_frame.repeat(phi, axis=0),
                                                high=highest_frame.repeat(phi, axis=0),
                                                dtype=np.float64)

        ### Setup the Stacking Frames Mechanism ###
        self.frame_stack = deque(maxlen=phi)

    def preprocess_image(self, observation):
        """
        Takes image and preprocesses it into wanted shape and channels (black and white)
        """
        observation = observation.astype(np.float32) # Make sure input is a float32 for cv2 bit-depth requirements
        black_and_white = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(black_and_white, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_observation = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_observation = new_observation / 255 # scale to (0, 1)
        return new_observation

    def reset(self):
        """
        Reset the frame stack deque by emptying and adding in the intitial observation, plus zeroes in remaining positions
        """
        self.frame_stack.clear()
        observation = self.env.reset()
        observation = self.preprocess_image(observation)

        # Fill the first index in DeQue with observation
        self.frame_stack.append(observation)

        # Fill the remaining positions in deque with zeros
        for _ in range(self.frame_stack.maxlen - 1):
            self.frame_stack.append(np.zeros_like(observation))

        return np.array(self.frame_stack).reshape(self.observation_space.high.shape)

    def observation(self, observation):
        """
        Append observation and return array
        """
        observation = self.preprocess_image(observation)
        self.frame_stack.append(observation)
        return np.array(self.frame_stack).reshape(self.observation_space.high.shape)


def build_env(env_name, repeat=4, phi=4, shape=(84,84,1)):
    """
    Build the environment with:
    repeat: number of repeat actions
    phi: number of frames to stack
    shape: final shape to feed into network
    """

    env = gym.make(env_name)
    env = FlickerReductionAndRepeat(repeat=repeat, env=env)
    env = PreprocessAndStackFrames(shape=shape, phi=phi, env=env)
    return env


if __name__ == "__main__":
    env = build_env("PongNoFrameskip-v4")

    for i in range(10):
        if i % 1 == 0:
            render = True
        else:
            render = True
        done = False
        score = 0
        observation = env.reset()
        while not done:
            if render:
                env.render()
            action = env.action_space.sample() # get current action for the current observation in q_eval
            new_observation, reward, done, info = env.step(action)
            score += reward




































