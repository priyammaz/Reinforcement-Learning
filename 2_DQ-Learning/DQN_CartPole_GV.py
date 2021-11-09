import sys
sys.path.append("Models/model_architecture")
sys.path.append("utils")

import numpy as np
import torch
import gym

from DeepQNetworkDense import DQNDense
from utils import TimeCapsule, plot_reward

class Alfred():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,model_name,
                 total_memories=100000, eps_end=0.01, eps_desc=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_desc = eps_desc
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.total_memories = total_memories
        self.input_dims = [input_dims[0]]
        self.batch_size = batch_size
        self.eps_end = eps_end
        self.model_name = model_name

        self.DQN = DQNDense(self.lr, output_actions=n_actions, input_shape=self.input_dims, model_name=self.model_name)

        self.timecapsule = TimeCapsule(self.total_memories)

    def store_transition(self, observation, action, reward, new_observation, done):
        self.timecapsule.memorize(observation, action, reward, done, new_observation)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.DQN.device)
            actions = self.DQN.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def lower_random(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_desc
        else:
            self.epsilon = self.eps_min

    def learn(self):
        # If not enough memories, skip
        if self.timecapsule.stored_mems < self.batch_size:
            return

        self.DQN.optimizer.zero_grad()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        observations, actions, rewards, new_observations, dones = self.timecapsule.recall(self.batch_size)

        ### Convert all data to tensors
        observations = torch.tensor(observations).to(self.DQN.device)
        new_observations = torch.tensor(new_observations).to(self.DQN.device)
        rewards = torch.tensor(rewards).to(self.DQN.device)
        dones = torch.tensor(dones).to(self.DQN.device)

        # Pass through current observation and get the q of actions taken
        q_actions_taken = self.DQN.forward(observations)[batch_index, actions]

        # Pass through the new observation and get the max q value
        with torch.no_grad():
            max_q = self.DQN.forward(new_observations)

        # If the game completed, set the q next value to 0
        max_q[dones] = 0.0

        ### Calculate Target ###
        target = rewards + self.gamma * torch.max(max_q, dim=1)[0]

        ### LEARNING ###
        loss = self.DQN.loss(target, q_actions_taken).to(self.DQN.device)
        loss.backward()
        self.DQN.optimizer.step()

        ### Lower epsilon ###
        self.lower_random()



if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env.reset()

    input_shape = env.observation_space.shape
    action_space = env.action_space.n

    agent = Alfred(gamma=0.95, epsilon=1.0, batch_size=100, n_actions=action_space, model_name='cartpole_DQN',
                  eps_end=0.01, input_dims=input_shape, lr=0.001)

    scores, eps_history = [], []
    avg_scores, min_scores, max_scores = [], [], []
    n_games = 500

    for i in range(n_games):
        if i % 10 == 0:
            render = True
        else:
            render = False

        score = 0
        done = False
        observation = env.reset()

        while not done:
            if render:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-30:])
        avg_scores.append(avg_score)
        min_score = np.min(scores[-30:])
        min_scores.append(min_score)
        max_score = np.max(scores[-30:])
        max_scores.append(max_score)

        print("episode {}, score {}, average_score {}".format(i, round(score, 2), round(avg_score, 2)))

    x = [i + 1 for i in range(n_games)]

    plot_reward(episodes=x, avg_score=avg_scores, min_score=min_scores, max_score=max_scores, game_name="CartPole")


