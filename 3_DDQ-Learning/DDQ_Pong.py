import sys
sys.path.append("Models/model_architecture")
sys.path.append("utils")

from utils import TimeCapsule2, build_env, plot_reward
from DeepQNetworkConvolution import DQNConv
import numpy as np
import torch
import torch.nn as nn

class Alfred():
    def __init__(self, lr, output_actions, input_shape, total_memories,
                 batch_size, gamma, min_epsilon=0.05,
                 epsilon_decrement=5e-5, replace=1000):
        self.lr = lr
        self.output_actions = output_actions
        self.input_shape = input_shape
        self.total_memories = total_memories
        self.batch_size = batch_size
        self.epsilon = 1
        self.min_epsilon = min_epsilon
        self.epsilon_decrement = epsilon_decrement
        self.replace = replace
        self.gamma = gamma

        self.name = "pong"
        self.num_actions_taken = 0

        self.timecapsule = TimeCapsule2(self.total_memories, input_shape)
        self.training_network = DQNConv(lr=self.lr,
                                        output_actions=self.output_actions,
                                        input_shape=self.input_shape,
                                        model_name=self.name + "_training_network_bn")
        self.copy_network = DQNConv(lr=self.lr,
                                    output_actions=self.output_actions,
                                    input_shape=self.input_shape,
                                    model_name=self.name + "_copy_network_bn")

        # if torch.cuda.device_count() > 1:
        #     print("Training on ", torch.cuda.device_count(), "GPU's!")
        #     self.training_network = nn.DataParallel(self.training_network)
        #     self.copy_network = nn.DataParallel(self.copy_network)
        #
        # self.training_network.to(self.training_network.device)
        # self.copy_network.to(self.copy_network.device)


    def store_transition(self, observation, action, reward, new_observation, done):
        self.timecapsule.memorize(observation, action, reward, done, new_observation)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = torch.tensor([observation], dtype=torch.float32).to(self.training_network.device)
            q_actions = self.training_network.forward(observation)
            action = torch.argmax(q_actions).item()
        else:
            action = np.random.choice([act for act in range(self.output_actions)])

        return action

    def lower_random(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon - self.epsilon_decrement
        else:
            self.epsilon = self.min_epsilon

    def copy_network_weights(self):
        if self.num_actions_taken % self.replace == 0:
            self.copy_network.load_state_dict(self.training_network.state_dict())

    def save_models(self):
        self.training_network.save_state()
        self.copy_network.save_state()

    def load_models(self):
        self.training_network.load_state()
        self.copy_network.load_state()


    def learn(self):
        if self.timecapsule.stored_mems < self.batch_size:
            return
        else:
            self.training_network.optimizer.zero_grad()
            self.copy_network_weights()

            observations, actions, rewards, new_observations, dones = self.timecapsule.recall(self.batch_size)
            batch_index = np.arange(self.batch_size)

            ### Convert all data to tensors
            observations = torch.tensor(observations).to(self.training_network.device)
            new_observations = torch.tensor(new_observations).to(self.training_network.device)
            rewards = torch.tensor(rewards).to(self.training_network.device)
            dones = torch.tensor(dones).to(self.training_network.device)

            # Pass through current observation and get the q of actions taken
            q_actions_taken = self.training_network.forward(observations)[batch_index, actions]

            # Get the target q value
            q_max_target = self.copy_network.forward(new_observations)

            # If the game completed, there is no next, so set to 0
            q_max_target[dones] = 0.0

            ### Calculate Target ###
            target = rewards + self.gamma * torch.max(q_max_target, dim=1)[0]

            ### LEARNING ###
            loss = self.training_network.loss(target, q_actions_taken).to(self.training_network.device)
            loss.backward()
            self.training_network.optimizer.step()

            ### Lower epsilon ###
            self.lower_random()

            ### Increment Learning Counter ###
            self.num_actions_taken += 1

if __name__ == "__main__":
    env = build_env("PongNoFrameskip-v4")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    when_render = 1
    alfred = Alfred(lr=0.0001, output_actions=env.action_space.n,
                    input_shape=(env.observation_space.shape),
                    total_memories=200000, batch_size=64, gamma=0.99)

    if load_checkpoint:
        alfred.load_models()

    num_steps = 0
    scores, epsilons, steps_array, min_scores, max_scores, avg_scores = [], [], [], [], [], []

    counter = 0
    for i in range(n_games):
        if counter % when_render == 0:
            render = True
        else:
            render = False
        done = False
        score = 0
        observation = env.reset()


        while not done:
            if render:
                env.render()
            action = alfred.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            num_steps += 1
            alfred.store_transition(observation, action, reward, new_observation, done)
            alfred.learn()
            observation = new_observation
        scores.append(score)
        steps_array.append(num_steps)
        avg_score = np.mean(scores[-100:])
        min_score = np.min(scores[-100:])
        max_score = np.max(scores[-100:])

        min_scores.append(min_score)
        max_scores.append(max_score)
        avg_scores.append(avg_score)
        epsilons.append(alfred.epsilon)

        print("Episode {}, Score {}, Avg Score {}, Best Score {}, Epsilon {}, Steps {}".format(counter,
                                                                                               score,
                                                                                               round(avg_score,2),
                                                                                               round(best_score,2),
                                                                                               round(alfred.epsilon,2),
                                                                                               num_steps))
        if avg_score > best_score:
            alfred.save_models()
            best_score = avg_score
        counter += 1

    plot_reward(episodes=list(range(n_games)),
                avg_score=avg_scores,
                min_score=min_scores,
                max_score=max_scores,
                epsilon=epsilons,
                game_name=alfred.name)













