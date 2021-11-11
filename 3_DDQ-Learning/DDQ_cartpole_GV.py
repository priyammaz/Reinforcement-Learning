import sys
sys.path.append("Models/model_architecture")
sys.path.append("utils")

from utils import TimeCapsule2, plot_reward
from DeepQNetworkConvolution import DQNConv
import numpy as np
import torch
import gym
import cv2

class Alfred():
    def __init__(self, lr, output_actions, input_shape, total_memories,
                 batch_size, gamma, min_epsilon=0.01,
                 epsilon_decrement=5e-4, replace=1000):
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
                                        model_name=self.name + "_training_network")
        self.copy_network = DQNConv(lr=self.lr,
                                    output_actions=self.output_actions,
                                    input_shape=self.input_shape,
                                    model_name=self.name + "_copy_network")

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
            with torch.no_grad():
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
    env = gym.make("CartPole-v0")
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    when_render = 1
    shape = (1,100,150)
    alfred = Alfred(lr=0.001, output_actions=env.action_space.n,
                    input_shape=shape,
                    total_memories=5000, batch_size=64, gamma=0.95)

    if load_checkpoint:
        alfred.load_models()
        
    def get_screen(shape):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = env.render(mode='rgb_array')
        screen = np.ascontiguousarray(screen, dtype=np.float32)         
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)        
        screen = cv2.resize(screen, shape[1:], interpolation=cv2.INTER_AREA)        
        screen = np.array(screen, dtype=np.uint8).reshape(shape)        
        return screen/255
        


    num_steps = 0
    scores, eps_history, steps_array, avg_scores, min_scores, max_scores = [], [], [], [], [], []

    for i in range(n_games):
        if i % when_render == 0:
            render = True
        else:
            render = False

        done = False
        score = 0
        
        env.reset()
        observation = get_screen(shape)


        while not done:
            if render:
                env.render()

            action = alfred.choose_action(observation)

            new_state, reward, done, info = env.step(action)
            
            new_observation = get_screen(shape)

            score += reward

            if not load_checkpoint:

                alfred.store_transition(observation, action, reward, new_observation, done)
                alfred.learn()

            observation = new_observation

        scores.append(score)
        steps_array.append(num_steps)

        avg_score = np.mean(scores[-30:])
        avg_scores.append(avg_score)
        min_score = np.min(scores[-30:])
        min_scores.append(min_score)
        max_score = np.max(scores[-30:])
        max_scores.append(max_score)
        print("Episode {}, Score {}, Avg Score {}, Best Score {}, Epsilon {}, Steps {}".format(i,
                                                                                               score,
                                                                                               round(avg_score,2),
                                                                                               round(best_score,2),
                                                                                               round(alfred.epsilon,2),
                                                                                               num_steps))
        if avg_score > best_score:
            if not load_checkpoint:
                alfred.save_models()
            best_score = avg_score
            
    x = [i + 1 for i in range(n_games)]
    plot_reward(episodes=x, avg_score=avg_scores, min_score=min_scores, max_score=max_scores, game_name="CartPole")













