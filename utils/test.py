from utils import build_env
import gym
import matplotlib.pyplot as plt
import numpy as np

env = build_env("PongNoFrameskip-v4")

for i in range(1):
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
        action = env.action_space.sample()  # get current action for the current observation in q_eval
        new_observation, reward, done, info = env.step(action)
        score += reward
