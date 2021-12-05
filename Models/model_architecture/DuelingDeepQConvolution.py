import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, lr, output_actions, input_shape, model_name, model_store_path="Models/model_store/"):
        super(DuelingDeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1))

        self.conv1bn = nn.BatchNorm2d(32)
        self.conv2bn = nn.BatchNorm2d(64)
        self.conv3bn = nn.BatchNorm2d(64)

        self.dense_input_shape = None
        if self.dense_input_shape is None:
            zeros = torch.zeros((1, ) + input_shape)
            with torch.no_grad():
                conv_out = self.convolutions(zeros)
                conv_shape = conv_out.size()
                self.dense_input_shape = int(np.prod(conv_shape))

        self.fc1 = nn.Linear(self.dense_input_shape, 512)
        self.value = nn.Linear(512, 1) # Value function
        self.advantage = nn.Linear(512, output_actions)

        ### OPTIMIZER LOSS AND DEVICE ###
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        ### MODEL CHECKPOINT CODE
        self.model_store_path = model_store_path
        self.model_name = model_name
        self.store_path = os.path.join(model_store_path, model_name)

    def convolutions(self, observation):
        x = F.relu(self.conv1bn(self.conv1(observation)))
        x = F.relu(self.conv2bn(self.conv2(x)))
        x = F.relu(self.conv3bn(self.conv3(x)))
        return x

    def forward(self, observation):
        x = self.convolutions(observation)
        x = x.view(-1, self.dense_input_shape)
        x = F.relu(self.fc1(x))
        value = self.value(x)
        actions = self.advantage(x)
        return value, actions

    def save_state(self):
        print("### Updating Model ###")
        torch.save(self.state_dict(), self.store_path)

    def load_state(self):
        print("### Loading Model ###")
        self.load_state_dict(torch.load(self.store_path))
