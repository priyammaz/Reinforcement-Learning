import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNDense(nn.Module):
    def __init__(self, lr, output_actions, input_shape, model_name, model_store_path="Models/model_store/"):
        super(DQNDense, self).__init__()
        self.input_shape = input_shape
        self.output_actions = output_actions
        self.lr = lr

        # Layers
        self.fc1 = nn.Linear(*self.input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.n_actions)

        ### OPTIMIZER LOSS AND DEVICE ###
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        ### MODEL CHECKPOINT CODE
        self.model_store_path = model_store_path
        self.model_name = model_name
        self.store_path = os.path.join(model_store_path, model_name)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

    def save_state(self):
        print("### Updating Model ###")
        torch.save(self.state_dict(), self.store_path)

    def load_state(self):
        print("### Loading Model ###")
        self.load_state_dict(torch.load(self.store_path))



