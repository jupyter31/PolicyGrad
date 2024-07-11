import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from src.torch_utils import np2torch, batch_iterator
from src.model.mlp import build_mlp
import gym


class BaselineNetwork(nn.Module):

    def __init__(self, env: gym.Env, config: Dict):
        super().__init__()
        self.config = config
        self.env = env
        self.lr = self.config["hyper_params"]["learning_rate"]
        self.device = torch.device("cpu")
        if self.config["model_training"]["device"] == "gpu":
            if torch.cuda.is_available(): 
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")

        input_size = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            output_size = self.env.action_space.n  # Discrete action space
        elif isinstance(self.env.action_space, gym.spaces.Box):
            output_size = self.env.action_space.shape[0]  # Continuous action space
        else:
            raise ValueError("Unsupported action space type")
        n_layers = self.config["hyper_params"]["n_layers"]
        size = self.config["hyper_params"]["layer_size"]
        self.network = build_mlp(input_size, 1, n_layers, size)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations: torch.tensor) -> torch.tensor:
        """
        Pytorch forward method used to perform a forward pass of inputs(observations)
        through the network.

        Args:
            observations: observation of state from the environment
                        (shape [batch size, dim(observation space)])

        Returns:
            output: networks predicted baseline value for a given observation
                (shape [batch size])
        """
        output = self.network(observations)
        output = output.squeeze()
        # print(output.ndim)
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns: np.array, observations: np.array) -> np.array:
        """
        Args:
            returns: the history of discounted future returns for each step (shape [batch size])
            observations: observations at each step (shape [batch size, dim(observation space)])

        Returns:
            advantages (np.array): returns - baseline values  (shape [batch size])
        """
        observations = np2torch(observations, device=self.device)
        with torch.no_grad():
            values = self.forward(observations).cpu().numpy()
        return returns - values

    def update_baseline(self, returns: np.array, observations: np.array):
        """
        Performs back propagation to update the weights of the baseline network according to MSE loss

        Args:
            returns: the history of discounted future returns for each step (shape [batch size])
            observations: observations at each step (shape [batch size, dim(observation space)])
        """
        returns = np2torch(returns, device=self.device)
        observations = np2torch(observations, device=self.device)

        for obs_batch, returns_batch in batch_iterator(observations, returns, batch_size=1000, shuffle=True):
            self.optimizer.zero_grad()

            values = self.forward(obs_batch)
            loss = nn.MSELoss()(values, returns_batch)

            loss.backward()
            self.optimizer.step()
