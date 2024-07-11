import numpy as np
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from src.torch_utils import np2torch


class BasePolicy(ABC):

    def __init__(self, device):
        ABC.__init__(self)
        self.device = device

    @abstractmethod
    def action_distribution(self, observations: torch.tensor) -> torch.distributions.Distribution:
        """
        Defines the conditional probability distribution over actions given an observation
        from the environment. Returns an object representing the policy's conditional distribution(s)
        given the observations. The distribution will have a batch shape matching that of observations,
        to allow for a different distribution for each observation in the batch.

        Args:
            observations:  observation of state from the environment
                        (shape [batch size, dim(observation space)])

        Returns:
            distribution: represents the conditional distributions over actions given the observations.
        """
        pass

    def act(self, observations: torch.tensor) -> np.array:
        """
        Samples actions to be used to act in the environment
        """
        observations = np2torch(observations, device=self.device)
        distribution = self.action_distribution(observations)
        actions = distribution.sample()
        sampled_actions = actions.cpu().numpy()
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network, device):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, device)
        self.network = network
        self.device = device

    def action_distribution(self, observations: torch.tensor) -> torch.distributions.Categorical:
        """
        Args:
            observations:  observation of states from the environment
                        (shape [batch size, dim(observation space)])

        Returns:
            distribution: represent the conditional distribution over
                         actions given a particular observation
        https://pytorch.org/docs/stable/distributions.html#categorical
        """
        logits = self.network(observations.to(self.device))
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):

    def __init__(self, network, action_dim, device):
        nn.Module.__init__(self)
        BasePolicy.__init__(self, device)
        self.network = network
        self.device = device
        # https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=device))

    def std(self) -> torch.tensor:
        """
        Returns the standard deviation for each dimension of the policy's actions
        (shape [dim(action space)])
        """
        return torch.exp(self.log_std)

    def action_distribution(self, observations: torch.tensor) -> torch.distributions.Distribution:
        mean = self.network(observations)
        std = self.std()
        distribution = torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
        return distribution
