import numpy as np
import torch
import gym
import os
from typing import Any, Dict, List, Tuple
from src.log import get_logger, export_plot
from src.torch_utils import np2torch
from src.model import BaselineNetwork, CategoricalPolicy, GaussianPolicy, build_mlp


class PolicyGradient():
    """
    Class for implementing a policy gradient algorithm.

    Args:
        env: an OpenAI Gym environment
        config: class with hyperparameters
        logger: logger instance from the logging module
        seed: fixed seed

    """

    def __init__(self, env: gym.Env, config: Dict[str, Any], seed: int, logger=None):
        # directory for training outputs
        if not os.path.exists(config["output"]["output_path"].format(seed)):
            os.makedirs(config["output"]["output_path"].format(seed))

        # store hyperparameters
        self.config = config
        self.seed = seed

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config["output"]["log_path"].format(seed))
        self.env = env
        self.env.reset(seed=self.seed)

        # discrete vs continuous action space
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.is_discrete else self.env.action_space.shape[0]
        )

        self.lr = self.config["hyper_params"]["learning_rate"]

        self.device = torch.device("cpu")
        if config["model_training"]["device"] == "gpu":
            if torch.cuda.is_available(): 
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")

        self.init_policy()

        if config["model_training"]["use_baseline"]:
            self.baseline_network = BaselineNetwork(env, config).to(self.device)

        try:
            if self.config["model_training"]["compile"] == True:
                self.network = torch.compile(self.network, mode=self.config["model_training"]["compile_mode"])
                self.policy = torch.compile(self.policy, mode=self.config["model_training"]["compile_mode"])
                if config["model_training"]["use_baseline"]:
                    self.baseline_network = torch.compile(self.baseline_network, mode=self.config["model_training"]["compile_mode"])
                print("Model compiled")
        except Exception as err:
            print(f"Model compile not supported: {err}")

    def init_policy(self):
        """
        Create a network that maps vectors of size |self.observation| to vectors of size |self.action_dim|
        according to |self.config| and puts in to correct device.

        Depending on the action space instantiate CategoricalPolicy if the space is discrete,
        and GaussianPolicy if action space is continuous.
        """
        self.network = build_mlp(self.observation_dim, self.action_dim, self.config["hyper_params"]["n_layers"], self.config["hyper_params"]["layer_size"])
        self.network.to(self.device)

        if self.is_discrete:
            self.policy = CategoricalPolicy(self.network, self.device)
        else:
            self.policy = GaussianPolicy(self.network, self.action_dim, self.device)

        # policy optimizer
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def init_averages(self):
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards: List, scores_eval: List):
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        pass

    def sample_path(self, env: gym.Env, num_episodes: int = None) -> Tuple[List, List]:
        """
        Sample trajectories from the environment.

        Args:
            num_episodes (int): the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym environment

        Returns:
            paths (list): a list of paths. Each path in paths is a dictionary with
                        path["observation"] a numpy array of ordered observations in the path
                        path["actions"] a numpy array of the corresponding actions in the path
                        path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards (list): the sum of all rewards encountered during this "path"
        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while num_episodes or t < self.config["hyper_params"]["batch_size"]:
            state, info = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config["hyper_params"]["max_ep_len"]):
                states.append(state)
                action = self.policy.act(states[-1][None])[0]
                state, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if terminated or truncated or step == self.config["hyper_params"]["max_ep_len"] - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config["hyper_params"][
                    "batch_size"
                ]:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths: List) -> np.array:
        """
        Calculate the returns G_t for each timestep.

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q_π(s_t, a_t):

        G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            T = len(rewards)
            gamma = self.config["hyper_params"]["gamma"]
            # Compute the returns G_t for each timestep
            returns = np.zeros(T)
            for t in range(T - 1, -1, -1):
                if t == T - 1:
                    returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + gamma * returns[t + 1]
            all_returns.append(returns)
        returns = np.concatenate(all_returns)
        return returns

    def normalize_advantage(self, advantages: np.array) -> np.array:
        """
        Normalize the advantages so that they have a mean of 0 and standard deviation of 1.

        Args:
            advantages: (shape [batch size])
        Returns:
            normalized_advantages: (shape [batch size])

        """
        mean = np.mean(advantages)
        std = np.std(advantages)
        normalized_advantages = (advantages - mean) / (std + 1e-8)
        return normalized_advantages

    def calculate_advantage(self, returns: np.array, observations: np.array) -> np.array:
        """
        Calculates the advantage for each of the observations

        Args:
            returns: shape [batch size]
            observations: shape [batch size, dim(observation space)]

        Returns:
            advantages (np.array): shape [batch size]
        """
        if self.config["model_training"]["use_baseline"]:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if self.config["model_training"]["normalize_advantage"]:
            advantages = self.normalize_advantage(advantages)

        return advantages

    def update_policy(self, observations: np.array, actions: np.array, advantages: np.array):
        """
        Args:
            observations: shape [batch size, dim(observation space)]
            actions: shape [batch size, dim(action space)] if continuous
                                [batch size] (and integer type) if discrete
            advantages: shape [batch size]
        """
        observations = np2torch(observations, device=self.device)
        actions = np2torch(actions, device=self.device)
        advantages = np2torch(advantages, device=self.device)

        # Compute the log probabilities of the actions given the observations
        distribution = self.policy.action_distribution(observations)
        log_probs = distribution.log_prob(actions)

        # we want to maximize policy's performance while torch tries to minimize the loss,
        # so we should negate it.
        policy_loss = -(log_probs * advantages).mean()

        # Update the policy parameters
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def train(self):
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        # set policy to device
        self.policy = self.policy.to(self.device)

        for t in range(self.config["hyper_params"]["num_batches"]):

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)

            # advantage will depend on the baseline implementation
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            if self.config["model_training"]["use_baseline"]:
                self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages)

            # logging
            if t % self.config["model_training"]["summary_freq"] == 0:
                self.update_averages(total_rewards, all_total_rewards)
                self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config["env"]["record"] and (
                last_record > self.config["model_training"]["record_freq"]
            ):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")

        if (
            self.evaluate(
                self.env,
                num_episodes=self.config["model_training"]["num_episodes_eval"],
            )
            >= self.config["env"]["min_expected_reward"]
        ):
            torch.save(
                self.policy.state_dict(),
                "submission/{}-{}-model-weights.pt".format(
                    self.config["env"]["env_name"],
                    "baseline"
                    if self.config["model_training"]["use_baseline"]
                    else "no-baseline",
                ),
            )
            np.save(
                "submission/{}-{}-scores.npy".format(
                    self.config["env"]["env_name"],
                    "baseline"
                    if self.config["model_training"]["use_baseline"]
                    else "no-baseline",
                ),
                averaged_total_rewards,
            )

        torch.save(
            self.policy.state_dict(),
            self.config["output"]["model_output"].format(self.seed),
        )
        np.save(
            self.config["output"]["scores_output"].format(self.seed),
            averaged_total_rewards,
        )
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config["env"]["env_name"],
            self.config["output"]["plot_output"].format(self.seed),
        )

    def evaluate(self, env=None, num_episodes=1, logging=False):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        if logging:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(
                avg_reward, sigma_reward
            )
            self.logger.info(msg)
        return avg_reward

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        env = gym.make(
            self.config["env"]["env_name"],
            render_mode="rgb_array"
        )
        env.reset(seed=self.seed)
        env = gym.wrappers.RecordVideo(
            env,
            self.config["output"]["record_path"].format(self.seed),
            step_trigger=lambda x: x % 100 == 0,
        )
        self.evaluate(env, 1)

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # record one game at the beginning
        if self.config["env"]["record"]:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config["env"]["record"]:
            self.record()
