import torch
import numpy as np
import torch.nn.functional as F
from mat_vat_agents.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _shuffle_agent_grid(x, y):
    '''
    生成两张索引矩阵，用来在保持“同列=同 agent”不打乱的前提下，随机打散 batch 中时间步 / 线程的顺序
    '''
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, obs_space, act_space, use_available, device):
        self.capacity = args.buffer_capacity
        self.max_agents = args.max_agents
        self.use_available = use_available
        self.device = device

        obs_shape = get_shape_from_obs_space(obs_space)
        act_shape = get_shape_from_act_space(act_space)
        if isinstance(act_shape, int):
            act_shape = (act_shape,)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        # # +1的原因：方便在最后一步填 next_obs / next_value，与 GAE 算法对齐
        # self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
        #                           dtype=np.float32)
        # self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)
        #
        # self.value_preds = np.zeros(
        #     (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # self.returns = np.zeros_like(self.value_preds)
        # self.advantages = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        #
        # if act_space.__class__.__name__ == 'Discrete':
        #     self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
        #                                      dtype=np.float32)
        # else:
        #     self.available_actions = None
        #
        # act_shape = get_shape_from_act_space(act_space)
        #
        # self.actions = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        # self.action_log_probs = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        # self.rewards = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        #
        # self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        # self.bad_masks = np.ones_like(self.masks)
        # self.active_masks = np.ones_like(self.masks)
        #
        # self.step = 0

        self.obs = np.zeros((self.capacity, self.max_agents, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.max_agents, *act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.action_log_probs = np.zeros((self.capacity, self.max_agents, 1), dtype=np.float32)
        self.agent_masks = np.zeros((self.capacity, self.max_agents, 1), dtype=np.float32)  # ★

        if self.use_available:
            self.available_actions = np.ones((self.capacity, self.max_agents, act_space.n), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def insert(self, obs, actions, action_log_probs,
               rewards, agent_masks, available_actions=None):
        """
        rollout过程每步调用，写入step=t的经历，同时把下一时刻观测写到step+1
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.action_log_probs[self.ptr] = action_log_probs
        self.rewards[self.ptr] = rewards
        self.agent_masks[self.ptr] = agent_masks

        if self.use_available and available_actions is not None:
            self.available_actions[self.ptr] = available_actions

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs[idx], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            rewards=torch.as_tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            logp=torch.as_tensor(self.action_log_probs[idx], dtype=torch.float32, device=self.device),
            masks=torch.as_tensor(self.agent_masks[idx], dtype=torch.float32, device=self.device)
        )
        if self.use_available:
            batch["ava"] = torch.as_tensor(self.available_actions[idx], dtype=torch.float32, device=self.device)
        return batch