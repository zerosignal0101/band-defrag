import torch
import numpy as np
from band_defrag.utils.mat_model_utils import get_shape_from_obs_space, get_shape_from_act_space
from band_defrag.utils.mat_model_utils import check


class TransformerPolicy:
    """
    MAT Policy  class. Wraps actor and critic networks to compute actions and value function predictions.
    算法壳，对上包含get_actions、evaluate_actions、get_values等通用接口，方便与replay buffer、trainer、runner对接；对下包装与管理MultiAgentTransformer模型
    功能：
        创建/保存/加载网络
        建立优化器、学习率调度
        在rollout、update两阶段执行前向推断
    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        self.max_agents = args.max_agents
        self.lr = args.lr
        self.lr_alpha = args.lr_alpha
        self.opti_eps = args.opti_eps  # 避免计算更新时除以0
        self.weight_decay = args.weight_decay  # 控制模型的复杂度，防止过拟合
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            # print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        # print("obs_dim: ", self.obs_dim)
        # print("share_obs_dim: ", self.share_obs_dim)
        # print("act_dim: ", self.act_dim)

        self.tpdv = dict(dtype=torch.float32, device=device)  # 统一把新建张量送到相同 device / dtype。

        if self.algorithm_name in ["mat", "mat_dec"]:
            from band_defrag.mat_algorithm.model.mat_transformer import MultiAgentTransformer as MAT
        else:
            raise NotImplementedError

        self.transformer = MAT(self.obs_dim, self.act_dim, self.max_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               action_type=self.action_type, device=self.device)

        # self.optimizer = torch.optim.Adam(self.transformer.parameters(),
        #                                   lr=self.lr, eps=self.opti_eps,  # eps：避免计算更新时除以0
        #                                   weight_decay=self.weight_decay)  # 权重衰减：控制模型的复杂度，防止过拟合
        actor_params = list(self.transformer.encoder.parameters()) + list(self.transformer.decoder.parameters())
        self.actor_opt = torch.optim.Adam(actor_params, lr=self.lr)
        # self.actor_opt = torch.optim.Adam(self.transformer.decoder.parameters(), lr=self.lr)
        self.q1_opt = torch.optim.Adam(self.transformer.q1.parameters(), lr=self.lr)
        self.q2_opt = torch.optim.Adam(self.transformer.q2.parameters(), lr=self.lr)
        self.alpha_opt = torch.optim.Adam([self.transformer.log_alpha], lr=self.lr_alpha)

    # def lr_decay(self, episode, episodes):
    #     """
    #     Decay the actor and critic learning rates. 线性衰减学习率
    #     :param episode: (int) current training episode.
    #     :param episodes: (int) total number of training episodes.
    #     """
    #     update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    # @torch.no_grad()
    def get_actions(self, obs, agent_mask, available_actions=None, deterministic=False):
        """
        Rollout阶段，在环境交互/收集数据时调用，用网络“玩游戏”收集数据
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic. MAT中的state全局状态
        :param obs (np.ndarray): local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        obs = check(obs).to(**self.tpdv)
        agent_mask = check(agent_mask).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actions, action_log_probs = self.transformer.get_actions(obs,
                                                                 agent_mask,
                                                                 available_actions,
                                                                 deterministic)

        return actions, action_log_probs


    # def get_values(self, cent_obs, obs, available_actions=None):
    #     """
    #     仅价值预测，与get_actions前半相同
    #     Get value function predictions.
    #     :param cent_obs (np.ndarray): centralized input to the critic.
    #
    #     :return values: (torch.Tensor) value function predictions.
    #     """
    #     cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
    #     obs = obs.reshape(-1, self.num_agents, self.obs_dim)
    #     if available_actions is not None:
    #         available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
    #
    #     values = self.transformer.get_values(cent_obs, obs, available_actions)
    #
    #     values = values.view(-1, 1)
    #
    #     return values

    def get_alpha(self):
        return self.transformer.alpha

    def evaluate_actions(self, obs, agent_mask, actions, available_actions=None):
        """
        Policy Update阶段，计算log pi(a)、entropy、V(s)供PPO损失
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        agent_mask = check(agent_mask).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        action_log_probs, q1, q2 = self.transformer(obs, actions, agent_mask, available_actions)
        return action_log_probs, q1, q2

    def act(self, obs, agent_mask, available_actions=None, deterministic=True):
        """
        简写，用于evaluation，仅返回动作
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        actions, _, = self.get_actions(obs, agent_mask, available_actions, deterministic)

        return actions

    def save(self, save_dir, episode):
        '''
        保存transformer纯权重，不包含优化器
        '''
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        '''
        加载transformer纯权重
        '''
        transformer_state_dict = torch.load(model_dir, weights_only=False)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    # Pytorch热切换，开启/关闭dropout
    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()

