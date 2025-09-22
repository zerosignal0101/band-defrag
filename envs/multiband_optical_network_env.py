import time

import gym
import numpy as np
import pickle
import random
import copy
import sys
import os
import itertools
import torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)           # 旧 API
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from base_functions import (release_service, one_link_transmission, check_action,
                            compute_current_service_utilization_increment, compute_current_service_GSNR_increment,
                            compute_related_services_GSNR_increment,
                            select_all_related_services, get_r_f)
import numba
from numba import jit, njit, prange
from concurrent.futures import ThreadPoolExecutor

GSNR_min = 15.0
GSNR_max = 25.5
GSNR_SPAN = GSNR_max - GSNR_min
W = 80
frequencies = np.concatenate([
                        np.linspace(184.4, 190.25, W // 2),
                        np.linspace(190.75, 196.6, W // 2)
                    ])


def _norm_gsnr(raw, low=-1.0, high=10.0):
    raw = raw.copy()

    # 1) 记录哪些位置原本就是 0（空槽 / sentinel）
    mask_zero = (raw == -1)

    # 2) 常规裁剪并线形缩放到 [0,1]
    raw = np.clip(raw, low, high)  # 把异常值压到区间端点
    norm = (raw - low) / (high - low)  # 线性映射

    # 3) 把原来是 0 的位置重新置 0
    norm[mask_zero] = 0.0

    return norm


def _raman_gain_triangular(df_THz: float,
                          df_rise:  float = 13.0,
                          df_fall:  float = 15.0,
                          g_peak:   float = 0.48) -> float:
    """Triangular approximation of Raman-gain profile."""
    if df_THz <= 0.0 or df_THz >= df_fall:
        return 0.0
    if df_THz <= df_rise:
        return g_peak / df_rise * df_THz          # 上升段
    return g_peak * (df_fall - df_THz) / (df_fall - df_rise)    # 下降段

# 向量化包装，便于批量调用
g_R = np.vectorize(_raman_gain_triangular)

class MultibandOpticalNetworkEnv(gym.Env):
    def __init__(self, topology, service_dict, service_to_be_sorting, max_agents, blocked_service):
        super(MultibandOpticalNetworkEnv, self).__init__()
        # 加载拓扑
        # with open(topology_file, 'rb') as f:
        #     self.topology = pickle.load(f)
        self.topology = topology
        # self.sorted_service_dict = sorted(service_dict.items(), key=lambda x: x[1].utilization)
        self.service_dict = service_dict
        self.service_to_be_sorting = service_to_be_sorting
        self.num_agents = len(self.service_to_be_sorting) if self.service_to_be_sorting != {} else 0
        self.max_agents = max_agents
        self.blocked_service = blocked_service

        self.service_ids = list(self.service_to_be_sorting.keys())
        # self.observation_space = gym.spaces.Box(low=0, high=1+1e-6, shape=(81,2), dtype=float)
        # self.share_observation_space = self.observation_space
        self.observation_space = [gym.spaces.Box(low=0, high=1 + 1e-6, shape=(163,), dtype=float)
                                  for n in range(self.max_agents)]
        self.share_observation_space = self.observation_space.copy()
        # self.action_space = gym.spaces.Box(low=0, high=79, shape=(1,), dtype=int)
        self.action_space = [gym.spaces.Discrete(80) for n in range(self.max_agents)]

        self.episode_over = True

        self.agent_mask = np.ones((self.max_agents, 1), dtype=np.float32)
        self.agent_mask[self.num_agents:] = 0

    def reset(self):
        self.service_ids = list(self.service_to_be_sorting.keys())
        self.num_agents = len(self.service_to_be_sorting) if self.service_to_be_sorting != {} else 0
        self.num_agents = min(self.num_agents, self.max_agents)
        self.service_ids = self.service_ids[:self.num_agents]

        self.agent_mask = np.ones((self.max_agents, 1), dtype=np.float32)
        self.agent_mask[self.num_agents:] = 0

        src = self.blocked_service.source_id
        dst = self.blocked_service.destination_id
        self.blocked_service.path = self.topology.graph['ksp'][str(src), str(dst)][0].node_list

        obs = self.get_observation()
        available_actions = np.ones((self.max_agents, 80))

        for i in self.service_ids:
            current_service = self.service_to_be_sorting[i]
            index = self.service_ids.index(i)
            for j in range((len(current_service.path) - 1)):
                u = current_service.path[j]
                v = current_service.path[j + 1]
                for k in range(80):
                    if self.topology[u][v]['wavelength_service'][k] != 0:
                        available_actions[index][k] = 0
        return obs, available_actions, self.agent_mask

    def get_observation(self):
        observation = []

        for i in self.service_ids:
            current_service = self.service_to_be_sorting[i]
            path = current_service.path
            # gsnr_min = np.full(W, np.inf, dtype=np.float32)
            link_length = 0
            link_length_max = 0
            ISRS_per_link = []
            for j in range((len(path) - 1)):
                u = current_service.path[j]
                v = current_service.path[j + 1]
                # link_gsnr = np.ones(W, dtype=np.float32) * -1
                # 该链路已占用的槽： service_id ≠ 0
                occupied = self.topology[u][v]['wavelength_service']  # list/array 长度 W
                occupied_idx = [idx for idx, svc in enumerate(occupied) if svc != 0]
                # 预生成一个 (W,) 的增益向量
                I_vec = np.zeros(W, dtype=np.float32)
                for k in range(W):
                    # if self.topology[u][v]['wavelength_SNR'][k] != 0 and self.topology[u][v]['wavelength_SNR'][k] != i:
                    #     service_id = self.topology[u][v]['wavelength_service'][k]
                    #     tmp_service = self.service_dict.get(service_id, None)
                    #     link_gsnr[k] = self.topology[u][v]['wavelength_SNR'][k] - tmp_service.snr_requirement

                    if occupied[k] != 0:  # 已被业务占用，不作为“可用载波”
                        continue
                    f_i = frequencies[k]
                    if occupied_idx:  # 至少有一个干扰信号才计算
                        df = np.abs(f_i - frequencies[occupied_idx])  # shape (n_occ,)
                        I_vec[k] = g_R(df).mean()  # 只累加 g_R

                # link_gsnr = np.asarray(self.topology[u][v]["wavelength_SNR"], dtype=np.float32)
                # gsnr_min = np.minimum(gsnr_min, link_gsnr)
                link_length += self.topology[u][v]["length"]
                if self.topology[u][v]["length"] > link_length_max:
                    link_length_max = self.topology[u][v]["length"]

                # # 对每个 *空闲* 槽 i，累加所有占用槽 j 的 g_R(|f_i-f_j|)
                # for k in range(W):
                #     if occupied[k] != 0:  # 已被业务占用，不作为“可用载波”
                #         continue
                #     f_i = frequencies[k]
                #     if occupied_idx:  # 至少有一个干扰信号才计算
                #         df = np.abs(f_i - frequencies[occupied_idx])  # shape (n_occ,)
                #         I_vec[k] = g_R(df).mean()  # 只累加 g_R

                ISRS_per_link.append(I_vec)

            ISRS_per_link = np.stack(ISRS_per_link, axis=0)  # shape (L, W)

            ISRS_max = np.max(ISRS_per_link, axis=0, initial=0.0)    # (W,)
            ISRS_mean = np.mean(ISRS_per_link, axis=0)               # (W,)

            # gsnr_norm = _norm_gsnr(gsnr_min)
            # print('gsnr_min:', gsnr_min, gsnr_norm)

            bitrate_norm = current_service.bit_rate / 500.0  # 最大业务比特率为500Gbps
            link_length_mean = link_length / (len(path)-1) / 2600000.0  # 最长链路为2600km
            link_length_max = link_length_max / 2600000.0
            # observation.append(np.concatenate([gsnr_norm, ISRS_max, ISRS_mean, np.asarray([link_length_max], np.float32), np.asarray([link_length_mean], np.float32), np.asarray([bitrate_norm], np.float32)]))
            observation.append(np.concatenate(
                [ISRS_max, ISRS_mean, np.asarray([link_length_max], np.float32),
                 np.asarray([link_length_mean], np.float32), np.asarray([bitrate_norm], np.float32)]))

        # padding 到固定 max_agents
        padding_vec = np.ones(W*2 + 3, dtype=np.float32) * -1
        while len(observation) < self.max_agents:
            observation.append(padding_vec.copy())

        return np.stack(observation, axis=0)

    def make_step(self, actions):
        origin_topology = copy.deepcopy(self.topology)
        origin_service_dict = copy.deepcopy(self.service_dict)
        actions = actions[:self.num_agents]
        list_service_to_be_sorting = list(self.service_to_be_sorting.items())

        for i in range(len(actions)):
            if (isinstance(actions, (list, tuple))):
                action = actions[i]
            else:
                action = int(actions[i, 0].item())
            tmp_topology = copy.deepcopy(self.topology)
            tmp_service_dict = copy.deepcopy(self.service_dict)
            _, tmp_service = list_service_to_be_sorting[i]
            allocation, Power, path_GSNR = check_action(action, tmp_topology, tmp_service, tmp_service_dict)
            if allocation:
                self.topology = tmp_topology
                self.service_dict = tmp_service_dict

        all_related_services = select_all_related_services(self.service_dict, self.blocked_service)
        reward1 = 0.0
        reward2 = 0.0
        for sid, service in all_related_services.items():
            origin_srv = origin_service_dict[sid]
            curr_srv = self.service_dict[sid]

            if sid in self.service_to_be_sorting:  # ★ 被选业务
                reward1 += compute_current_service_utilization_increment(
                    origin_topology, origin_srv,
                    self.topology, curr_srv,
                    self.blocked_service)
            else:  # ☆ 未选业务
                reward2 += compute_current_service_GSNR_increment(
                    origin_topology, origin_srv, self.topology,
                    curr_srv, self.blocked_service)
        reward = reward1 + reward2
        if len(self.service_to_be_sorting) > 0:
            reward1 = reward1 / 100 / len(self.service_to_be_sorting)
        if (len(all_related_services) - len(self.service_to_be_sorting)) > 0:
            reward2 = reward2 / 2 / (len(all_related_services) - len(self.service_to_be_sorting))
        return reward / len(all_related_services) * 5, reward1, reward2
