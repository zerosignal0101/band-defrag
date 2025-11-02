import time
from collections import Counter
import gym
from numpy.typing import NDArray
from typing import List, Literal, Union, Optional, Annotated, Tuple, Dict
import numpy as np
import random
import copy
import torch

from band_defrag.network_sim.allocators import CHANNEL_NUM, CENTER_FREQUENCIES, AllocatedService, get_available_bands, \
    try_allocate_service_on_path_wavelength
from band_defrag.network_sim.service_generator import NetworkService


def _raman_gain_triangular(df_THz: float,
                           df_rise: float = 13.0,
                           df_fall: float = 15.0,
                           g_peak: float = 0.48) -> float:
    """Triangular approximation of Raman-gain profile."""
    if df_THz <= 0.0 or df_THz >= df_fall:
        return 0.0
    if df_THz <= df_rise:
        return g_peak / df_rise * df_THz  # 上升段
    return g_peak * (df_fall - df_THz) / (df_fall - df_rise)  # 下降段


# 向量化包装，便于批量调用
g_R = np.vectorize(_raman_gain_triangular)


def get_service_overlap_counts(
        edge_key_list: List[Tuple[int, int]],
        allocation_status: Dict[Tuple[int, int], np.ndarray],
        allocated_service_idx: Dict[Tuple[int, int], np.ndarray]
) -> Dict[int, int]:
    service_counter = Counter()

    for edge_key in edge_key_list:
        status_arr = allocation_status.get(edge_key)
        service_arr = allocated_service_idx.get(edge_key)

        if status_arr is None or service_arr is None:
            continue

        # 向量化提取有效业务ID
        valid_indices = np.where(status_arr)[0]  # 被占用频点的索引
        valid_services = service_arr[valid_indices]  # 对应的业务ID
        unique_services = set(valid_services)

        service_counter.update(unique_services)

    return dict(service_counter)


class MultibandOpticalNetworkEnv:
    def __init__(
            self,
            max_agents: int,
            allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
            allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
            edge_distances: Dict[Tuple[int, int], float],  # RO
            ksp_cache: Dict[Tuple[int, int], List[List[int]]],  # RO
            allocated_service_dict: Dict[int, AllocatedService],  # RW
            blocked_service: NetworkService  # RO
    ):
        super(MultibandOpticalNetworkEnv, self).__init__()

        self.max_agents = max_agents

        # Read-Only props
        self.ori_allocation_status = copy.deepcopy(allocation_status)
        self.ori_allocated_service_idx = copy.deepcopy(allocated_service_idx)
        self.edge_distances = edge_distances
        self.ksp_cache = ksp_cache
        self.ori_allocated_service_dict = copy.deepcopy(allocated_service_dict)
        self.blocked_service = blocked_service

        service_key = (min(blocked_service.source_id, blocked_service.destination_id),
                       max(blocked_service.source_id, blocked_service.destination_id))
        blocked_service_path = self.ksp_cache[service_key][0]
        self.blocked_edge_key_list = [
            (min(blocked_service_path[i], blocked_service_path[i + 1]),
             max(blocked_service_path[i], blocked_service_path[i + 1]))
            for i in range(len(blocked_service_path) - 1)
        ]

        # Current allocation status in Env
        self.allocation_status = copy.deepcopy(allocation_status)
        self.allocated_service_idx = copy.deepcopy(allocated_service_idx)
        self.allocated_service_dict = copy.deepcopy(allocated_service_dict)

        # Overlap service detect
        service_overlap_counts = get_service_overlap_counts(
            self.blocked_edge_key_list, self.allocation_status, self.allocated_service_idx
        )
        self.sorted_service_id_list = sorted(service_overlap_counts.items(), key=lambda item: item[1], reverse=True)

        # gym spaces
        self.observation_space = [gym.spaces.Box(low=0, high=1 + 1e-6, shape=(163,), dtype=float)
                                  for n in range(self.max_agents)]
        self.share_observation_space = self.observation_space.copy()
        self.action_space = [gym.spaces.Discrete(80) for n in range(self.max_agents)]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.allocation_status = copy.deepcopy(self.ori_allocation_status)
        self.allocated_service_idx = copy.deepcopy(self.ori_allocated_service_idx)
        self.allocated_service_dict = copy.deepcopy(self.ori_allocated_service_dict)

        sorted_service_id_list = self.sorted_service_id_list

        available_action_list = []
        for index, (service_id_for_action, service_overlap_count) in enumerate(sorted_service_id_list):
            if index >= self.max_agents:
                break
            service_data_for_action = self.allocated_service_dict.get(service_id_for_action)
            service_path_for_action = service_data_for_action.path
            service_edge_key_list_for_action = [
                (min(service_path_for_action[i], service_path_for_action[i + 1]),
                 max(service_path_for_action[i], service_path_for_action[i + 1]))
                for i in range(len(service_path_for_action) - 1)
            ]
            available_bands = get_available_bands(service_edge_key_list_for_action, self.allocation_status)
            available_action_list.append(available_bands)

        num_agent_needed = len(available_action_list)
        # Padded if the action_list is not long enough
        num_rows_to_pad = self.max_agents - num_agent_needed
        if num_rows_to_pad > 0:
            padding_rows = [np.ones(CHANNEL_NUM, dtype=bool) for _ in range(num_rows_to_pad)]
            available_action_list.extend(padding_rows)

        available_actions = np.vstack(available_action_list)

        # Agent mask, limited to num_agent_needed
        agent_mask = np.ones(self.max_agents, dtype=np.float64)
        agent_mask[num_agent_needed:] = 0.0

        observation = self._get_observation()

        return observation, available_actions, agent_mask

    def _get_observation(self) -> NDArray[np.float64]:
        observation = []
        sorted_service_id_list = self.sorted_service_id_list
        for index, (service_id_related, service_overlap_count) in enumerate(sorted_service_id_list):
            if index >= self.max_agents:
                break
            service_data_related = self.allocated_service_dict.get(service_id_related)
            service_path_related = service_data_related.path
            service_edge_key_list_related = [
                (min(service_path_related[i], service_path_related[i + 1]),
                 max(service_path_related[i], service_path_related[i + 1]))
                for i in range(len(service_path_related) - 1)
            ]
            link_total_length = 0.0
            edge_distance_max = 0.0
            ISRS_per_edge = []
            for edge_key in service_edge_key_list_related:
                allocation_status_on_edge_key = self.allocation_status.get(edge_key)
                I_vec = np.zeros(CHANNEL_NUM, dtype=np.float64)
                allocation_wave_idx = np.where(allocation_status_on_edge_key)[0]
                allocated_frequencies = CENTER_FREQUENCIES[allocation_wave_idx]
                if allocation_wave_idx.size > 0:
                    available_status_on_edge_key = ~ allocation_status_on_edge_key
                    available_frequencies = CENTER_FREQUENCIES[available_status_on_edge_key]
                    if available_frequencies.size > 0:
                        df_matrix = np.abs(
                            available_frequencies[:, np.newaxis] - allocated_frequencies[np.newaxis, :]
                        ) / 1e12  # THz TODO: Check if the delta frequency unit is THz
                        g_R_contributions = g_R(df_matrix)
                        # 沿干扰源维度（列）求和，得到每个可用槽的总干扰
                        # 结果是一个 (n_available,) 的向量
                        # TODO: Check if the method is mean or sum
                        total_interference_for_available_slots = np.sum(g_R_contributions, axis=1)
                        # 将计算出的干扰值填充回 I_vec 数组的对应位置
                        I_vec[available_status_on_edge_key] = total_interference_for_available_slots

                edge_distance = self.edge_distances[edge_key]
                link_total_length += edge_distance
                if edge_distance > edge_distance_max:
                    edge_distance_max = edge_distance

                ISRS_per_edge.append(I_vec)

            ISRS_max = np.max(ISRS_per_edge, axis=0, initial=0.0)  # (W,)
            ISRS_mean = np.mean(ISRS_per_edge, axis=0)  # (W,)

            bitrate_norm = service_data_related.bit_rate_requirement / 500.0  # 最大业务比特率为500Gbps
            link_length_norm = link_total_length / len(service_edge_key_list_related) / 2600000.0  # 最长链路为2600km
            edge_distance_max_norm = edge_distance_max / 2600000.0

            observation.append(np.concatenate(
                [ISRS_max, ISRS_mean, np.asarray([edge_distance_max_norm], np.float64),
                 np.asarray([link_length_norm], np.float64), np.asarray([bitrate_norm], np.float64)]))

        padding_vec = np.ones(CHANNEL_NUM * 2 + 3, dtype=np.float64) * -1.0
        num_rows_to_pad = self.max_agents - len(observation)
        if num_rows_to_pad > 0:
            padding_rows = [padding_vec for _ in range(num_rows_to_pad)]
            observation.extend(padding_rows)

        return np.stack(observation, axis=0)

    def step(self, actions: NDArray[int]):
        for new_wavelength, (service_id, _) in zip(actions, self.sorted_service_id_list):
            service_data = self.allocated_service_dict[service_id]
            if new_wavelength == service_data.service_id:
                continue
            is_reallocation_success, allocated_service = try_allocate_service_on_path_wavelength(
                service=NetworkService(**service_data.model_dump()),
                path=service_data.path,
                wavelength=new_wavelength,
                allocated_service_idx=self.allocated_service_idx,
                allocation_status=self.allocation_status,
                allocated_service_dict=self.allocated_service_dict,
                edge_distances=self.edge_distances
            )

    def step_on_status(
            self, actions: NDArray[int],
            allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
            allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
            allocated_service_dict: Dict[int, AllocatedService],  # RW
    ) -> List[AllocatedService]:
        reallocated_services = []
        for new_wavelength, (service_id, _) in zip(actions, self.sorted_service_id_list):
            service_data = self.allocated_service_dict[service_id]
            if new_wavelength == service_data.service_id:
                continue
            is_reallocation_success, allocated_service = try_allocate_service_on_path_wavelength(
                service=NetworkService(**service_data.model_dump()),
                path=service_data.path,
                wavelength=new_wavelength,
                allocated_service_idx=allocated_service_idx,
                allocation_status=allocation_status,
                allocated_service_dict=allocated_service_dict,
                edge_distances=self.edge_distances
            )
            if is_reallocation_success:
                reallocated_services.append(allocated_service)

        return reallocated_services
