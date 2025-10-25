import numpy as np
import copy

from networkx.classes import DiGraph
from tqdm import tqdm
import random
import time
import torch
from typing import Dict, Any, Tuple, List, Optional

# 导入你的自定义类和函数
from band_defrag.network_loader.multiband_optical_network_env import MultibandOpticalNetworkEnv
from band_defrag.utils.network_utils import release_service, random_fit, select_sorting_services
from band_defrag.mat_algorithm.model.mat_policy import TransformerPolicy
from band_defrag.models.service import Service

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

EVENT_ALLOCATION = 'ALLOCATION'
EVENT_RELEASE_EXPIRED = 'RELEASE_EXPIRED'
EVENT_REALLOCATION = 'REALLOCATION'


def blocking_test(
        initial_topology: DiGraph,
        all_incoming_services: Dict[str, Service],
        max_agent: int,
        policy: TransformerPolicy,
        progress_desc: str = "Services"
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Simulates optical network service blocking for two strategies:
    1. No Defragmentation: Services are allocated as they arrive; if resources are
       insufficient, they are blocked.
    2. MAT Defragmentation: If an arriving service cannot be accommodated, a
       defragmentation attempt is made using a Reinforcement Learning policy,
       after which the service is re-attempted.

    Args:
        initial_topology (Network): The initial state of the optical network topology.
        all_incoming_services (Dict[str, Service]): A dictionary of all services
            to be tested, keyed by service ID, ordered by arrival time.
        max_agent (int): Maximum number of agents (re-allocatable services) for
            the defragmentation environment.
        policy (Any): The trained RL policy used for selecting defragmentation actions.
        progress_desc (str): Description for the tqdm progress bar.

    Returns:
        Tuple[Dict[str, int], List[Dict[str, Any]]]:
            - A dictionary containing blocking counts for both strategies.
            - A timeline of events for the MAT defragmentation strategy. Each event
              is a dictionary detailing a state change (allocation, release, etc.),
              allowing for reconstruction of the network state at any point in time.
    """

    # --- 1. 初始化变量 ---
    # 策略 1: 不重排 (No Defragmentation)
    topology_s1 = copy.deepcopy(initial_topology)
    service_dict_s1: Dict[str, Service] = {}
    blocknum_s1 = 0

    # 策略 2: MAT重排 (MAT Defragmentation)
    topology_s2 = copy.deepcopy(initial_topology)
    service_dict_s2: Dict[str, Service] = {}
    blocknum_s2 = 0
    defrag_attempts_s2 = 0

    # 用于存储所有状态变化事件的时间线
    defrag_timeline_events: List[Dict[str, Any]] = []

    # --- 2. 模拟服务到达与处理 ---
    for incoming_service in tqdm(
            all_incoming_services.values(),
            total=len(all_incoming_services),
            desc=progress_desc,
            leave=True
    ):
        arrival_time = incoming_service.arrival_time

        # --- 释放到期业务 (为策略1处理) ---
        # (This part for s1 does not need to change)
        expired_s1 = [
            s for s in service_dict_s1.values()
            if s.departure_time <= arrival_time
        ]
        for service in expired_s1:
            release_service(topology_s1, service, service_dict_s1)

        # --- 为策略2释放到期业务并记录事件 ---
        expired_s2 = [
            s for s in service_dict_s2.values()
            if s.departure_time <= arrival_time
        ]
        for service in expired_s2:
            # 在释放前记录事件
            defrag_timeline_events.append({
                'timestamp': arrival_time,
                'event_type': EVENT_RELEASE_EXPIRED,
                'service_id': service.service_id,
                'details': {'departure_time': service.departure_time}
            })
            # 执行释放
            release_service(topology_s2, service, service_dict_s2)

        # ====== 策略 1: 不重排 (No Defragmentation) ======
        # (This part for s1 does not need to change)
        current_service_s1 = copy.deepcopy(incoming_service)
        path_s1, _, _ = random_fit(topology_s1, current_service_s1, service_dict_s1)
        if path_s1 is None:
            blocknum_s1 += 1

        # ====== 策略 2: MAT重排 (MAT Defragmentation) ======
        current_service_s2 = copy.deepcopy(incoming_service)
        path_s2, _, _ = random_fit(topology_s2, current_service_s2, service_dict_s2)

        # 如果新服务无法直接分配，尝试重排
        if path_s2 is None:
            # 预设新业务的路径，用于选择重叠业务 (select_sorting_services 需要)
            src_node = str(current_service_s2.source_id)
            dst_node = str(current_service_s2.destination_id)
            current_service_s2.path = topology_s2.graph['ksp'][src_node, dst_node][0].node_list

            # 选择需要重排的服务
            services_to_be_sorted = select_sorting_services(
                service_dict_s2, current_service_s2, arrival_time, max_agent, 0.4
            )

            if services_to_be_sorted:
                defrag_attempts_s2 += 1

                # 保存重排前，所有待重排业务的原始状态，用于后续比较
                initial_sorted_states = {
                    svc_id: copy.deepcopy(service_dict_s2[svc_id])
                    for svc_id in services_to_be_sorted
                }

                # 创建并执行重排环境
                env = MultibandOpticalNetworkEnv(
                    topology_s2, service_dict_s2, services_to_be_sorted, max_agent, current_service_s2
                )
                obs, ava, agent_mask = env.reset()
                obs = np.expand_dims(obs, 0)
                ava = np.expand_dims(ava, 0)
                agent_mask = np.expand_dims(agent_mask, 0)

                with torch.no_grad():
                    actions, _ = policy.get_actions(obs, agent_mask, ava, deterministic=True)
                actions_unbatched = actions[0]  # 去掉 batch 维

                # 执行重排操作
                _, _, _ = env.make_step(actions_unbatched)

                # 更新拓扑和服务字典到重排后的状态
                topology_s2 = env.topology
                service_dict_s2 = env.service_dict

                # --- 记录重构事件 ---
                for svc_id, original_svc_state in initial_sorted_states.items():
                    new_svc_state = service_dict_s2.get(svc_id)

                    old_path_nodes = original_svc_state.path
                    old_path_wavelength = original_svc_state.wavelength

                    new_path_nodes = new_svc_state.path
                    new_path_wavelength = new_svc_state.wavelength
                    if (new_path_nodes != old_path_nodes or
                            new_path_wavelength != old_path_wavelength):
                        new_service_to_dict = new_svc_state.to_dict()
                        new_service_to_dict['defrag_service_id'] = current_service_s2.service_id
                        defrag_timeline_events.append({
                            'timestamp': arrival_time,
                            'event_type': EVENT_REALLOCATION,
                            'service_id': svc_id,
                            'details': new_service_to_dict
                        })

            # 再次尝试分配新业务
            path_s2, _, _ = random_fit(topology_s2, current_service_s2, service_dict_s2)

        # --- 记录新业务的最终分配/阻塞状态 ---
        if path_s2 is None:
            blocknum_s2 += 1
            # 阻塞本身不是一个改变网络状态的事件，所以我们不记录它
            # 但如果你想记录“尝试分配失败”的事件，也可以在这里添加
        else:
            # 成功分配，记录 ALLOCATION 事件
            # 注意: random_fit 已经将 service_dict_s2 更新了
            allocated_service = service_dict_s2[current_service_s2.service_id]
            defrag_timeline_events.append({
                'timestamp': arrival_time,
                'event_type': EVENT_ALLOCATION,
                'service_id': allocated_service.service_id,
                'details': allocated_service.to_dict()
            })

    return {
        'blocknum1': blocknum_s1,
        'blocknum2': blocknum_s2
    }, defrag_timeline_events  # <-- 返回事件时间线
