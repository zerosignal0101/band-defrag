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


def blocking_test(
        initial_topology: DiGraph,
        all_incoming_services: Dict[str, Service],
        max_agent: int,
        policy: TransformerPolicy,
        progress_desc: str = "Services"
) -> Tuple[Dict[str, int], Dict[str, Service], List[Dict[str, Any]]]:
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
        Tuple[Dict[str, int], Dict[str, Service], List[Dict[str, Any]]]:
            - A dictionary containing blocking counts for both strategies
              ('blocknum1' for no defrag, 'blocknum2' for MAT defrag).
            - The final state of the service dictionary for the MAT defragmentation strategy.
            - A list of dictionaries, each representing a defragmentation event.
              Each event includes details about the trigger service, reallocations,
              and the final allocation status.
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
    defrag_attempts_s2 = 0  # 记录MAT策略下尝试重排的次数

    # 用于存储重排事件的列表
    defragmentation_events: List[Dict[str, Any]] = []

    # --- 辅助函数：释放过期服务 ---
    def _release_expired_services(
            current_topology: DiGraph,
            current_service_dict: Dict[str, Service],
            current_time: float
    ) -> None:
        """
        Helper function to release services that have departed by the current time.
        Modifies current_topology and current_service_dict in place.
        """
        # Iterate over a copy of keys to safely modify the dictionary during iteration
        for svc_id in list(current_service_dict.keys()):
            service = current_service_dict[svc_id]
            if service.departure_time <= current_time:
                release_service(current_topology, service, current_service_dict)

    # --- 2. 模拟服务到达与处理 ---
    for incoming_service in tqdm(
            all_incoming_services.values(),
            total=len(all_incoming_services),
            desc=progress_desc,
            leave=True
    ):
        arrival_time = incoming_service.arrival_time

        # --- 释放到期业务 (为两种策略分别处理) ---
        _release_expired_services(topology_s1, service_dict_s1, arrival_time)
        _release_expired_services(topology_s2, service_dict_s2, arrival_time)

        # ====== 策略 1: 不重排 (No Defragmentation) ======
        # 为策略1复制一份当前服务，避免相互影响
        current_service_s1 = copy.deepcopy(incoming_service)
        path_s1, wavelength_s1, _ = random_fit(topology_s1, current_service_s1, service_dict_s1)
        if path_s1 is None:
            blocknum_s1 += 1

        # ====== 策略 2: MAT重排 (MAT Defragmentation) ======
        # 为策略2复制一份当前服务
        current_service_s2 = copy.deepcopy(incoming_service)

        # 尝试直接分配新服务
        path_s2, wavelength_s2, _ = random_fit(topology_s2, current_service_s2, service_dict_s2)

        is_defrag_attempted = False
        current_defrag_record: Optional[Dict[str, Any]] = None

        # 如果新服务无法直接分配，尝试重排
        if path_s2 is None:
            # 预设新业务的路径，用于选择重叠业务 (select_sorting_services 需要)
            src_node = str(current_service_s2.source_id)
            dst_node = str(current_service_s2.destination_id)
            # 确保拓扑中存在KSP路径信息
            if src_node in topology_s2.graph['ksp'] and dst_node in topology_s2.graph['ksp'][src_node]:
                current_service_s2.path = topology_s2.graph['ksp'][src_node, dst_node][0].node_list
            else:
                # 如果没有KSP路径，则该服务无法处理，直接阻塞
                blocknum_s2 += 1
                continue  # Skip to next incoming service

            # 选择需要重排的服务
            services_to_be_sorted = select_sorting_services(
                service_dict_s2, current_service_s2, arrival_time, max_agent, 0.4
            )

            if services_to_be_sorted:
                is_defrag_attempted = True
                defrag_attempts_s2 += 1

                # 初始化重排事件记录
                current_defrag_record = {
                    'trigger_service_id': current_service_s2.service_id,
                    'arrival_time': arrival_time,
                    'is_initial_blocked': True,  # 记录触发重排是因为初始分配失败
                    'reallocations': []
                }

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

                # 记录重排业务的变化
                for svc_id, original_svc_state in initial_sorted_states.items():
                    new_svc_state = service_dict_s2.get(svc_id)  # 获取重排后的服务对象

                    old_path_nodes = original_svc_state.path
                    old_path_wavelength = original_svc_state.wavelength

                    new_path_nodes = new_svc_state.path if new_svc_state else None
                    new_path_wavelength = new_svc_state.wavelength if new_svc_state else None

                    # 如果路径或波长发生变化，或者服务在重排过程中被释放 (未重新分配)
                    if (new_path_nodes != old_path_nodes or
                            new_path_wavelength != old_path_wavelength or
                            new_svc_state is None):  # 服务被释放且未成功重分配
                        current_defrag_record['reallocations'].append({
                            'service_id': svc_id,
                            'old_path': old_path_nodes,
                            'old_wavelength': old_path_wavelength,
                            'new_path': new_path_nodes,
                            'new_wavelength': new_path_wavelength,
                            'status': 'relocated' if new_svc_state else 'released_during_defrag'
                        })

            # 重排后再次尝试分配原始阻塞的新服务 (如果执行了重排)
            path_s2, wavelength_s2, _ = random_fit(topology_s2, current_service_s2, service_dict_s2)

        # --- 记录 MAT 策略的最终阻塞状态和重排结果 ---
        if path_s2 is None:
            blocknum_s2 += 1
            if current_defrag_record:
                current_defrag_record['final_blocked'] = True
                current_defrag_record['new_service_allocation'] = None
        else:
            if current_defrag_record:  # 如果重排被尝试了
                current_defrag_record['final_blocked'] = False
                current_defrag_record['new_service_allocation'] = {
                    'path': path_s2,
                    'wavelength': wavelength_s2
                }

        # 将重排事件添加到列表 (只有当它被实际尝试时)
        if current_defrag_record:
            defragmentation_events.append(current_defrag_record)

    return {
        'blocknum1': blocknum_s1,
        'blocknum2': blocknum_s2
    }, service_dict_s2, defragmentation_events
