import networkx as nx
import numpy as np
import copy

from tqdm import tqdm
import random
import time
import torch
from typing import Dict, Any, Tuple, List, Optional

from band_defrag.environments.multiband_optical_network_env import MultibandOpticalNetworkEnv
from band_defrag.mat_algorithm.model.mat_policy import TransformerPolicy
from band_defrag.network_sim.allocators import release_service, ksp_allocate_service
from band_defrag.network_sim.network_state import prepare_network_state
from band_defrag.network_sim.service_generator import NetworkService

EVENT_ALLOCATION = 'ALLOCATION'
EVENT_RELEASE_EXPIRED = 'RELEASE_EXPIRED'
EVENT_REALLOCATION = 'REALLOCATION'


def blocking_test(
        initial_topology: nx.DiGraph | nx.Graph,
        incoming_services: List[NetworkService],
        max_agents: int,
        policy: TransformerPolicy,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    block_num_ksp_only = blocking_test_ksp_only(
        initial_topology=initial_topology,
        incoming_services=incoming_services,
    )

    block_num_with_mat, defrag_timeline_events = blocking_test_with_mat(
        initial_topology=initial_topology,
        incoming_services=incoming_services,
        max_agents=max_agents,
        policy=policy,
    )

    block_compare_dict = {
        'block_num_ksp_only': block_num_ksp_only,
        'block_num_with_mat': block_num_with_mat,
    }

    return block_compare_dict, defrag_timeline_events


def blocking_test_with_mat(
        initial_topology: nx.DiGraph | nx.Graph,
        incoming_services: List[NetworkService],
        max_agents: int,
        policy: TransformerPolicy,
):
    (allocation_status, allocated_service_idx,
     edge_distances, ksp_cache, allocated_service_dict,
     idx_to_node_id, node_id_to_idx) = prepare_network_state(initial_topology)

    block_num = 0
    defrag_timeline_events = []

    for incoming_service in tqdm(
            incoming_services,
            total=len(incoming_services),
            desc='Service allocation with MAT algorithm',
            leave=True
    ):
        arrival_time = incoming_service.arrival_time
        expired_service = [
            s for s in allocated_service_dict.values()
            if s.departure_time <= arrival_time
        ]
        for allocated_service in expired_service:
            release_service(
                allocation_status=allocation_status,
                allocated_service_idx=allocated_service_idx,
                allocated_service_dict=allocated_service_dict,
                service=allocated_service,
            )
            defrag_timeline_events.append({
                'timestamp': arrival_time,
                'event_type': EVENT_RELEASE_EXPIRED,
                'service_id': allocated_service.service_id,
                'details': {'departure_time': allocated_service.departure_time}
            })

        is_allocation_success, allocated_service = ksp_allocate_service(
            allocation_status=allocation_status,
            allocated_service_idx=allocated_service_idx,
            edge_distances=edge_distances,
            ksp_cache=ksp_cache,
            allocated_service_dict=allocated_service_dict,
            service=incoming_service
        )

        if not is_allocation_success:
            env = MultibandOpticalNetworkEnv(
                max_agents=max_agents,
                allocation_status=allocation_status,
                edge_distances=edge_distances,
                ksp_cache=ksp_cache,
                allocated_service_dict=allocated_service_dict,
                allocated_service_idx=allocated_service_idx,
                blocked_service=incoming_service,
            )
            observation, available_actions, agent_mask = env.reset()
            # Expand dimension
            observation = np.expand_dims(observation, 0)
            available_actions = np.expand_dims(available_actions, 0)
            agent_mask = np.expand_dims(agent_mask, 0)

            with torch.no_grad():
                actions, _ = policy.get_actions(observation, agent_mask, available_actions, deterministic=True)
            actions_unbatched = actions[0]  # 去掉 batch 维

            reallocated_services = env.step_on_status(
                actions=actions_unbatched,
                allocation_status=allocation_status,
                allocated_service_idx=allocated_service_idx,
                allocated_service_dict=allocated_service_dict,
            )

            for reallocated_service_data in reallocated_services:
                defrag_timeline_events.append({
                    'timestamp': arrival_time,
                    'event_type': EVENT_REALLOCATION,
                    'service_id': incoming_service.service_id,
                    'details': {
                        'source_id': idx_to_node_id[reallocated_service_data.source_id],
                        'destination_id': idx_to_node_id[reallocated_service_data.destination_id],
                        'arrival_time': reallocated_service_data.arrival_time,
                        'departure_time': reallocated_service_data.departure_time,
                        'bit_rate_requirement': reallocated_service_data.bit_rate_requirement,
                        'snr_requirement': reallocated_service_data.snr_requirement,
                        'path': [
                            idx_to_node_id[idx] for idx in reallocated_service_data.path
                        ],
                        'wavelength': reallocated_service_data.wavelength,
                        'power': reallocated_service_data.power,
                        'defrag_service_id': incoming_service.service_id
                    }
                })

            is_allocation_success, allocated_service = ksp_allocate_service(
                allocation_status=allocation_status,
                allocated_service_idx=allocated_service_idx,
                edge_distances=edge_distances,
                ksp_cache=ksp_cache,
                allocated_service_dict=allocated_service_dict,
                service=incoming_service
            )

        if not is_allocation_success:
            block_num += 1
        else:
            defrag_timeline_events.append({
                'timestamp': arrival_time,
                'event_type': EVENT_ALLOCATION,
                'service_id': allocated_service.service_id,
                'details': {
                    'source_id': idx_to_node_id[allocated_service.source_id],
                    'destination_id': idx_to_node_id[allocated_service.destination_id],
                    'arrival_time': arrival_time,
                    'departure_time': allocated_service.departure_time,
                    'bit_rate_requirement': allocated_service.bit_rate_requirement,
                    'snr_requirement': allocated_service.snr_requirement,
                    'path': [
                        idx_to_node_id[idx] for idx in allocated_service.path
                    ],
                    'wavelength': allocated_service.wavelength,
                    'power': allocated_service.power
                }
            })

    return block_num, defrag_timeline_events


def blocking_test_ksp_only(
        initial_topology: nx.DiGraph | nx.Graph,
        incoming_services: List[NetworkService],
):
    (allocation_status, allocated_service_idx,
     edge_distances, ksp_cache, allocated_service_dict,
     idx_to_node_id, node_id_to_idx) = prepare_network_state(initial_topology)

    block_num = 0

    for incoming_service in tqdm(
            incoming_services,
            total=len(incoming_services),
            desc='Service allocation with KSP only',
            leave=True
    ):
        arrival_time = incoming_service.arrival_time
        expired_service = [
            s for s in allocated_service_dict.values()
            if s.departure_time <= arrival_time
        ]
        for allocated_service in expired_service:
            release_service(
                allocation_status=allocation_status,
                allocated_service_idx=allocated_service_idx,
                allocated_service_dict=allocated_service_dict,
                service=allocated_service,
            )

        is_allocation_success, allocated_service = ksp_allocate_service(
            allocation_status=allocation_status,
            allocated_service_idx=allocated_service_idx,
            edge_distances=edge_distances,
            ksp_cache=ksp_cache,
            allocated_service_dict=allocated_service_dict,
            service=incoming_service
        )

        if not is_allocation_success:
            block_num += 1

    return block_num
