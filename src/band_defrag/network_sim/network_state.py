import networkx as nx
import numpy as np
from numpy.typing import NDArray
from itertools import islice
import copy
from tqdm import tqdm
import random
import time
import torch
from typing import Dict, Any, Tuple, List, Optional

from band_defrag.network_sim.allocators import AllocatedService, CHANNEL_NUM


MAX_PATH_NUM = 5


def prepare_network_state(
        topology_loaded: nx.DiGraph | nx.Graph
) -> Tuple[
    Dict[Tuple[int, int], NDArray[bool]],  # allocation_status (RW)
    Dict[Tuple[int, int], NDArray[int]],  # allocated_service_idx (RW)
    Dict[Tuple[int, int], float],  # edge_distances (RO)
    Dict[Tuple[int, int], List[List[int]]],  # ksp_cache (RO)
    Dict[int, AllocatedService],  # allocated_service_dict (RW)
    Dict[int, str],  # idx_to_node_id (RO)
    Dict[str, int],  # node_id_to_idx (RO)
]:
    topology = nx.Graph(topology_loaded)

    idx_to_node_id = {}
    node_id_to_idx = {}
    for idx, node_id in enumerate(topology.nodes()):
        idx_to_node_id[idx] = node_id
        node_id_to_idx[node_id] = idx

    ksp_cache = {}
    allocation_status = {}
    allocated_service_idx = {}
    edge_distances = {}

    node_num = len(topology.nodes)
    for i in range(node_num - 1):
        for j in range(i + 1, node_num):
            pair_key = (i, j)
            sliced_paths = islice(
                nx.shortest_simple_paths(
                    topology, idx_to_node_id[i], idx_to_node_id[j], weight='weight')
                , MAX_PATH_NUM
            )
            ksp_cache[pair_key] = [
                [node_id_to_idx[node] for node in path] for path in sliced_paths
            ]

    for u, v, data in topology.edges(data=True):
        source_idx = node_id_to_idx[u]
        destination_idx = node_id_to_idx[v]
        edge_key = (min(source_idx, destination_idx), max(source_idx, destination_idx))
        allocation_status[edge_key] = np.zeros(CHANNEL_NUM, dtype=bool)
        edge_distances[edge_key] = data['weight'] * 1e3  # Unit: km -> m
        allocated_service_idx[edge_key] = np.zeros(CHANNEL_NUM, dtype=int)

    return allocation_status, allocated_service_idx, edge_distances, ksp_cache, {}, idx_to_node_id, node_id_to_idx
