from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional, Annotated, Tuple
import numpy as np
import networkx as nx
import random


class NetworkService(BaseModel):
    service_id: int
    source_id: int
    destination_id: int
    arrival_time: float
    departure_time: float
    bit_rate_requirement: int


class AllocatedService(NetworkService):
    power: float
    path: List[int]
    wavelength: int
    snr_requirement: float
    utilization: float


def random_get_node_pairs(topology: nx.DiGraph, pair_num: int) -> List[Tuple[int, int]]:
    node_pairs = []
    for i in range(pair_num):
        source_id = random.randint(0, len(topology.nodes) - 1)
        destination_id = random.randint(0, len(topology.nodes) - 2)
        if destination_id >= source_id:
            destination_id += 1
        node_pairs.append((source_id, destination_id))

    return node_pairs


def generate_services(
        topology: nx.DiGraph, service_num: int,
        avg_arrival_interval: float, avg_holding_time: float
) -> List[NetworkService]:
    node_pairs = random_get_node_pairs(topology, service_num)
    services = []

    # 定义仿真参数
    lambda_rate = 1 / avg_arrival_interval  # 到达率
    mu_rate = 1 / avg_holding_time  # 持续时间的倒数

    current_time = 0.0
    # First service arrival time
    current_time += np.random.exponential(1 / lambda_rate)

    for service_id, (source_id, destination_id) in enumerate(node_pairs):
        # Service time
        arrival_time = current_time
        holding_time = np.random.exponential(1 / mu_rate)
        departure_time = arrival_time + holding_time

        # Update current time
        next_arrival_interval = np.random.exponential(1 / lambda_rate)
        current_time += next_arrival_interval

        # 用自定义函数做加权采样，小比特率业务比重大
        bit_rate_candidates = np.arange(100, 501, 10)  # 每10为一个档
        # 权重：反比于比特率，越小越重
        weights = 1 / bit_rate_candidates
        weights = weights / np.sum(weights)

        bit_rate_requirement = np.random.choice(bit_rate_candidates, p=weights)

        service = NetworkService(
            service_id=service_id, source_id=source_id, destination_id=destination_id,
            arrival_time=arrival_time, departure_time=departure_time,
            bit_rate_requirement=bit_rate_requirement
        )
        services.append(service)

    return services
