from pydantic import BaseModel, Field
from typing import List, Literal, Union, Optional, Annotated, Tuple, Dict
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
    snr_requirement: float


def generate_services(
        ksp_cache: Dict[Tuple[int, int], List[List[int]]], service_num: int,
        avg_arrival_interval: float, avg_holding_time: float
) -> List[NetworkService]:
    node_pairs = []

    edge_keys = list(ksp_cache.keys())
    for _ in range(service_num):
        edge_key = random.choice(edge_keys)
        edge_key = random.choice([edge_key, (edge_key[1], edge_key[0])])
        node_pairs.append(edge_key)

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

        if bit_rate_requirement > 700:
            snr_requirement = 26.5 - 1
        elif bit_rate_requirement > 600:
            snr_requirement = 25.0 - 1
        elif bit_rate_requirement > 500:
            snr_requirement = 23.5 - 1
        elif bit_rate_requirement > 400:
            snr_requirement = 21.0 - 1
        elif bit_rate_requirement > 300:
            snr_requirement = 18.7 - 1
        else:
            snr_requirement = 15

        service = NetworkService(
            service_id=service_id, source_id=source_id, destination_id=destination_id,
            arrival_time=arrival_time, departure_time=departure_time,
            bit_rate_requirement=bit_rate_requirement, snr_requirement=snr_requirement
        )
        services.append(service)

    return services
