from typing import List, Literal, Union, Optional, Annotated, Tuple, Dict
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import random

from band_defrag.network_sim.service_generator import NetworkService


class AllocatedService(NetworkService):
    power: float
    path: List[int]
    wavelength: int
    gsnr: float
    utilization: float


CHANNEL_NUM = 80


def get_available_bands(
        path: List[int],
        allocation_status: Dict[Tuple[int, int], NDArray[bool]]
) -> NDArray[bool]:
    """
    返回路径上所有空闲波段（布尔数组，True=空闲）
    """
    if not path:  # 空路径处理
        return np.ones(CHANNEL_NUM, dtype=bool)

    # 初始状态为全True（全空闲）
    combined_status = np.ones(CHANNEL_NUM, dtype=bool)
    for i in range((len(path) - 1)):
        # 更新逻辑：空闲 = 当前空闲 & 链路空闲
        link_key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
        combined_status &= ~allocation_status[link_key]

    return combined_status


def ksp_allocate_service(
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],
        ksp_cache: Dict[Tuple[int, int], List[List[int]]],
        service: NetworkService
):
    ksp_paths = ksp_cache[(min(service.source_id, service.destination_id),
                           max(service.source_id, service.destination_id))]

    ref_power_c = np.full(CHANNEL_NUM // 2, 0.002754, dtype=np.float64)
    ref_power_l = np.full(CHANNEL_NUM // 2, 0.003890, dtype=np.float64)
    ref_power = np.concatenate([ref_power_l, ref_power_c])

    is_allocated = False
    for k, path in enumerate(ksp_paths):
        if is_allocated:
            break

        # 1. 获取路径可用波段（布尔数组）
        available_bands = get_available_bands(path, allocation_status)

        # 2. 提取空闲波段索引（关键优化点）
        free_bands = np.where(available_bands)[0].tolist()

        # 3. 若无空闲波段，跳过此路径
        if not free_bands:
            continue

        # 4. 随机打乱空闲波段顺序（更高效）
        random.shuffle(free_bands)  # 原地打乱，比random.sample更高效

        # 5. 只遍历真正的空闲波段
        for wave_id in free_bands:
            # 分配波段到路径上所有链路
            for i in range((len(path) - 1)):
                link_key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                allocation_status[link_key][wave_id] = True

            # 分配成功，退出当前路径（假设一次分配一个波段）
            is_allocated = True
            break

    pass
