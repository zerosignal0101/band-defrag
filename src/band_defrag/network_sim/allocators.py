from typing import List, Literal, Union, Optional, Annotated, Tuple, Dict
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import random

from band_defrag.network_sim.gsnr_calculator import one_link_transmission
from band_defrag.network_sim.service_generator import NetworkService


class AllocatedService(NetworkService):
    power: float
    path: List[int]
    wavelength: int
    gsnr: float
    utilization: float


CHANNEL_NUM = 80


def get_available_bands(
        link_key_list: List[Tuple[int, int]],
        allocation_status: Dict[Tuple[int, int], NDArray[bool]]
) -> NDArray[bool]:
    """
    返回路径上所有空闲波段（布尔数组，True=空闲）
    """
    if not link_key_list:  # 空路径处理
        return np.ones(CHANNEL_NUM, dtype=bool)

    # 初始状态为全True（全空闲）
    combined_status = np.ones(CHANNEL_NUM, dtype=bool)
    for link_key in link_key_list:
        combined_status &= ~allocation_status[link_key]

    return combined_status


POWER_BAND_C = 0.002754
POWER_BAND_L = 0.003890
CENTER_FREQUENCIES = np.concatenate([
    np.linspace(184.4e12, 190.25e12, CHANNEL_NUM // 2),
    np.linspace(190.75e12, 196.6e12, CHANNEL_NUM // 2)
])


def ksp_allocate_service(
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],
        allocation_service_idx: Dict[Tuple[int, int], NDArray[int]],
        link_distances: Dict[Tuple[int, int], float],
        ksp_cache: Dict[Tuple[int, int], List[List[int]]],
        service_dict: Dict[int, NetworkService],
        service: NetworkService
):
    ksp_paths = ksp_cache[(min(service.source_id, service.destination_id),
                           max(service.source_id, service.destination_id))]

    ref_power_c = np.full(CHANNEL_NUM // 2, POWER_BAND_C, dtype=np.float64)
    ref_power_l = np.full(CHANNEL_NUM // 2, POWER_BAND_L, dtype=np.float64)
    ref_power = np.concatenate([ref_power_l, ref_power_c])

    is_allocated = False
    for k, path in enumerate(ksp_paths):
        # Prepare links on the path
        link_key_list = [
            (min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)
        ]

        if is_allocated:
            break

        available_bands = get_available_bands(link_key_list, allocation_status)
        free_bands = np.where(available_bands)[0].tolist()

        if not free_bands:
            continue

        random.shuffle(free_bands)
        for wave_id in free_bands:
            is_power_check_success = False
            for link_key in link_key_list:
                power = ref_power * (~ available_bands)
                power[wave_id] = POWER_BAND_C if wave_id > CHANNEL_NUM // 2 else POWER_BAND_L
                # print(f"Current power on {link_key}: \n", power)
                _, gsnr = one_link_transmission(
                    link_distances[link_key], CHANNEL_NUM, power, CENTER_FREQUENCIES
                )
                # print(f"Calculated gsnr on {link_key}: \n", gsnr)
                if service.snr_requirement > gsnr[wave_id]:
                    print(f'Service SNR not satisfied {service.snr_requirement} > {gsnr[wave_id]}')
                    break

                is_associated_service_failed = False
                for wave_id_checking in range(CHANNEL_NUM):
                    if wave_id_checking == wave_id:
                        continue
                    if allocation_status[link_key][wave_id_checking]:
                        service_id_checking = allocation_service_idx[link_key][wave_id_checking]
                        service_data_checking = service_dict.get(service_id_checking, None)
                        if service_data_checking.snr_requirement > gsnr[wave_id_checking]:
                            print(f'Associated service SNR not satisfied '
                                  f'{service_data_checking.snr_requirement} > {gsnr[wave_id_checking]}')
                            is_associated_service_failed = True
                            break

                if is_associated_service_failed:
                    break

                is_power_check_success = True

            if not is_power_check_success:
                print(f"Power check failed on {wave_id}.")
                break

            for link_key in link_key_list:
                allocation_status[link_key][wave_id] = True
                allocation_service_idx[link_key][wave_id] = service.service_id

            # 分配成功，退出当前路径（假设一次分配一个波段）
            is_allocated = True
            print(f'Allocate success on {wave_id} with path: \n', path)
            break

    pass
