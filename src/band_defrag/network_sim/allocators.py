from typing import List, Literal, Union, Optional, Annotated, Tuple, Dict
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import random

from band_defrag.network_sim.gsnr_calculator import one_link_transmission
from band_defrag.network_sim.service_generator import NetworkService


class AllocatedService(NetworkService):
    path: List[int]
    wavelength: int


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
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
        allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
        link_distances: Dict[Tuple[int, int], float],  # RO
        ksp_cache: Dict[Tuple[int, int], List[List[int]]],  # RO
        allocated_service_dict: Dict[int, AllocatedService],  # RW
        service: NetworkService  # RO
) -> Tuple[bool, AllocatedService | None]:
    # Checking if this service has been allocated, that should not happen
    assert allocated_service_dict.get(service.service_id) is None, 'Already allocated, that should not happen'

    ksp_paths = ksp_cache[(min(service.source_id, service.destination_id),
                           max(service.source_id, service.destination_id))]

    ref_power_c = np.full(CHANNEL_NUM // 2, POWER_BAND_C, dtype=np.float64)
    ref_power_l = np.full(CHANNEL_NUM // 2, POWER_BAND_L, dtype=np.float64)
    ref_power = np.concatenate([ref_power_l, ref_power_c])

    is_allocated = False
    allocated_service = None
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
                        service_id_checking = allocated_service_idx[link_key][wave_id_checking]
                        service_data_checking = allocated_service_dict.get(service_id_checking, None)
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
                allocated_service_idx[link_key][wave_id] = service.service_id

            # 分配成功，退出当前路径（假设一次分配一个波段）
            is_allocated = True
            # print(f'Allocate success on {wave_id} with path: \n', path)
            allocated_service = AllocatedService(
                **service.model_dump(),
                path=path,
                wavelength=wave_id,
            )
            allocated_service_dict[allocated_service.service_id] = allocated_service
            break

    return is_allocated, allocated_service


def release_service(
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
        allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
        link_distances: Dict[Tuple[int, int], float],  # RO (not directly used for release but part of context)
        ksp_cache: Dict[Tuple[int, int], List[List[int]]],  # RO (not directly used for release but part of context)
        allocated_service_dict: Dict[int, AllocatedService],  # RW
        service: AllocatedService  # RO
):
    """
    移除一个业务的分配。
    """
    # 1. 检查业务是否存在于已分配列表中
    allocated_data = allocated_service_dict.get(service.service_id)
    if allocated_data is None:
        # 如果业务不存在，可能是一个错误或重复释放，根据需求决定是抛出异常还是静默处理
        print(f"Warning: Service ID {service.service_id} not found in allocated services. Skipping release.")
        return

    # 2. 获取业务的路径和波长信息
    path = allocated_data.path
    wavelength = allocated_data.wavelength

    # 3. 准备路径上的链接列表
    link_key_list = [
        (min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)
    ]

    # 4. 遍历路径上的所有链接，更新分配状态
    for link_key in link_key_list:
        if link_key not in allocation_status:
            # 这通常不应该发生，除非数据不一致
            print(f"Error: Link {link_key} not found in allocation_status during release for service {service.service_id}.")
            continue

        # 将该链接上的该波段标记为释放（空闲）
        allocation_status[link_key][wavelength] = False
        # 将该链接上的该波段的业务ID标记为无效（例：-1 或 0 如果ID从1开始）
        # 假设服务ID为非负数，-1 是一个安全的值来表示未分配
        allocated_service_idx[link_key][wavelength] = -1

    # 5. 从已分配业务字典中移除该业务
    del allocated_service_dict[service.service_id]

    # 6. GSNR 重新评估（仅作说明，在此模型下非必要）
    # 当一个业务被释放时，它所产生的干扰消失，这会导致其路径上其他所有已分配业务的 GSNR 值增加或保持不变。
    # 因此，在此 GSNR 模型下，释放业务并不会导致其他现有业务的 SNR 需求突然不满足。
    # 如果需要跟踪每个业务的实时 GSNR，或者在更复杂的模型中释放可能导致其他业务面临风险，
    # 那么这里需要遍历所有受影响链接上的活性业务，重新计算它们的 GSNR。
    # 但就“正确释放”而言，上述步骤已经足够。
    #
    # for link_key in link_key_list:
    #     # Re-calculate power (excluding the released service)
    #     power = ref_power * allocation_status[link_key] # Note: ref_power would need to be passed or accessed
    #     _, gsnr_after_release = one_link_transmission(
    #         link_distances[link_key], CHANNEL_NUM, power, CENTER_FREQUENCIES
    #     )
    #     for check_wave_id in range(CHANNEL_NUM):
    #         if allocation_status[link_key][check_wave_id]: # If a service is still allocated here
    #             # Update internal GSNR record for service_id_checking if needed
    #             pass
    print(f"Service ID {service.service_id} successfully released.")
