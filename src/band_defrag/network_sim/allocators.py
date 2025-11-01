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
        edge_key_list: List[Tuple[int, int]],
        allocation_status: Dict[Tuple[int, int], NDArray[bool]]
) -> NDArray[bool]:
    """
    返回路径上所有空闲波段（布尔数组，True=空闲）
    """
    if not edge_key_list:  # 空路径处理
        return np.ones(CHANNEL_NUM, dtype=bool)

    # 初始状态为全True（全空闲）
    combined_status = np.ones(CHANNEL_NUM, dtype=bool)
    for edge_key in edge_key_list:
        # allocation_status[edge_key] is True for OCCUPIED bands
        # We need bands where all edges are FREE, so we combine NOT(occupied)
        combined_status &= ~allocation_status[edge_key]

    return combined_status


POWER_BAND_C = 0.002754
POWER_BAND_L = 0.003890
CENTER_FREQUENCIES = np.concatenate([
    np.linspace(184.4e12, 190.25e12, CHANNEL_NUM // 2),
    np.linspace(190.75e12, 196.6e12, CHANNEL_NUM // 2)
])
REF_POWER = np.concatenate([
    np.full(CHANNEL_NUM // 2, POWER_BAND_L, dtype=np.float64),
    np.full(CHANNEL_NUM // 2, POWER_BAND_C, dtype=np.float64)
])


def try_allocate_service_on_path_wavelength(
        service: NetworkService,
        path: List[int],
        wavelength: int,
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
        allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
        edge_distances: Dict[Tuple[int, int], float],  # RO
        allocated_service_dict: Dict[int, AllocatedService],  # RW
) -> Tuple[bool, AllocatedService | None]:
    """
    尝试将服务分配到指定的路径和波长。
    执行所有必要的SNR检查（新服务和受影响的现有服务）。
    如果检查通过，则执行分配并返回True和AllocatedService对象。
    若检查失败，返回False和None。
    """
    edge_key_list = [
        (min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)
    ]

    for edge_key in edge_key_list:
        # Step 1: Calculate the power profile for this specific edge, assuming the new service is allocated
        # Start with the current occupation status of this edge
        # allocation_status[edge_key] is True for OCCUPIED bands on this specific edge
        current_edge_occupied_bands = allocation_status[edge_key].copy()

        # Temporarily mark the new service's wavelength as occupied for this edge for the SNR calculation
        # This reflects the state AFTER the new service is allocated
        current_edge_occupied_bands[wavelength] = True

        # Create the power array for one_link_transmission.
        # Channels that are occupied transmit power (from REF_POWER), others transmit 0.
        power_for_gsnr_calc = np.where(current_edge_occupied_bands, REF_POWER, 0.0)

        # Step 2: Perform SNR check for the new service (at 'wavelength')
        _, gsnr_values = one_link_transmission(
            edge_distances[edge_key], CHANNEL_NUM, power_for_gsnr_calc, CENTER_FREQUENCIES
        )

        if service.snr_requirement > gsnr_values[wavelength]:
            # print(f'Service {service.service_id} SNR not satisfied on edge {edge_key} at wave {wavelength}: '
            #       f'Required {service.snr_requirement:.2f} > Actual {gsnr_values[wavelength]:.2f}')
            return False, None  # New service's SNR requirement not met

        # Step 3: Perform SNR checks for all existing services on this edge that might be affected
        # The gsnr_values calculated above are *already* considering the new service being present.
        for wave_id_checking in range(CHANNEL_NUM):
            if wave_id_checking == wavelength:
                continue  # Skip the new service, it's already checked

            # If there's an existing service on this wavelength on this edge
            if allocation_status[edge_key][wave_id_checking]:
                service_id_checking = allocated_service_idx[edge_key][wave_id_checking]
                service_data_checking = allocated_service_dict.get(service_id_checking)

                # Check if the existing service's SNR requirement is still met with the new service
                if service_data_checking and service_data_checking.snr_requirement > gsnr_values[wave_id_checking]:
                    # print(f'Associated service {service_id_checking} SNR not satisfied on edge {edge_key} at wave {wave_id_checking}: '
                    #       f'Required {service_data_checking.snr_requirement:.2f} > Actual {gsnr_values[wave_id_checking]:.2f} '
                    #       f'due to new service {service.service_id} at {wavelength}')
                    return False, None  # Existing service's SNR requirement not met

    # Release old allocation if exist
    if allocated_service_dict.get(service.service_id):
        release_service(
            allocation_status=allocation_status,
            allocated_service_idx=allocated_service_idx,
            allocated_service_dict=allocated_service_dict,
            service=allocated_service_dict[service.service_id],
        )

    # If all checks pass for all edges:
    # Step 4: Perform the actual allocation by updating the allocation status across all edges in the path
    for edge_key in edge_key_list:
        allocation_status[edge_key][wavelength] = True
        allocated_service_idx[edge_key][wavelength] = service.service_id

    # Create and store the allocated service object
    allocated_service = AllocatedService(
        **service.model_dump(),
        path=path,
        wavelength=wavelength,
    )
    allocated_service_dict[allocated_service.service_id] = allocated_service

    # print(f"Service {service.service_id} successfully allocated on path {path} at wavelength {wavelength}")
    return True, allocated_service


def ksp_allocate_service(
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
        allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
        edge_distances: Dict[Tuple[int, int], float],  # RO
        ksp_cache: Dict[Tuple[int, int], List[List[int]]],  # RO
        allocated_service_dict: Dict[int, AllocatedService],  # RW
        service: NetworkService  # RO
) -> Tuple[bool, AllocatedService | None]:
    # Checking if this service has been allocated, that should not happen
    assert allocated_service_dict.get(service.service_id) is None, \
        f'Service {service.service_id} already allocated, that should not happen'

    # Retrieve KSP paths for the service's source and destination
    # Ensure consistent key ordering for ksp_cache
    cache_key = (min(service.source_id, service.destination_id),
                 max(service.source_id, service.destination_id))
    ksp_paths = ksp_cache.get(cache_key, [])

    if not ksp_paths:
        # print(f"No KSP paths found for service {service.service_id} between {service.source_id} and {service.destination_id}")
        return False, None

    # Iterate through each KSP path
    for path in ksp_paths:
        # Prepare edge keys for the current path
        edge_key_list = [
            (min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)
        ]

        # Find bands that are currently free on *all* edges of this specific path
        available_bands_on_path = get_available_bands(edge_key_list, allocation_status)
        free_bands = np.where(available_bands_on_path)[0].tolist()

        if not free_bands:
            # print(f"No free bands available on path {path} for service {service.service_id}, trying next path.")
            continue  # No free bands available on this path, try next path

        random.shuffle(free_bands)  # Randomize the order of trying free bands

        # Iterate through each available wavelength on the current path
        for wave_id in free_bands:
            # Attempt to allocate the service on the current path and wavelength
            is_allocated, allocated_service = try_allocate_service_on_path_wavelength(
                service=service,
                path=path,
                wavelength=wave_id,
                allocation_status=allocation_status,
                allocated_service_idx=allocated_service_idx,
                edge_distances=edge_distances,
                allocated_service_dict=allocated_service_dict,
            )
            if is_allocated:
                # Allocation successful, return immediately
                return True, allocated_service
            # If not allocated (due to SNR check failure on one or more edges),
            # the loop continues to try the next free_band on this same path.

    # If no suitable path and wavelength combination is found after checking all KSP paths
    # and their available bands
    # print(f"No suitable path/wavelength found for service {service.service_id}")
    return False, None


def release_service(
        allocation_status: Dict[Tuple[int, int], NDArray[bool]],  # RW
        allocated_service_idx: Dict[Tuple[int, int], NDArray[int]],  # RW
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

    # 3. 准备路径上的边列表
    edge_key_list = [
        (min(path[i], path[i + 1]), max(path[i], path[i + 1])) for i in range(len(path) - 1)
    ]

    # 4. 遍历路径上的所有边，更新分配状态
    for edge_key in edge_key_list:
        if edge_key not in allocation_status:
            # 这通常不应该发生，除非数据不一致
            print(f"Error: Link {edge_key} not found in allocation_status during release for service {service.service_id}.")
            continue

        # 将该边上的该波段标记为释放（空闲）
        allocation_status[edge_key][wavelength] = False
        # 将该边上的该波段的业务ID标记为无效（例：-1 或 0 如果ID从1开始）
        # 假设服务ID为非负数，-1 是一个安全的值来表示未分配
        allocated_service_idx[edge_key][wavelength] = -1

    # 5. 从已分配业务字典中移除该业务
    del allocated_service_dict[service.service_id]

    # 6. GSNR 重新评估（仅作说明，在此模型下非必要）
    # 当一个业务被释放时，它所产生的干扰消失，这会导致其路径上其他所有已分配业务的 GSNR 值增加或保持不变。
    # 因此，在此 GSNR 模型下，释放业务并不会导致其他现有业务的 SNR 需求突然不满足。
    # 如果需要跟踪每个业务的实时 GSNR，或者在更复杂的模型中释放可能导致其他业务面临风险，
    # 那么这里需要遍历所有受影响边上的活性业务，重新计算它们的 GSNR。
    # 但就“正确释放”而言，上述步骤已经足够。
    #
    # for edge_key in edge_key_list:
    #     # Re-calculate power (excluding the released service)
    #     power = ref_power * allocation_status[edge_key] # Note: ref_power would need to be passed or accessed
    #     _, gsnr_after_release = one_link_transmission(
    #         edge_distances[edge_key], CHANNEL_NUM, power, CENTER_FREQUENCIES
    #     )
    #     for check_wave_id in range(CHANNEL_NUM):
    #         if allocation_status[edge_key][check_wave_id]: # If a service is still allocated here
    #             # Update internal GSNR record for service_id_checking if needed
    #             pass
    # print(f"Service ID {service.service_id} successfully released.")
