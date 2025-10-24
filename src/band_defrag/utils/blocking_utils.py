import numpy as np
import copy
from tqdm import tqdm
import random
import time
import torch

from band_defrag.network_loader.multiband_optical_network_env import MultibandOpticalNetworkEnv
from band_defrag.utils.network_utils import release_service, random_fit, select_sorting_services

SEED = 42
random.seed(SEED)
np.random.seed(SEED)           # 旧 API
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def blocking_test(topology, services, max_agent, policy, progress_desc="Services"):
    # 初始化变量
    service_dict = {}
    blocknum1 = 0  # 不整理的阻塞数
    blocknum2 = 0  # MAT的阻塞数
    denum2 = 0

    # MAT 用
    topology2 = copy.deepcopy(topology)
    service_dict2 = copy.deepcopy(service_dict)

    # <<<<<<< 新增：用于存储重排事件的列表 >>>>>>>
    defragmentation_events = []

    for tmp_service in tqdm(
            services.values(),  # 直接迭代 dict 值就行
            total=len(services),  # 精确总量，方便计算 %
            desc=progress_desc,
            leave=True):
        time0 = tmp_service.arrival_time

        # 释放到期业务
        static_dict = copy.deepcopy(service_dict)
        for i in static_dict.keys():
            if static_dict[i].departure_time <= time0:
                release_service(topology, static_dict[i], service_dict)
        static_dict2 = copy.deepcopy(service_dict2)
        for i in static_dict2.keys():
            if static_dict2[i].departure_time <= time0:
                release_service(topology2, static_dict2[i], service_dict2)

        # ====== 策略 1: 不重排 ======
        path, wavelength, info = random_fit(topology, tmp_service, service_dict)
        if path is None:
            blocknum1 += 1

        # ====== 策略 2: MAT重排 ======
        tmp_service2 = copy.deepcopy(tmp_service)

        path2, wavelength2, info = random_fit(topology2, tmp_service2, service_dict2)

        # 如果新服务无法直接分配，尝试重排
        if path2 is None:
            # 保存重排前的服务状态
            origin_service_dict2_for_defrag = {
                svc_id: copy.deepcopy(svc)
                for svc_id, svc in service_dict2.items()
            }  # 只记录需要重排的会有性能提升

            src = tmp_service2.source_id
            dst = tmp_service2.destination_id
            # 临时设定新业务的预设路径，用于选择重叠业务
            tmp_service2.path = topology.graph['ksp'][str(src), str(dst)][0].node_list

            services_to_be_sorting = select_sorting_services(service_dict2, tmp_service2, time0, max_agent, 0.4)

            # <<<<<<< 新增：重排事件记录 >>>>>>>
            current_defrag_event = {
                'trigger_service_id': tmp_service2.service_id,
                'arrival_time': time0,
                'departure_time': tmp_service2.departure_time,
                'reallocations': []  # 存储被重排服务的旧/新状态
            }
            is_defrag_attempted = False

            if len(services_to_be_sorting) > 0:
                is_defrag_attempted = True
                denum2 += 1
                env = MultibandOpticalNetworkEnv(topology2, service_dict2, services_to_be_sorting, max_agent,
                                                 tmp_service2)
                obs, ava, agent_mask = env.reset()
                obs = np.expand_dims(obs, 0)
                ava = np.expand_dims(ava, 0)
                agent_mask = np.expand_dims(agent_mask, 0)

                # start_time2 = time.time() # Optional: keep time measurement
                with torch.no_grad():
                    actions, logp = policy.get_actions(obs, agent_mask, ava, deterministic=True)
                actions, logp = actions[0], logp[0]  # 去掉 batch 维

                reward, _, __ = env.make_step(actions)  # 执行重排操作

                # end_time2 = time.time() # Optional: keep time measurement
                # tmp_time2 += end_time2 - start_time2 # Optional: keep time measurement

                topology2 = env.topology  # 更新拓扑
                service_dict2 = env.service_dict  # 更新服务字典

                # 记录重排业务的变化
                for svc_id, original_svc in services_to_be_sorting.items():
                    old_path = origin_service_dict2_for_defrag[svc_id].path
                    old_wavelength = origin_service_dict2_for_defrag[svc_id].wavelength

                    new_svc = service_dict2.get(svc_id)
                    if new_svc:
                        new_path = new_svc.path
                        new_wavelength = new_svc.wavelength
                    else:  # 服务被释放且未成功重分配
                        new_path = None
                        new_wavelength = None

                    if old_path != new_path or old_wavelength != new_wavelength:
                        current_defrag_event['reallocations'].append({
                            'service_id': svc_id,
                            'old_path': old_path,
                            'old_wavelength': old_wavelength,
                            'new_path': new_path,
                            'new_wavelength': new_wavelength
                        })

            # 重排后再次尝试分配原始阻塞的新服务
            path2, wavelength2, info = random_fit(topology2, tmp_service2, service_dict2)

            if path2 is None:
                blocknum2 += 1
                current_defrag_event['final_blocked'] = True
                current_defrag_event['new_service_allocation'] = None
            else:
                current_defrag_event['final_blocked'] = False
                current_defrag_event['new_service_allocation'] = {
                    'path': path2,
                    'wavelength': wavelength2
                }

            # 只有当尝试进行重排时才添加事件
            if is_defrag_attempted:
                defragmentation_events.append(current_defrag_event)

        # 如果新服务一开始就分配成功，就不记录重排事件
        else:
            # 记录新服务最初分配的路径和波长
            # 如果需要记录所有新业务的分配信息，可以在这里添加
            pass

    return {
        'blocknum1': blocknum1,
        'blocknum2': blocknum2
    }, service_dict2, defragmentation_events  # <<<<<<< 返回重排事件 >>>>>>>
