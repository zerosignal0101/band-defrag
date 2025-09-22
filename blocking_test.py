import numpy as np
import copy
from tqdm import tqdm
import random
import time
import torch

from envs.multiband_optical_network_env import MultibandOpticalNetworkEnv
from base_functions import release_service, random_fit, select_sorting_services
SEED = 42
random.seed(SEED)
np.random.seed(SEED)           # 旧 API
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def blocking_test(topology, services, max_agent, policy, progress_desc="Services"):
    # 初始化变量
    service_dict = {}
    # for key, value in list(services.items())[:100]:
    #     service_dict[key] = value
    blocknum1 = 0  # 不整理的阻塞数
    blocknum2 = 0  # MAT的阻塞数

    denum2 = 0

    # MAT 用
    topology2 = copy.deepcopy(topology)
    service_dict2 = copy.deepcopy(service_dict)

    for tmp_service in tqdm(
            services.values(),           # 直接迭代 dict 值就行
            total=len(services),          # 精确总量，方便计算 %
            desc=progress_desc,
            leave=True):
        time0 = tmp_service.arrival_time
        # print('id/src/dst/bitrate:', tmp_service.service_id, tmp_service.source_id, tmp_service.destination_id, tmp_service.bit_rate)
        # 释放到期业务
        static_dict = copy.deepcopy(service_dict)
        for i in static_dict.keys():
            if static_dict[i].holding_time <= time0:
                release_service(topology, static_dict[i], service_dict)
        static_dict2 = copy.deepcopy(service_dict2)
        for i in static_dict2.keys():
            if static_dict2[i].holding_time <= time0:
                release_service(topology2, static_dict2[i], service_dict2)

        path, wavelength, info = random_fit(topology, tmp_service, service_dict)
        if path == None:
            blocknum1 += 1

        tmp_service2 = copy.deepcopy(tmp_service)

        path2, wavelength2, info = random_fit(topology2, tmp_service2, service_dict2)
        if path2 == None:
            origin_service_dict2 = copy.deepcopy(service_dict2)

            tmp_time2 = 0
            src = tmp_service2.source_id
            dst = tmp_service2.destination_id
            tmp_service2.path = topology.graph['ksp'][str(src), str(dst)][0].node_list
            services_to_be_sorting = select_sorting_services(service_dict2, tmp_service2, time0, max_agent, 0.4)

            if len(services_to_be_sorting) > 0:
                denum2 += 1
                env = MultibandOpticalNetworkEnv(topology2, service_dict2, services_to_be_sorting, max_agent, tmp_service2)
                obs, ava, agent_mask = env.reset()
                obs = np.expand_dims(obs, 0)  # (1,N,obs_dim)
                # print('obs:', obs)
                ava = np.expand_dims(ava, 0)  # (1,N,act_dim)  或 None
                agent_mask = np.expand_dims(agent_mask, 0)  # (1,N,1)
                start_time2 = time.time()
                with torch.no_grad():
                    actions, logp = policy.get_actions(obs, agent_mask, ava, deterministic=True)
                actions, logp = actions[0], logp[0]  # 去掉 batch 维
                actions_flat = actions.view(-1)
                reward, _, __ = env.make_step(actions)

                # print('actions:', actions_flat.tolist())
                # print('reward2:', tmp_service2.bit_rate, reward, utilization_incre, GSNR_incre)

                end_time2 = time.time()
                tmp_time2 += end_time2 - start_time2

                topology2 = env.topology
                service_dict2 = env.service_dict

                path2, wavelength2, info = random_fit(topology2, tmp_service2, service_dict2)
                if path2 == None:
                    blocknum2 += 1
                # else:
                #     print('success!!!!')

            else:
                blocknum2 += 1


    print("blocking_num: ", blocknum1, blocknum2)
    print('整理次数：', denum2)

    return {
        'blocknum1': blocknum1,
        'blocknum2': blocknum2
    }, service_dict2
