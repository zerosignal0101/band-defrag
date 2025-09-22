import numpy as np
import random
import sys
import torch

from envs.multiband_optical_network_env import MultibandOpticalNetworkEnv
from mat_algorithm.algorithms.mat.algorithm.transformer_policy import TransformerPolicy
from mat_algorithm.config import get_config
from base_functions import read_graphml_as_topology, parse_args, new_service_dict
from blocking_test import blocking_test


rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(42)
random.seed(42)
np.set_printoptions(threshold=np.inf)
max_agent = 30

graphml_file = "network_sweden_example.graphml"  # 前端生成
erlang = 200  # 前端指定，业务负载率
service_num = 200  # 前端指定,仿真业务个数
topology, wkey = read_graphml_as_topology(graphml_file, relabel_to_int=True)
args = sys.argv[1:]
parser = get_config()
all_args = parse_args(args, parser)
dummy_env = MultibandOpticalNetworkEnv(None, None, {}, max_agent, None)
policy = TransformerPolicy(all_args, dummy_env.observation_space[0], dummy_env.action_space[0], device=torch.device("cpu"))
policy.restore("mat_algorithm/1_transformer_5000.pt")

services = new_service_dict(topology, erlang, service_num)
result, service_dict = blocking_test(topology, services, max_agent, policy)
print(result)
print(result['blocknum1']/service_num, result['blocknum2']/service_num)  # 不进行碎片整理的阻塞率 & 进行碎片整理后的阻塞率
for sid, srv in service_dict.items():
    print(f"Service {sid}: path={srv.path}, wavelength={srv.wavelength}")  # 所有网络中存在业务的编号、路由和载波编号