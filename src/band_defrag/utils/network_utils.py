import networkx as nx
import numpy as np
from itertools import islice
import copy
from numpy import log10, abs, arange, arcsinh, isfinite, log, mean, pi, sum, zeros
import math
from tqdm import tqdm
import random
from numba import njit, prange
import time
import torch
import sys

from band_defrag.models.path import Path
from band_defrag.models.service import Service

SEED = 42
random.seed(SEED)
np.random.seed(SEED)  # 旧 API
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def parse_args(args, parser):
    parser.add_argument('--n_agent', type=int, default=30)
    all_args = parser.parse_known_args(args)[0]

    return all_args


def get_k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight_attr='weight'):
    return float(np.sum([graph[path[i]][path[i + 1]][weight_attr] for i in range(len(path) - 1)]))


def read_graphml_as_topology(file, relabel_to_int=True, edge_weight_key_preference=('weight', 'length')):
    """
    读取 GraphML，并返回一个满足你后续流程(topology.graph[...] 等)的无向图。
    - relabel_to_int: 是否把节点重标为 "0","1","2"...（字符串），与你原始 txt 读取保持一致
    - edge_weight_key_preference: 指定最短路权重优先使用哪一个属性
    """
    G_loaded = nx.read_graphml(file)  # 读进来可能是 MultiGraph，也可能是 Graph
    # 如果是 MultiGraph，先合并成简单图：保留第一条边的属性（或你也可以自定义规则）
    if isinstance(G_loaded, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph()
        H.add_nodes_from(G_loaded.nodes(data=True))
        for u, v, key, data in G_loaded.edges(keys=True, data=True):
            if H.has_edge(u, v):
                # 已有边时，可根据需要挑选更小的length/weight；此处保留已有
                continue
            H.add_edge(u, v, **data)
        G = H
    else:
        G = G_loaded

    # 统一成无向图（若原图是有向且你想保留方向，可以改为 nx.DiGraph）
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)

    # 可选：把节点重标为 "0","1","2"...（字符串）
    if relabel_to_int:
        mapping = {node: str(i) for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
        # 同时把 name 属性设为节点字符串 id
        for n in G.nodes():
            G.nodes[n]['name'] = n
    else:
        # 不重标也给个 name（用原 id 的字符串）
        for n in G.nodes():
            G.nodes[n]['name'] = str(n)

    # 确定用于最短路的权重字段
    # 优先使用 edge_weight_key_preference 中的第一个存在于任意边属性里的键
    edge_keys_present = set()
    for _, _, d in G.edges(data=True):
        edge_keys_present.update(d.keys())
    weight_attr = None
    for k in edge_weight_key_preference:
        if k in edge_keys_present:
            weight_attr = k
            break
    # 若都没有，就创建一个默认 weight=1、length=1
    if weight_attr is None:
        weight_attr = 'weight'
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            G[u][v]['length'] = 1.0
    else:
        # 确保两者都有（方便后续 get_path_weight 可用任一）
        for u, v, d in G.edges(data=True):
            if 'weight' not in d and weight_attr == 'length':
                d['weight'] = float(d.get('length', 1.0))
            if 'length' not in d and weight_attr == 'weight':
                d['length'] = float(d.get('weight', 1.0))

    # 标准化边的必要属性：id/index（连续编号），并转成 float/int
    edge_id = 0
    for u, v in G.edges():
        G[u][v]['id'] = edge_id
        G[u][v]['index'] = edge_id
        # 规范化数值类型
        for key in ['weight', 'length']:
            if key in G[u][v]:
                try:
                    G[u][v][key] = float(G[u][v][key])
                except Exception:
                    G[u][v][key] = 1.0
        edge_id += 1

    # 初始化每条边的 80 个频隙属性
    for u, v in G.edges():
        wavelength_power = np.zeros(80, dtype=float)
        wavelength_utilization = np.zeros(80, dtype=float)
        wavelength_SNR = np.zeros(80, dtype=float)
        wavelength_service = np.zeros(80, dtype=int)
        wavelength_bitrate = np.zeros(80, dtype=float)
        G[u][v]['wavelength_power'] = wavelength_power
        G[u][v]['wavelength_utilization'] = wavelength_utilization
        G[u][v]['wavelength_SNR'] = wavelength_SNR
        G[u][v]['wavelength_service'] = wavelength_service
        G[u][v]['wavelength_bitrate'] = wavelength_bitrate
        G[u][v]['numsp'] = 0
        # edge_id 已有

    # 计算 k 最短路并写入 graph-level 属性
    k_paths = 1
    k_shortest_paths = {}
    path_counter = 0
    nodes_list = list(G.nodes())
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            s, t = nodes_list[i], nodes_list[j]
            paths = list(islice(nx.shortest_simple_paths(G, s, t, weight=weight_attr), k_paths))
            lengths = [get_path_weight(G, p, weight_attr if weight_attr else 'weight') for p in paths]
            objs = []
            for p, L in zip(paths, lengths):
                objs.append(Path(path_id=path_counter, node_list=p, length=L))
                path_counter += 1
            k_shortest_paths[(s, t)] = objs
            k_shortest_paths[(t, s)] = objs

    G.graph['name'] = 'sweden'
    G.graph['ksp'] = k_shortest_paths
    G.graph['k_paths'] = k_paths

    # node_indices / 每个节点的 index
    G.graph['node_indices'] = []
    for idx, node in enumerate(G.nodes()):
        G.graph['node_indices'].append(node)
        G.nodes[node]['index'] = idx

    return G, weight_attr


def process_topology(topology, edge_weight_key_preference=('weight', 'length'), num_channels=80):
    # 如果是 MultiGraph，先合并成简单图：保留第一条边的属性（或你也可以自定义规则）
    if isinstance(topology, (nx.MultiGraph, nx.MultiDiGraph)):
        H = nx.Graph()
        H.add_nodes_from(topology.nodes(data=True))
        for u, v, key, data in topology.edges(keys=True, data=True):
            if H.has_edge(u, v):
                # 已有边时，可根据需要挑选更小的length/weight；此处保留已有
                continue
            H.add_edge(u, v, **data)
        G = H
    else:
        G = topology

    # 统一成无向图（若原图是有向且你想保留方向，可以改为 nx.DiGraph）
    if isinstance(G, nx.DiGraph):
        G = nx.Graph(G)

    # 确定用于最短路的权重字段
    # 优先使用 edge_weight_key_preference 中的第一个存在于任意边属性里的键
    edge_keys_present = set()
    for _, _, d in G.edges(data=True):
        edge_keys_present.update(d.keys())
    weight_attr = None
    for k in edge_weight_key_preference:
        if k in edge_keys_present:
            weight_attr = k
            break
    # 若都没有，就创建一个默认 weight=1、length=1
    if weight_attr is None:
        weight_attr = 'weight'
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
            G[u][v]['length'] = 1.0
    else:
        # 确保两者都有（方便后续 get_path_weight 可用任一）
        for u, v, d in G.edges(data=True):
            if 'weight' not in d and weight_attr == 'length':
                d['weight'] = float(d.get('length', 1.0))
            if 'length' not in d and weight_attr == 'weight':
                d['length'] = float(d.get('weight', 1.0))

    # 标准化边的必要属性：id/index（连续编号），并转成 float/int
    edge_id = 0
    for u, v in G.edges():
        G[u][v]['id'] = edge_id
        G[u][v]['index'] = edge_id
        # 规范化数值类型
        for key in ['weight', 'length']:
            if key in G[u][v]:
                try:
                    G[u][v][key] = float(G[u][v][key])
                except Exception:
                    G[u][v][key] = 1.0
        edge_id += 1

    # 初始化每条边的 num_channels 个频隙属性
    for u, v in G.edges():
        wavelength_power = np.zeros(num_channels, dtype=float)
        wavelength_utilization = np.zeros(num_channels, dtype=float)
        wavelength_SNR = np.zeros(num_channels, dtype=float)
        wavelength_service = np.zeros(num_channels, dtype=int)
        wavelength_bitrate = np.zeros(num_channels, dtype=float)
        G[u][v]['wavelength_power'] = wavelength_power
        G[u][v]['wavelength_utilization'] = wavelength_utilization
        G[u][v]['wavelength_SNR'] = wavelength_SNR
        G[u][v]['wavelength_service'] = wavelength_service
        G[u][v]['wavelength_bitrate'] = wavelength_bitrate
        G[u][v]['numsp'] = 0
        # edge_id 已有

    # 计算 k 最短路并写入 graph-level 属性
    k_paths = 1
    k_shortest_paths = {}
    path_counter = 0
    nodes_list = list(G.nodes())
    for i in range(len(nodes_list)):
        for j in range(i + 1, len(nodes_list)):
            s, t = nodes_list[i], nodes_list[j]
            paths = list(islice(nx.shortest_simple_paths(G, s, t, weight=weight_attr), k_paths))
            lengths = [get_path_weight(G, p, weight_attr if weight_attr else 'weight') for p in paths]
            
            # 存储正向路径 (s -> t)
            forward_objs = []
            for p, L in zip(paths, lengths):
                forward_objs.append(Path(path_id=path_counter, node_list=p, length=L))
                path_counter += 1
            k_shortest_paths[(s, t)] = forward_objs
            
            # 存储反向路径 (t -> s)
            backward_objs = []
            for obj in forward_objs:
                # 反转节点列表但保持相同的路径ID和长度
                reversed_path = Path(
                    path_id=obj.path_id,
                    node_list=list(reversed(obj.node_list)),  # 反转节点顺序
                    length=obj.length  # 长度保持不变
                )
                backward_objs.append(reversed_path)
            k_shortest_paths[(t, s)] = backward_objs

    G.graph['name'] = 'sweden'
    G.graph['ksp'] = k_shortest_paths
    G.graph['k_paths'] = k_paths

    # node_indices / 每个节点的 index
    G.graph['node_indices'] = []
    for idx, node in enumerate(G.nodes()):
        G.graph['node_indices'].append(node)
        G.nodes[node]['index'] = idx

    return G, weight_attr


def new_service(topology, upper_bitrate, services_processed_since_reset):
    '''
    生成一个随机业务
    '''
    src, __, dst, ___ = _get_node_pair(topology)

    # 用自定义函数做加权采样，小比特率业务比重大
    bit_rate_candidates = np.arange(100, upper_bitrate, 10)  # 每10为一个档
    # 权重：反比于比特率，越小越重

    # Exponential weighting
    beta = 0.005
    weights_exp = np.exp(-beta * bit_rate_candidates)
    weights_exp = weights_exp / weights_exp.sum()

    bit_rate = np.random.choice(bit_rate_candidates, p=weights_exp)
    # bit_rate = random.randint(100, 520)

    service = Service(service_id=services_processed_since_reset,
                      source_id=src, destination_id=dst, bit_rate=bit_rate)

    # services_processed_since_reset += 1

    return service


def new_service_dict(topology, avg_arrival_interval, avg_holding_time, service_arrival_time_max, upper_bitrate=601):
    # 定义仿真参数
    lambda_rate = 1 / avg_arrival_interval # 到达率
    mu_rate = 1 / avg_holding_time  # 持续时间的倒数
    # 初始化变量
    time = 0
    calls_in_progress = 0
    total_calls = 0
    call_arrivals = []
    call_departures = []
    service_dict = {}
    # 开始仿真
    while time < service_arrival_time_max:
        # 下一次到达时间（到达间隔时间服从参数为λ的指数分布）
        time_to_next_arrival = np.random.exponential(1 / lambda_rate)
        time += time_to_next_arrival
        if time >= service_arrival_time_max:
            break

        # 记录呼叫到达时间
        call_arrivals.append(time)
        # 计算呼叫持续时间（服从参数为μ的指数分布）
        call_duration = np.random.exponential(1 / mu_rate)
        call_departure_time = time + call_duration

        # 更新正在进行的呼叫数量
        calls_in_progress += 1

        tmp_service = new_service(topology, upper_bitrate, total_calls + 1)
        total_calls += 1
        tmp_service.arrival_time = time
        tmp_service.departure_time = call_departure_time
        service_dict[tmp_service.service_id] = tmp_service

    return service_dict


# one_link_transmission references
# [1] A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems
# [2] Modeling and mitigation of fiber nonlinearity in wideband optical signal transmission

# 1. one_link_transmission function
@njit(nogil=True, cache=True)
def get_alpha(frequency):
    '''
    计算不同频率处的alpha值
    '''
    wavelength0 = 1550e-9
    wavelength = 3e8 / frequency
    alpha0 = 0.162
    alpha1 = -7.3764e-5
    alpha2 = 3.7685e-6
    alpha = alpha2 * (wavelength - wavelength0) ** 2 + alpha1 * (wavelength - wavelength0) + alpha0
    #print('alpha:', alpha)
    return alpha / 4.343 / 1e3


@njit(nogil=True, cache=True)
def get_r_f(frequency, P_total):
    '''
    计算SRS的中间过程
    '''
    delta_f = 15e12
    f_m1 = 184.325e12
    f_M1 = 190.325e12
    f_m2 = 190.675e12
    f_M2 = 196.675e12
    B_t = 12e12
    if frequency - delta_f < f_m1 and frequency + delta_f > f_M1:
        return P_total * frequency
    elif frequency - delta_f > f_m1 and frequency + delta_f < f_M1:
        return 0.0
    elif frequency - delta_f < f_m1 and frequency + delta_f < f_M1:
        return P_total / B_t * (frequency ** 2 / 2 - frequency * f_m1 + (f_M1 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f > f_m1 and frequency + delta_f > f_M1:
        return P_total / B_t * (frequency * f_M1 - frequency ** 2 / 2 - (f_m1 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f < f_m2 and frequency + delta_f > f_M2:
        return P_total * frequency
    elif frequency - delta_f > f_m2 and frequency + delta_f < f_M2:
        return 0.0
    elif frequency - delta_f < f_m2 and frequency + delta_f < f_M2:
        return P_total / B_t * (frequency ** 2 / 2 - frequency * f_m2 + (f_M2 ** 2 - delta_f ** 2) / 2)
    elif frequency - delta_f > f_m2 and frequency + delta_f > f_M2:
        return P_total / B_t * (frequency * f_M2 - frequency ** 2 / 2 - (f_m2 ** 2 - delta_f ** 2) / 2)
    else:
        return 0.0


@njit(parallel=True, nogil=True, cache=True)
def get_Pi_z(distance, P_total, frequencies, Power, i, channels):
    '''
    计算经过distance传输后各信道的功率值
    '''
    C_r = 0.028 / 1e3 / 1e12
    alpha = get_alpha(frequencies[i])
    L_eff = (1 - np.exp(-alpha * distance)) / alpha
    tmp_frequency = frequencies[i]
    r_f = get_r_f(tmp_frequency, P_total)
    tmp_sum = 0.0
    for j in prange(channels):
        alpha_j = get_alpha(frequencies[j])
        L_eff_j = (1 - np.exp(-alpha_j * distance)) / alpha_j
        r_f_j = get_r_f(frequencies[j], P_total)
        # print('r_fj:', r_f_j)
        # print('re:',Power[j] * np.exp(-C_r*L_eff_j*r_f_j))
        # 确保每一步都是标量操作
        power_j = Power[j]
        exp_term = np.exp(-C_r * L_eff_j * r_f_j)
        contribution = power_j * exp_term  # 将乘积转换为标量

        tmp_sum += contribution
    result = Power[i] * (np.exp(-C_r * L_eff * r_f) * P_total) / tmp_sum
    # print("Power after z:", result)
    return result


@njit(nogil=True, cache=True)
def lin2db(value):
    '''
    ln -> dB
    '''
    return 10 * log10(value)


def watt2dbm(value):
    '''
    W -> dBm
    '''
    return lin2db(value * 1e3)


@njit(nogil=True, cache=True)
def removenan(x):
    '''
    除掉nan值
    '''
    x[~isfinite(x)] = 0
    return x


def todB(x):
    '''
    转化为dB
    '''
    return 10 * log(x) / log(10)


@njit(parallel=True, nogil=True, cache=True)
def calculate_ASE_noise(Att, fi, Bch, distance):
    channels, n = fi.shape
    c = 3e8
    n_sp = 1.41  # n_sp = NF(4.5dB)/2
    h = 6.62607015e-34
    RefLambda = 1575e-9  # 假设这是你的常量
    num_of_spans = int(np.ceil(distance / 100e3))
    Length = 100 * 1e3 * np.ones(num_of_spans)
    single_ASE = np.zeros((channels, n))

    for j in prange(n):
        for i in prange(channels):
            a_i = Att[i, j]  # \alpha of COI in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            length_i = Length[j]
            single_ASE[i, j] = 2 * n_sp * h * (f_i + c / RefLambda) * B_i * (np.exp(a_i * length_i) - 1)

    return np.sum(single_ASE, axis=1)


@njit(parallel=True, nogil=True, cache=True)
def _numba_one_link_transmission(distance, channels, Power, frequencies):
    '''
    计算链路 ASE 噪声和调整后的 Power（考虑信号再分配）。
    输入的 Power 不会被修改。
    '''
    # 复制 Power 避免副作用
    Power_out = np.copy(Power)
    P_total = np.sum(Power_out)

    for i in prange(channels):
        Power_out[i] = get_Pi_z(100e3, P_total, frequencies, Power, i, channels)

    num_of_spans = int(np.ceil(distance / 100e3))
    RefFreq = 190.5e12
    fi = np.array([(f - RefFreq) for f in frequencies]).reshape(-1, 1)

    Bch = 150e9 * np.ones((channels, num_of_spans))
    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))

    ASE_noise = calculate_ASE_noise(Att, fi, Bch, distance)

    return ASE_noise, Power_out

    # return copy_Power


@njit(nogil=True, cache=True)
def cal_eps(B_i, f_i, a_i, mean_L, beta2, beta3):
    return (3 / 10) * log(1 + (6 / a_i) / (
            mean_L * arcsinh(pi ** 2 / 2 * abs(mean(beta2) + 2 * pi * mean(beta3) * f_i) / a_i * B_i ** 2)))


@njit(nogil=True, cache=True)
def cal_SPM(phi_i, T_i, B_i, a, a_bar, gamma):
    return 4 / 9 * gamma ** 2 / B_i ** 2 * pi / (phi_i * a_bar * (2 * a + a_bar)) \
        * ((T_i - a ** 2) / a * arcsinh(phi_i * B_i ** 2 / a / pi) + ((a + a_bar) ** 2 - T_i) / (a + a_bar) * arcsinh(
            phi_i * B_i ** 2 / (a + a_bar) / pi))


@njit(nogil=True, cache=True)
def cal_XPM(Pi, Pk, phi_ik, T_k, B_i, B_k, a, a_bar, gamma):
    if Pi == 0:
        return 0
    else:
        return 32 / 27 * sum((Pk / Pi) ** 2 * gamma ** 2 / (B_k * phi_ik * a_bar * (2 * a + a_bar))
                             * ((T_k - a ** 2) / a * np.arctan(phi_ik * B_i / a)
                                + ((a + a_bar) ** 2 - T_k) / (a + a_bar) * np.arctan(phi_ik * B_i / (a + a_bar)))
                             )


@njit(parallel=True, nogil=True, cache=True)
def calculate_NLI_noise(
        Att, Att_bar, Cr, Pch, fi, Bch, Length, D, S, gamma, RefLambda):
    """
    Returns nonlinear interference power and coefficient for each WDM
    channel.

    Format:
    - channel dependent quantities have the format of a N_ch x n matrix,
      where N_ch is the number of channels slots and n is the number of spans.
    - channel independent quantities have the format of a 1 x n matrix
    - channel and span independent quantities are scalars

    INPUTS:
        Att: attenuation coefficient [Np/m] of channel i of span j,
            format: N_ch x n matrix
        Att_bar: attenuation coefficient (bar) [Np/m] of channel i of span j,
            format: N_ch x n matrix
        Cr[i,j]: the slope of the linear regression of the normalized Raman gain spectrum [1/W/m/Hz] of channel i of span j,
            format: N_ch x n matrix

        Pch[i,j]:  the launch power [W] of channel i of span j,
            format: N_ch x n matrix
        fi[i,j]: center frequency relative to the reference frequency (3e8/RefLambda) [Hz]
            of channel i of span j, format: N_ch x n matrix
        Bch[i,j]: the bandwidth [Hz] of channel i of span j,
            format: N_ch x n matrix

        Length[j]: the span length [m] of span j,
            format: 1 x n vector
        D[j]: the dispersion coefficient [s/m^2] of span j,
            format: 1 x n vector
        S[j]: the span length [s/m^3] of span j,
            format: 1 x n vector
        gamma[j]: the span length [1/W/m] of span j,
            format: 1 x n vector
        RefLambda: is the reference wavelength (where beta2, beta3 are defined) [m],
            format: 1 x 1 vector

        coherent: boolean for coherent or incoherent NLI accumulation across multiple fiber spans

    RETURNS:
        NLI: Nonlinear Interference Power[W],
            format: N_ch x 1 vector
        eta_n: Nonlinear Interference coeffcient [1/W^2],
            format: N_ch x 1 matrix
    """
    channels, n = Att.shape

    c = 3e8

    a = Att
    a_bar = Att_bar
    L = Length
    P_ij = Pch
    # print("Power:",P_ij)
    Ptot = np.sum(P_ij, axis=0)

    beta2 = -D * RefLambda ** 2 / (2 * pi * c)
    beta3 = RefLambda ** 2 / (2 * pi * c) ** 2 * (RefLambda ** 2 * S + 2 * RefLambda * D)
    # n_sp = 1.41  # n_sp = NF(4.5dB)/2  from "modeling and mitigation of fiber nonlinearity in wideband optical signal transmission"
    # h = 6.62607015*1e-34

    # Average Coherence Factor
    # mean_att_i = mean(a, axis=1)  # average attenuation coefficent for channel i
    mean_att_i = np.empty(a.shape[0])
    for i in range(a.shape[0]):
        mean_att_i[i] = np.sum(a[i]) / a.shape[1]  # 计算每行的均值
    mean_L = np.mean(L)  # average fiber length

    eta_SPM = zeros((channels, n))
    eta_XPM = zeros((channels, n))

    for j in prange(n):
        """ Calculation of nonlinear interference (NLI) power in fiber span j """
        for i in prange(channels):
            """ Compute the NLI of each COI """
            not_i = arange(channels) != i
            a_i = a[i, j]  # \alpha of COI in fiber span j
            a_k = a[not_i, j]  # \alpha of INT in fiber span j
            a_bar_i = a_bar[i, j]  # \bar{\alpha} of COI in fiber span j
            a_bar_k = a_bar[not_i, j]  # \bar{\alpha} of INT in fiber span j
            f_i = fi[i, j]  # f_i of COI in fiber span j
            #print(f_i)
            f_k = fi[not_i, j]  # f_k of INT in fiber span j
            B_i = Bch[i, j]  # B_i of COI in fiber span j
            B_k = Bch[not_i, j]  # B_k of INT in fiber span j
            Cr_i = Cr[i, j]  # Cr  of COI in fiber span j
            Cr_k = Cr[not_i, j]  # Cr  of INT in fiber span j
            P_i = P_ij[i, j]  # P_i of COI in fiber span j
            P_k = P_ij[not_i, j]  # P_k of INT in fiber span j

            phi_i = 3 / 2 * pi ** 2 * (beta2[j] + pi * beta3[j] * (f_i + f_i))  # \phi_i of COI in fiber span j
            phi_ik = 2 * pi ** 2 * (f_k - f_i) * (
                    beta2[j] + pi * beta3[j] * (f_i + f_k))  # \phi_ik of COI-INT pair in fiber span j

            T_i = (a_i + a_bar_i - f_i * Ptot[j] * Cr_i) ** 2  # T_i of COI in fiber span j
            T_k = (a_k + a_bar_k - f_k * Ptot[j] * Cr_k) ** 2  # T_k of INT in fiber span j

            eta_SPM[i, j] = cal_SPM(phi_i, T_i, B_i, a_i, a_bar_i, gamma[j]) * n ** cal_eps(B_i, f_i, mean_att_i[i],
                                                                                            mean_L, beta2,
                                                                                            beta3)  # computation of SPM contribution in fiber span j
            eta_XPM[i, j] = cal_XPM(P_i, P_k, phi_ik, T_k, B_i, B_k, a_k, a_bar_k,
                                    gamma[j])  # computation of XPM contribution in fiber span j

    nonzero_mask = P_ij[:, 0] != 0  # 创建一个布尔掩码，表示 P_ij[:, 0] 是否不等于零
    eta_n = np.zeros(channels)  # 先将所有值初始化为零
    # eta_n[nonzero_mask] = np.sum(
    #     (eta_SPM[nonzero_mask] + eta_XPM[nonzero_mask]),
    #     axis=1)
    eta_n[nonzero_mask] = np.sum(eta_SPM[nonzero_mask] + eta_XPM[nonzero_mask], axis=1)

    # computation of NLI normalized to transmitter power, see Ref. [1, Eq. (5)]
    NLI = P_ij[:, 0] ** 3 * eta_n  # Ref. [1, Eq. (1)]

    #print("NLI:",NLI)
    return NLI


@njit(parallel=True, nogil=True, cache=True)
def calculate_GSNR(Power, noise, channels):
    GSNR = np.zeros(channels)
    for i in prange(channels):
        if Power[i] != 0:
            GSNR[i] = lin2db(Power[i] / noise[i])
        else:
            GSNR[i] = 0
    return GSNR


@njit(parallel=True, nogil=True, cache=True)
def tile_implementation(fi, num_of_spans):
    rows, cols = fi.shape
    expanded = np.empty((rows, num_of_spans))  # 创建一个新的数组，大小为 (rows, 8)
    for i in prange(rows):
        for j in prange(num_of_spans):
            expanded[i, j] = fi[i, 0]  # 将每一行复制到新的数组中
    return expanded


def one_link_transmission(distance, channels, Power, frequencies):
    '''
    单链路传输计算，返回链路上每个信道的 GSNR。
    不修改原始 Power 数组，避免副作用。
    '''
    # 获取 ASE 噪声和更新后的 Power（考虑 SRS 等效）
    ASE_noise, adjusted_Power = _numba_one_link_transmission(
        distance, channels, Power, frequencies
    )

    # 物理参数设置
    num_of_spans = math.ceil(distance / 100e3)
    RefFreq = 190.5e12  # Hz

    fi = [(f - RefFreq) for f in frequencies]
    fi = np.array(fi).reshape(-1, 1)
    fi = tile_implementation(fi, num_of_spans)

    Att = 0.2 / 4.343 / 1e3 * np.ones((channels, num_of_spans))
    Att_bar = Att
    Cr = 0.028 / 1e3 / 1e12 * np.ones((channels, num_of_spans))

    # 使用 adjusted_Power 生成 Pch
    tmpPower = 0.0031622776602 * ((adjusted_Power != 0).astype(int))
    tmpPower = tmpPower.reshape(channels, 1)
    Pch = tile_implementation(tmpPower, num_of_spans)

    Bch = np.full((channels, num_of_spans), 150e9)
    Length = 100e3 * np.ones(num_of_spans)
    D = 17e-12 / 1e-9 / 1e3 * np.ones(num_of_spans)
    S = 0.067e-12 / 1e-9 / 1e3 / 1e-9 * np.ones(num_of_spans)
    gamma = 1.21 / 1e3 * np.ones(num_of_spans)
    RefLambda = 1575e-9

    # 计算非线性干扰
    NLI_noise = calculate_NLI_noise(Att, Att_bar, Cr, Pch, fi, Bch, Length, D, S, gamma, RefLambda)

    # 总噪声
    noise = NLI_noise + ASE_noise

    # 计算 GSNR
    GSNR = calculate_GSNR(adjusted_Power, noise, channels)

    return adjusted_Power, GSNR


# # 使用one_link_transmission的一个实例
# distance = 600e3
# channels = 80
# Power = 0.0031622776602 * np.ones(channels)
# # Power[0:10] = 0
# # Power[40:50] = 0
# frequencies = np.concatenate([np.linspace(184.4e12, 190.25e12, channels // 2), np.linspace(190.75e12, 196.6e12, channels // 2)])
# GSNR = one_link_transmission(distance, channels, Power, frequencies)
# print(GSNR)


# 2. generate_service
# 固定随机数
rng1 = random.Random(42)
rng2 = random.Random(42)
services_processed_since_reset = 1


def _get_node_pair(topology):
    """
    Uses the `node_request_probabilities` variable to generate a source and a destination.

    :return: source node, source node id, destination node, destination node id
    """
    # 按概率生成源节点
    # print('node:', len(topology.nodes()), len(node_request_probabilities))
    node_request_probabilities = 1 / len(topology.nodes()) * np.ones(len(topology.nodes()), dtype=float)

    src = rng1.choices([x for x in topology.nodes()], weights=node_request_probabilities)[
        0]  #Escoge aleatoriamente un nodo, si se le pasa probabilidad la tiene en cuenta, sino genera trafico uniforme.
    # 原节点id
    src_id = topology.graph['node_indices'].index(src)
    new_node_probabilities = np.copy(node_request_probabilities)
    # 防止原节点被选为目的节点
    new_node_probabilities[src_id] = 0.
    new_node_probabilities = new_node_probabilities / np.sum(new_node_probabilities)
    # 选定目的节点
    dst = rng1.choices([x for x in topology.nodes()], weights=new_node_probabilities)[0]
    dst_id = topology.graph['node_indices'].index(dst)
    return src, src_id, dst, dst_id


# 3. allocate a service
def random_fit(topology, service:Service, service_dict, channels, frequencies):
    '''
    input: topology, service, service.bitrate \in [400,800]
    output: path_node_list, wavelength j
    '''
    src = service.source_id
    dst = service.destination_id
    bit_rate = service.bit_rate
    total_utilization = 0
    allocation = False
    reason = None
    if bit_rate > 700:
        service.snr_requirement = 26.5-1
    elif bit_rate > 600:
        service.snr_requirement = 25.0-1
    elif bit_rate > 500:
        service.snr_requirement = 23.5-1
    elif bit_rate > 400:
        service.snr_requirement = 21.0-1
    elif bit_rate > 300:
        service.snr_requirement = 18.7-1
    else:
        service.snr_requirement = 15

    for path in topology.graph['ksp'][str(src), str(dst)]:
        path_start_time = time.time()
        start_wavelength = rng1.randint(0, channels - 1)
        wave_reason = np.zeros(channels)
        for offset in range(channels):
            j = (start_wavelength + offset) % channels
            allocation = True
            outer_break = False
            Power = np.zeros((len(path.node_list), channels), dtype=float)
            path_GSNR = np.zeros((len(path.node_list), channels), dtype=float)
            for i in range((len(path.node_list) - 1)):
                if outer_break:
                    break
                u = path.node_list[i]
                v = path.node_list[i + 1]

                # 检查波长是否空闲
                if not (topology[u][v]['wavelength_power'][j] == 0 or (
                np.isnan(topology[u][v]['wavelength_power'][j]))):
                    wave_reason[j] = 1
                    allocation = False
                    break
                else:
                    # 检查SNR是否满足要求
                    # 获取链路参数
                    distance = topology[u][v]['length']
                    Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
                    if j < 40:
                        service.power = 0.002754
                    else:
                        service.power = 0.003890
                    Power[i][j] = service.power
                    tmp = copy.deepcopy(Power[i])
                    tmp = np.array(tmp)
                    # 计算链路的GSNR
                    Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
                    path_GSNR[i] = GSNR

                    if service.snr_requirement > GSNR[j]:
                        wave_reason[j] = 2
                        allocation = False
                        break
                    else:
                        # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                        for m in range(channels):
                            if m != j and topology[u][v]['wavelength_service'][m] != 0:
                                tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                                if tmp_service.snr_requirement >= GSNR[m]:
                                    allocation = False
                                    wave_reason[j] = 3
                                    outer_break = True
                                    break

            if allocation:
                service.path = path.node_list
                service.wavelength = j
                for i in range(len(path.node_list) - 1):
                    u = path.node_list[i]
                    v = path.node_list[i + 1]
                    topology[u][v]['wavelength_power'] = Power[i]
                    topology[u][v]['wavelength_SNR'] = path_GSNR[i]
                    topology[u][v]['wavelength_bitrate'][j] = service.bit_rate
                    if topology[u][v]['wavelength_SNR'][j] >= 26.5 - 1:
                        capacity = 800
                    elif topology[u][v]['wavelength_SNR'][j] >= 25.0 - 1:
                        capacity = 700
                    elif topology[u][v]['wavelength_SNR'][j] >= 23.5 - 1:
                        capacity = 600
                    elif topology[u][v]['wavelength_SNR'][j] >= 21.0 - 1:
                        capacity = 500
                    elif topology[u][v]['wavelength_SNR'][j] >= 18.7 - 1:
                        capacity = 400
                    elif topology[u][v]['wavelength_SNR'][j] >= 15:
                        capacity = 200
                    else:
                        capacity = 0.1
                    topology[u][v]['wavelength_utilization'][j] = bit_rate / capacity
                    total_utilization += bit_rate / capacity
                    topology[u][v]['wavelength_service'][j] = service.service_id

                    # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
                    for wave in range(channels):
                        if wave != j and topology[u][v]['wavelength_service'][wave] != 0:
                            service_id = topology[u][v]['wavelength_service'][wave]
                            tmp_service = service_dict.get(service_id, None)
                            # print('tmp_service:', service_id)
                            if topology[u][v]['wavelength_SNR'][wave] >= 26.5 - 1:
                                capacity = 800
                            elif topology[u][v]['wavelength_SNR'][wave] >= 25.0 - 1:
                                capacity = 700
                            elif topology[u][v]['wavelength_SNR'][wave] >= 23.5 - 1:
                                capacity = 600
                            elif topology[u][v]['wavelength_SNR'][wave] >= 21.0 - 1:
                                capacity = 500
                            elif topology[u][v]['wavelength_SNR'][wave] >= 18.7 - 1:
                                capacity = 400
                            elif topology[u][v]['wavelength_SNR'][wave] >= 15:
                                capacity = 200
                            else:
                                capacity = 0.1
                            topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity
                            tmp_service.utilization = tmp_service.bit_rate / capacity
                            service_dict[tmp_service.service_id] = tmp_service
                service.utilization = total_utilization / len(service.path)
                service_dict[service.service_id] = service
                return path.node_list, j, []
        path_end_time = time.time()
        # print(f"路径计算总耗时: {path_end_time - path_start_time:.6f} 秒")
    end_time = time.time()
    # print(f"函数总运行时间: {end_time - start_time:.6f} 秒")
    # print('分配失败', reason)
    return None, None, wave_reason


def check_action(action, topology, service, service_dict):
    '''
    耗时0.0x，好像还是过长？
    '''
    # ① 若 action 就是当前波长，直接成功返回，不修改拓扑
    if action == service.wavelength:
        return True, None, None
    release_service(topology, service, service_dict)

    allocation = True
    outer_break = False
    total_utilization = 0
    Power = np.zeros((len(service.path), 80), dtype=float)
    path_GSNR = np.zeros((len(service.path), 80), dtype=float)

    for i in range((len(service.path) - 1)):
        if outer_break:
            break
        u = service.path[i]
        v = service.path[i + 1]
        # 检查波长是否空闲
        if not (topology[u][v]['wavelength_power'][action] == 0
                or (np.isnan(topology[u][v]['wavelength_power'][action]))):
            allocation = False
            break
        else:
            # 检查SNR是否满足要求
            # 获取链路参数
            distance = topology[u][v]['length']
            channels = 80
            Power[i] = copy.deepcopy(topology[u][v]['wavelength_power'])
            if action < 40:
                service.power = 0.00275422870
            else:
                service.power = 0.00389045144
            Power[i][action] = service.power

            frequencies = np.concatenate([
                np.linspace(184.4e12, 190.25e12, channels // 2),
                np.linspace(190.75e12, 196.6e12, channels // 2)
            ])
            tmp = copy.deepcopy(topology[u][v]['wavelength_power'])
            tmp[action] = service.power
            tmp = np.array(tmp)

            Power_after_transmission, GSNR = one_link_transmission(distance, channels, tmp, frequencies)
            path_GSNR[i] = GSNR

            if service.snr_requirement > GSNR[action]:
                allocation = False
                reason = "GSNR not satisfied!!"
                break
            else:
                # 检查该业务会不会对其它业务有影响，如有影响，则拒绝
                for m in range(80):
                    if m != action and topology[u][v]['wavelength_service'][m] != 0:
                        tmp_service = service_dict.get(topology[u][v]['wavelength_service'][m], None)
                        if tmp_service.snr_requirement >= GSNR[m]:
                            allocation = False
                            reason = 'interference!'
                            outer_break = True
                            break

    if allocation:
        service.path = service.path
        service.wavelength = action
        for i in range(len(service.path) - 1):
            u = service.path[i]
            v = service.path[i + 1]
            topology[u][v]['wavelength_power'] = Power[i]
            topology[u][v]['wavelength_SNR'] = path_GSNR[i]
            topology[u][v]['wavelength_bitrate'][action] = service.bit_rate
            if topology[u][v]['wavelength_SNR'][action] >= 26.5 - 1:
                capacity = 800
            elif topology[u][v]['wavelength_SNR'][action] >= 25.0 - 1:
                capacity = 700
            elif topology[u][v]['wavelength_SNR'][action] >= 23.5 - 1:
                capacity = 600
            elif topology[u][v]['wavelength_SNR'][action] >= 21.0 - 1:
                capacity = 500
            elif topology[u][v]['wavelength_SNR'][action] >= 18.7 - 1:
                capacity = 400
            elif topology[u][v]['wavelength_SNR'][action] >= 15:
                capacity = 200
            else:
                capacity = 0
            topology[u][v]['wavelength_utilization'][action] = service.bit_rate / capacity
            total_utilization += service.bit_rate / capacity
            topology[u][v]['wavelength_service'][action] = service.service_id

            # # 重新更新涉及链路上所有波长处的带宽利用率！！！！！
            for wave in range(80):
                if wave != action and topology[u][v]['wavelength_service'][wave] != 0:
                    service_id = topology[u][v]['wavelength_service'][wave]
                    tmp_service = service_dict.get(service_id, None)
                    # print('tmp_service:', service_id)
                    if topology[u][v]['wavelength_SNR'][wave] >= 26.5 - 1:
                        capacity = 800
                    elif topology[u][v]['wavelength_SNR'][wave] >= 25.0 - 1:
                        capacity = 700
                    elif topology[u][v]['wavelength_SNR'][wave] >= 23.5 - 1:
                        capacity = 600
                    elif topology[u][v]['wavelength_SNR'][wave] >= 21.0 - 1:
                        capacity = 500
                    elif topology[u][v]['wavelength_SNR'][wave] >= 18.7 - 1:
                        capacity = 400
                    elif topology[u][v]['wavelength_SNR'][wave] >= 15:
                        capacity = 200
                    else:
                        capacity = 0
                    topology[u][v]['wavelength_utilization'][wave] = tmp_service.bit_rate / capacity
                    tmp_service.utilization = tmp_service.bit_rate / capacity
                    service_dict[tmp_service.service_id] = tmp_service

        service.utilization = total_utilization / len(service.path)
        service_dict[service.service_id] = service
    return allocation, Power, path_GSNR


def compute_current_service_utilization_increment(origin_topology, origin_service, topology, service, blocked_service):
    origin_utilization = 0
    current_utilization = 0
    service_path = service.path
    blocked_path = blocked_service.path
    cnt = 0

    # 构造链路集合用于匹配（无向边，确保顺序一致）
    blocked_links = set((min(blocked_path[i], blocked_path[i + 1]), max(blocked_path[i], blocked_path[i + 1]))
                        for i in range(len(blocked_path) - 1))

    for i in range((len(service_path) - 1)):
        u = service_path[i]
        v = service_path[i + 1]
        link = (min(u, v), max(u, v))  # 无向边匹配
        if link in blocked_links:
            origin_utilization += origin_topology[u][v]['wavelength_utilization'][origin_service.wavelength]
            current_utilization += topology[u][v]['wavelength_utilization'][service.wavelength]
            cnt += 1
    reward = (current_utilization - origin_utilization) / cnt * 100
    # print('reward1:', reward)
    return reward


def compute_current_service_GSNR_increment(origin_topology, origin_service, topology, service, blocked_service):
    origin_GSNR = 0
    current_GSNR = 0
    service_path = service.path
    blocked_path = blocked_service.path

    # 构造链路集合用于匹配（无向边，确保顺序一致）
    blocked_links = set((min(blocked_path[i], blocked_path[i + 1]), max(blocked_path[i], blocked_path[i + 1]))
                        for i in range(len(blocked_path) - 1))

    cnt = 0
    for i in range((len(service_path) - 1)):
        u = service_path[i]
        v = service_path[i + 1]
        link = (min(u, v), max(u, v))  # 无向边匹配
        if link in blocked_links:
            origin_GSNR += origin_topology[u][v]['wavelength_SNR'][origin_service.wavelength]
            current_GSNR += topology[u][v]['wavelength_SNR'][service.wavelength]
            cnt += 1
    if cnt != 0:
        reward = (current_GSNR - origin_GSNR) / cnt
    else:
        reward = 0
    # print('reward2:', reward)
    return reward * 2


def compute_related_services_GSNR_increment(origin_topology, origin_service, topology, service, blocked_service):
    origin_GSNR = 0
    current_GSNR = 0
    service_path = service.path
    blocked_path = blocked_service.path

    # 构造链路集合用于匹配（无向边，确保顺序一致）
    blocked_links = set((min(blocked_path[i], blocked_path[i + 1]), max(blocked_path[i], blocked_path[i + 1]))
                        for i in range(len(blocked_path) - 1))

    cnt = 0
    for i in range((len(service_path) - 1)):
        u = service_path[i]
        v = service_path[i + 1]
        link = (min(u, v), max(u, v))  # 无向边匹配
        if link in blocked_links:
            for j in range(80):
                if (topology[u][v]['wavelength_service'][j] != 0 and topology[u][v]['wavelength_service'][
                    j] != origin_service.wavelength
                        and topology[u][v]['wavelength_service'][j] != service.wavelength):
                    origin_GSNR += origin_topology[u][v]['wavelength_SNR'][j]
                    current_GSNR += topology[u][v]['wavelength_SNR'][j]
                    cnt += 1
    if cnt != 0:
        reward = (current_GSNR - origin_GSNR) / cnt
    else:
        reward = 0
    return reward * 5


def path_to_links(path):
    """ 将路径节点序列转换为 **链路集合** """
    return {(path[i], path[i + 1])
            for i in range(len(path) - 1)}


def select_sorting_services(service_dict, blocked_service, current_time, num_agent, upper_utilization,
                            sort_by: str = "default"):
    '''
    从service_dict中选出与blocked_service.path有重合链路的service，按照重合链路数排序，重合链路数多的排在前面。
    如果重合链路数相同，则按照带宽利用率和剩余时间（current_time - service.arrival_time）排序，带宽利用率低、剩余时间长的排在前面。

    Parameters
    ----------
    sort_by : str
        - "default"   -> ①重叠链路数(降) ②带宽利用率(升) ③剩余时间(降)
        - "arrival"   -> 按到达时间 arrival_time 降序（“最新到达”优先）
        - "remaining" -> 按剩余时间 remaining_time 降序（剩余时间越长越优）
        - "path_len"  -> 按路径长度 len(path) 降序（“跳数”多的优先）
        - "utilization" -> 按带宽利用率 utilization 升序
    :return: (dict) 排序后的service对象字典
    '''

    blocked_links = path_to_links(blocked_service.path)
    candidates = []

    for svc in service_dict.values():
        # 只考虑利用率是标量且低于阈值的业务
        if not (np.isscalar(svc.utilization) and svc.utilization < upper_utilization):
            continue

        overlap_cnt = len(blocked_links & path_to_links(svc.path))
        if overlap_cnt == 0:  # 无重叠则跳过
            continue

        remaining_time = current_time - svc.arrival_time
        candidates.append((svc, overlap_cnt, remaining_time))

    if not candidates:
        return {}

    # === 选择排序规则 ===
    if sort_by == "arrival":  # 到达时间降序
        key_fn = lambda x: (-x[0].arrival_time,)
    elif sort_by == "remaining":  # 剩余时间降序
        key_fn = lambda x: (-x[2],)
    elif sort_by == "path_len":  # 路径长度降序
        key_fn = lambda x: (-len(x[0].path),)
    elif sort_by == "utilization":
        key_fn = lambda x: (x[0].utilization)
    else:  # default 复现旧行为
        key_fn = lambda x: (-x[1],  # overlap_cnt (降)
                            x[0].utilization,  # utilization (升)
                            -x[2])  # remaining_time (降)

    # 排序并截取前 num_agent 条
    candidates.sort(key=key_fn)
    top_services = candidates[:num_agent]

    # 输出 {service_id: service_obj}
    return {svc.service_id: svc for svc, *_ in top_services}


def select_all_related_services(service_dict, blocked_service):
    '''
    从service_dict中选出与blocked_service.path有重合链路的service
    :return: (dict) service对象字典
    '''

    overlapping_services = []
    blocked_links = path_to_links(blocked_service.path)  # 计算 blocked_service 的链路集合

    for service in service_dict.values():
        # 计算重合链路数
        service_links = path_to_links(service.path)  # 将 service 的路径转换为链路集合
        overlap_count = len(blocked_links & service_links)  # 计算与 blocked_service 的重合链路数

        # if overlap_count > 0 and service.utilization < upper_utilization:
        if overlap_count > 0:
            overlapping_services.append(service)
            # print('overlap:', service.service_id, service.utilization, overlap_count, remaining_time)

    # 返回服务对象列表
    return {s.service_id: s for s in overlapping_services}


# 4. release a service
def release_service(topology, service: Service, service_dict, frequencies=None):
    '''
    service_dict: 业务字典，键为service_id，值为service对象
    释放指定的某个业务，并更新相关链路的状态
    '''
    # 更新链路状态
    del service_dict[service.service_id]
    if frequencies is not None:
        channels = len(frequencies)
    else:
        channels = 80
        frequencies = np.concatenate([
            np.linspace(184.4e12, 190.25e12, channels // 2),
            np.linspace(190.75e12, 196.6e12, channels // 2)
        ])
    for i in range(len(service.path) - 1):
        u = service.path[i]
        v = service.path[i + 1]
        # 清除波长上的业务信息
        topology[u][v]['wavelength_power'][service.wavelength] = 0
        topology[u][v]['wavelength_utilization'][service.wavelength] = 0
        topology[u][v]['wavelength_SNR'][service.wavelength] = 0
        topology[u][v]['wavelength_service'][service.wavelength] = 0
        topology[u][v]['wavelength_bitrate'][service.wavelength] = 0

        # 重新计算 wavelength_power -> SNR
        power_vector = np.array(topology[u][v]['wavelength_power'])
        if np.allclose(power_vector, 0):
            topology[u][v]['wavelength_SNR'] = np.zeros(channels)
            topology[u][v]['wavelength_utilization'] = np.zeros(channels)
            topology[u][v]['wavelength_bitrate'] = np.zeros(channels)
            topology[u][v]['wavelength_service'] = np.zeros(channels, dtype=int)
            continue

        distance = topology[u][v]['length']
        Power_after_transmission, GSNR = one_link_transmission(distance, channels, power_vector, frequencies)

        # print('power_after_transmission:', Power_after_transmission)
        topology[u][v]['wavelength_SNR'] = GSNR

        # 更新该链路所有传输载波的利用率
        for j in range(channels):
            if topology[u][v]['wavelength_service'][j] != 0:
                service_id = topology[u][v]['wavelength_service'][j]
                tmp_service1 = service_dict.get(int(service_id), None)
                if tmp_service1 == None:
                    print('id:', j, service_id, service.service_id, service.wavelength)
                    print('ids:', service_dict.keys())
                if topology[u][v]['wavelength_SNR'][j] >= 26.5 - 1:
                    capacity = 800
                elif topology[u][v]['wavelength_SNR'][j] >= 25.0 - 1:
                    capacity = 700
                elif topology[u][v]['wavelength_SNR'][j] >= 23.5 - 1:
                    capacity = 600
                elif topology[u][v]['wavelength_SNR'][j] >= 21.0 - 1:
                    capacity = 500
                elif topology[u][v]['wavelength_SNR'][j] >= 18.7 - 1:
                    capacity = 400
                elif topology[u][v]['wavelength_SNR'][j] >= 15:
                    capacity = 200
                else:
                    capacity = 1e-6
                topology[u][v]['wavelength_utilization'][j] = tmp_service1.bit_rate / capacity
                tmp_service1.utilization = tmp_service1.bit_rate / capacity
                service_dict[tmp_service1.service_id] = tmp_service1
                # print('service_id:', tmp_service.service_id, 'bitrate:', bit_rate, 'capacity:', capacity)
