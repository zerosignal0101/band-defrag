import networkx as nx
import numpy as np
from itertools import islice
import pickle
from back_utils import Path

def get_k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def get_path_weight(graph, path, weight_attr='weight'):
    return float(np.sum([graph[path[i]][path[i+1]][weight_attr] for i in range(len(path) - 1)]))

def read_graphml_as_topology(file, relabel_to_int=True, edge_weight_key_preference=('weight','length')):
    """
    读取 GraphML，并返回一个满足你后续流程(topology.graph[...] 等)的无向图。
    - relabel_to_int: 是否把节点重标为 "0","1","2"...（字符串），与你原始 txt 读取保持一致
    - edge_weight_key_preference: 指定最短路权重优先使用哪一个属性
    """
    G_loaded = nx.read_graphml(file)   # 读进来可能是 MultiGraph，也可能是 Graph
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

graphml_file = "network_sweden_example.graphml"
topology, wkey = read_graphml_as_topology(graphml_file, relabel_to_int=True)

# 落盘（与你原例保持一致的命名规则）
out_path = './sweden_topology_1path.h5'
with open(out_path, 'wb') as f:
    pickle.dump(topology, f)

print(f"Done. weight attribute used = '{wkey}'. Saved to: {out_path}")
print(topology.nodes)
print(topology['0']['4'])