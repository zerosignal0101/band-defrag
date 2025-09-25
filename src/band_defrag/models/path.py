class Path:

    def __init__(self, path_id, node_list, length, best_modulation=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length # 路径长度
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1
