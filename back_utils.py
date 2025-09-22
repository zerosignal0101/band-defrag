from itertools import islice
import networkx as nx
import numpy as np


class Path:

    def __init__(self, path_id, node_list, length, best_modulation=None):
        self.path_id = path_id
        self.node_list = node_list
        self.length = length # 路径长度
        self.best_modulation = best_modulation
        self.hops = len(node_list) - 1


class Service:

    def __init__(self, service_id, source, source_id, destination=None, destination_id=None, arrival_time=None,
                 holding_time=None, bit_rate=None, modulation=None, power=None):
        self.service_id = service_id
        self.source = source
        self.source_id = source_id
        self.destination = destination
        self.destination_id = destination_id
        self.arrival_time = arrival_time
        self.holding_time = holding_time
        self.bit_rate = bit_rate
        self.modulation = modulation
        self.power = power
        self.path = None
        self.wavelength = None
        self.snr_requirement = 0  # 所需的SNR
        self.GSNR = 0
        self.utilization = 0
    def __str__(self):
        msg = '{'
        msg += '' if self.bit_rate is None else f'br: {self.bit_rate}, '
        # msg += '' if self.service_class is None else f'cl: {self.service_class}, '
        return f'Serv. {self.service_id} ({self.source} -> {self.destination})' + msg
