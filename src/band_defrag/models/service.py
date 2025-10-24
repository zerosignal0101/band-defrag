class Service:

    def __init__(self, service_id, source_id, destination_id=None, arrival_time=None,
                 departure_time=None, bit_rate=None, modulation=None, power=None):
        self.service_id = service_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.arrival_time = arrival_time
        self.departure_time = departure_time
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
        return f'Serv. {self.service_id} ({self.source_id} -> {self.destination_id})' + msg
    
    def to_dict(self) -> dict:
        """将 Service 对象实例转换为字典，以便序列化为 JSON"""
        return {
            "service_id": self.service_id,
            "source_id": self.source_id,
            "destination_id": self.destination_id,
            "arrival_time": self.arrival_time,
            "departure_time": self.departure_time,
            "bit_rate": self.bit_rate,
            "modulation": self.modulation,
            "power": self.power,
            "path": self.path,
            "wavelength": self.wavelength,
            "snr_requirement": self.snr_requirement,
            "GSNR": self.GSNR,
            "utilization": self.utilization,
        }
