'''
HBM 模拟器，用于模拟 HBM 的读写操作。
使用方式：使用 for 循环模拟时钟周期，在每个周期中调用 read 或 write 方法。
包括内容：
1. 读写操作的延迟
2. 读写操作的带宽
3. 读写操作的冲突
4. 读写操作的优先级
5. 读写操作的缓存
'''


class HBMOperation:
    def __init__(self):
        self.working = False

    def read(self, address: int, size: int):
        pass

    def write(self, address: int, size: int):
        pass


class HBM(HBMOperation):
    def __init__(self, channels: int, channel_width_bits: int, capacity_per_channel_gb: float, base_latency_cycles: int, channel_latency_cycles: int):
        self.channels = channels
        self.channel_width_bits = channel_width_bits
        self.capacity_per_channel_gb = capacity_per_channel_gb
        self.base_latency_cycles = base_latency_cycles
        self.channel_latency_cycles = channel_latency_cycles

        # when call read or write, if the working is True, then the read or write will be blocked
        self.working = False
