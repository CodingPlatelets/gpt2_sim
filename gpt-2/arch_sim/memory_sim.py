import numpy as np
from torch import nn
import torch
from collections import OrderedDict


class SRAMOverflowError(Exception):
    """SRAM溢出"""
    pass


class DDROverflowError(Exception):
    """DDR溢出"""
    pass


class _HardwareSimulator:
    def __init__(self):
        self._initialized = False

    def initialize(self, read_delay=5, write_delay=10, sram_capacity=1024 * 1024,
                   ddr_capacity=8*1024*1024, ddr_read_delay=100, ddr_write_delay=80,
                   ddr_to_sram_delay=50):
        # SRAM attributes
        self.sram = OrderedDict()  # 模拟SRAM存储权重/中间结果
        self.sram_capacity = sram_capacity  # SRAM容量
        self.sram_used = 0                   # 已使用的SRAM容量
        self.cycles = 0                      # 时钟周期计数
        self.access_log = []                 # 记录内存访问
        self.read_delay = read_delay          # 读取延迟
        self.write_delay = write_delay        # 写入延迟

        # DDR attributes
        self.ddr = OrderedDict()             # 模拟DDR存储
        self.ddr_capacity = ddr_capacity     # DDR容量
        self.ddr_used = 0                    # 已使用的DDR容量
        self.ddr_read_delay = ddr_read_delay  # DDR读取延迟
        self.ddr_write_delay = ddr_write_delay  # DDR写入延迟
        self.ddr_to_sram_delay = ddr_to_sram_delay  # DDR到SRAM的传输延迟
        self._initialized = True

    def _calc_tensor_size(self, tensor):
        """计算张量占用的字节数"""
        if isinstance(tensor, torch.Tensor):
            return tensor.numel() * tensor.element_size()
        elif isinstance(tensor, np.ndarray):
            return tensor.nbytes
        else:
            raise TypeError("Unsupported data type")

    def sram_load(self, key, data, evict_if_full=True):
        """数据直接写入SRAM，如果空间不足则抛出异常或触发替换"""
        data_size = self._calc_tensor_size(data)

        # 检查容量是否超限
        if data_size > self.sram_capacity:
            raise SRAMOverflowError(
                f"Data size ({data_size} bytes) exceeds total SRAM capacity"
            )
        # 检查是否需要替换旧数据
        while self.sram_used + data_size > self.sram_capacity:
            if not evict_if_full:
                raise SRAMOverflowError(
                    f"SRAM full (used={self.sram_used}/{self.sram_capacity} bytes)")
            # 驱逐最老的数据到DDR
            self._evict_to_ddr()

        # 存储数据并更新状态
        self.sram[key] = data
        self.sram_used += data_size
        self.cycles += self.write_delay
        self.access_log.append(f"LOAD {key} ({data_size} bytes) to SRAM")

    def sram_read(self, key):  # -> Any:
        """SRAM读取数据（更新访问顺序）"""
        if key not in self.sram:
            # 当数据不在SRAM时，尝试从DDR加载
            if key in self.ddr:
                self._load_from_ddr_to_sram(key)
            else:
                raise KeyError(
                    f"Segmentation fault: Key {key} not found in either SRAM or DDR")

        # 将访问项移到字典末尾（模拟LRU的最近使用）
        data = self.sram.pop(key)
        self.sram[key] = data

        self.cycles += self.read_delay  # 模拟读取延迟
        self.access_log.append(f"READ {key} from SRAM")
        return data

    def _evict_to_ddr(self):
        """将最老的SRAM数据驱逐到DDR"""
        if not self.sram:
            return

        oldest_key, oldest_data = next(iter(self.sram.items()))
        evicted_size = self._calc_tensor_size(oldest_data)

        # 检查DDR是否有足够空间
        while self.ddr_used + evicted_size > self.ddr_capacity:
            self._evict_oldest_from_ddr()

        # 转移数据到DDR
        self.ddr[oldest_key] = oldest_data
        self.ddr_used += evicted_size

        # 从SRAM中删除
        del self.sram[oldest_key]
        self.sram_used -= evicted_size

        # 记录访问和延迟
        self.cycles += self.ddr_write_delay
        self.access_log.append(
            f"EVICT {oldest_key} ({evicted_size} bytes) from SRAM to DDR")

    def _load_from_ddr_to_sram(self, key):
        """从DDR加载数据到SRAM"""
        data = self.ddr.pop(key)
        data_size = self._calc_tensor_size(data)
        self.ddr_used -= data_size

        # 确保SRAM有足够空间
        while self.sram_used + data_size > self.sram_capacity:
            self._evict_to_ddr()

        # 加载到SRAM
        self.sram[key] = data
        self.sram_used += data_size

        # 记录访问和延迟
        self.cycles += self.ddr_read_delay + self.ddr_to_sram_delay
        self.access_log.append(
            f"LOAD {key} ({data_size} bytes) from DDR to SRAM")

    def _evict_oldest(self):
        """现在逻辑改为将SRAM中最老的数据移到DDR"""
        self._evict_to_ddr()

    def _evict_oldest_from_ddr(self):
        """从DDR中驱逐最老的数据"""
        if not self.ddr:
            return

        oldest_key, oldest_data = next(iter(self.ddr.items()))
        evicted_size = self._calc_tensor_size(oldest_data)

        # 从DDR中删除
        del self.ddr[oldest_key]
        self.ddr_used -= evicted_size

        # 记录访问
        self.access_log.append(
            f"EVICT {oldest_key} ({evicted_size} bytes) from DDR")

    def ddr_load(self, key, data, evict_if_full=True):
        """直接将数据加载到DDR"""
        data_size = self._calc_tensor_size(data)

        # 检查容量是否超限
        if data_size > self.ddr_capacity:
            raise DDROverflowError(
                f"Data size ({data_size} bytes) exceeds total DDR capacity"
            )

        # 检查是否需要替换旧数据
        while self.ddr_used + data_size > self.ddr_capacity:
            if not evict_if_full:
                raise DDROverflowError(
                    f"DDR full (used={self.ddr_used}/{self.ddr_capacity} bytes)")
            self._evict_oldest_from_ddr()

        # 存储数据并更新状态
        self.ddr[key] = data
        self.ddr_used += data_size
        self.cycles += self.ddr_write_delay
        self.access_log.append(f"LOAD {key} ({data_size} bytes) to DDR")
