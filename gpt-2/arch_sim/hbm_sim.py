from typing import Tuple, Dict, Optional, Union, List, Any, Callable
from collections import deque


class HBMSimulator:
    """
    高带宽内存 (HBM) 简易模拟器
    - 支持16个通道(channel)
    - 每个通道数据位宽512位(64字节)
    - 返回访问数据和消耗周期
    """

    def __init__(
        self,
        num_channels: int = 16,
        channel_width_bits: int = 512,
        capacity_per_channel_gb: float = 1.0,
        base_latency_cycles: int = 20,
        channel_latency_cycles: int = 5,
    ):
        """
        初始化HBM模拟器

        参数:
            num_channels: HBM通道数量
            channel_width_bits: 每个通道的位宽(比特)
            capacity_per_channel_gb: 每个通道的容量(GB)
            base_latency_cycles: 基础访问延迟(周期)
            channel_latency_cycles: 通道切换延迟(周期)
        """
        self.num_channels = num_channels
        self.channel_width_bits = channel_width_bits
        self.channel_width_bytes = channel_width_bits // 8

        # 计算每个通道的容量（字节）
        self.capacity_per_channel_bytes = int(
            capacity_per_channel_gb * 1024 * 1024 * 1024)
        self.total_capacity_bytes = self.capacity_per_channel_bytes * num_channels

        # 延迟参数
        self.base_latency_cycles = base_latency_cycles
        self.channel_latency_cycles = channel_latency_cycles

        # 初始化内存数据（使用字典存储，键为地址，值为数据）
        self.memory: Dict[int, bytes] = {}

        # 统计信息
        self.total_reads = 0
        self.total_writes = 0
        self.total_read_cycles = 0
        self.total_write_cycles = 0

        # 通道状态跟踪 - 记录每个通道上次访问的周期
        self.channel_last_access: Dict[int, int] = {
            i: 0 for i in range(num_channels)}
        self.current_cycle = 0

        print(f"已初始化HBM模拟器: {num_channels}个通道, 每通道{channel_width_bits}位宽, "
              f"总容量{self.total_capacity_bytes / (1024*1024*1024):.2f}GB")

    def _get_channel_from_address(self, address: int) -> int:
        """从地址计算对应的通道号, 使用分散存储模式(default)"""
        return address % self.num_channels

    def _calculate_latency(self, address: int, is_read: bool = True) -> int:
        """
        计算访问延迟(周期数)

        参数:
            address: 访问地址
            is_read: 是否为读操作

        返回:
            消耗的周期数
        """
        channel = self._get_channel_from_address(address)

        # 基础延迟
        latency = self.base_latency_cycles

        # 通道冲突检查 - 如果该通道最近被访问过，需要额外等待
        channel_last_cycle = self.channel_last_access[channel]
        if channel_last_cycle > self.current_cycle:
            # 通道还在被占用，需要等待
            latency += channel_last_cycle - self.current_cycle

        # 更新通道访问时间
        self.channel_last_access[channel] = self.current_cycle + latency

        # 读写操作可能有不同的延迟
        if not is_read:
            # 写操作通常需要额外周期
            latency += 2

        return latency

    def read(self, address: int, size_bytes: Optional[int] = None) -> Tuple[Union[bytes, None], int]:
        """
        从HBM读取数据

        参数:
            address: 读取的起始地址
            size_bytes: 读取的字节数，默认为一个通道宽度

        返回:
            (数据, 消耗的周期数)
        """
        if address >= self.total_capacity_bytes:
            raise ValueError(f"地址{address}超出总容量范围")

        if size_bytes is None:
            size_bytes = self.channel_width_bytes

        # 计算延迟
        latency = self._calculate_latency(address, is_read=True)

        # 检查数据是否存在
        if address in self.memory:
            data = self.memory[address][:size_bytes]
        else:
            # 模拟未初始化的内存返回零值
            data = bytes(size_bytes)

        # 更新时钟周期
        self.current_cycle += latency

        # 更新统计信息
        self.total_reads += 1
        self.total_read_cycles += latency

        return data, latency

    def write(self, address: int, data: bytes) -> int:
        """
        向HBM写入数据

        参数:
            address: 写入的起始地址
            data: 要写入的数据

        返回:
            消耗的周期数
        """
        if address >= self.total_capacity_bytes:
            raise ValueError(f"地址{address}超出总容量范围")

        # 计算延迟
        latency = self._calculate_latency(address, is_read=False)

        # 写入数据
        self.memory[address] = data

        # 更新时钟周期
        self.current_cycle += latency

        # 更新统计信息
        self.total_writes += 1
        self.total_write_cycles += latency

        return latency

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            "total_reads": self.total_reads,
            "total_writes": self.total_writes,
            "total_read_cycles": self.total_read_cycles,
            "total_write_cycles": self.total_write_cycles,
            "avg_read_latency": self.total_read_cycles / max(1, self.total_reads),
            "avg_write_latency": self.total_write_cycles / max(1, self.total_writes),
            "current_cycle": self.current_cycle,
        }

    def reset_stats(self) -> None:
        """重置统计信息"""
        self.total_reads = 0
        self.total_writes = 0
        self.total_read_cycles = 0
        self.total_write_cycles = 0
        self.current_cycle = 0
        self.channel_last_access = {i: 0 for i in range(self.num_channels)}


class PendingOperation:
    """表示一个待完成的内存操作"""

    def __init__(self, completion_cycle: int, data: Any = None, callback: Optional[Callable] = None):
        """
        初始化待完成操作

        参数:
            completion_cycle: 操作完成的周期
            data: 与操作相关的数据
            callback: 操作完成时调用的回调函数
        """
        self.completion_cycle = completion_cycle
        self.data = data
        self.callback = callback

    def is_complete(self, current_cycle: int) -> bool:
        """检查操作是否已完成"""
        return current_cycle >= self.completion_cycle

    def complete(self) -> Any:
        """完成操作并返回数据"""
        if self.callback:
            self.callback(self.data)
        return self.data


class HBMBuffer:
    """
    HBM同步缓冲区，支持流水线模式的读写操作
    可以发起异步读写操作，并在适当的时机检查和获取结果
    """

    def __init__(self, hbm_simulator: HBMSimulator, buffer_size: int = 32):
        """
        初始化HBM缓冲区

        参数:
            hbm_simulator: HBM模拟器实例
            buffer_size: 缓冲区大小(条目数)
        """
        self.hbm = hbm_simulator
        self.buffer_size = buffer_size
        self.pending_reads: deque[PendingOperation] = deque()
        self.pending_writes: deque[PendingOperation] = deque()
        self.current_cycle = 0

    def async_read(self, address: int, size_bytes: Optional[int] = None, callback: Optional[Callable] = None) -> int:
        """
        异步读取数据，立即返回而不等待完成

        参数:
            address: 读取的起始地址
            size_bytes: 读取的字节数
            callback: 读操作完成时的回调函数

        返回:
            操作标识符(操作在队列中的位置)
        """
        if len(self.pending_reads) >= self.buffer_size:
            raise RuntimeError(f"读缓冲区已满 (大小: {self.buffer_size})")

        # 模拟发起读取操作，但不等待结果
        data, latency = self.hbm.read(address, size_bytes)
        completion_cycle = self.current_cycle + latency

        # 创建待完成的操作并加入队列
        op = PendingOperation(completion_cycle, data, callback)
        self.pending_reads.append(op)

        # 返回操作ID (在队列中的索引)
        return len(self.pending_reads) - 1

    def async_write(self, address: int, data: bytes, callback: Optional[Callable] = None) -> int:
        """
        异步写入数据，立即返回而不等待完成

        参数:
            address: 写入的起始地址
            data: 要写入的数据
            callback: 写操作完成时的回调函数

        返回:
            操作标识符(操作在队列中的位置)
        """
        if len(self.pending_writes) >= self.buffer_size:
            raise RuntimeError(f"写缓冲区已满 (大小: {self.buffer_size})")

        # 模拟发起写入操作，但不等待结果
        latency = self.hbm.write(address, data)
        completion_cycle = self.current_cycle + latency

        # 创建待完成的操作并加入队列
        op = PendingOperation(completion_cycle, data, callback)
        self.pending_writes.append(op)

        # 返回操作ID (在队列中的索引)
        return len(self.pending_writes) - 1

    def tick(self, cycles: int = 1) -> None:
        """
        推进时钟周期

        参数:
            cycles: 要推进的周期数
        """
        self.current_cycle += cycles

    def check_completed_reads(self) -> List[Any]:
        """
        检查并返回所有已完成的读操作的数据

        返回:
            已完成读操作的数据列表
        """
        completed_data = []

        # 检查队列前端的操作是否已完成
        while self.pending_reads and self.pending_reads[0].is_complete(self.current_cycle):
            op = self.pending_reads.popleft()
            completed_data.append(op.complete())

        return completed_data

    def check_completed_writes(self) -> int:
        """
        检查并处理已完成的写操作

        返回:
            已完成的写操作数
        """
        completed_count = 0

        # 检查队列前端的操作是否已完成
        while self.pending_writes and self.pending_writes[0].is_complete(self.current_cycle):
            op = self.pending_writes.popleft()
            op.complete()
            completed_count += 1

        return completed_count

    def wait_all_reads(self) -> List[Any]:
        """
        等待所有读操作完成并返回数据

        返回:
            所有已完成读操作的数据列表
        """
        if not self.pending_reads:
            return []

        # 找到最后一个操作的完成周期
        max_cycle = max(op.completion_cycle for op in self.pending_reads)

        # 推进时钟到所有操作都完成
        if max_cycle > self.current_cycle:
            self.tick(max_cycle - self.current_cycle)

        # 收集所有完成的操作
        return self.check_completed_reads()

    def wait_all_writes(self) -> int:
        """
        等待所有写操作完成

        返回:
            完成的写操作数
        """
        if not self.pending_writes:
            return 0

        # 找到最后一个操作的完成周期
        max_cycle = max(op.completion_cycle for op in self.pending_writes)

        # 推进时钟到所有操作都完成
        if max_cycle > self.current_cycle:
            self.tick(max_cycle - self.current_cycle)

        # 处理所有完成的操作
        return self.check_completed_writes()

    def get_stats(self) -> Dict[str, int]:
        """获取缓冲区状态统计"""
        return {
            "current_cycle": self.current_cycle,
            "pending_reads": len(self.pending_reads),
            "pending_writes": len(self.pending_writes),
            "buffer_size": self.buffer_size,
        }


# 使用示例
if __name__ == "__main__":
    # 创建HBM模拟器实例
    hbm = HBMSimulator()

    # 创建缓冲区
    buffer = HBMBuffer(hbm)

    # 写入一些测试数据
    test_data = bytes([i % 256 for i in range(64)])
    write_cycles = hbm.write(0, test_data)
    print(f"写入地址0消耗{write_cycles}个周期")

    # 读取数据
    data, read_cycles = hbm.read(0)
    print(f"从地址0读取数据消耗{read_cycles}个周期")
    print(f"读取的数据前10字节: {data[:10]}")

    # 使用缓冲区进行流水线操作
    print("\n--- 使用缓冲区进行流水线操作 ---")

    # 发起多个异步写入
    for i in range(5):
        addr = i * 64
        data = bytes([i + j % 256 for j in range(64)])
        buffer.async_write(addr, data)
        print(f"异步写入地址{addr}")

    # 发起多个异步读取
    for i in range(5):
        addr = i * 64
        buffer.async_read(addr)
        print(f"异步读取地址{addr}")

    # 等待所有写入完成
    buffer.wait_all_writes()
    print("所有写入操作已完成")

    # 等待并获取所有读取结果
    results = buffer.wait_all_reads()
    print(f"所有读取操作已完成，获得{len(results)}个结果")

    # 获取统计信息
    buffer_stats = buffer.get_stats()
    hbm_stats = hbm.get_stats()
    print(f"缓冲区统计信息: {buffer_stats}")
    print(f"HBM统计信息: {hbm_stats}")
