"""
基础内存模拟器
支持地址和偏移量访问，可配置内存大小和访问长度
"""

import struct
from typing import Union, List, Optional
import logging

class MemorySimulator:
    """
    基础内存模拟器类
    
    功能特性：
    - 可配置内存大小
    - 支持地址和偏移量访问
    - 固定可调整的访问长度
    - 支持读写操作
    - 边界检查和错误处理
    """
    
    def __init__(self, memory_size: int = 1024 * 1024, access_length: int = 4):
        """
        初始化内存模拟器
        
        Args:
            memory_size (int): 内存大小（字节），默认1MB
            access_length (int): 每次访问的固定长度（字节），默认4字节
        """
        if memory_size <= 0:
            raise ValueError("内存大小必须大于0")
        if access_length <= 0:
            raise ValueError("访问长度必须大于0")
            
        self.memory_size = memory_size
        self.access_length = access_length
        self.memory = bytearray(memory_size)  # 使用bytearray作为内存存储
        self.access_count = 0  # 访问计数器
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def _validate_address(self, address: int, offset: int = 0) -> int:
        """
        验证地址和偏移量的有效性
        
        Args:
            address (int): 基地址
            offset (int): 偏移量
            
        Returns:
            int: 实际访问地址
            
        Raises:
            ValueError: 地址无效时抛出异常
        """
        actual_address = address + offset
        
        if actual_address < 0:
            raise ValueError(f"地址不能为负数: {actual_address}")
        
        if actual_address + self.access_length > self.memory_size:
            raise ValueError(
                f"访问超出内存边界: 地址={actual_address}, "
                f"访问长度={self.access_length}, 内存大小={self.memory_size}"
            )
            
        return actual_address
    
    def read(self, address: int, offset: int = 0) -> bytes:
        """
        从指定地址读取数据
        
        Args:
            address (int): 基地址
            offset (int): 偏移量，默认为0
            
        Returns:
            bytes: 读取的数据
        """
        actual_address = self._validate_address(address, offset)
        data = bytes(self.memory[actual_address:actual_address + self.access_length])
        self.access_count += 1
        
        self.logger.debug(f"读取: 地址={actual_address}, 数据={data.hex()}")
        return data
    
    def write(self, address: int, data: Union[bytes, bytearray, int], offset: int = 0):
        """
        向指定地址写入数据
        
        Args:
            address (int): 基地址
            data (Union[bytes, bytearray, int]): 要写入的数据
            offset (int): 偏移量，默认为0
        """
        actual_address = self._validate_address(address, offset)
        
        # 处理不同类型的数据
        if isinstance(data, int):
            # 将整数转换为字节，使用小端序
            if self.access_length == 1:
                data = struct.pack('<B', data & 0xFF)
            elif self.access_length == 2:
                data = struct.pack('<H', data & 0xFFFF)
            elif self.access_length == 4:
                data = struct.pack('<I', data & 0xFFFFFFFF)
            elif self.access_length == 8:
                data = struct.pack('<Q', data & 0xFFFFFFFFFFFFFFFF)
            else:
                # 对于其他长度，转换为字节并截断或填充
                data = data.to_bytes(self.access_length, byteorder='little', signed=False)
        elif isinstance(data, (bytes, bytearray)):
            # 确保数据长度匹配访问长度
            if len(data) > self.access_length:
                data = data[:self.access_length]  # 截断
            elif len(data) < self.access_length:
                data = data + b'\x00' * (self.access_length - len(data))  # 填充零
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")
        
        # 写入数据
        self.memory[actual_address:actual_address + self.access_length] = data
        self.access_count += 1
        
        self.logger.debug(f"写入: 地址={actual_address}, 数据={data.hex()}")
    
    def read_int(self, address: int, offset: int = 0, signed: bool = False) -> int:
        """
        从指定地址读取整数
        
        Args:
            address (int): 基地址
            offset (int): 偏移量，默认为0
            signed (bool): 是否为有符号整数，默认为False
            
        Returns:
            int: 读取的整数值
        """
        data = self.read(address, offset)
        
        # 根据访问长度解析整数
        if self.access_length == 1:
            return struct.unpack('<b' if signed else '<B', data)[0]
        elif self.access_length == 2:
            return struct.unpack('<h' if signed else '<H', data)[0]
        elif self.access_length == 4:
            return struct.unpack('<i' if signed else '<I', data)[0]
        elif self.access_length == 8:
            return struct.unpack('<q' if signed else '<Q', data)[0]
        else:
            # 对于其他长度，使用通用方法
            return int.from_bytes(data, byteorder='little', signed=signed)
    
    def write_int(self, address: int, value: int, offset: int = 0):
        """
        向指定地址写入整数
        
        Args:
            address (int): 基地址
            value (int): 要写入的整数值
            offset (int): 偏移量，默认为0
        """
        self.write(address, value, offset)
    
    def fill(self, address: int, value: int, count: int, offset: int = 0):
        """
        用指定值填充连续的内存区域
        
        Args:
            address (int): 起始地址
            value (int): 填充值
            count (int): 填充次数
            offset (int): 偏移量，默认为0
        """
        for i in range(count):
            current_offset = offset + i * self.access_length
            self.write(address, value, current_offset)
    
    def copy(self, src_address: int, dst_address: int, count: int, 
             src_offset: int = 0, dst_offset: int = 0):
        """
        在内存中复制数据
        
        Args:
            src_address (int): 源地址
            dst_address (int): 目标地址
            count (int): 复制次数
            src_offset (int): 源偏移量，默认为0
            dst_offset (int): 目标偏移量，默认为0
        """
        for i in range(count):
            current_src_offset = src_offset + i * self.access_length
            current_dst_offset = dst_offset + i * self.access_length
            data = self.read(src_address, current_src_offset)
            self.write(dst_address, data, current_dst_offset)
    
    def dump(self, address: int, length: int, offset: int = 0) -> List[str]:
        """
        转储内存内容为十六进制字符串列表
        
        Args:
            address (int): 起始地址
            length (int): 转储长度（字节）
            offset (int): 偏移量，默认为0
            
        Returns:
            List[str]: 十六进制字符串列表
        """
        actual_address = address + offset
        if actual_address < 0 or actual_address + length > self.memory_size:
            raise ValueError("转储范围超出内存边界")
        
        result = []
        for i in range(0, length, 16):  # 每行显示16字节
            line_start = actual_address + i
            line_end = min(line_start + 16, actual_address + length)
            line_data = self.memory[line_start:line_end]
            
            # 格式化为十六进制
            hex_str = ' '.join(f'{b:02x}' for b in line_data)
            # 格式化为ASCII（可打印字符）
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in line_data)
            
            result.append(f"{line_start:08x}: {hex_str:<48} |{ascii_str}|")
        
        return result
    
    def reset(self):
        """重置内存，将所有字节设为0"""
        self.memory = bytearray(self.memory_size)
        self.access_count = 0
        self.logger.info("内存已重置")
    
    def get_stats(self) -> dict:
        """
        获取内存模拟器统计信息
        
        Returns:
            dict: 统计信息字典
        """
        return {
            'memory_size': self.memory_size,
            'access_length': self.access_length,
            'access_count': self.access_count,
            'memory_usage': f"{self.memory_size / (1024 * 1024):.2f} MB"
        }
    
    def set_access_length(self, new_length: int):
        """
        动态调整访问长度
        
        Args:
            new_length (int): 新的访问长度
        """
        if new_length <= 0:
            raise ValueError("访问长度必须大于0")
        
        old_length = self.access_length
        self.access_length = new_length
        self.logger.info(f"访问长度已从 {old_length} 字节调整为 {new_length} 字节")
    
    def __str__(self) -> str:
        """返回内存模拟器的字符串表示"""
        stats = self.get_stats()
        return (f"MemorySimulator(size={stats['memory_usage']}, "
                f"access_length={stats['access_length']}B, "
                f"accesses={stats['access_count']})")
    
    def __repr__(self) -> str:
        """返回内存模拟器的详细表示"""
        return (f"MemorySimulator(memory_size={self.memory_size}, "
                f"access_length={self.access_length})")


# 示例使用函数
def demo():
    """演示内存模拟器的基本用法"""
    print("=== 内存模拟器演示 ===")
    
    # 创建一个1KB的内存模拟器，每次访问4字节
    mem = MemorySimulator(memory_size=1024, access_length=4)
    print(f"创建内存模拟器: {mem}")
    
    # 写入一些数据
    print("\n--- 写入数据 ---")
    mem.write_int(0, 0x12345678)  # 在地址0写入整数
    mem.write_int(4, 0xABCDEF00)  # 在地址4写入整数
    mem.write(8, b"Hello")        # 在地址8写入字符串
    
    # 读取数据
    print("\n--- 读取数据 ---")
    value1 = mem.read_int(0)
    value2 = mem.read_int(4)
    data = mem.read(8)
    
    print(f"地址0的值: 0x{value1:08x}")
    print(f"地址4的值: 0x{value2:08x}")
    print(f"地址8的数据: {data}")
    
    # 使用偏移量访问
    print("\n--- 使用偏移量访问 ---")
    mem.write_int(100, 0xDEADBEEF, offset=8)  # 在地址108写入
    value3 = mem.read_int(100, offset=8)      # 从地址108读取
    print(f"地址108的值: 0x{value3:08x}")
    
    # 转储内存内容
    print("\n--- 内存转储 ---")
    dump_lines = mem.dump(0, 32)
    for line in dump_lines:
        print(line)
    
    # 显示统计信息
    print(f"\n--- 统计信息 ---")
    stats = mem.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    demo()
