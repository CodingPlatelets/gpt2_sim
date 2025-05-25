"""
内存模拟器测试文件
测试基础内存模拟器的各种功能
"""

import pytest
import sys
import os

# 添加当前目录到路径，以便导入mem_sim
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mem_sim import MemorySimulator


def test_basic_initialization():
    """测试基本初始化"""
    # 默认参数
    mem = MemorySimulator()
    assert mem.memory_size == 1024 * 1024
    assert mem.access_length == 4
    assert mem.access_count == 0
    
    # 自定义参数
    mem2 = MemorySimulator(memory_size=2048, access_length=8)
    assert mem2.memory_size == 2048
    assert mem2.access_length == 8


def test_invalid_initialization():
    """测试无效初始化参数"""
    with pytest.raises(ValueError):
        MemorySimulator(memory_size=0)
    
    with pytest.raises(ValueError):
        MemorySimulator(memory_size=-1)
    
    with pytest.raises(ValueError):
        MemorySimulator(access_length=0)
    
    with pytest.raises(ValueError):
        MemorySimulator(access_length=-1)


def test_basic_read_write():
    """测试基本读写操作"""
    mem = MemorySimulator(memory_size=1024, access_length=4)
    
    # 写入整数
    mem.write_int(0, 0x12345678)
    value = mem.read_int(0)
    assert value == 0x12345678
    
    # 写入字节数据
    mem.write(4, b"test")
    data = mem.read(4)
    assert data == b"test"
    
    # 检查访问计数
    assert mem.access_count == 4  # 2次写入 + 2次读取


def test_offset_access():
    """测试偏移量访问"""
    mem = MemorySimulator(memory_size=1024, access_length=4)
    
    # 使用偏移量写入
    mem.write_int(100, 0xDEADBEEF, offset=8)
    
    # 使用偏移量读取
    value = mem.read_int(100, offset=8)
    assert value == 0xDEADBEEF
    
    # 验证实际地址是108 (100 + 8)
    direct_value = mem.read_int(108)
    assert direct_value == 0xDEADBEEF


def test_boundary_checking():
    """测试边界检查"""
    mem = MemorySimulator(memory_size=16, access_length=4)
    
    # 正常访问
    mem.write_int(0, 0x12345678)
    mem.write_int(4, 0xABCDEF00)
    mem.write_int(8, 0x11223344)
    mem.write_int(12, 0x55667788)
    
    # 超出边界的访问应该抛出异常
    with pytest.raises(ValueError):
        mem.write_int(16, 0x99AABBCC)  # 地址16超出边界
    
    with pytest.raises(ValueError):
        mem.read_int(16)  # 地址16超出边界
    
    with pytest.raises(ValueError):
        mem.write_int(14, 0x99AABBCC)  # 地址14+4=18超出边界
    
    # 负地址
    with pytest.raises(ValueError):
        mem.write_int(-1, 0x12345678)
    
    with pytest.raises(ValueError):
        mem.write_int(0, 0x12345678, offset=-1)


def test_different_access_lengths():
    """测试不同的访问长度"""
    # 1字节访问
    mem1 = MemorySimulator(memory_size=64, access_length=1)
    mem1.write_int(0, 0xFF)
    assert mem1.read_int(0) == 0xFF
    
    # 2字节访问
    mem2 = MemorySimulator(memory_size=64, access_length=2)
    mem2.write_int(0, 0x1234)
    assert mem2.read_int(0) == 0x1234
    
    # 8字节访问
    mem8 = MemorySimulator(memory_size=64, access_length=8)
    mem8.write_int(0, 0x123456789ABCDEF0)
    assert mem8.read_int(0) == 0x123456789ABCDEF0


def test_signed_integers():
    """测试有符号整数"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 写入负数
    mem.write_int(0, -1)
    
    # 读取为无符号数
    unsigned_value = mem.read_int(0, signed=False)
    assert unsigned_value == 0xFFFFFFFF
    
    # 读取为有符号数
    signed_value = mem.read_int(0, signed=True)
    assert signed_value == -1


def test_fill_operation():
    """测试填充操作"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 填充5个位置，每个位置4字节
    mem.fill(0, 0x12345678, 5)
    
    # 验证填充结果
    for i in range(5):
        value = mem.read_int(i * 4)
        assert value == 0x12345678


def test_copy_operation():
    """测试复制操作"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 在源位置写入数据
    mem.write_int(0, 0x11111111)
    mem.write_int(4, 0x22222222)
    mem.write_int(8, 0x33333333)
    
    # 复制到目标位置
    mem.copy(src_address=0, dst_address=32, count=3)
    
    # 验证复制结果
    assert mem.read_int(32) == 0x11111111
    assert mem.read_int(36) == 0x22222222
    assert mem.read_int(40) == 0x33333333


def test_memory_dump():
    """测试内存转储"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 写入一些数据
    mem.write(0, b"Hello World!")
    
    # 转储内存
    dump_lines = mem.dump(0, 16)
    assert len(dump_lines) == 1  # 16字节应该只有一行
    assert "Hello World!" in dump_lines[0]


def test_dynamic_access_length():
    """测试动态调整访问长度"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 初始访问长度为4
    assert mem.access_length == 4
    
    # 调整为8字节
    mem.set_access_length(8)
    assert mem.access_length == 8
    
    # 测试新的访问长度
    mem.write_int(0, 0x123456789ABCDEF0)
    value = mem.read_int(0)
    assert value == 0x123456789ABCDEF0
    
    # 无效的访问长度
    with pytest.raises(ValueError):
        mem.set_access_length(0)


def test_reset_operation():
    """测试重置操作"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 写入一些数据
    mem.write_int(0, 0x12345678)
    mem.write_int(4, 0xABCDEF00)
    
    # 重置内存
    mem.reset()
    
    # 验证内存已清零
    assert mem.read_int(0) == 0
    assert mem.read_int(4) == 0
    assert mem.access_count == 2  # 重置后访问计数也应该清零


def test_statistics():
    """测试统计信息"""
    mem = MemorySimulator(memory_size=1024, access_length=4)
    
    # 执行一些操作
    mem.write_int(0, 0x12345678)
    mem.read_int(0)
    
    stats = mem.get_stats()
    assert stats['memory_size'] == 1024
    assert stats['access_length'] == 4
    assert stats['access_count'] == 2
    assert '1.00' in stats['memory_usage']  # 1024字节 = 0.001MB


def test_data_type_handling():
    """测试不同数据类型的处理"""
    mem = MemorySimulator(memory_size=64, access_length=4)
    
    # 测试整数
    mem.write(0, 0x12345678)
    assert mem.read_int(0) == 0x12345678
    
    # 测试字节数组
    mem.write(4, bytearray([0x11, 0x22, 0x33, 0x44]))
    data = mem.read(4)
    assert data == b'\x11\x22\x33\x44'
    
    # 测试字节串
    mem.write(8, b'\xAA\xBB\xCC\xDD')
    data = mem.read(8)
    assert data == b'\xAA\xBB\xCC\xDD'
    
    # 测试不支持的数据类型
    with pytest.raises(TypeError):
        mem.write(12, "string")  # 字符串不被支持


def run_comprehensive_test():
    """运行综合测试"""
    print("=== 运行内存模拟器综合测试 ===")
    
    # 创建内存模拟器
    mem = MemorySimulator(memory_size=1024, access_length=4)
    print(f"创建内存模拟器: {mem}")
    
    # 测试基本读写
    print("\n1. 测试基本读写...")
    mem.write_int(0, 0x12345678)
    value = mem.read_int(0)
    assert value == 0x12345678
    print("✓ 基本读写测试通过")
    
    # 测试偏移量访问
    print("\n2. 测试偏移量访问...")
    mem.write_int(100, 0xDEADBEEF, offset=8)
    value = mem.read_int(100, offset=8)
    assert value == 0xDEADBEEF
    print("✓ 偏移量访问测试通过")
    
    # 测试边界检查
    print("\n3. 测试边界检查...")
    try:
        mem.write_int(1024, 0x12345678)  # 应该失败
        assert False, "边界检查失败"
    except ValueError:
        print("✓ 边界检查测试通过")
    
    # 测试填充操作
    print("\n4. 测试填充操作...")
    mem.fill(200, 0xAAAAAAAA, 5)
    for i in range(5):
        value = mem.read_int(200 + i * 4)
        assert value == 0xAAAAAAAA
    print("✓ 填充操作测试通过")
    
    # 测试复制操作
    print("\n5. 测试复制操作...")
    mem.copy(src_address=0, dst_address=300, count=1)
    value = mem.read_int(300)
    assert value == 0x12345678
    print("✓ 复制操作测试通过")
    
    # 测试内存转储
    print("\n6. 测试内存转储...")
    dump_lines = mem.dump(0, 32)
    assert len(dump_lines) >= 1
    print("✓ 内存转储测试通过")
    
    # 显示最终统计
    print(f"\n=== 测试完成 ===")
    stats = mem.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n所有测试通过！✓")


if __name__ == "__main__":
    run_comprehensive_test() 