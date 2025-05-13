from hbm_sim import HBMSimulator, HBMBuffer
import unittest
import sys
import os
from typing import List

# 添加模块路径
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))
# 导入HBM模拟器相关类


class TestHBMSimulator(unittest.TestCase):
    """测试HBM模拟器基本功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.hbm = HBMSimulator(
            num_channels=16,
            channel_width_bits=512,
            capacity_per_channel_gb=1.0
        )

    def test_init(self):
        """测试初始化参数"""
        self.assertEqual(self.hbm.num_channels, 16)
        self.assertEqual(self.hbm.channel_width_bits, 512)
        self.assertEqual(self.hbm.channel_width_bytes, 64)
        self.assertEqual(self.hbm.capacity_per_channel_bytes,
                         1024 * 1024 * 1024)
        self.assertEqual(self.hbm.total_capacity_bytes,
                         16 * 1024 * 1024 * 1024)

    def test_channel_mapping(self):
        """测试地址到通道的映射"""
        for i in range(32):
            channel = self.hbm._get_channel_from_address(i)
            self.assertEqual(channel, i % 16)

    def test_read_write(self):
        """测试基本的读写操作"""
        # 准备测试数据
        test_data = bytes([i % 256 for i in range(64)])

        # 写入数据
        write_cycles = self.hbm.write(128, test_data)
        self.assertTrue(write_cycles > 0)

        # 读取数据
        read_data, read_cycles = self.hbm.read(128)
        self.assertTrue(read_cycles > 0)
        self.assertEqual(read_data, test_data)

        # 读取未初始化的数据
        empty_data, _ = self.hbm.read(256)
        self.assertEqual(empty_data, bytes(64))

    def test_stats(self):
        """测试统计信息"""
        # 执行一些操作
        for i in range(5):
            self.hbm.write(i * 64, bytes([i] * 64))
            self.hbm.read(i * 64)

        # 获取统计信息
        stats = self.hbm.get_stats()

        # 验证统计信息
        self.assertEqual(stats["total_reads"], 5)
        self.assertEqual(stats["total_writes"], 5)
        self.assertTrue(stats["total_read_cycles"] > 0)
        self.assertTrue(stats["total_write_cycles"] > 0)
        self.assertTrue(stats["avg_read_latency"] > 0)
        self.assertTrue(stats["avg_write_latency"] > 0)

    def test_reset_stats(self):
        """测试重置统计信息"""
        # 执行一些操作
        self.hbm.write(0, bytes([0] * 64))
        self.hbm.read(0)

        # 重置统计信息
        self.hbm.reset_stats()

        # 验证统计已重置
        stats = self.hbm.get_stats()
        self.assertEqual(stats["total_reads"], 0)
        self.assertEqual(stats["total_writes"], 0)
        self.assertEqual(stats["total_read_cycles"], 0)
        self.assertEqual(stats["total_write_cycles"], 0)

    def test_invalid_address(self):
        """测试无效地址访问"""
        # 尝试访问超出范围的地址
        with self.assertRaises(ValueError):
            self.hbm.write(self.hbm.total_capacity_bytes + 1, bytes([0] * 64))

        with self.assertRaises(ValueError):
            self.hbm.read(self.hbm.total_capacity_bytes + 1)


class TestHBMBuffer(unittest.TestCase):
    """测试HBM缓冲区功能"""

    def setUp(self):
        """测试前的准备工作"""
        self.hbm = HBMSimulator()
        self.buffer = HBMBuffer(self.hbm, buffer_size=10)

    def test_async_read_write(self):
        """测试异步读写操作"""
        # 准备测试数据
        test_data = [bytes([i] * 64) for i in range(5)]

        # 执行异步写入
        write_ids = []
        for i in range(5):
            write_id = self.buffer.async_write(i * 64, test_data[i])
            write_ids.append(write_id)

        # 验证写入ID正确
        self.assertEqual(write_ids, [0, 1, 2, 3, 4])

        # 等待所有写入完成
        completed_writes = self.buffer.wait_all_writes()
        self.assertEqual(completed_writes, 5)

        # 执行异步读取
        read_ids = []
        for i in range(5):
            read_id = self.buffer.async_read(i * 64)
            read_ids.append(read_id)

        # 验证读取ID正确
        self.assertEqual(read_ids, [0, 1, 2, 3, 4])

        # 等待所有读取完成
        results = self.buffer.wait_all_reads()
        self.assertEqual(len(results), 5)

        # 验证读取的数据正确
        for i in range(5):
            self.assertEqual(results[i], test_data[i])

    def test_buffer_full(self):
        """测试缓冲区已满的情况"""
        # 将缓冲区填满
        for i in range(10):
            self.buffer.async_write(i * 64, bytes([i] * 64))

        # 尝试继续写入，应该失败
        with self.assertRaises(RuntimeError):
            self.buffer.async_write(1000, bytes([0] * 64))

        # 完成一些写入操作 - 推进足够的周期使操作完成
        self.buffer.tick(100)  # 添加足够的周期确保操作完成
        completed = self.buffer.check_completed_writes()
        self.assertTrue(completed > 0, "应该有一些操作完成")

        # 现在应该可以继续写入
        self.buffer.async_write(1000, bytes([0] * 64))

    def test_tick_and_check(self):
        """测试时钟推进和检查完成操作"""
        # 执行一些异步操作
        for i in range(3):
            self.buffer.async_write(i * 64, bytes([i] * 64))

        # 推进少量周期
        self.buffer.tick(10)

        # 检查是否有完成的写入
        first_completed = self.buffer.check_completed_writes()
        
        # 推进足够多的周期确保所有操作完成
        self.buffer.tick(100)

        # 再次检查
        second_completed = self.buffer.check_completed_writes()
        
        # 两次检查的总和应该等于3
        self.assertEqual(first_completed + second_completed, 3, 
                         f"两次完成的操作总数应为3，但得到{first_completed}+{second_completed}")

    def test_callback(self):
        """测试回调函数"""
        # 用于跟踪回调函数调用的列表
        callback_results = []

        # 定义回调函数
        def on_read_complete(data):
            callback_results.append(data)

        # 写入数据
        self.buffer.async_write(0, bytes([42] * 64))
        self.buffer.wait_all_writes()

        # 使用回调读取数据
        self.buffer.async_read(0, callback=on_read_complete)

        # 等待读取完成
        self.buffer.wait_all_reads()

        # 验证回调函数被调用
        self.assertEqual(len(callback_results), 1)
        self.assertEqual(callback_results[0], bytes([42] * 64))


class TestPipelinedOperations(unittest.TestCase):
    """测试流水线操作模式"""

    def setUp(self):
        """测试前的准备工作"""
        self.hbm = HBMSimulator(
            num_channels=16,
            channel_width_bits=512,
            capacity_per_channel_gb=1.0
        )
        self.buffer = HBMBuffer(self.hbm, buffer_size=32)

    def test_pipeline_efficiency(self):
        """测试流水线效率"""
        # 准备一些测试地址，确保它们映射到不同通道
        addresses = [ch * 256 for ch in range(16)]

        # 记录开始周期
        start_cycle = self.buffer.current_cycle

        # 串行执行16次读写操作
        for addr in addresses:
            # 写入然后立即读取
            self.buffer.async_write(addr, bytes([addr % 256] * 64))
            self.buffer.wait_all_writes()

            self.buffer.async_read(addr)
            self.buffer.wait_all_reads()

        # 记录串行执行的总周期数
        serial_cycles = self.buffer.current_cycle - start_cycle

        # 重置统计信息
        self.buffer.current_cycle = 0
        self.hbm.reset_stats()

        # 流水线执行16次读写操作
        # 先执行所有写入
        for addr in addresses:
            self.buffer.async_write(addr, bytes([addr % 256] * 64))

        # 等待所有写入完成
        self.buffer.wait_all_writes()

        # 再执行所有读取
        for addr in addresses:
            self.buffer.async_read(addr)

        # 等待所有读取完成
        self.buffer.wait_all_reads()

        # 记录流水线执行的总周期数
        pipelined_cycles = self.buffer.current_cycle

        # 验证流水线执行比串行执行更有效率
        self.assertLess(pipelined_cycles, serial_cycles)
        print(f"串行执行周期: {serial_cycles}, 流水线执行周期: {pipelined_cycles}")
        print(f"性能提升: {serial_cycles / pipelined_cycles:.2f}x")


if __name__ == "__main__":
    unittest.main()
