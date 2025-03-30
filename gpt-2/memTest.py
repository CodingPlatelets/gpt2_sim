import unittest
import torch
from attention_sim import _HardwareSimulator, SRAMOverflowError, DDROverflowError


class TestHardwareSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = _HardwareSimulator()
        # 设置较小的内存容量便于测试
        self.sim.initialize(
            sram_capacity=1024,     # 1KB SRAM
            ddr_capacity=4096,      # 4KB DDR
            read_delay=5,
            write_delay=10,
            ddr_read_delay=50,
            ddr_write_delay=40,
            ddr_to_sram_delay=20
        )

    def test_sram_lru_eviction(self):
        """测试SRAM的LRU淘汰策略"""
        # 1. 写入两个半满数据块
        data_a = torch.randn(8, 16)  # 512 bytes
        data_b = torch.randn(8, 16)  # 512 bytes
        self.sim.sram_load("A", data_a)
        self.sim.sram_load("B", data_b)

        # 2. 读取A使其成为"最近使用"
        _ = self.sim.sram_read("A")

        # 3. 写入新数据应淘汰B（因为A更近被使用）
        data_c = torch.randn(8, 16)  # 512 bytes
        self.sim.sram_load("C", data_c)

        # 4. 验证B被淘汰到DDR，而非完全删除
        self.assertNotIn("B", self.sim.sram)    # B不在SRAM中
        self.assertIn("B", self.sim.ddr)        # B在DDR中
        self.assertIn("A", self.sim.sram)       # A仍在SRAM中
        self.assertIn("C", self.sim.sram)       # C在SRAM中

        # 5. 检查日志记录
        logs = "\n".join(self.sim.access_log)
        self.assertIn("EVICT B", logs)         # 检查是否淘汰了B
        self.assertIn("from SRAM to DDR", logs)  # 检查淘汰目标是DDR

    def test_ddr_load_and_access(self):
        """测试直接加载到DDR和从DDR访问数据"""
        # 1. 直接加载数据到DDR
        data_a = torch.randn(16, 16)  # 1024 bytes
        self.sim.ddr_load("A", data_a)
        
        # 2. 验证数据只在DDR中
        self.assertNotIn("A", self.sim.sram)
        self.assertIn("A", self.sim.ddr)
        
        # 3. 通过sram_read尝试读取DDR中的数据
        cycles_before = self.sim.cycles
        data = self.sim.sram_read("A")
        
        # 4. 验证数据已加载到SRAM
        self.assertIn("A", self.sim.sram)
        
        # 5. 验证延迟计算正确 (DDR读取 + DDR到SRAM传输 + SRAM读取)
        delay = self.sim.ddr_read_delay + self.sim.ddr_to_sram_delay + self.sim.read_delay
        self.assertEqual(self.sim.cycles - cycles_before, delay)
        
        # 6. 检查返回的数据正确
        torch.testing.assert_close(data, data_a)

    def test_sram_overflow_handling(self):
        """测试SRAM溢出时正确流转数据到DDR"""
        # 1. 填满SRAM
        data_a = torch.randn(16, 16)  # 1024 bytes
        self.sim.sram_load("A", data_a)
        
        # 2. 继续加载数据应触发SRAM到DDR的转移
        data_b = torch.randn(8, 8)    # 256 bytes
        self.sim.sram_load("B", data_b)
        
        # 3. 验证A被转移到DDR，B在SRAM
        self.assertNotIn("A", self.sim.sram)
        self.assertIn("A", self.sim.ddr)
        self.assertIn("B", self.sim.sram)
        
        # 4. 加载更多数据，验证B仍在SRAM（没有被转移）
        data_c = torch.randn(8, 8)    # 256 bytes
        self.sim.sram_load("C", data_c)
        
        self.assertIn("B", self.sim.sram)
        self.assertIn("C", self.sim.sram)

    def test_ddr_overflow(self):
        """测试DDR溢出处理"""
        # 1. 填满DDR
        data_blocks = []
        for i in range(8):  # 8 blocks of 512 bytes = 4KB (填满DDR)
            data = torch.randn(8, 16)  # 512 bytes
            self.sim.ddr_load(f"DDR_{i}", data)
            data_blocks.append(data)
        
        # 2. 验证所有数据都在DDR中
        for i in range(8):
            self.assertIn(f"DDR_{i}", self.sim.ddr)
        
        # 3. 再加载一个数据块，应触发DDR的LRU淘汰
        new_data = torch.randn(8, 16)  # 512 bytes
        self.sim.ddr_load("DDR_new", new_data)
        
        # 4. 验证最早加载的数据被淘汰
        self.assertNotIn("DDR_0", self.sim.ddr)
        self.assertIn("DDR_new", self.sim.ddr)
        
        # 5. 检查日志记录
        logs = "\n".join(self.sim.access_log)
        self.assertIn("EVICT DDR_0", logs)
        self.assertIn("from DDR", logs)

    def test_data_retrieval_flow(self):
        """测试数据检索流程 - SRAM缓存未命中时，从DDR获取"""
        # 1. 直接加载到DDR
        data_orig = torch.randn(4, 4)  # 64 bytes
        self.sim.ddr_load("flow_test", data_orig)
        
        # 2. 填满SRAM防止直接加载
        self.sim.sram_load("filler", torch.randn(16, 16))  # 1024 bytes
        
        # 3. 尝试读取DDR中的数据
        data_read = self.sim.sram_read("flow_test")
        
        # 4. 验证数据流转 - filler被移到DDR，flow_test被移到SRAM
        self.assertNotIn("filler", self.sim.sram)
        self.assertIn("filler", self.sim.ddr)
        self.assertIn("flow_test", self.sim.sram)
        
        # 5. 验证读取的数据正确
        torch.testing.assert_close(data_read, data_orig)

    def test_ddr_eviction_when_full(self):
        """测试使用evict_if_full=False时的行为"""
        # 1. 填满DDR
        for i in range(8):  # 8 blocks of 512 bytes = 4KB
            data = torch.randn(8, 16)  # 512 bytes
            self.sim.ddr_load(f"block_{i}", data)
            
        # 2. 尝试在不允许淘汰的情况下加载新数据
        with self.assertRaises(DDROverflowError):
            self.sim.ddr_load("overflow_test", torch.randn(8, 16), evict_if_full=False)
            
        # 3. 使用默认参数应正常工作
        self.sim.ddr_load("should_work", torch.randn(8, 16))
        self.assertIn("should_work", self.sim.ddr)

    def test_memory_stats(self):
        """测试内存使用统计正确性"""
        # 1. 初始状态
        self.assertEqual(self.sim.sram_used, 0)
        self.assertEqual(self.sim.ddr_used, 0)
        
        # 2. 加载数据后
        data_a = torch.randn(8, 8)  # 256 bytes
        self.sim.sram_load("stats_test", data_a)
        
        self.assertEqual(self.sim.sram_used, 256)
        
        # 3. 移动到DDR后
        self.sim.sram_load("filler", torch.randn(16, 16))  # 1024 bytes (触发淘汰)
        
        self.assertEqual(self.sim.sram_used, 1024)  # 只有filler在SRAM
        self.assertEqual(self.sim.ddr_used, 256)     # stats_test在DDR
        
        # 4. 从DDR读回后
        _ = self.sim.sram_read("stats_test")
        
        # filler应该被淘汰到DDR，stats_test从DDR移到SRAM
        self.assertEqual(self.sim.sram_used, 256)     # 只有stats_test在SRAM
        self.assertEqual(self.sim.ddr_used, 1024)     # 只有filler在DDR

    def test_cycles_accounting(self):
        """测试时钟周期计数正确性"""
        # 重置计数
        self.sim.cycles = 0
        
        # 1. SRAM写入
        self.sim.sram_load("cycle_a", torch.randn(4, 4))
        self.assertEqual(self.sim.cycles, self.sim.write_delay)
        
        # 2. SRAM读取
        self.sim.cycles = 0
        _ = self.sim.sram_read("cycle_a")
        self.assertEqual(self.sim.cycles, self.sim.read_delay)
        
        # 3. DDR写入
        self.sim.cycles = 0
        self.sim.ddr_load("cycle_b", torch.randn(4, 4))
        self.assertEqual(self.sim.cycles, self.sim.ddr_write_delay)
        
        # 4. DDR到SRAM的加载
        self.sim.cycles = 0
        _ = self.sim.sram_read("cycle_b")  # 触发从DDR到SRAM的加载
        expected_delay = self.sim.ddr_read_delay + self.sim.ddr_to_sram_delay + self.sim.read_delay
        self.assertEqual(self.sim.cycles, expected_delay)


if __name__ == "__main__":
    unittest.main()
