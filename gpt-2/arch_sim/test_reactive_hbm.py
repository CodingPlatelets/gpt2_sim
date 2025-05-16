from reactive_hbm import (
    Observable, Observer, Subscription,
    ReactiveHBMOperation, ReactiveHBMBuffer,
    HBMSimulator
)
import pytest
import sys
import os
import threading
import time
from typing import List, Any
from collections import deque

# 添加模块路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 导入要测试的模块


# 辅助函数用于打印测试信息
def print_test_info(test_name):
    """打印测试信息的辅助函数"""
    print(f"\n{'='*50}")
    print(f"开始测试: {test_name}")
    print(f"{'='*50}")

def print_test_success(test_name):
    """打印测试成功信息的辅助函数"""
    print(f"\n{'-'*50}")
    print(f"✅ 测试通过: {test_name}")
    print(f"{'-'*50}")

# 固定装置(fixtures)
@pytest.fixture(scope="session", autouse=True)
def test_session_start():
    """在测试会话开始时打印信息"""
    print("\n\n⭐⭐⭐ 开始响应式HBM模拟器测试会话 ⭐⭐⭐\n")
    yield
    print("\n\n✨✨✨ 所有测试完成 ✨✨✨\n")

@pytest.fixture
def hbm_simulator():
    """创建并返回一个HBM模拟器实例"""
    return HBMSimulator()


@pytest.fixture
def reactive_buffer(hbm_simulator):
    """创建并返回一个标准大小的响应式缓冲区"""
    return ReactiveHBMBuffer(hbm_simulator, buffer_size=20)


@pytest.fixture
def large_buffer(hbm_simulator):
    """创建并返回一个大型响应式缓冲区"""
    return ReactiveHBMBuffer(hbm_simulator, buffer_size=100)


# 测试Observable和Observer基本功能
def test_subscribe_notify():
    """测试订阅和通知机制"""
    print_test_info("订阅和通知机制")
    
    observable = Observable()
    received_data = []
    completion_called = [False]
    
    class TestObserver(Observer):
        def on_next(self, value):
            received_data.append(value)
            print(f"  接收数据: {value}")
        
        def on_error(self, error):
            pytest.fail(f"不应该收到错误: {error}")
        
        def on_completed(self):
            completion_called[0] = True
            print("  已收到完成信号")
    
    # 订阅Observable
    subscription = observable.subscribe(TestObserver())
    print("✓ 成功订阅Observable")
    
    # 发出数据项和完成信号
    observable.notify("data1")
    observable.notify("data2")
    observable.notify_completed()
    
    # 验证结果
    assert received_data == ["data1", "data2"]
    assert completion_called[0] is True
    
    # 测试取消订阅
    subscription.unsubscribe()
    print("✓ 成功取消订阅")
    observable.notify("data3")  # 该数据不应该被接收
    assert len(received_data) == 2  # 仍然是2项
    
    print_test_success("订阅和通知机制")


def test_error_handling():
    """测试错误处理"""
    print_test_info("错误处理")
    
    observable = Observable()
    error_received = [False]
    test_error = ValueError("测试错误")
    
    class ErrorObserver(Observer):
        def on_next(self, value):
            pass
        
        def on_error(self, error):
            error_received[0] = True
            print(f"  正确接收到错误: {error}")
            assert error == test_error
        
        def on_completed(self):
            pass
    
    observable.subscribe(ErrorObserver())
    print("✓ 成功订阅Observable")
    
    observable.notify_error(test_error)
    
    assert error_received[0] is True
    print_test_success("错误处理")


def test_operators():
    """测试操作符功能（map, filter）"""
    print_test_info("操作符功能")
    
    # 手动创建Observable发出数据，而不是使用from_list
    source = Observable()
    results = []
    completion_called = [False]
    
    # 创建转换链: map->filter->map
    print("✓ 创建转换链: map->filter->map")
    result = source \
        .map(lambda x: x * 10) \
        .filter(lambda x: x > 20) \
        .map(lambda x: f"num:{x}")
    
    class ResultObserver(Observer):
        def on_next(self, value):
            results.append(value)
            print(f"  接收转换后的数据: {value}")
        
        def on_completed(self):
            completion_called[0] = True
            print("  转换链完成")
    
    result.subscribe(ResultObserver())
    
    # 手动发出数据项和完成信号
    for i in range(1, 6):
        source.notify(i)
    source.notify_completed()
    
    # 验证结果
    assert results == ["num:30", "num:40", "num:50"]
    assert completion_called[0] is True
    
    print_test_success("操作符功能")


# 测试ReactiveHBMOperation功能
def test_read_operation(hbm_simulator, reactive_buffer):
    """测试读操作的响应式功能"""
    print_test_info("读操作响应式功能")
    
    # 先写入一些数据
    test_addr = 128
    hbm_simulator.write(test_addr, bytes([42] * 64))
    print(f"✓ 写入测试数据到地址 {test_addr}")
    
    # 创建响应式读操作
    read_op = reactive_buffer.read(test_addr)
    print(f"✓ 创建响应式读操作")
    
    data_received = []
    completed = [False]
    
    class ReadObserver(Observer):
        def on_next(self, value):
            data_received.append(value)
            print(f"  接收读取的数据: {list(value[:4])}")
        
        def on_completed(self):
            completed[0] = True
            print("  读操作完成")
    
    read_op.subscribe(ReadObserver())
    
    # 等待操作完成
    reactive_buffer.wait_until_idle()
    
    # 验证结果
    assert len(data_received) == 1
    assert data_received[0][:4] == bytes([42] * 4)
    assert completed[0] is True
    
    print_test_success("读操作响应式功能")


def test_write_operation(hbm_simulator, reactive_buffer):
    """测试写操作的响应式功能"""
    print_test_info("写操作响应式功能")
    
    test_addr = 256
    test_data = bytes([0xAA] * 64)
    print(f"✓ 准备写入数据到地址 {test_addr}")
    
    write_op = reactive_buffer.write(test_addr, test_data)
    
    address_received = []
    completed = [False]
    
    class WriteObserver(Observer):
        def on_next(self, value):
            address_received.append(int(value))
            print(f"  接收写操作完成通知，地址: {int(value)}")
        
        def on_completed(self):
            completed[0] = True
            print("  写操作完成")
    
    write_op.subscribe(WriteObserver())
    
    # 等待操作完成
    reactive_buffer.wait_until_idle()
    
    # 验证结果
    assert address_received == [test_addr]
    assert completed[0] is True
    
    # 验证数据确实写入了HBM
    read_data, _ = hbm_simulator.read(test_addr)
    assert read_data == test_data
    print(f"✓ 成功验证写入的数据")
    
    print_test_success("写操作响应式功能")


def test_operation_chaining(hbm_simulator, reactive_buffer):
    """测试操作链接"""
    print_test_info("操作链接")
    
    test_addr = 512
    test_data = bytes([0xBB] * 64)
    
    # 写入操作完成后立即读取
    write_then_read_data = []
    
    class ChainObserver(Observer):
        def __init__(self, buffer):
            self.buffer = buffer
        
        def on_next(self, value):
            # 写入完成后的地址
            addr = int(value)
            print(f"  写入操作完成，地址：{addr}")
            
            # 读取刚刚写入的地址
            print(f"  立即启动读取操作")
            read_op = self.buffer.read(addr)
            
            class ReadObserver(Observer):
                def on_next(self, data):
                    write_then_read_data.append(data)
                    print(f"  读取操作完成，数据: {list(data[:4])}")
            
            read_op.subscribe(ReadObserver())
    
    # 执行写入并链接到读取
    print(f"✓ 执行写入操作并链接读取")
    write_op = reactive_buffer.write(test_addr, test_data)
    write_op.subscribe(ChainObserver(reactive_buffer))
    
    # 等待所有操作完成
    reactive_buffer.wait_until_idle()
    
    # 验证结果
    assert len(write_then_read_data) == 1
    assert write_then_read_data[0] == test_data
    
    print_test_success("操作链接")


# 测试流水线操作
def test_pipeline_read(hbm_simulator, reactive_buffer):
    """测试流水线读取操作"""
    print_test_info("流水线读取")
    
    # 准备一些测试数据
    addresses = [i * 64 for i in range(10)]
    print(f"✓ 准备10个测试地址: {addresses[:3]}...")
    
    for addr in addresses:
        hbm_simulator.write(addr, bytes([addr % 256] * 64))
    print(f"✓ 已写入所有测试数据")
    
    # 执行流水线读取
    results = []
    completion_called = [False]
    
    class PipelineObserver(Observer):
        def on_next(self, value):
            results.append(value)
            if len(results) % 3 == 0 or len(results) == len(addresses):
                print(f"  已接收 {len(results)}/{len(addresses)} 个读取结果")
        
        def on_completed(self):
            completion_called[0] = True
            print("  流水线读取完成")
    
    print(f"✓ 启动流水线读取操作")
    read_results = reactive_buffer.pipeline_read(addresses)
    read_results.subscribe(PipelineObserver())
    
    # 等待所有操作完成
    reactive_buffer.wait_until_idle()
    
    # 验证结果
    assert len(results) == 10
    # 验证数据内容
    for i, data in enumerate(results):
        expected_value = addresses[i] % 256
        assert data[0] == expected_value
    assert completion_called[0] is True
    
    print_test_success("流水线读取")


def test_pipeline_write(hbm_simulator, reactive_buffer):
    """测试流水线写入操作"""
    print_test_info("流水线写入")
    
    # 准备要写入的数据对
    address_data_pairs = [(i * 128, bytes([i] * 64)) for i in range(5)]
    print(f"✓ 准备5个写入数据对")
    
    # 执行流水线写入
    addresses_received = []
    completion_called = [False]
    
    class PipelineObserver(Observer):
        def on_next(self, value):
            addresses_received.append(int(value))
            print(f"  收到写入完成通知: 地址 {int(value)}")
        
        def on_completed(self):
            completion_called[0] = True
            print("  流水线写入完成")
    
    print(f"✓ 启动流水线写入操作")
    write_results = reactive_buffer.pipeline_write(address_data_pairs)
    write_results.subscribe(PipelineObserver())
    
    # 等待所有操作完成
    reactive_buffer.wait_until_idle()
    
    # 验证结果 - 收到的地址可能顺序不同
    assert len(addresses_received) == 5
    for addr, _ in address_data_pairs:
        assert addr in addresses_received
    assert completion_called[0] is True
    
    # 验证数据确实写入了HBM
    print(f"✓ 验证所有数据已正确写入")
    for addr, expected_data in address_data_pairs:
        actual_data, _ = hbm_simulator.read(addr)
        assert actual_data == expected_data
    
    print_test_success("流水线写入")


def test_write_then_read_pipeline(hbm_simulator, reactive_buffer):
    """测试写入后读取的流水线操作"""
    print_test_info("写入后读取的流水线")
    
    # 准备要写入的数据对
    addresses = [i * 64 for i in range(8)]
    address_data_pairs = [(addr, bytes([addr % 256] * 64))
                           for addr in addresses]
    
    # 等待写入完成事件
    write_completed = threading.Event()
    
    # 读取结果
    read_results = []
    read_completed = threading.Event()
    
    print("✓ 设置写入完成后自动执行读取的链式操作")
    
    class ReadObserver(Observer):
        def on_next(self, value):
            read_results.append(value)
            if len(read_results) % 4 == 0 or len(read_results) == len(addresses):
                print(f"  已接收 {len(read_results)}/{len(addresses)} 个读取结果")
        
        def on_completed(self):
            read_completed.set()
            print("  流水线读取完成")
    
    class WriteObserver(Observer):
        def __init__(self, buffer, addresses):
            self.buffer = buffer
            self.addresses = addresses
        
        def on_completed(self):
            # 写入完成后开始读取
            write_completed.set()
            print("  流水线写入完成，开始流水线读取")
            read_pipeline = self.buffer.pipeline_read(self.addresses)
            read_pipeline.subscribe(ReadObserver())
    
    # 执行流水线写入
    print("✓ 启动流水线写入操作")
    write_results = reactive_buffer.pipeline_write(address_data_pairs)
    write_results.subscribe(WriteObserver(reactive_buffer, addresses))
    
    # 等待操作完成
    write_completed.wait(timeout=2.0)
    read_completed.wait(timeout=2.0)
    
    # 验证结果
    assert write_completed.is_set(), "写入操作未完成"
    assert read_completed.is_set(), "读取操作未完成"
    assert len(read_results) == 8
    
    print_test_success("写入后读取的流水线")


# 测试并发和线程安全性
def test_concurrent_operations(hbm_simulator, large_buffer):
    """测试并发操作的处理"""
    print_test_info("并发操作处理")
    
    # 创建多个线程同时执行读写操作
    threads = []
    results = []
    
    # 用于同步线程
    start_event = threading.Event()
    thread_completed = [0]
    
    def worker_thread(thread_id, buffer):
        # 等待开始信号
        start_event.wait()
        print(f"  线程 {thread_id} 开始执行")
        
        # 执行一些读写操作
        try:
            for i in range(10):
                addr = thread_id * 1000 + i * 64
                data = bytes([thread_id] * 64)
                
                # 写入然后读取
                write_op = buffer.write(addr, data)
                
                class ReadAfterWrite(Observer):
                    def __init__(self, buffer, addr):
                        self.buffer = buffer
                        self.addr = addr
                    
                    def on_completed(self):
                        # 写入完成后读取
                        read_op = self.buffer.read(addr)
                        
                        class ReadObserver(Observer):
                            def on_next(self, value):
                                results.append((thread_id, addr, value[:1]))
                        
                        read_op.subscribe(ReadObserver())
                
                write_op.subscribe(ReadAfterWrite(buffer, addr))
            
            # 线程完成
            with threading.Lock():
                thread_completed[0] += 1
                print(f"  线程 {thread_id} 完成，已完成 {thread_completed[0]}/5 个线程")
        except Exception as e:
            pytest.fail(f"线程{thread_id}发生异常: {e}")
    
    # 创建几个工作线程
    print("✓ 创建5个并发工作线程")
    for i in range(5):
        t = threading.Thread(target=worker_thread, args=(i, large_buffer))
        threads.append(t)
        t.start()
    
    # 让所有线程同时开始
    print("✓ 同时启动所有线程")
    start_event.set()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    # 等待所有操作完成
    large_buffer.wait_until_idle()
    
    # 验证结果
    print(f"✓ 验证结果: {len(results)} 个操作已完成")
    assert len(results) == 50  # 5个线程，每个10个操作
    # 验证每个线程的数据正确
    for thread_id, addr, value in results:
        assert value == bytes([thread_id])
    
    print_test_success("并发操作处理")


def test_buffer_overflow_handling(hbm_simulator):
    """测试缓冲区溢出处理"""
    print_test_info("缓冲区溢出处理")
    
    # 创建一个小型缓冲区
    buffer_size = 10
    small_buffer = ReactiveHBMBuffer(hbm_simulator, buffer_size=buffer_size)
    print(f"✓ 创建大小为 {buffer_size} 的小型缓冲区")
    
    # 尝试提交超过缓冲区大小的操作
    num_operations = 20
    print(f"✓ 尝试提交 {num_operations} 个操作 (超过缓冲区大小)")
    
    for i in range(num_operations):
        small_buffer.write(i * 64, bytes([i % 256] * 64))
        if i % 5 == 0:
            print(f"  已提交 {i}/{num_operations} 个操作")
    
    # 等待所有操作完成
    print("✓ 等待所有操作完成")
    small_buffer.wait_until_idle()
    
    # 验证所有操作都完成了
    assert len(small_buffer.pending_operations) == 0
    print("✓ 确认所有操作都已完成")
    
    # 验证HBM中的数据正确
    print("✓ 验证所有数据都已正确写入")
    for i in range(num_operations):
        data, _ = hbm_simulator.read(i * 64)
        assert data[0] == i % 256
    
    print_test_success("缓冲区溢出处理")


# 参数化测试示例
@pytest.mark.parametrize("buffer_size,num_operations", [
    (5, 10),    # 小缓冲区，多操作
    (20, 15),   # 中等缓冲区，少操作
    (50, 100),  # 大缓冲区，大量操作
])
def test_various_buffer_sizes(hbm_simulator, buffer_size, num_operations):
    """测试不同缓冲区大小下的性能"""
    print_test_info(f"不同缓冲区大小: 大小={buffer_size}, 操作数={num_operations}")
    
    buffer = ReactiveHBMBuffer(hbm_simulator, buffer_size=buffer_size)
    print(f"✓ 创建大小为 {buffer_size} 的缓冲区")
    
    # 执行指定数量的操作
    print(f"✓ 执行 {num_operations} 个写入操作")
    start_time = time.time()
    
    for i in range(num_operations):
        buffer.write(i * 64, bytes([i % 256] * 64))
        if i % (num_operations // 4) == 0:
            print(f"  已完成 {i}/{num_operations} 个操作")
    
    # 等待所有操作完成
    buffer.wait_until_idle()
    
    # 计算执行时间
    elapsed = time.time() - start_time
    print(f"✓ 所有操作完成，耗时: {elapsed:.4f}秒")
    print(f"✓ 每操作平均用时: {elapsed/num_operations*1000:.2f}毫秒")
    
    # 验证所有操作都完成了
    assert len(buffer.pending_operations) == 0
    
    # 验证统计数据合理
    stats = buffer.get_stats()
    assert stats["buffer_size"] == buffer_size
    assert stats["pending_operations"] == 0
    assert stats["current_cycle"] > 0
    
    print_test_success(f"不同缓冲区大小: 大小={buffer_size}, 操作数={num_operations}")
