from typing import Callable, List, Dict, Any, Optional, Union, TypeVar, Generic
from collections import deque
import threading
import time
from hbm_sim import HBMSimulator


T = TypeVar('T')  # 用于泛型

class Observer(Generic[T]):
    """观察者接口，用于接收Observable发出的事件"""
    
    def on_next(self, value: T) -> None:
        """处理新的数据项"""
        pass
    
    def on_error(self, error: Exception) -> None:
        """处理错误"""
        pass
    
    def on_completed(self) -> None:
        """处理完成事件"""
        pass


class Observable(Generic[T]):
    """可观察对象，代表一个事件或数据流"""
    
    def __init__(self):
        self.observers: List[Observer[T]] = []
    
    def subscribe(self, observer: Observer[T]) -> 'Subscription':
        """订阅此Observable的事件"""
        self.observers.append(observer)
        return Subscription(self, observer)
    
    def notify(self, value: T) -> None:
        """通知所有观察者有新的数据"""
        for observer in self.observers:
            observer.on_next(value)
    
    def notify_error(self, error: Exception) -> None:
        """通知所有观察者发生了错误"""
        for observer in self.observers:
            observer.on_error(error)
    
    def notify_completed(self) -> None:
        """通知所有观察者流已完成"""
        for observer in list(self.observers):  # 创建副本以避免修改迭代中的列表
            observer.on_completed()
    
    def map(self, transform_func: Callable[[T], Any]) -> 'Observable':
        """转换Observable的数据"""
        result = Observable()
        
        class MapObserver(Observer[T]):
            def on_next(self, value: T) -> None:
                try:
                    transformed = transform_func(value)
                    result.notify(transformed)
                except Exception as e:
                    result.notify_error(e)
            
            def on_error(self, error: Exception) -> None:
                result.notify_error(error)
            
            def on_completed(self) -> None:
                result.notify_completed()
        
        self.subscribe(MapObserver())
        return result
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Observable':
        """过滤Observable的数据"""
        result = Observable()
        
        class FilterObserver(Observer[T]):
            def on_next(self, value: T) -> None:
                try:
                    if predicate(value):
                        result.notify(value)
                except Exception as e:
                    result.notify_error(e)
            
            def on_error(self, error: Exception) -> None:
                result.notify_error(error)
            
            def on_completed(self) -> None:
                result.notify_completed()
        
        self.subscribe(FilterObserver())
        return result
    
    @staticmethod
    def from_list(items: List[T]) -> 'Observable[T]':
        """从列表创建Observable"""
        result = Observable[T]()
        
        def emit_items():
            for item in items:
                result.notify(item)
            result.notify_completed()
        
        # 在新线程中发射项目，模拟异步行为
        threading.Thread(target=emit_items).start()
        return result
    
    @staticmethod
    def merge(*observables: 'Observable[T]') -> 'Observable[T]':
        """合并多个Observable"""
        result = Observable[T]()
        active_count = len(observables)
        
        class MergeObserver(Observer[T]):
            def on_next(self, value: T) -> None:
                result.notify(value)
            
            def on_error(self, error: Exception) -> None:
                result.notify_error(error)
            
            def on_completed(self) -> None:
                nonlocal active_count
                active_count -= 1
                if active_count == 0:
                    result.notify_completed()
        
        for obs in observables:
            obs.subscribe(MergeObserver())
        
        return result


class Subscription:
    """表示Observable和Observer之间的订阅关系"""
    
    def __init__(self, observable: Observable, observer: Observer):
        self.observable = observable
        self.observer = observer
        self.is_unsubscribed = False
    
    def unsubscribe(self) -> None:
        """取消订阅"""
        if not self.is_unsubscribed:
            if self.observer in self.observable.observers:
                self.observable.observers.remove(self.observer)
            self.is_unsubscribed = True


class ReactiveHBMOperation(Observable[bytes]):
    """表示响应式的HBM操作，可观察操作结果"""
    
    def __init__(self, 
                 operation_type: str, 
                 address: int, 
                 data: Optional[bytes] = None,
                 completion_cycle: int = 0):
        super().__init__()
        self.operation_type = operation_type  # 'read' 或 'write'
        self.address = address
        self.data = data
        self.completion_cycle = completion_cycle
        self.is_completed = False
    
    def complete(self, result_data: Optional[bytes] = None) -> None:
        """完成操作并通知观察者"""
        if not self.is_completed:
            self.is_completed = True
            try:
                # 对于读操作，通知数据；对于写操作，通知地址
                if self.operation_type == 'read':
                    self.notify(result_data or self.data)
                else:
                    self.notify(bytes(f"{self.address}".encode()))
                
                # 确保在独立的try块中调用notify_completed，以便数据通知后一定会触发完成事件
                try:
                    self.notify_completed()
                except Exception as e:
                    print(f"操作完成事件通知出错: {e}")
            except Exception as e:
                print(f"操作数据通知出错: {e}")
                # 即使数据通知失败，也尝试发送完成事件
                try:
                    self.notify_completed()
                except Exception as e2:
                    print(f"操作完成事件通知出错: {e2}")


class ReactiveHBMBuffer:
    """响应式HBM缓冲区，支持基于事件的流水线操作"""
    
    def __init__(self, hbm_simulator: HBMSimulator, buffer_size: int = 64):
        """
        初始化响应式HBM缓冲区
        
        参数:
            hbm_simulator: HBM模拟器实例
            buffer_size: 缓冲区大小(条目数)
        """
        self.hbm = hbm_simulator
        self.buffer_size = buffer_size
        self.pending_operations: deque[ReactiveHBMOperation] = deque()
        self.current_cycle = 0
        
        # 创建操作流Observable
        self.operation_stream = Observable()
        self.read_stream = Observable()
        self.write_stream = Observable()
        
        # 设置模拟时钟更新间隔（模拟时钟周期）
        self.clock_interval_ms = 1
        self._clock_running = False
        self._clock_thread = None
        
        # 同步锁，用于线程安全
        self._lock = threading.Lock()
    
    def read(self, address: int, size_bytes: Optional[int] = None) -> ReactiveHBMOperation:
        """
        执行读操作并返回可观察的结果
        
        参数:
            address: 读取的起始地址
            size_bytes: 读取的字节数，默认为一个通道宽度
            
        返回:
            可观察的读操作结果
        """
        with self._lock:
            if len(self.pending_operations) >= self.buffer_size:
                # 等待一些操作完成
                self._wait_for_space()
            
            # 从HBM读取数据
            data, latency = self.hbm.read(address, size_bytes)
            completion_cycle = self.current_cycle + latency
            
            # 创建响应式操作
            operation = ReactiveHBMOperation('read', address, data, completion_cycle)
            self.pending_operations.append(operation)
            
            # 通知操作流和读操作流
            self.operation_stream.notify(operation)
            self.read_stream.notify(operation)
            
            # 确保时钟在运行
            self._ensure_clock_running()
            
            return operation
    
    def _wait_for_space(self, timeout_ms: int = 1000) -> bool:
        """等待缓冲区有空闲空间"""
        start_time = time.time()
        while len(self.pending_operations) >= self.buffer_size:
            # 释放锁，允许时钟线程处理完成的操作
            self._lock.release()
            time.sleep(0.001)  # 短暂休眠
            self._lock.acquire()
            
            # 检查超时
            if time.time() - start_time > timeout_ms / 1000:
                return False
        return True
    
    def write(self, address: int, data: bytes) -> ReactiveHBMOperation:
        """
        执行写操作并返回可观察的结果
        
        参数:
            address: 写入的起始地址
            data: 要写入的数据
            
        返回:
            可观察的写操作结果
        """
        with self._lock:
            if len(self.pending_operations) >= self.buffer_size:
                # 等待一些操作完成
                self._wait_for_space()
            
            # 向HBM写入数据
            latency = self.hbm.write(address, data)
            completion_cycle = self.current_cycle + latency
            
            # 创建响应式操作
            operation = ReactiveHBMOperation('write', address, data, completion_cycle)
            self.pending_operations.append(operation)
            
            # 通知操作流和写操作流
            self.operation_stream.notify(operation)
            self.write_stream.notify(operation)
            
            # 确保时钟在运行
            self._ensure_clock_running()
            
            return operation
    
    def pipeline_read(self, addresses: List[int]) -> Observable:
        """
        执行流水线读取操作
        
        参数:
            addresses: 要读取的地址列表
            
        返回:
            包含所有读取结果的Observable
        """
        results = Observable()
        completed_count = [0]  # 使用列表以便在闭包中修改
        total_count = len(addresses)
        
        # 批量发起读取操作
        for i, addr in enumerate(addresses):
            try:
                operation = self.read(addr)
                
                class PipelineObserver(Observer[bytes]):
                    def on_next(self, value: bytes) -> None:
                        try:
                            results.notify(value)
                        except Exception as e:
                            print(f"流水线读取通知错误: {e}")
                    
                    def on_completed(self) -> None:
                        try:
                            completed_count[0] += 1
                            if completed_count[0] >= total_count:
                                results.notify_completed()
                        except Exception as e:
                            print(f"流水线读取完成通知错误: {e}")
                
                operation.subscribe(PipelineObserver())
            except Exception as e:
                print(f"启动流水线读取操作错误 (地址 {addr}): {e}")
        
        return results
    
    def pipeline_write(self, address_data_pairs: List[tuple]) -> Observable:
        """
        执行流水线写入操作
        
        参数:
            address_data_pairs: (地址, 数据)对的列表
            
        返回:
            表示所有写入完成的Observable
        """
        results = Observable()
        completed_count = [0]  # 使用列表以便在闭包中修改
        total_count = len(address_data_pairs)
        
        # 批量发起写入操作
        for addr, data in address_data_pairs:
            try:
                operation = self.write(addr, data)
                
                class PipelineObserver(Observer[bytes]):
                    def on_next(self, value: bytes) -> None:
                        try:
                            results.notify(value)
                        except Exception as e:
                            print(f"流水线写入通知错误: {e}")
                    
                    def on_completed(self) -> None:
                        try:
                            completed_count[0] += 1
                            if completed_count[0] >= total_count:
                                results.notify_completed()
                        except Exception as e:
                            print(f"流水线写入完成通知错误: {e}")
                
                operation.subscribe(PipelineObserver())
            except Exception as e:
                print(f"启动流水线写入操作错误 (地址 {addr}): {e}")
        
        return results
    
    def _clock_tick(self) -> None:
        """模拟时钟滴答，处理已完成的操作"""
        while self._clock_running:
            try:
                completed_ops = []
                
                with self._lock:
                    # 增加周期
                    self.current_cycle += 1
                    
                    # 检查是否有完成的操作
                    for op in list(self.pending_operations):
                        if op.completion_cycle <= self.current_cycle:
                            self.pending_operations.remove(op)
                            completed_ops.append(op)
                
                # 在锁外通知完成的操作
                for op in completed_ops:
                    try:
                        op.complete()
                    except Exception as e:
                        print(f"操作完成处理错误: {e}")
                
                # 确保操作完成通知后再检查是否需要停止时钟
                with self._lock:
                    # 如果没有待处理的操作，停止时钟
                    if not self.pending_operations:
                        # 给所有完成的操作一点时间来发送它们的通知
                        if completed_ops:
                            # 如果刚处理过操作，再等待一个周期
                            continue
                        else:
                            self._clock_running = False
                            break
            except Exception as e:
                print(f"时钟周期处理错误: {e}")
            
            # 休眠一段时间，模拟时钟周期
            time.sleep(self.clock_interval_ms / 1000)
    
    def _ensure_clock_running(self) -> None:
        """确保模拟时钟在运行"""
        if not self._clock_running:
            self._clock_running = True
            self._clock_thread = threading.Thread(target=self._clock_tick)
            self._clock_thread.daemon = True
            self._clock_thread.start()
    
    def stop_clock(self) -> None:
        """停止模拟时钟"""
        with self._lock:
            self._clock_running = False
        if self._clock_thread:
            self._clock_thread.join(timeout=0.5)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区状态统计"""
        with self._lock:
            return {
                "current_cycle": self.current_cycle,
                "pending_operations": len(self.pending_operations),
                "buffer_size": self.buffer_size,
            }
    
    def wait_until_idle(self, timeout_ms: int = 5000) -> bool:
        """等待直到所有操作完成"""
        start_time = time.time()
        last_pending_count = -1
        
        while True:
            with self._lock:
                pending_count = len(self.pending_operations)
                
                # 如果没有待处理操作，且上一次检查时也没有，则认为已完全空闲
                if pending_count == 0:
                    if last_pending_count == 0:
                        # 再等待一小段时间确保所有操作都完成了它们的通知
                        time.sleep(0.01)
                        return True
                    last_pending_count = 0
                else:
                    last_pending_count = pending_count
            
            # 短暂睡眠，避免过度占用CPU
            time.sleep(0.01)
            
            # 检查超时
            if time.time() - start_time > timeout_ms / 1000:
                return False


# 使用示例
if __name__ == "__main__":
    # 创建HBM模拟器实例
    hbm = HBMSimulator()
    
    # 创建响应式缓冲区，增大缓冲区大小
    reactive_buffer = ReactiveHBMBuffer(hbm, buffer_size=128)
    
    print("=== 响应式HBM缓冲区示例 ===")
    
    # 示例1: 单个读写操作
    print("\n示例1: 单个响应式读写操作")
    
    # 写入数据
    write_op = reactive_buffer.write(0, bytes([1, 2, 3, 4]))
    
    # 通过观察者模式获取结果
    class WriteObserver(Observer[bytes]):
        def on_next(self, value: bytes) -> None:
            print(f"写入完成，地址: {int(value)}")
        
        def on_completed(self) -> None:
            print("写入操作已完成")
    
    write_op.subscribe(WriteObserver())
    
    # 读取数据
    read_op = reactive_buffer.read(0)
    
    class ReadObserver(Observer[bytes]):
        def on_next(self, value: bytes) -> None:
            print(f"读取完成，数据: {list(value[:4])}")
        
        def on_completed(self) -> None:
            print("读取操作已完成")
    
    read_op.subscribe(ReadObserver())
    
    # 等待这些操作完成
    time.sleep(0.1)
    
    # 示例2: 流水线操作
    print("\n示例2: 流水线操作")
    
    # 准备地址列表，确保映射到不同通道
    read_addresses = [ch * 64 for ch in range(8)]
    
    # 先写入一些数据
    write_pairs = [(addr, bytes([addr % 256] * 64)) for addr in read_addresses]
    
    processing_done = threading.Event()
    
    class PipelineReadObserver(Observer[bytes]):
        def __init__(self):
            self.count = 0
        
        def on_next(self, value: bytes) -> None:
            self.count += 1
            print(f"读取#{self.count}完成: {list(value[:4])}")
        
        def on_completed(self) -> None:
            print("所有流水线读取已完成")
            processing_done.set()
    
    class PipelineWriteObserver(Observer[bytes]):
        def __init__(self):
            self.count = 0
        
        def on_next(self, value: bytes) -> None:
            self.count += 1
            if self.count == 1:  # 只打印第一个结果避免输出过多
                print(f"写入完成，地址: {int(value)}")
        
        def on_completed(self) -> None:
            print("所有流水线写入已完成")
            
            # 现在执行流水线读取
            print("开始流水线读取...")
            read_results = reactive_buffer.pipeline_read(read_addresses)
            read_results.subscribe(PipelineReadObserver())
    
    # 开始流水线写入
    print("开始流水线写入...")
    write_results = reactive_buffer.pipeline_write(write_pairs)
    write_results.subscribe(PipelineWriteObserver())
    
    # 等待处理完成或超时
    processing_done.wait(timeout=2.0)
    
    # 等待所有操作完成
    reactive_buffer.wait_until_idle()
    
    print("\n=== 统计信息 ===")
    print(f"HBM统计: {hbm.get_stats()}")
    print(f"缓冲区统计: {reactive_buffer.get_stats()}")
    
    # 停止时钟以允许程序干净退出
    reactive_buffer.stop_clock() 