import numpy as np
import matplotlib.pyplot as plt
import multiprocessing  # 添加此行
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class MACUnit:
    def __init__(self):
        self.busy = False         # 标记单元是否正在工作
        self.remaining_cycles = 0  # 剩余需要的计算周期数
        self.result = None        # 计算结果
        self.a = None             # 输入操作数A
        self.b = None             # 输入操作数B
        self.done = False          # 标记计算是否完成

    def assign(self, a, b):
        """分配乘法任务给MAC单元"""
        self.a = a
        self.b = b
        self.busy = True
        self.remaining_cycles = 1  # Assume 1 cycle latency
        self.done = False
        
    def tick(self):
        """模拟时钟周期推进，返回计算完成的结果或None"""
        if self.busy:
            self.remaining_cycles -= 1
            if self.remaining_cycles == 0:
                self.result = self.a * self.b
                self.busy = False
                self.done = True
                return self.result
        return None
    

    def is_done(self):
        return self.done

    def clear(self):
        self.done = False
        self.result = None


class MACLine:
    """
    MAC单元阵列模拟类
    模拟具有多个并行MAC单元的计算线路
    """
    def __init__(self, mac_width):
        self.mac_units = [MACUnit() for _ in range(mac_width)]  # 创建多个MAC单元
        self.pending_blocks = [] # 存储待处理的块

    def enqueue_block(self, a_block, b_block):
        # 将块添加到待处理队列
        self.pending_blocks.append((a_block, b_block))

    def tick(self):
        # Assign next block if available and MAC units are free
        if self.pending_blocks:
            ready = all(not unit.busy for unit in self.mac_units)
            if ready:
                a_block, b_block = self.pending_blocks.pop(0)
                for i in range(len(self.mac_units)):
                    self.mac_units[i].assign(a_block[i], b_block[i])

        results = []
        for unit in self.mac_units:
            result = unit.tick()
            if unit.is_done():
                results.append(unit.result)
                unit.clear()
        return results

    def is_active(self):
        # 检查是否有单元正在工作或有待处理的块
        return self.pending_blocks or any(u.busy for u in self.mac_units)


class AdderTree:
    """
    加法树模拟类
    实现二叉树结构的加法网络，用于高效汇总多个部分积
    """
    def __init__(self):
        self.values = []

    def add_values(self, values):
        # 将新计算的值添加到加法树
        self.values.extend(values)

    def reduce(self):
        """执行二叉树结构的加法归约，返回最终结果"""
        values = self.values
        while len(values) > 1:
            if len(values) % 2 != 0:
                values.append(0)
            values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
        self.values = []
        return values[0] if values else 0


# class MACPipelineSimulator:
#     """
#     MAC流水线矩阵乘法模拟器
#     整合MAC线路和加法树, 用于模拟具有多个并行MAC单元的计算线路
#     """
#     def __init__(self, mac_width=4):
#         self.mac_width = mac_width  # Number of MAC units in one line
#         self.clock_cycle = 0        # Current clock cycle
#         self.mac_line = MACLine(mac_width)    # MAC计算线路
#         self.adder_tree = AdderTree()         # 加法树

#     def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
#         """ 
#         执行矩阵乘法 A x B = C 模拟硬件流水线过程
#         """
#         m, k1 = A.shape
#         k2, n = B.shape
#         assert k1 == k2, "Inner dimensions must match"

#         C = np.zeros((m, n))

#         for i in range(m):
#             for j in range(n):
#                 self.clock_cycle = 0
#                 mac_line = MACLine(self.mac_width)
#                 adder_tree = AdderTree()

#                 # Schedule all blocks
#                 for block_start in range(0, k1, self.mac_width):
#                     a_block = A[i, block_start:block_start + self.mac_width]
#                     b_block = B[block_start:block_start + self.mac_width, j]

#                     if len(a_block) < self.mac_width:
#                         a_block = np.pad(a_block, (0, self.mac_width - len(a_block)))
#                         b_block = np.pad(b_block, (0, self.mac_width - len(b_block)))

#                     mac_line.enqueue_block(a_block, b_block)


#                 while mac_line.is_active():
#                     self.clock_cycle += 1
#                     results = mac_line.tick()
#                     adder_tree.add_values(results)

#                 result = adder_tree.reduce()
#                 C[i][j] = result
#                 self.log.append(f"C[{i}][{j}] computed in {self.clock_cycle} cycles => {result}")
#         return C
        
class MACPipelineSimulator:
    def __init__(self, mac_width=4, parallel_mode='process', num_workers=None):
        self.mac_width = mac_width
        self.log = []
        self.parallel_mode = parallel_mode  # 'process', 'thread', 或 'serial'
        self.num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    
    def _compute_element(self, params):
        """计算结果矩阵的单个元素，作为并行任务"""
        i, j, A, B = params
        clock_cycle = 0
        mac_line = MACLine(self.mac_width)
        adder_tree = AdderTree()
        
        k = A.shape[1]  # A是m x k矩阵
        
        # 调度所有数据块
        for block_start in range(0, k, self.mac_width):
            a_block = A[i, block_start:block_start + self.mac_width].copy()
            b_block = B[block_start:block_start + self.mac_width, j].copy()
            
            if len(a_block) < self.mac_width:
                a_block = np.pad(a_block, (0, self.mac_width - len(a_block)))
                b_block = np.pad(b_block, (0, self.mac_width - len(b_block)))
            
            mac_line.enqueue_block(a_block, b_block)
        
        # 运行模拟
        while mac_line.is_active():
            clock_cycle += 1
            results = mac_line.tick()
            adder_tree.add_values(results)
        
        result = adder_tree.reduce()
        log_entry = f"C[{i}][{j}] computed in {clock_cycle} cycles => {result}"
        
        return (i, j, result, log_entry)
    
    def multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """执行矩阵乘法 A x B = C，支持并行计算"""
        m, k1 = A.shape
        k2, n = B.shape
        assert k1 == k2, "Inner dimensions must match"
        
        C = np.zeros((m, n))
        self.log = []
        
        # 准备任务参数
        tasks = [(i, j, A, B) for i in range(m) for j in range(n)]
        
        if self.parallel_mode == 'serial':
            # 串行执行（原始逻辑）
            results = [self._compute_element(task) for task in tasks]
        elif self.parallel_mode == 'thread':
            # 多线程并行执行
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._compute_element, tasks))
        
        elif self.parallel_mode == 'process':
            # 多进程并行执行（适合计算密集型任务）
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._compute_element, tasks))
        
        # 收集结果
        for i, j, value, log_entry in results:
            C[i, j] = value
            self.log.append(log_entry)
        
        return C

    def print_log(self):
        for entry in self.log:
            print(entry)
                    
        

    def print_log(self):
        for entry in self.log:
            print(entry)

def verify_result(m=4, k=7, n=5, random_seed=42):
    """
    生成随机矩阵 A (m x k) 和 B (k x n)，
    计算 MACLineMatrixMultiplier 计算的 C 与 NumPy 直接矩阵乘法的比较。
    """
    np.random.seed(random_seed)
    A = np.random.randint(0, 10, size=(m, k))
    B = np.random.randint(0, 10, size=(k, n))
    # A = np.random.uniform(0, 10, size=(m, k))
    # B = np.random.uniform(0, 10, size=(k, n))

    sim = MACPipelineSimulator(mac_width=4)
    C_mac = sim.multiply(A, B)
    sim.print_log()
    
    C_np = A @ B

    if np.allclose(C_mac, C_np):
        print("\n验证成功:结果一致!")
    else:
        print("\n验证失败:结果不一致!")
        
if __name__ == "__main__":
    verify_result(5,100,5)
