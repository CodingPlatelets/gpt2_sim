import numpy as np
import math
import time
import logging
from scipy.sparse import csr_matrix, lil_matrix

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SparseMatrixDistribution")

def min_bits_needed(bit_width):
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))

class Distribution:
    def __init__(self):
        self.data_buffer = []
        self.data_mask_1 = None
        self.data_mask_2 = None
    
    def get_data(self, data):
        self.data_buffer = data

class ShiftUnit:
    def __init__(self, bit_mask, ec_idx, values_len, bit_width=4, offset=0):
        self.bit_width = bit_width
        self.bit_mask = bit_mask
        self.ec_idx = ec_idx
        self.offset = offset
        self.min_bits_num = min_bits_needed(bit_width)
        self.zero_count = []
        self.zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.index = [0] * len(self.ec_idx)
        self.values_len = values_len

    def get_zero_count(self):
        self.zero_count = []
        count_zeros = 0
        for i in range(self.bit_width):
            self.zero_count.append(count_zeros)
            bit = (self.bit_mask >> (self.bit_width - 1 - i)) & 1
            if bit == 0:
                count_zeros += 1
    
    def get_zero_count_bit_vec(self):
        bit_vec = []
        for count in self.zero_count:
            bit_vec.append(bin(count)[2:].zfill(self.min_bits_num))

        for bit in bit_vec:
            for i in range(self.min_bits_num):
                self.zero_count_bit_vec[i].append(int(bit[self.min_bits_num - i - 1]))

    def shift(self):
        shifted_ec_idx = self.ec_idx.copy()
        
        for bit_level in range(self.min_bits_num):
            temp_result = [0] * len(shifted_ec_idx)
            for i in range(len(self.zero_count_bit_vec[bit_level])):
                if self.zero_count_bit_vec[bit_level][i] == 1:
                    target_idx = i - (1 << bit_level)
                    if target_idx < len(temp_result):
                        temp_result[target_idx] = shifted_ec_idx[i]
                else:
                    temp_result[i] = shifted_ec_idx[i]
            
            shifted_ec_idx = temp_result.copy()
     
        self.index = [0] * self.values_len
        for i in range(len(shifted_ec_idx)):
            target_idx = i + self.offset
            if target_idx < self.values_len:  
                self.index[target_idx] = shifted_ec_idx[i]

    def print_state(self):
        """打印 ShiftUnit 对象的关键信息"""
        print("\n==== ShiftUnit 状态 ====")
        print(f"bit_width: {self.bit_width}")
        print(f"offset: {self.offset}")
        print(f"bit_mask: {bin(self.bit_mask)[2:].zfill(self.bit_width)}")
        print(f"min_bits_num: {self.min_bits_num}")
        
        print(f"zero_count: {self.zero_count}")
        
        print("\nzero_count_bit_vec:")
        for i, bit_vec in enumerate(self.zero_count_bit_vec):
            print(f"  Bit {i}: {bit_vec}")
        
        print(f"Input ec_idx: {self.ec_idx}")
        print(f"Output index: {self.index}")
        print("=======================")


    def __call__(self):           
        self.get_zero_count()
        self.get_zero_count_bit_vec()
        self.shift()

class MFIU:
    def __init__(self, width=4, bit_width=4):
        self.width = width
        self.bit_width = bit_width
        self.A_bit_mask_vec = [0] * width
        self.B_bit_mask_vec = [0] * width
        self.A_row_offset_vec = [0] * width
        self.B_col_offset_vec = [0] * width
        self.AB_bit_vec = [0] * width
        self.AB_prefix_sum = [] 
        self.ec_idx_vec = [[] for _ in range(width)]
        

    def get_mask_offset_values(self, mask_A_row, mask_B_col, offset_A_row, offset_B_col, values_A, values_B):
        
        assert(len(mask_B_col) * len(mask_A_row) == self.width)
        idx = 0
        for i in range(len(mask_B_col)):
            for j in range(len(mask_A_row)):
                self.B_bit_mask_vec[idx] = mask_B_col[i]
                self.B_col_offset_vec[idx] = offset_B_col[i]
                self.A_bit_mask_vec[idx] = mask_A_row[j]
                self.A_row_offset_vec[idx] = offset_A_row[j]
                idx += 1
        self.values_A = values_A
        self.values_B = values_B

    def bitvec_to_bitseq(self, bitvec):
        all_bits = []
        for val in bitvec:
            for i in range(self.bit_width):
                bit = (val >> (self.bit_width - 1 - i)) & 1
                all_bits.append(bit)
        return all_bits
    
    def ecseq_to_ecvec(self, ecseq):
        for i in range(self.width):
            temp = []
            for j in range(self.bit_width):
                temp.append(ecseq[i * self.bit_width + j])
            self.ec_idx_vec[i] = temp
    
    def init_shift_unit(self):
        self.shift_unit_a = [ShiftUnit(self.A_bit_mask_vec[i], self.ec_idx_vec[i], len(self.values_A), self.bit_width, self.A_row_offset_vec[i]) for i in range(self.width)]
        self.shift_unit_b = [ShiftUnit(self.B_bit_mask_vec[i], self.ec_idx_vec[i], len(self.values_B), self.bit_width, self.B_col_offset_vec[i]) for i in range(self.width)]

    def print_shift(self):
        print("shiftA index")
        for shift in self.shift_unit_a:
            print(shift.index)
        print("shiftB index")
        for shift in self.shift_unit_b:
            print(shift.index)
        
    def print_state(self):
        """打印MFIU对象的所有成员值"""
        print("==== MFIU状态 ====")
        print(f"width: {self.width}")
        print(f"bit_width: {self.bit_width}")
        print(f"A_bit_mask_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.A_bit_mask_vec]}")
        print(f"B_bit_mask_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.B_bit_mask_vec]}")
        print(f"AB_bit_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.AB_bit_vec]}")
        print(f"AB_prefix_sum: {self.AB_prefix_sum}")
        print(f"ec_idx_vec: {self.ec_idx_vec}")
        self.print_shift()
        print("==================")


    def __call__(self):
        self.AB_bit_vec = [a & b for a, b in zip(self.A_bit_mask_vec, self.B_bit_mask_vec)]
        
        bit_seq = np.array(self.bitvec_to_bitseq(self.AB_bit_vec))
        
        self.AB_prefix_sum = np.cumsum(bit_seq).tolist()
        
        ec_idx_seq = np.where(bit_seq, self.AB_prefix_sum, 0).tolist()
        
        self.ecseq_to_ecvec(ec_idx_seq)

        self.init_shift_unit()
        for shift in self.shift_unit_a:
            shift()
        for shift in self.shift_unit_b:
            shift()

# 新增处理单元(PE)类
class ProcessingElement:
    """矩阵计算处理单元"""
    def __init__(self, pe_id):
        self.pe_id = pe_id
        self.busy = False
        self.result_buffer = None
        self.current_task = None
        logger.debug(f"创建处理单元: PE-{pe_id}")
    
    def compute(self, a_value, b_value):
        """执行标量乘法"""
        self.busy = True
        # 模拟计算延迟
        time.sleep(0.001)
        result = a_value * b_value
        self.result_buffer = result
        self.busy = False
        return result
    
    def is_busy(self):
        return self.busy
    
    def get_result(self):
        result = self.result_buffer
        self.result_buffer = None
        return result

# 内存单元类
class MemoryUnit:
    """模拟硬件内存单元，存储稀疏矩阵数据"""
    def __init__(self, name, capacity=1024):
        self.name = name
        self.capacity = capacity
        self.data = None
        self.busy = False
        logger.debug(f"创建内存单元: {name}, 容量: {capacity}")
    
    def load(self, data):
        """加载数据到内存单元"""
        if isinstance(data, csr_matrix) or isinstance(data, np.ndarray):
            self.data = data
            logger.debug(f"内存单元 {self.name} 加载数据: 形状{data.shape}")
            return True
        else:
            logger.error(f"内存单元 {self.name} 只接受CSR矩阵或numpy数组")
            return False
    
    def read(self):
        """读取数据"""
        self.busy = True
        time.sleep(0.0005)  # 模拟内存访问延迟
        self.busy = False
        return self.data
    
    def is_busy(self):
        return self.busy

# 控制器类
class Controller:
    """控制单元，协调整个系统的运行"""
    def __init__(self, clock_freq=1000):
        self.clock_freq = clock_freq  # 时钟频率，模拟时钟周期
        self.clock_cycle = 0
        logger.info(f"创建控制单元，时钟频率: {clock_freq} Hz")
    
    def tick(self):
        """模拟一个时钟周期"""
        self.clock_cycle += 1
        time.sleep(1.0/self.clock_freq)  # 模拟时钟周期
        return self.clock_cycle

# 数据分发网络
class DistributionNetwork:
    """基于ShiftUnit和MFIU的数据分发网络"""
    def __init__(self, num_pes=4, bit_width=4):
        self.num_pes = num_pes
        self.bit_width = bit_width
        self.processing_elements = [ProcessingElement(i) for i in range(num_pes)]
        self.mfiu = MFIU(num_pes, bit_width)
        self.ready = False
        logger.info(f"创建分发网络，包含 {num_pes} 个处理单元")
    
    def configure(self, mask_A_row, mask_B_col, offset_A_row, offset_B_col, values_A, values_B):
        """配置分发网络"""
        self.mfiu.get_mask_offset_values(mask_A_row, mask_B_col, offset_A_row, offset_B_col, values_A, values_B)
        self.mfiu()
        self.ready = True
        return self.ready
    
    def get_values_for_pe(self):
        """获取每个PE需要的数据对"""
        if not self.ready:
            logger.error("分发网络未配置")
            return []
            
        pairs = []
        for i in range(self.num_pes):
            a_indices = self.mfiu.shift_unit_a[i].index
            b_indices = self.mfiu.shift_unit_b[i].index
            
            # 过滤有效索引（非零）
            valid_indices = [(a_idx, b_idx) for a_idx, b_idx in zip(a_indices, b_indices) if a_idx > 0 and b_idx > 0]
            pairs.append(valid_indices)
            
        return pairs
    
    def compute_batch(self, values_A, values_B):
        """批量执行计算任务"""
        pe_pairs = self.get_values_for_pe()
        results = [[] for _ in range(self.num_pes)]
        
        for pe_idx, pe_data_pairs in enumerate(pe_pairs):
            pe = self.processing_elements[pe_idx]
            for a_idx, b_idx in pe_data_pairs:
                if a_idx < len(values_A) and b_idx < len(values_B):
                    a_val = values_A[a_idx-1]  # 索引从1开始，-1调整为从0开始
                    b_val = values_B[b_idx-1]
                    result = pe.compute(a_val, b_val)
                    results[pe_idx].append(result)
                
        return results

# 主矩阵分发网络模拟器
class MatrixDistributionNetwork:
    """稀疏矩阵分发网络模拟器"""
    def __init__(self, num_pes=4, memory_capacity=10240, clock_freq=1000, bit_width=4):
        self.mat_A_memory = MemoryUnit("MatrixA", memory_capacity)
        self.mat_B_memory = MemoryUnit("MatrixB", memory_capacity)
        self.result_memory = MemoryUnit("Result", memory_capacity)
        self.distribution_network = DistributionNetwork(num_pes, bit_width)
        self.controller = Controller(clock_freq)
        self.bit_width = bit_width
        self.num_pes = num_pes
        logger.info(f"初始化稀疏矩阵分发网络，PE数量: {num_pes}, 内存容量: {memory_capacity}")
    
    def load_matrices(self, A, B):
        """加载输入矩阵"""
        # 确保是CSR格式
        if not isinstance(A, csr_matrix):
            A = csr_matrix(A)
        if not isinstance(B, csr_matrix):
            B = csr_matrix(B)
            
        logger.info(f"加载矩阵 A: {A.shape}, 非零元素: {A.nnz}")
        logger.info(f"加载矩阵 B: {B.shape}, 非零元素: {B.nnz}")
        
        if A.shape[1] != B.shape[0]:
            logger.error(f"矩阵形状不兼容: A({A.shape}) 和 B({B.shape})")
            raise ValueError("矩阵维度不兼容")
            
        self.mat_A_memory.load(A)
        self.mat_B_memory.load(B)
        return True
    
    def get_values_offset_mask(self, matrix):
        """从CSR矩阵提取值、偏移和掩码"""
        values = matrix.data
        col_indices = matrix.indices
        row_ptr = matrix.indptr
        
        # 获取矩阵大小
        num_rows = len(row_ptr) - 1
        num_cols = matrix.shape[1]
        
        # 创建每行的二进制掩码
        masks = []
        
        for row in range(num_rows):
            start = row_ptr[row]
            end = row_ptr[row + 1]
            
            # 确保掩码宽度符合bit_width需求
            if num_cols > self.bit_width:
                # 如果列数超过bit_width，我们需要处理多个块
                logger.warning(f"矩阵列数{num_cols}超过bit_width{self.bit_width}，将分块处理")
                # 这里简化处理，只取前bit_width列
                effective_cols = min(num_cols, self.bit_width)
            else:
                effective_cols = num_cols
            
            row_mask = 0
            for i in range(start, end):
                col = col_indices[i]
                if col < effective_cols:  # 仅处理有效范围内的列
                    row_mask |= (1 << (effective_cols - 1 - col))
            
            masks.append(row_mask)
        
        return values, row_ptr, masks
    
    def multiply(self):
        """执行矩阵乘法运算"""
        A = self.mat_A_memory.read()
        B = self.mat_B_memory.read()
        
        if A is None or B is None:
            logger.error("矩阵数据未加载")
            return None
            
        m, k = A.shape
        k2, n = B.shape
        
        logger.info(f"开始矩阵乘法: A({m}x{k}) * B({k2}x{n})")
        
        # 提取CSR格式的相关信息
        values_A, offset_A, masks_A = self.get_values_offset_mask(A)
        values_B, offset_B, masks_B = self.get_values_offset_mask(B)
        
        # 创建结果矩阵
        result = lil_matrix((m, n))
        
        # 模拟计算过程
        start_time = time.time()
        total_cycles = 0
        
        # 根据分块大小处理矩阵
        # 这里简化模型，仅演示一个块的处理
        block_size = min(self.bit_width, k)
        
        # 为分发网络配置数据
        pe_allocation_count = min(self.num_pes, len(masks_A) * len(masks_B))
        actual_masks_A = masks_A[:min(len(masks_A), self.num_pes)]
        actual_masks_B = masks_B[:min(len(masks_B), self.num_pes)]
        
        self.distribution_network.configure(
            actual_masks_A, 
            actual_masks_B, 
            offset_A[:len(actual_masks_A)], 
            offset_B[:len(actual_masks_B)], 
            values_A, 
            values_B
        )
        
        # 执行计算
        results = self.distribution_network.compute_batch(values_A, values_B)
        
        # 更新计算周期
        total_cycles += self.controller.tick()
        
        # 收集结果到稀疏矩阵
        for pe_idx, pe_results in enumerate(results):
            row_idx = pe_idx // len(actual_masks_B)
            col_idx = pe_idx % len(actual_masks_B)
            
            if row_idx < m and col_idx < n:
                # 将PE的结果累加到对应的位置
                for res in pe_results:
                    result[row_idx, col_idx] += res
        
        elapsed_time = time.time() - start_time
        logger.info(f"矩阵乘法完成，用时: {elapsed_time:.4f}秒，时钟周期: {total_cycles}")
        
        # 转换为CSR格式并存储结果
        result_csr = result.tocsr()
        self.result_memory.load(result_csr)
        
        return result_csr
    
    def get_result(self):
        """获取计算结果"""
        return self.result_memory.read()

# 测试函数
def test_matrix_distribution_network():
    """测试稀疏矩阵分发网络模拟器"""
    # 创建小型稀疏矩阵用于测试
    A = np.array([
        [1, 0, 2, 0],
        [0, 3, 0, 0],
        [0, 0, 4, 0],
        [5, 0, 0, 6]
    ])
    
    B = np.array([
        [1, 0, 0, 2],
        [0, 3, 0, 0],
        [4, 0, 5, 0],
        [0, 6, 0, 0]
    ])
    
    # 转换为CSR格式
    A_csr = csr_matrix(A)
    B_csr = csr_matrix(B)
    
    # 创建网络实例
    network = MatrixDistributionNetwork(num_pes=4, bit_width=4)
    
    # 加载矩阵
    network.load_matrices(A_csr, B_csr)
    
    # 执行矩阵乘法
    logger.info("开始执行矩阵乘法")
    result = network.multiply()
    
    # 验证结果
    expected = A_csr.dot(B_csr)
    is_correct = np.allclose((result - expected).data, 0)
    logger.info(f"结果验证: {'正确' if is_correct else '错误'}")
    logger.info(f"结果矩阵形状: {result.shape}, 非零元素: {result.nnz}")
    
    # 显示结果
    print("\n计算结果矩阵:")
    print(result.toarray())
    print("\n期望结果矩阵:")
    print(expected.toarray())
    
    return result, expected

def test_larger_sparse_matrix():
    """测试更大规模的稀疏矩阵"""
    # 创建稀疏矩阵
    m, k, n = 8, 4, 8  # 保持较小规模以便于测试
    density = 0.3
    
    # 创建随机稀疏矩阵
    A = sparse.random(m, k, density=density, format='csr')
    B = sparse.random(k, n, density=density, format='csr')
    
    # 创建网络实例
    network = MatrixDistributionNetwork(num_pes=4, bit_width=4)
    
    # 加载矩阵
    network.load_matrices(A, B)
    
    # 执行矩阵乘法
    logger.info("开始执行矩阵乘法 - 大型稀疏矩阵")
    result = network.multiply()
    
    # 验证结果
    expected = A.dot(B)
    
    # 显示结果摘要
    print("\n大型稀疏矩阵结果摘要:")
    print(f"结果矩阵形状: {result.shape}, 非零元素: {result.nnz}")
    print(f"期望矩阵形状: {expected.shape}, 非零元素: {expected.nnz}")
    
    return result, expected

if __name__ == "__main__":
    logger.info("启动稀疏矩阵分发网络模拟器")
    
    # 测试小型矩阵
    print("\n=== 测试1: 小型矩阵 ===")
    result1, expected1 = test_matrix_distribution_network()
    
    # 测试较大规模稀疏矩阵
    print("\n=== 测试2: 较大稀疏矩阵 ===")
    result2, expected2 = test_larger_sparse_matrix()
    
    logger.info("模拟器测试完成")