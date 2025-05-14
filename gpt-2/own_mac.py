import numpy as np
import logging
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SparseMatrixPipeline")

class CSCMatrix:
    """压缩稀疏列(CSC)格式的稀疏矩阵表示"""
    def __init__(self, data, row_indices, col_ptrs, shape):
        self.data = data
        self.row_indices = row_indices
        self.col_ptrs = col_ptrs
        self.shape = shape
    
    @classmethod
    def from_dense(cls, dense_matrix):
        rows, cols = dense_matrix.shape
        data = []
        row_indices = []
        col_ptrs = [0]
        
        for j in range(cols):
            for i in range(rows):
                if dense_matrix[i, j] != 0:
                    data.append(dense_matrix[i, j])
                    row_indices.append(i)
            col_ptrs.append(len(data))
        
        return cls(data, row_indices, col_ptrs, dense_matrix.shape)

class ProcessingElement:
    """处理元素(PE)单元"""
    def __init__(self, pe_id):
        """初始化处理元素"""
        self.pe_id = pe_id
        self.accumulators = {}  # 按列索引存储累积值
        
        # 输入阶段数据
        self.a_value = None
        self.b_value = None
        self.output_col = None
        self.current_col = None
        
        # 计算阶段数据
        self.compute_a_value = None
        self.compute_b_value = None
        self.compute_col = None
        
        # 流水线控制
        self.stage_input_valid = False
        self.stage_compute_valid = False
    
    def load_data(self, a_value, b_value, output_col, valid):
        """加载输入数据，确保值被正确捕获"""
        if valid:
            self.a_value = a_value  # A矩阵元素
            self.b_value = b_value  # B矩阵元素  
            self.output_col = output_col  # 输出列索引
            self.current_col = output_col  # 当前处理的列
            self.stage_input_valid = True  # 标记输入有效
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        result = False
        
        # 先处理计算阶段 - 这样可以保证流水线不会堵塞
        if self.stage_compute_valid:
            # 确保当前列的累加器存在 - 使用compute_col而不是current_col
            if self.compute_col not in self.accumulators:
                self.accumulators[self.compute_col] = 0.0
                
            # 准确计算并累加结果 - 使用已暂存的计算阶段变量
            product = self.compute_a_value * self.compute_b_value
            self.accumulators[self.compute_col] += product
            
            # 不再更新过时的累加器字段
            self.stage_compute_valid = False
            result = True
        
        # 然后处理输入阶段 - 确保不会跳过任何数据
        if self.stage_input_valid:
            # 保存当前输入数据的拷贝，防止被覆盖
            self.compute_a_value = self.a_value
            self.compute_b_value = self.b_value
            self.compute_col = self.current_col
            
            self.stage_compute_valid = True
            self.stage_input_valid = False
            result = True
        
        return result
    
    def reset_accumulator(self):
        """重置累加器，返回累积值"""

        self.accumulated_value = 0.0
        self.accumulators = {}  # 清空所有列的累加器

    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        return self.stage_input_valid or self.stage_compute_valid

class DistributionNetwork:
    """分发网络，负责将CSC格式矩阵数据分发到PE"""
    def __init__(self, num_pes):
        self.num_pes = num_pes
        self.a_matrix = None
        
        # PE队列
        self.queues = [deque() for _ in range(num_pes)]
        
        # 流水线阶段
        self.stage = "idle"  # idle, active
        self.stage_load_b_valid = False
        self.stage_fetch_a_valid = False
        self.stage_distribute_valid = False
        
        # 当前处理的数据
        self.current_b_elements = []  # 当前处理的B元素
        self.ready_pairs = []         # 准备好发送到PE的数据对
    
    def load_matrices(self, a_matrix, b_csc):
        """加载矩阵数据"""
        self.a_matrix = a_matrix
        
        # 将CSC格式矩阵按列分配到队列
        for col_idx in range(len(b_csc.col_ptrs) - 1):
            start = b_csc.col_ptrs[col_idx]
            end = b_csc.col_ptrs[col_idx + 1]
            
            # 只处理有非零元素的列
            if end > start:
                # 轮询分配元素到不同的队列，而不是按列索引分配
                for i in range(start, end):
                    b_value = b_csc.data[i]
                    b_row_idx = b_csc.row_indices[i]
                    queue_idx = col_idx % self.num_pes  
                    self.queues[queue_idx].append((b_value, b_row_idx, col_idx))
    
    def clock_cycle(self):
        """
        分发网络实现真正的流水线架构，允许三个阶段并行运行
        """
        # 保存返回的数据
        output_data = []
        
        # 第1阶段：如果分发阶段有效，返回数据并立即清空
        if self.stage_distribute_valid:
            output_data = self.ready_pairs
            self.ready_pairs = []
            self.stage_distribute_valid = False
        
        # 第2阶段：如果获取A元素阶段有效，处理并进入分发阶段
        if self.stage_fetch_a_valid:
            self.ready_pairs = []
            for item in self.current_b_elements:
                if item is not None:
                    b_value, b_row_idx, output_col, pe_idx = item
                    a_value = self.a_matrix[0, b_row_idx]
                    self.ready_pairs.append((a_value, b_value, output_col, pe_idx))
            
            self.stage_fetch_a_valid = False
            self.stage_distribute_valid = True
        
        # 第3阶段：如果当前没有在获取A元素，尝试从队列获取B元素
        if not self.stage_fetch_a_valid:
            self.current_b_elements = []
            has_data = False
            
            for pe_idx in range(self.num_pes):
                if self.queues[pe_idx]:
                    b_value, b_row_idx, output_col = self.queues[pe_idx].popleft()
                    self.current_b_elements.append((b_value, b_row_idx, output_col, pe_idx))
                    has_data = True
                else:
                    self.current_b_elements.append(None)
            
            if has_data:
                self.stage_fetch_a_valid = True
        
        return output_data
    
    def is_busy(self):
        """检查分发网络是否处于忙碌状态"""
        return self.stage_load_b_valid or self.stage_fetch_a_valid or self.stage_distribute_valid
    
    def is_empty(self):
        """检查所有队列是否为空"""
        return all(len(queue) == 0 for queue in self.queues)
    
    def get_queue_lengths(self):
        """获取所有队列的长度"""
        return [len(queue) for queue in self.queues]

class SparseMatrixPipeline:
    """稀疏矩阵乘法硬件模拟器"""
    def __init__(self, num_pes=8):
        self.num_pes = num_pes
        self.clock = 0
        self.pes = [ProcessingElement(i) for i in range(num_pes)]
        self.distribution_network = DistributionNetwork(num_pes)
        
        # 结果收集
        self.result = None
        self.result_ready = False
    
    
    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        for pe in self.pes:
            pe.reset_accumulator()
        
        if hasattr(self, 'B') and self.B is not None:
            self.result = np.zeros((self.A.shape[0], self.B.shape[1]))
        
        self.result_ready = False
    
    def is_active(self):
        """检查模拟器是否仍在活动状态"""
        network_active = not self.distribution_network.is_empty() or self.distribution_network.is_busy()
        pes_active = any(pe.is_busy() for pe in self.pes)
        return network_active or pes_active
    
    def print_stage_valid(self):
        """打印流水线各阶段状态"""
        dn = self.distribution_network
        logger.debug(
            f"Clock {self.clock}: "
            f"Network(load_b={dn.stage_load_b_valid}, "
            f"fetch_a={dn.stage_fetch_a_valid}, "
            f"distribute={dn.stage_distribute_valid}), "
            f"PE_busy={sum(1 for pe in self.pes if pe.is_busy())}"
        )
    
    def clock_cycle(self):
        """执行一个时钟周期的模拟"""
        # 第1阶段: 处理PE的计算
        for pe in self.pes:
            pe.clock_cycle()
        
        # 第2阶段: 运行分发网络一个周期并获取输出数据
        ready_pairs = self.distribution_network.clock_cycle()
        if ready_pairs:
            for a_value, b_value, output_col, pe_idx in ready_pairs:
                self.pes[pe_idx].load_data(a_value, b_value, output_col, True)
        
        # 检查是否完成
        if not self.distribution_network.is_busy() and \
            all(not pe.is_busy() for pe in self.pes) and \
            self.distribution_network.is_empty() and \
            not self.result_ready:
            
            # 收集结果 - 从每个PE的所有列收集累积值
            for pe in self.pes:
                for col, value in pe.accumulators.items():
                    if value != 0:  # 只累加非零值
                        self.result[0, col] += value
            
            self.result_ready = True
            return True
        
        self.clock += 1
        return self.result_ready
    
    def run_simulation(self, max_cycles=10000):
        """运行完整模拟"""
        logger.info("开始稀疏矩阵乘法流水线模拟")
        self.reset()
        
        while self.clock < max_cycles:
            if self.clock % 100 == 0 or self.clock == 0:
                queue_lengths = self.distribution_network.get_queue_lengths()
                busy_pes = sum(1 for pe in self.pes if pe.is_busy())
                logger.info(f"周期 {self.clock}: 最大有效队列长度={max(queue_lengths)}, 活跃PE={busy_pes}") 

            if self.clock_cycle():
                logger.info(f"模拟完成，共用时 {self.clock} 个时钟周期")
                return True
            
        logger.warning(f"模拟达到最大周期限制 ({max_cycles})，尚未完成计算")
        return False
    
    def generate_matrices(self, a_rows=1, a_cols=128, b_rows=128, b_cols=128, sparsity=0.8, seed=42):
        """生成测试矩阵A(稠密)和B(稀疏)"""
        np.random.seed(seed)
        
        # 生成稠密矩阵A
        self.A = np.random.rand(a_rows, a_cols)
        
        # 生成稀疏矩阵B
        dense_B = np.random.rand(b_rows, b_cols)
        mask = np.random.rand(b_rows, b_cols) < sparsity  # 生成掩码，True表示元素将被置0
        self.B_dense = np.where(mask, 0, dense_B)
        
        # 转换为CSC格式
        self.B = CSCMatrix.from_dense(self.B_dense)
        
        # 计算参考结果
        self.reference_result = np.matmul(self.A, self.B_dense)
        
        # 初始化结果矩阵
        self.result = np.zeros((a_rows, b_cols))
        
        logger.info(f"矩阵A形状: {self.A.shape}, 矩阵B形状: {self.B_dense.shape}")
        logger.info(f"矩阵B非零元素: {len(self.B.data)}, 稀疏度: {1 - len(self.B.data)/(b_rows*b_cols):.2f}")
        
        # 将矩阵加载到分发网络
        self.distribution_network.load_matrices(self.A, self.B)
    
    def verify_result(self):
        """验证计算结果与NumPy参考结果比对"""
        if not hasattr(self, 'reference_result'):
            logger.warning("没有参考结果可供验证")
            return False

        error = np.abs(self.result - self.reference_result).max()
        logger.info(f"最大误差: {error}")
        
        if error < 1e-10:
            logger.info("验证成功: 结果与参考值一致")
            return True
        else:
            logger.error("验证失败: 结果与参考值不一致")
            logger.debug(f"模拟结果:\n{self.result}")
            logger.debug(f"参考结果:\n{self.reference_result}")
            return False
            
    def run_large_matrix_multiplication(self, a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, 
                                      sparsity=0.8, block_size=128, seed=42):
        """
        通过分块方式处理大规模矩阵乘法
        
        参数:
            a_rows, a_cols: A矩阵的维度 (1x4096)
            b_rows, b_cols: B矩阵的维度 (4096x4096)
            sparsity: B矩阵的稀疏度
            block_size: 列块大小，应与PE数量匹配
            seed: 随机种子
        """
        logger.info(f"开始大规模矩阵乘法 ({a_rows}x{a_cols}) * ({b_rows}x{b_cols})")
        
        # 生成完整的A矩阵 (1x4096)
        np.random.seed(seed)
        self.A = np.random.rand(a_rows, a_cols)
        
        # 创建完整的参考B矩阵用于验证
        dense_B_full = np.random.rand(b_rows, b_cols)
        mask = np.random.rand(b_rows, b_cols) < sparsity
        self.B_dense_full = np.where(mask, 0, dense_B_full)
        
        # 使用NumPy计算参考结果
        self.reference_result = np.matmul(self.A, self.B_dense_full)
        
        # 计算需要的列块数量
        num_blocks = b_cols // block_size
        if b_cols % block_size != 0:
            num_blocks += 1
        
        # 初始化全局结果矩阵（只创建一次）
        global_result = np.zeros((a_rows, b_cols))
        
        # 迭代处理每个列块
        total_cycles = 0
        for block in range(num_blocks):
            # 计算当前块的列范围
            start_col = block * block_size
            end_col = min((block + 1) * block_size, b_cols)
            actual_block_size = end_col - start_col
            
            logger.info(f"处理列块 {block+1}/{num_blocks}: 列 {start_col} 到 {end_col-1}")
            
            # 提取当前块的B矩阵 (b_rows x actual_block_size)
            B_block = self.B_dense_full[:, start_col:end_col]
            
            # 重置PE状态
            for pe in self.pes:
                pe.reset_accumulator()
            
            # 重置模拟器状态
            self.clock = 0
            self.result_ready = False
            
            # 将B矩阵块转换为CSC格式
            self.B = CSCMatrix.from_dense(B_block)
            
            # 为当前块创建一个块大小的结果矩阵
            self.result = np.zeros((a_rows, actual_block_size))
            
            # 将矩阵加载到分发网络并运行模拟
            self.distribution_network = DistributionNetwork(self.num_pes)
            self.distribution_network.load_matrices(self.A, self.B)
            self.run_simulation()
            
            # 将块结果复制到全局结果矩阵的对应位置
            for col in range(actual_block_size):
                global_col = start_col + col
                global_result[0, global_col] = self.result[0, col]
            
            total_cycles += self.clock
            logger.info(f"列块 {block+1} 完成，用时 {self.clock} 个周期")
        
        # 最后将全局结果设置为最终结果
        self.result = global_result
        
        logger.info(f"大规模矩阵乘法完成，总用时 {total_cycles} 个周期")
        return self.verify_result()

def main():
    """主函数"""
    PE_SIZE = 128
    # 创建模拟器
    simulator = SparseMatrixPipeline(num_pes = PE_SIZE)
    
    # 选择运行标准矩阵乘法或大规模矩阵乘法
    use_large_matrix = True
    
    if use_large_matrix:
        # 运行大规模矩阵乘法 (分块处理)
        simulator.run_large_matrix_multiplication(
            a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, 
            sparsity=0.8, block_size=PE_SIZE)
    else:
        # 运行标准矩阵乘法
        simulator.generate_matrices(a_rows=1, a_cols=128, b_rows=128, b_cols=128, sparsity=0.8)
        simulator.run_simulation()
    
    # 验证结果
    simulator.verify_result()
    
    # 不打印整个结果矩阵，只显示部分样本
    logger.info(f"结果矩阵样本（:\n{simulator.result[0, :10]}")
    logger.info(f"预期结果样本:\n{simulator.reference_result[0, :10]}")
    logger.info(f"最大误差: {np.max(np.abs(simulator.result - simulator.reference_result))}")

if __name__ == "__main__":
    main()