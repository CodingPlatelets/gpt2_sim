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
        self.pe_id = pe_id
        self.accumulated_value = 0.0
        self.output_col = None
        
        # 流水线阶段状态
        self.stage_input_valid = False
        self.stage_compute_valid = False
        
        # 输入数据
        self.a_value = None
        self.b_value = None
    
    def load_data(self, a_value, b_value, output_col, valid):
        """加载输入数据"""
        self.a_value = a_value
        self.b_value = b_value
        self.output_col = output_col
        self.stage_input_valid = valid
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        # 阶段2：计算
        if self.stage_compute_valid:
            self.accumulated_value += self.a_value * self.b_value
            self.stage_compute_valid = False
            return True
        
        # 阶段1：输入到计算
        if self.stage_input_valid:
            self.stage_compute_valid = True
            self.stage_input_valid = False
        
        return False
    
    def reset_accumulator(self):
        """重置累加器，返回累积值"""
        result = self.accumulated_value
        self.accumulated_value = 0.0
        return result
    
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
        分发网络主要分为三个阶段
        1. 加载B元素阶段：从队列中获取B元素
        2. 获取A元素阶段：根据取得的B元素，从A矩阵中获取与之对应的A元素
        3. 分发阶段：将A和B元素组合成数据对，准备发送到PE
        """
        # 首先，检查是否有数据要分发
        output_data = []
        
        # 如果分发阶段有效，返回准备好的数据
        if self.stage_distribute_valid:
            output_data = self.ready_pairs
            self.ready_pairs = []
            self.stage_distribute_valid = False
            return output_data
        
        # 如果获取A元素阶段有效，处理并进入分发阶段
        if self.stage_fetch_a_valid:
            self.ready_pairs = []
            for item in self.current_b_elements:
                if item is not None:
                    b_value, b_row_idx, output_col, pe_idx = item
                    a_value = self.a_matrix[0, b_row_idx]
                    self.ready_pairs.append((a_value, b_value, output_col, pe_idx))
            
            self.stage_fetch_a_valid = False
            self.stage_distribute_valid = True
            return []
        
        # 如果加载B元素阶段有效，从队列获取元素并进入获取A阶段
        if self.stage_load_b_valid:
            self.current_b_elements = []
            for pe_idx in range(self.num_pes):
                if self.queues[pe_idx]:
                    b_value, b_row_idx, output_col = self.queues[pe_idx].popleft()
                    self.current_b_elements.append((b_value, b_row_idx, output_col, pe_idx))
                else:
                    self.current_b_elements.append(None)
            
            self.stage_load_b_valid = False
            self.stage_fetch_a_valid = True
            return []
        
        # 如果所有阶段都不活跃，且有数据待处理，启动加载B阶段
        if not (self.stage_load_b_valid or self.stage_fetch_a_valid or self.stage_distribute_valid):
            has_data = any(len(q) > 0 for q in self.queues)
            if has_data:
                self.stage_load_b_valid = True
        
        return []
    
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
        
        if self.B is not None:
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
        """
        整个稀疏矩阵乘法分为：
        
        """
        # 第1阶段: 处理PE的计算
        for pe in self.pes:
            pe.clock_cycle()
        
        # 第2阶段: 运行分发网络一个周期并获取输出数据
        ready_pairs = self.distribution_network.clock_cycle()
        
        # 第3阶段: 将数据加载到PE
        if ready_pairs:
            for a_value, b_value, output_col, pe_idx in ready_pairs:
                self.pes[pe_idx].load_data(a_value, b_value, output_col, True)
        
        # 检查是否完成
        if not self.distribution_network.is_busy() and \
            all(not pe.is_busy() for pe in self.pes) and \
            self.distribution_network.is_empty() and \
            not self.result_ready:
            
            # 收集结果
            for pe in self.pes:
                if pe.output_col is not None and pe.accumulated_value != 0:
                    self.result[0, pe.output_col] += pe.accumulated_value
            
            self.result_ready = True
            return True
        
        self.clock += 1
        return self.result_ready
    
    def run_simulation(self, max_cycles=10000):
        """运行完整模拟"""
        logger.info("开始稀疏矩阵乘法流水线模拟")
        self.reset()
        
        while self.clock < max_cycles:
            # 每隔100个周期打印一次状态
            if self.clock % 100 == 0 or self.clock == 0:
                queue_lengths = self.distribution_network.get_queue_lengths()
                busy_pes = sum(1 for pe in self.pes if pe.is_busy())
                logger.info(f"周期 {self.clock}: 队列长度={queue_lengths}, 活跃PE={busy_pes}") 

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

def main():
    """主函数"""
    # 创建模拟器
    simulator = SparseMatrixPipeline(num_pes = 128)

    # 加载矩阵
    simulator.generate_matrices(a_rows=1, a_cols=128, b_rows=128, b_cols=128, sparsity=0.8)
        
    # 运行模拟
    simulator.run_simulation()
    
    # 验证结果
    simulator.verify_result()
    
    # 打印结果和性能统计
    logger.info(f"结果矩阵:\n{simulator.result}")
    logger.info(f"总时钟周期: {simulator.clock}")

if __name__ == "__main__":
    main()


# import numpy as np
# import logging
# from collections import deque

# # 配置日志
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("SparseMatrixPipeline")

# class CSCMatrix:
#     """压缩稀疏列(CSC)格式的稀疏矩阵表示"""
#     def __init__(self, data, row_indices, col_ptrs, shape):
#         self.data = data
#         self.row_indices = row_indices
#         self.col_ptrs = col_ptrs
#         self.shape = shape
    
#     @classmethod
#     def from_dense(cls, dense_matrix):
#         rows, cols = dense_matrix.shape
#         data = []
#         row_indices = []
#         col_ptrs = [0]
        
#         for j in range(cols):
#             for i in range(rows):
#                 if dense_matrix[i, j] != 0:
#                     data.append(dense_matrix[i, j])
#                     row_indices.append(i)
#             col_ptrs.append(len(data))
        
#         return cls(data, row_indices, col_ptrs, dense_matrix.shape)

# class ProcessingElement:
#     """处理元素(PE)单元"""
#     def __init__(self, pe_id):
#         self.pe_id = pe_id
#         self.accumulated_value = 0.0
#         self.output_col = None
        
#         # 流水线阶段状态
#         self.stage_input_valid = False
#         self.stage_compute_valid = False
        
#         # 输入数据
#         self.a_value = None
#         self.b_value = None
    
#     def load_data(self, a_value, b_value, output_col, valid):
#         """加载输入数据"""
#         self.a_value = a_value
#         self.b_value = b_value
#         self.output_col = output_col
#         self.stage_input_valid = valid
    
#     def clock_cycle(self):
#         """执行一个时钟周期"""
#         # 阶段2：计算阶段 执行实际的乘法和累加操作，将结果存储到累加器中 模拟ALU执行计算的过程
#         if self.stage_compute_valid:
#             self.accumulated_value += self.a_value * self.b_value
#             self.stage_compute_valid = False
#             return True
        
#         # 阶段1：输入阶段 接收来自分发网络的数据（A矩阵元素和B矩阵元素）模拟数据加载到PE内部寄存器的过程
#         if self.stage_input_valid:
#             self.stage_compute_valid = True
#             self.stage_input_valid = False
        
#         return False
    
#     def reset_accumulator(self):
#         """重置累加器，返回累积值"""
#         result = self.accumulated_value
#         self.accumulated_value = 0.0
#         return result
    
#     def is_busy(self):
#         """检查PE是否正在处理数据"""
#         return self.stage_input_valid or self.stage_compute_valid

# class DistributionNetwork:
#     """分发网络，负责将CSC格式矩阵数据分发到PE"""
#     def __init__(self, num_pes):
#         self.num_pes = num_pes
#         self.a_matrix = None
        
#         # PE队列
#         self.queues = [deque() for _ in range(num_pes)]
        
#         # 流水线阶段
#         self.stage = "idle"  # idle, active
#         self.stage_load_b_valid = False
#         self.stage_fetch_a_valid = False
#         self.stage_distribute_valid = False
        
#         # 当前处理的数据
#         self.current_b_elements = []  # 当前处理的B元素
#         self.ready_pairs = []         # 准备好发送到PE的数据对
    
#     def load_matrices(self, a_matrix, b_csc):
#         """加载矩阵数据"""
#         self.a_matrix = a_matrix
        
#         # 将CSC格式矩阵按列分配到队列
#         for col_idx in range(len(b_csc.col_ptrs) - 1):
#             start = b_csc.col_ptrs[col_idx]
#             end = b_csc.col_ptrs[col_idx + 1]
            
#             # 只处理有非零元素的列
#             if end > start:
#                 # 轮询分配元素到不同的队列，而不是按列索引分配
#                 for i in range(start, end):
#                     b_value = b_csc.data[i]
#                     b_row_idx = b_csc.row_indices[i]
#                     queue_idx = col_idx % self.num_pes  
#                     self.queues[queue_idx].append((b_value, b_row_idx, col_idx))
    
#     def clock_cycle(self):

#         """
#         分发网络主要分为三个阶段
#         1. stage_load_b_valid     加载B元素阶段： 从队列中获取B元素
#         2. stage_fetch_a_valid    获取A元素阶段： 根据取得的B元素，从A矩阵中获取与之对应的A元素
#         3. stage_distribute_valid 分发阶段：     将A和B元素组合成数据对，准备发送到PE
#         """

#         #  # 如果首次调用且没有任何状态激活，但队列中有数据，则激活第一个阶段
#         # if not (self.stage_load_b_valid or self.stage_fetch_a_valid or self.stage_distribute_valid):
#         #     has_data = any(len(q) > 0 for q in self.queues)
#         #     if has_data:
#         #         self.stage_load_b_valid = True
#         #         self.stage = "active"

#         # 首先，检查是否有数据要分发
#         output_data = []
        
#         # 如果分发阶段有效，返回准备好的数据
#         if self.stage_distribute_valid:
#             output_data = self.ready_pairs
#             self.ready_pairs = []
#             self.stage_distribute_valid = False
#             return output_data
        
#         # 如果获取A元素阶段有效，处理并进入分发阶段
#         if self.stage_fetch_a_valid:
#             self.ready_pairs = []
#             for item in self.current_b_elements:
#                 if item is not None:
#                     b_value, b_row_idx, output_col, pe_idx = item
#                     a_value = self.a_matrix[0, b_row_idx]
#                     self.ready_pairs.append((a_value, b_value, output_col, pe_idx))
            
#             self.stage_fetch_a_valid = False
#             self.stage_distribute_valid = True
#             return []
        
#         # 如果加载B元素阶段有效，从队列获取元素并进入获取A阶段
#         if self.stage_load_b_valid:
#             self.current_b_elements = []
#             for pe_idx in range(self.num_pes):
#                 if self.queues[pe_idx]:
#                     b_value, b_row_idx, output_col = self.queues[pe_idx].popleft()
#                     self.current_b_elements.append((b_value, b_row_idx, output_col, pe_idx))
#                 else:
#                     self.current_b_elements.append(None)
            
#             self.stage_load_b_valid = False
#             self.stage_fetch_a_valid = True
#             return []
        
#         # 如果所有阶段都不活跃，且有数据待处理，启动加载B阶段
#         if not (self.stage_load_b_valid or self.stage_fetch_a_valid or self.stage_distribute_valid):
#             has_data = any(len(q) > 0 for q in self.queues)
#             if has_data:
#                 self.stage_load_b_valid = True
        
#         return []
    
#     def is_busy(self):
#         """检查分发网络是否处于忙碌状态"""
#         return self.stage_load_b_valid or self.stage_fetch_a_valid or self.stage_distribute_valid
    
#     def is_empty(self):
#         """检查所有队列是否为空"""
#         return all(len(queue) == 0 for queue in self.queues)
    
#     def get_queue_lengths(self):
#         """获取所有队列的长度"""
#         return [len(queue) for queue in self.queues]

# class SparseMatrixPipeline:
#     """稀疏矩阵乘法硬件模拟器"""
#     def __init__(self, num_pes=8):
#         self.num_pes = num_pes
#         self.clock = 0
#         self.pes = [ProcessingElement(i) for i in range(num_pes)]
#         self.distribution_network = DistributionNetwork(num_pes)
        
#         # 结果收集
#         self.result = None
#         self.result_ready = False
    
#     def load_matrices(self, a_matrix, b_csc):
#         """加载矩阵数据"""
#         self.A = a_matrix
#         self.B = b_csc
        
#         # 初始化结果矩阵
#         self.result = np.zeros((self.A.shape[0], self.B.shape[1]))
        
#         # 将矩阵加载到分发网络
#         self.distribution_network.load_matrices(self.A, self.B)
    
#     def load_csc_matrix(self, data, row_indices, col_ptrs, a_matrix, shape):
#         """直接加载CSC格式的矩阵数据"""
#         self.B = CSCMatrix(data, row_indices, col_ptrs, shape)
#         self.A = a_matrix
        
#         # 初始化结果矩阵
#         self.result = np.zeros((self.A.shape[0], shape[1]))
        
#         # 计算参考结果
#         b_dense = np.zeros(shape)
#         for col in range(shape[1]):
#             start = col_ptrs[col]
#             end = col_ptrs[col + 1] if col + 1 < len(col_ptrs) else len(data)
#             for i in range(start, end):
#                 row = row_indices[i]
#                 val = data[i]
#                 b_dense[row, col] = val
        
#         self.B_dense = b_dense
#         self.reference_result = np.matmul(self.A, b_dense)
        
#         # 将矩阵加载到分发网络
#         self.distribution_network.load_matrices(self.A, self.B)
    
#     def reset(self):
#         """重置模拟器状态"""
#         self.clock = 0
#         for pe in self.pes:
#             pe.reset_accumulator()
        
#         if self.B is not None:
#             self.result = np.zeros((self.A.shape[0], self.B.shape[1]))
        
#         self.result_ready = False
    
#     def is_active(self):
#         """检查模拟器是否仍在活动状态"""
#         network_active = not self.distribution_network.is_empty() or self.distribution_network.is_busy()
#         pes_active = any(pe.is_busy() for pe in self.pes)
#         return network_active or pes_active
    
#     def print_stage_valid(self):
#         """打印流水线各阶段状态"""
#         dn = self.distribution_network
#         logger.debug(
#             f"Clock {self.clock}: "
#             f"Network(load_b={dn.stage_load_b_valid}, "
#             f"fetch_a={dn.stage_fetch_a_valid}, "
#             f"distribute={dn.stage_distribute_valid}), "
#             f"PE_busy={sum(1 for pe in self.pes if pe.is_busy())}"
#         )
    
#     def clock_cycle(self):
#         """
#         整个稀疏矩阵乘法分为：
        
#         """
        
#         # 第1阶段: 处理PE的计算 更新所有PE的状态（执行计算）
#         for pe in self.pes:
#             pe.clock_cycle()
        
#         # 第2阶段: 运行分发网络一个clock并获取输出数据  将输出数据加载到对应PE
#         ready_pairs = self.distribution_network.clock_cycle()
#         if ready_pairs:
#             for a_value, b_value, output_col, pe_idx in ready_pairs:
#                 self.pes[pe_idx].load_data(a_value, b_value, output_col, True)
        
#         # 检查是否完成
#         if not self.distribution_network.is_busy() and \
#             all(not pe.is_busy() for pe in self.pes) and \
#             self.distribution_network.is_empty() and \
#             not self.result_ready:
            
#             # 收集结果
#             for pe in self.pes:
#                 if pe.output_col is not None and pe.accumulated_value != 0:
#                     self.result[0, pe.output_col] += pe.accumulated_value
            
#             self.result_ready = True
#             return True
        
#         self.clock += 1
#         return self.result_ready
    
#     def is_active(self):
#         """检查模拟器是否仍在活动状态"""
#         # 检查分发网络是否还有数据需要处理
#         network_active = not self.distribution_network.is_empty()
        
#         # 检查分发网络的任何阶段是否处于活跃状态
#         network_busy = self.distribution_network.is_busy()
        
#         # 检查任何PE是否仍在处理数据
#         pes_active = any(pe.is_busy() for pe in self.pes)
        
#         # 流水线活跃条件：网络有数据或网络忙碌或PE活跃
#         return network_active or network_busy or pes_active
    
#     def run_simulation(self, max_cycles=10000):
#         """运行完整模拟"""
#         logger.info("开始稀疏矩阵乘法流水线模拟")
#         self.reset()
        
#         # 首次运行一个周期，确保初始化阶段正常启动
#         self.clock_cycle()
        
#         # 类似macLine中的实现方式，检查流水线是否活跃
#         # 流水线活跃条件：1. 分发网络仍在处理数据 或 2. PE仍在计算 或 3. 结果未准备好
#         while (self.is_active() or not self.result_ready) and self.clock < max_cycles:
#             # 每隔100个周期打印一次状态
#             # if self.clock % 100 == 0:
#             queue_lengths = self.distribution_network.get_queue_lengths()
#             busy_pes = sum(1 for pe in self.pes if pe.is_busy())
#             # logger.info(f"周期 {self.clock}: 队列长度={queue_lengths}, 活跃PE={busy_pes}")
#             logger.info(f"周期 {self.clock}: 队列最大长度={max(queue_lengths)}, 活跃PE={busy_pes}")

#             # 添加更详细的流水线状态信息
#             # dn = self.distribution_network
#             # logger.debug(
#             #     f"网络状态: load_b={dn.stage_load_b_valid}, "
#             #     f"fetch_a={dn.stage_fetch_a_valid}, "
#             #     f"distribute={dn.stage_distribute_valid}"
#             # )

#             # 执行一个时钟周期
#             if self.clock_cycle():
#                 logger.info(f"模拟完成，共用时 {self.clock} 个时钟周期")
#                 return True
            
            
        
#         # 检查退出条件
#         if self.clock >= max_cycles:
#             logger.warning(f"模拟达到最大周期限制 ({max_cycles})，尚未完成计算")
#             return False
#         else:
#             logger.info(f"模拟完成，共用时 {self.clock} 个时钟周期")
#             return True
    
#     def generate_matrices(self, a_rows=1, a_cols=128, b_rows=128, b_cols=128, sparsity=0.8, seed=42):
#         """生成测试矩阵A(稠密)和B(稀疏)"""
#         np.random.seed(seed)
        
#         # 生成稠密矩阵A
#         self.A = np.random.rand(a_rows, a_cols)
        
#         # 生成稀疏矩阵B
#         dense_B = np.random.rand(b_rows, b_cols)
#         mask = np.random.rand(b_rows, b_cols) < sparsity  # 生成掩码，True表示元素将被置0
#         self.B_dense = np.where(mask, 0, dense_B)
        
#         # 转换为CSC格式
#         self.B = CSCMatrix.from_dense(self.B_dense)
        
#         # 计算参考结果
#         self.reference_result = np.matmul(self.A, self.B_dense)
        
#         # 初始化结果矩阵
#         self.result = np.zeros((a_rows, b_cols))
        
#         logger.info(f"矩阵A形状: {self.A.shape}, 矩阵B形状: {self.B_dense.shape}")
#         logger.info(f"矩阵B非零元素: {len(self.B.data)}, 稀疏度: {1 - len(self.B.data)/(b_rows*b_cols):.2f}")
        
#         # 将矩阵加载到分发网络
#         self.distribution_network.load_matrices(self.A, self.B)
    
#     def verify_result(self):
#         """验证计算结果与NumPy参考结果比对"""
#         if not hasattr(self, 'reference_result'):
#             logger.warning("没有参考结果可供验证")
#             return False

#         error = np.abs(self.result - self.reference_result).max()
#         logger.info(f"最大误差: {error}")
        
#         if error < 1e-10:
#             logger.info("验证成功: 结果与参考值一致")
#             return True
#         else:
#             logger.error("验证失败: 结果与参考值不一致")
#             logger.debug(f"模拟结果:\n{self.result}")
#             logger.debug(f"参考结果:\n{self.reference_result}")
#             return False

# def main():
#     """主函数"""
#     # 创建模拟器
#     simulator = SparseMatrixPipeline(num_pes = 128)

#     # 加载矩阵
#     simulator.generate_matrices(a_rows=1, a_cols=128, b_rows=128, b_cols=128, sparsity=0.8)
        
#     # 运行模拟
#     simulator.run_simulation()
    
#     # 验证结果
#     simulator.verify_result()
    
#     # 打印结果和性能统计
#     logger.info(f"结果矩阵:\n{simulator.result}")
#     logger.info(f"总时钟周期: {simulator.clock}")

# if __name__ == "__main__":
#     main()