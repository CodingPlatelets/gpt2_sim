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
    """
    处理元素(PE)单元 - 模拟硬件中的基本计算单元
    
    每个PE负责处理矩阵B的一整列，实现两级流水线：
    1. 输入阶段：接收矩阵A和B的元素值
    2. 计算阶段：执行乘法和累加操作
    """
    def __init__(self, pe_id):
        """初始化处理元素"""
        self.pe_id = pe_id
        self.accumulator = 0.0  # 当前列的累加器
        
        # 当前处理的列信息
        self.current_col = None  # 当前处理的全局列索引
        self.remaining_elements = 0  # 当前列剩余元素数
        self.is_empty_column = False  # 标记是否为空列
        
        # 输入阶段数据
        self.a_value = None
        self.b_value = None
        
        # 计算阶段数据
        self.compute_a_value = None
        self.compute_b_value = None
        
        # 流水线控制
        self.stage_input_valid = False
        self.stage_compute_valid = False
        
        # 状态跟踪
        self.column_complete = False       # 当前列是否已完成处理
        self.result_ready = False          # 结果是否已准备好被收集
    
    def assign_column(self, col_idx, element_count):
        """
        分配一个新列给此PE处理
            col_idx: 列索引
            element_count: 该列中非零元素的数量
        Returns:
            (prev_col, result): 如果有上一列的结果则返回，否则返回None
        """
        # 记录之前完成列的结果
        result = None
        prev_col = None
        
        if self.result_ready:
            prev_col = self.current_col
            result = self.accumulator
            self.result_ready = False
        
        # 更新到新列
        self.current_col = col_idx
        self.remaining_elements = element_count
        self.accumulator = 0.0  # 重置累加器
        
        if element_count == 0:
            # 对于空列，直接标记为完成，不需要进行计算
            self.is_empty_column = True
            self.column_complete = True
            self.result_ready = True  # 空列直接标记结果就绪
            logger.debug(f"PE {self.pe_id} 分配到空列 {col_idx}")
        else:
            # 对于非空列，设置为未完成状态，准备计算
            self.is_empty_column = False
            self.column_complete = False
            logger.debug(f"PE {self.pe_id} 分配到列 {col_idx}，包含 {element_count} 个元素")
        
        return prev_col, result
    
    def load_data(self, a_value, b_value, is_end_marker=False):
        """加载输入数据，支持空标记"""
        if is_end_marker:
            # 这是列结束标记
            self.column_complete = True
            self.result_ready = True
            return
        
        if not self.column_complete:
            # 只有在列未完成时才加载新数据
            self.a_value = a_value
            self.b_value = b_value
            self.stage_input_valid = True  # 标记输入阶段有效
    
    def clock_cycle(self):
        """执行一个时钟周期，返回是否有活动"""
        if self.is_empty_column:
            # 空列直接标记为完成，不需要执行计算
            return False
        
        result = False
        
        # 先处理计算阶段(第二级流水线)，避免数据阻塞
        if self.stage_compute_valid:
            # 计算并累加结果
            product = self.compute_a_value * self.compute_b_value
            self.accumulator += product
            
            # 减少剩余元素计数
            self.remaining_elements -= 1
            
            # 检查是否完成当前列
            if self.remaining_elements <= 0:
                self.column_complete = True
                self.result_ready = True
                logger.debug(f"PE {self.pe_id} 完成处理列 {self.current_col}")

            # 清除计算阶段标志，为下一个数据做准备
            self.stage_compute_valid = False
            result = True
        
        # 然后处理输入阶段(第一级流水线)
        if self.stage_input_valid:
            # 将输入数据传递到计算阶段
            self.compute_a_value = self.a_value
            self.compute_b_value = self.b_value
            # 设置计算阶段有效，清除输入阶段标志
            self.stage_compute_valid = True
            self.stage_input_valid = False
            result = True
        
        return result
    
    def is_column_complete(self):
        """检查当前列是否已完成"""
        return self.column_complete
    
    def is_result_ready(self):
        """检查结果是否准备好"""
        return self.result_ready
    
    def get_column_result(self):
        """获取当前列的结果并标记为已收集"""
        if self.result_ready:
            col = self.current_col
            result = self.accumulator
            self.result_ready = False  # 标记结果已被收集
            return col, result
        return None, 0.0
    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        return self.stage_input_valid or self.stage_compute_valid or (not self.column_complete)

class MatrixDataManager:
    """矩阵数据管理器，负责准备和分发列数据"""
    def __init__(self, a_matrix, b_csc, num_pes):
        self.a_matrix = a_matrix
        self.b_csc = b_csc
        self.num_pes = num_pes
        self.total_cols = b_csc.shape[1]
    
    def get_column_element_count(self, col_idx):
        """获取指定列的元素数量"""
        if col_idx >= self.total_cols or col_idx < 0:
            return 0
        return self.b_csc.col_ptrs[col_idx + 1] - self.b_csc.col_ptrs[col_idx]
    
    def get_column_data(self, col_idx):
        """获取指定列的所有数据"""
        if col_idx >= self.total_cols or col_idx < 0:
            return [] # 无效列索引返回空列表
        
        # 获取该列在CSC格式中的起始和结束位置
        start = self.b_csc.col_ptrs[col_idx]
        end = self.b_csc.col_ptrs[col_idx + 1]
        
        # 准备该列所有需要的数据对
        column_data = []
        for i in range(start, end):
            b_value = self.b_csc.data[i]
            b_row_idx = self.b_csc.row_indices[i]
            a_value = self.a_matrix[0, b_row_idx]
            column_data.append((a_value, b_value))
        
        return column_data

class SparseMatrixPipeline:
    """
        稀疏矩阵乘法硬件模拟器 - 主控制类
        
        负责整个模拟系统的协调和执行，包括:
        1. 初始化和分配任务到PE
        2. 执行时钟周期模拟
        3. 收集结果和验证
    """
    def __init__(self, num_pes=128):
        self.num_pes = num_pes                       # PE数量
        self.clock = 0                               # 时钟周期计数
        self.pes = [ProcessingElement(i) for i in range(num_pes)]  # 创建PE数组
        
        # 结果收集
        self.result = None  # 最终计算结果
        
        # 每个PE有一个队列，存储待处理的列数据
        self.pe_data_queues = [deque() for _ in range(num_pes)]
    
    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        self.pes = [ProcessingElement(i) for i in range(self.num_pes)]
        self.pe_data_queues = [deque() for _ in range(self.num_pes)]
    
    def run_dynamic_column_pipeline(self, a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, sparsity=0.8, seed=42):
        """
        动态列加载流水线：利用多个PE并行处理矩阵乘法，每个PE专门负责矩阵B中的一整列，当完成当前列的计算后，立即加载新的列进行处理
        1. 首先是将数据处理，即将矩阵B转换为CSC格式
        2. 将与B对应的A矩阵行数据匹配，一起加载进PE中
        3. 每个block调用PE进行处理，同时在每个block判断列数据是否执行完毕
        
        """
        logger.info(f"开始动态列加载流水线矩阵乘法 ({a_rows}x{a_cols}) * ({b_rows}x{b_cols})")
        
        # 生成测试矩阵
        np.random.seed(seed)
        self.A = np.random.rand(a_rows, a_cols)
        
        dense_B_full = np.random.rand(b_rows, b_cols)
        mask = np.random.rand(b_rows, b_cols) < sparsity
        self.B_dense_full = np.where(mask, 0, dense_B_full)
        
        # 计算参考结果
        self.reference_result = np.matmul(self.A, self.B_dense_full)
        
        # 初始化结果矩阵
        self.result = np.zeros((a_rows, b_cols))
        
        # 重置状态
        self.reset()
        
        # 将B矩阵转换为CSC格式
        B_csc = CSCMatrix.from_dense(self.B_dense_full)
        
        # 创建矩阵数据管理器
        data_manager = MatrixDataManager(self.A, B_csc, self.num_pes)
        
        # 记录下一个要加载的列
        next_col_to_load = self.num_pes
        
        # 记录已完成的列
        completed_columns = set()
        
        # 为所有PE分配初始列 (0-127)
        for pe_idx in range(self.num_pes):
            # 获取此PE负责的初始列数据
            col_idx = pe_idx
            element_count = data_manager.get_column_element_count(col_idx)
            column_data = data_manager.get_column_data(col_idx)
            
            # 将列数据分配列给PE
            prev_col, prev_result = self.pes[pe_idx].assign_column(col_idx, element_count)
            
            # 将列数据挨个加入到PE的输入队列中
            for a_value, b_value in column_data:
                self.pe_data_queues[pe_idx].append((a_value, b_value, False))
        
        # 主循环
        max_cycles = 200000
        last_status_time = 0
        status_interval = 1000
        
        while self.clock < max_cycles:
            # 检查是否所有列都已处理完
            if len(completed_columns) == b_cols:
                break
            
            # 1. 输入阶段：为每个PE提供数据
            for pe_idx in range(self.num_pes):
                pe = self.pes[pe_idx]
                
                # 检查PE是否已完成列且结果就绪
                if pe.is_result_ready():
                    # 获取结果
                    col, result = pe.get_column_result()
                    
                    if col is not None:
                        # 存储结果
                        self.result[0, col] = result
                        completed_columns.add(col)
                        # logger.info(f"PE {pe_idx} 完成列 {col}，结果为 {result:.6f}")
                        
                        # 为此PE分配新列(哪个pe结束了就继续分配下一个待分配的列)
                        new_col = next_col_to_load
                        next_col_to_load += 1
                        if new_col < b_cols:
                            # 获取新列数据
                            element_count = data_manager.get_column_element_count(new_col)
                            column_data = data_manager.get_column_data(new_col)
                            
                            # 分配新列给PE
                            pe.assign_column(new_col, element_count)
                            
                            # 将新列数据加入PE的队列
                            self.pe_data_queues[pe_idx].clear()  # 清空旧数据
                            for a_value, b_value in column_data:
                                self.pe_data_queues[pe_idx].append((a_value, b_value, False))
                            
                            logger.debug(f"为PE {pe_idx} 分配新列 {new_col}，有 {element_count} 个元素")
                
                # 如果PE需要数据且队列中有数据
                if not pe.is_column_complete() and self.pe_data_queues[pe_idx]:
                    if not pe.stage_input_valid:  # 确保PE准备好接收新数据
                        a_value, b_value, is_end_marker = self.pe_data_queues[pe_idx].popleft()
                        pe.load_data(a_value, b_value, is_end_marker)
            
            # 2. 计算阶段：所有PE执行计算
            for pe_idx in range(self.num_pes):
                self.pes[pe_idx].clock_cycle()
            
            # 定期打印状态
            if self.clock - last_status_time >= status_interval:
                last_status_time = self.clock
                completion_pct = len(completed_columns) * 100.0 / b_cols
                busy_pes = sum(1 for pe in self.pes if pe.is_busy())
                
                logger.info(f"周期 {self.clock}: 已完成 {len(completed_columns)}/{b_cols} 列 ({completion_pct:.1f}%), "
                           f"活跃PE={busy_pes}, 下一列={next_col_to_load}")
            
            self.clock += 1
        
        
        
        return self.verify_result()
    
    def verify_result(self):
        """验证计算结果与NumPy参考结果比对"""
        # 验证结果
        logger.info(f"动态列加载流水线矩阵乘法完成，总用时 {self.clock} 个周期")
        sample_size = 20
        logger.info(f"结果矩阵样本(前{sample_size}列):")
        logger.info(f"{self.result[0, :sample_size]}")
        logger.info(f"参考矩阵样本(前{sample_size}列):")
        logger.info(f"{self.reference_result[0, :sample_size]}")

        if not hasattr(self, 'reference_result'):
            logger.warning("没有参考结果可供验证")
            return False

        error = np.abs(self.result - self.reference_result).max()
        logger.info(f"最大误差: {error}")
        
        # 考虑浮点数计算误差，使用更宽松的容差
        if error < 1e-8:
            logger.info("验证成功: 结果与参考值一致")
            return True
        else:
            logger.error("验证失败: 结果与参考值不一致")
            logger.info(f"结果矩阵样本:\n{self.result[0, :10]}")
            logger.info(f"预期结果样本:\n{self.reference_result[0, :10]}")
            return False

def main():
    """主函数"""
    PE_SIZE = 128
    # 创建模拟器
    simulator = SparseMatrixPipeline(num_pes=PE_SIZE)
    # 稀疏度0.9 8192 快5000个clock 4096 快1000
    # 稀疏度0.8 8192 快10000个clock 4096 快2000
    # 使用动态列加载流水线矩阵乘法    409
    simulator.run_dynamic_column_pipeline(
        a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, 
        sparsity=0.8)

if __name__ == "__main__":
    main()