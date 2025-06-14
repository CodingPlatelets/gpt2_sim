import numpy as np
import logging
from collections import deque
import struct
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SparseMatrixPipeline")

def fp32_to_bf16(fp32_value):
    """将FP32值转换为BF16格式"""
    pipeline = FP32toBF16Pipeline()
    pipeline.run_simulation([(fp32_value, True)], print_states=False)
    return pipeline.outputs[0]["bf16"] if pipeline.outputs else 0

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式"""
    fp32_bits = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

def bf16_to_float_block(bf16_block):
    """将一块BF16值转换为FP32格式"""
    float_block = []
    for bf16 in bf16_block:
        float_block.append(bf16_to_float(bf16))
    return float_block

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

class MACUnit:
    """
    处理元素(PE)单元 - 使用BF16格式进行计算
    
    三级流水线：
    1. 输入阶段：接收矩阵A和B的BF16元素值
    2. 乘法阶段：使用BF16乘法流水线
    3. 累加阶段：使用BF16加法流水线
    """
    def __init__(self, pe_id):
        """初始化处理元素"""
        self.pe_id = pe_id
        
        # BF16流水线
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.add_pipeline = BF16AddPipeline()
        
        # 当前处理的列信息
        self.current_col = None
        self.remaining_elements = 0
        self.is_empty_column = False
        
        # BF16格式的累加器初值
        self.accumulator = fp32_to_bf16(0.0)
        
        # 输入阶段数据
        self.input_a_value = None
        self.input_b_value = None
        
        # 乘法阶段数据
        self.multiply_a_value = None
        self.multiply_b_value = None
        self.multiply_result = None
        
        # 加法阶段数据
        self.add_pending = False
        
        # 流水线控制
        self.stage_input_valid = False
        self.stage_multiply_valid = False
        self.stage_add_valid = False
        
        # 状态跟踪
        self.column_complete = False
        self.result_ready = False
    
    def assign_column(self, col_idx, element_count):
        """分配一个新列给此PE处理"""
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
        self.accumulator = fp32_to_bf16(0.0)  # 重置BF16累加器
        
        # 重置流水线状态
        self.stage_input_valid = False
        self.stage_multiply_valid = False
        self.stage_add_valid = False
        self.add_pending = False
        
        if element_count == 0:
            # 对于空列，直接标记为完成
            self.is_empty_column = True
            self.column_complete = True
            self.result_ready = True
            logger.debug(f"PE {self.pe_id} 分配到空列 {col_idx}")
        else:
            self.is_empty_column = False
            self.column_complete = False
            logger.debug(f"PE {self.pe_id} 分配到列 {col_idx}，包含 {element_count} 个元素")
        
        return prev_col, result
    
    def load_data(self, a_value, b_value, is_end_marker=False):
        """加载BF16格式的输入数据"""
        if is_end_marker:
            self.column_complete = True
            self.result_ready = True
            return
        
        if not self.column_complete and not self.stage_input_valid:
            self.input_a_value = a_value  # 已经是BF16格式
            self.input_b_value = b_value  # 已经是BF16格式
            self.stage_input_valid = True
            return True
        return False
    
    def clock_cycle(self):
        """执行一个时钟周期，返回PE是否活跃"""
        if self.is_empty_column:
            return False
        
        active = False
        
        # 阶段3: 累加结果 (如果有等待的加法操作)
        if self.stage_add_valid:
            add_result = self.add_pipeline.clock_cycle(self.accumulator, self.multiply_result, True)
            if add_result["valid_output"]:
                self.accumulator = self.add_pipeline.outputs.pop(0)
                
                # 减少剩余元素计数
                self.remaining_elements -= 1
                
                # 检查是否完成当前列
                if self.remaining_elements <= 0:
                    self.column_complete = True
                    self.result_ready = True
                    logger.debug(f"PE {self.pe_id} 完成处理列 {self.current_col}")
                
                self.stage_add_valid = False
                active = True
        
        # 阶段2: 执行乘法 (如果有有效的乘法数据)
        if self.stage_multiply_valid:
            multiply_result = self.multiply_pipeline.clock_cycle(self.multiply_a_value, self.multiply_b_value, True)
            if multiply_result["valid_output"]:
                self.multiply_result = self.multiply_pipeline.outputs.pop(0)
                self.stage_add_valid = True
                self.stage_multiply_valid = False
                active = True
        
        # 阶段1: 加载输入到乘法阶段 (如果有有效的输入数据)
        if self.stage_input_valid:
            self.multiply_a_value = self.input_a_value
            self.multiply_b_value = self.input_b_value
            self.stage_multiply_valid = True
            self.stage_input_valid = False
            active = True
        
        return active
    
    def is_ready_for_input(self):
        """检查PE是否准备好接收新的输入数据"""
        return not self.column_complete and not self.stage_input_valid
    
    def is_result_ready(self):
        """检查结果是否准备好"""
        return self.result_ready
    
    def get_column_result(self):
        """获取当前列的BF16结果并标记为已收集"""
        if self.result_ready:
            col = self.current_col
            result = self.accumulator  # 返回BF16格式的结果
            self.result_ready = False
            return col, result
        return None, None
    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        return (self.stage_input_valid or 
                self.stage_multiply_valid or 
                self.stage_add_valid or 
                (not self.column_complete))

class MatrixDataManager:
    """矩阵数据管理器，负责准备和分发BF16格式的列数据"""
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
        """获取指定列的所有数据，并转换为BF16格式"""
        if col_idx >= self.total_cols or col_idx < 0:
            return []
        
        # 获取该列在CSC格式中的起始和结束位置
        start = self.b_csc.col_ptrs[col_idx]
        end = self.b_csc.col_ptrs[col_idx + 1]
        
        # 准备该列所有需要的数据对，并转换为BF16格式
        column_data = []
        for i in range(start, end):
            b_value = self.b_csc.data[i]
            b_row_idx = self.b_csc.row_indices[i]
            a_value = self.a_matrix[0, b_row_idx]
            
            # 转换为BF16格式
            a_bf16 = fp32_to_bf16(a_value)
            b_bf16 = fp32_to_bf16(b_value)
            
            column_data.append((a_bf16, b_bf16))
        
        return column_data

class SparseMatrixPipeline:
    """稀疏矩阵乘法硬件模拟器 - 支持BF16计算"""
    def __init__(self, num_pes=128):
        self.num_pes = num_pes
        self.clock = 0
        self.pes = [MACUnit(i) for i in range(num_pes)]
        
        # 结果收集
        self.result = None
        
        # 每个PE的数据队列
        self.pe_data_queues = [deque() for _ in range(num_pes)]
        
        # 统计信息
        self.processed_elements = 0
        self.debug_mode = False
    
    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        self.pes = [MACUnit(i) for i in range(self.num_pes)]
        self.pe_data_queues = [deque() for _ in range(self.num_pes)]
        self.processed_elements = 0
    
    def run_dynamic_column_pipeline(self, a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, sparsity=0.8, seed=42):
        """使用BF16格式的动态列加载流水线矩阵乘法"""
        logger.info(f"开始BF16动态列加载流水线矩阵乘法 ({a_rows}x{a_cols}) * ({b_rows}x{b_cols})")
        
        # 生成测试矩阵
        np.random.seed(seed)
        self.A = np.random.rand(a_rows, a_cols).astype(np.float32)
        
        dense_B_full = np.random.rand(b_rows, b_cols).astype(np.float32)
        mask = np.random.rand(b_rows, b_cols) < sparsity
        self.B_dense_full = np.where(mask, 0, dense_B_full)
        
        # 计算参考结果
        self.reference_result = np.matmul(self.A, self.B_dense_full)
        
        # 初始化结果矩阵
        self.result = np.zeros((a_rows, b_cols), dtype=np.float32)
        
        # 重置状态
        self.reset()
        
        # 将B矩阵转换为CSC格式
        B_csc = CSCMatrix.from_dense(self.B_dense_full)
        
        # 统计非零元素数量
        nnz = len(B_csc.data)
        logger.info(f"矩阵B非零元素: {nnz}, 稀疏度: {1 - nnz/(b_rows*b_cols):.2f}")
        
        # 创建矩阵数据管理器
        data_manager = MatrixDataManager(self.A, B_csc, self.num_pes)
        
        # 记录下一个要加载的列
        next_col_to_load = self.num_pes
        
        # 记录已完成的列
        completed_columns = set()
        
        # 为所有PE分配初始列 (0 到 num_pes-1)
        for pe_idx in range(self.num_pes):
            col_idx = pe_idx
            # 跳过超出范围的列
            if col_idx >= b_cols:
                continue
                
            element_count = data_manager.get_column_element_count(col_idx)
            column_data = data_manager.get_column_data(col_idx)
            
            # 将列数据分配给PE
            self.pes[pe_idx].assign_column(col_idx, element_count)
            
            # 清空旧队列数据
            self.pe_data_queues[pe_idx].clear()
            
            # 将列数据加入到PE的队列中
            for a_bf16, b_bf16 in column_data:
                self.pe_data_queues[pe_idx].append((a_bf16, b_bf16, False))
            
            if element_count == 0:
                # 空列直接标记为已完成
                completed_columns.add(col_idx)
        
        # 主循环
        max_cycles = 10000  # 降低最大周期，防止无限循环
        last_status_time = 0
        status_interval = 100
        
        while self.clock < max_cycles:
            # 检查是否所有列都已处理完
            if len(completed_columns) == b_cols:
                logger.info(f"所有 {b_cols} 列已全部处理完成")
                break
            
            # 跟踪本周期是否有活跃处理
            cycle_active = False
            
            # 1. 从队列向PE提供数据
            for pe_idx in range(self.num_pes):
                pe = self.pes[pe_idx]
                
                # 处理PE完成的结果
                if pe.is_result_ready():
                    col, result = pe.get_column_result()
                    
                    if col is not None:
                        # 存储BF16格式的结果，转换回FP32
                        self.result[0, col] = bf16_to_float(result)
                        completed_columns.add(col)
                        
                        # 为此PE分配新列
                        if next_col_to_load < b_cols:
                            new_col = next_col_to_load
                            next_col_to_load += 1
                            
                            # 获取新列BF16数据
                            element_count = data_manager.get_column_element_count(new_col)
                            column_data = data_manager.get_column_data(new_col)
                            
                            # 分配新列给PE
                            pe.assign_column(new_col, element_count)
                            
                            # 将新列BF16数据加入PE的队列
                            self.pe_data_queues[pe_idx].clear()
                            for a_bf16, b_bf16 in column_data:
                                self.pe_data_queues[pe_idx].append((a_bf16, b_bf16, False))
                            
                            if element_count == 0:
                                # 空列直接标记为已完成
                                completed_columns.add(new_col)
                            
                            cycle_active = True
                
                # 如果PE准备好接收数据且队列中有数据
                if pe.is_ready_for_input() and self.pe_data_queues[pe_idx]:
                    a_bf16, b_bf16, is_end = self.pe_data_queues[pe_idx].popleft()
                    if pe.load_data(a_bf16, b_bf16, is_end):
                        self.processed_elements += 1
                        cycle_active = True
            
            # 2. 所有PE执行一个周期
            for pe in self.pes:
                if pe.clock_cycle():
                    cycle_active = True
            
            # 如果没有任何活动且没有列需要处理，尝试提前终止
            if not cycle_active and len(completed_columns) == b_cols:
                logger.info(f"检测到无活跃操作，所有列已完成，提前终止")
                break
            
            # 定期打印状态
            if self.clock - last_status_time >= status_interval:
                last_status_time = self.clock
                completion_pct = len(completed_columns) * 100.0 / b_cols
                busy_pes = sum(1 for pe in self.pes if pe.is_busy())
                
                logger.info(f"周期 {self.clock}: 已完成 {len(completed_columns)}/{b_cols} 列 ({completion_pct:.1f}%), "
                           f"活跃PE={busy_pes}, 下一列={next_col_to_load}")
            
            self.clock += 1
        
        logger.info(f"BF16动态列加载完成，总用时 {self.clock} 个周期")
        logger.info(f"处理非零元素: {self.processed_elements}/{nnz}")
        logger.info(f"平均每周期处理元素: {self.processed_elements/max(1, self.clock):.2f}")
        
        return self.verify_result()
    
    def verify_result(self):
        """验证计算结果与NumPy参考结果比对"""
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
        
        # 考虑BF16计算误差，使用更宽松的容差
        if error < 1e-1:
            logger.info("验证成功: 结果与参考值一致 (考虑BF16精度)")
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
    # 使用BF16动态列加载流水线矩阵乘法
    simulator.run_dynamic_column_pipeline(
        a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, 
        sparsity=0.8)

if __name__ == "__main__":
    main()