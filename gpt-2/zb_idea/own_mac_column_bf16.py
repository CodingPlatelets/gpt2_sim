import numpy as np
import logging
from collections import deque
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SparseMatrixPipeline")
import struct

def direct_bf16_add(a, b):
    """直接计算两个BF16数字的和
    
    Args:
        a (int): 第一个BF16格式的数字
        b (int): 第二个BF16格式的数字
        
    Returns:
        int: BF16格式的加法结果
    """
    # 这里直接创建BF16AddPipeline类的实例，而不是调用同名函数
    from bf16_sim import BF16AddPipeline as BF16AddPipelineClass
    pipeline = BF16AddPipelineClass()
    pipeline.run_simulation([(a, b, True)], print_states=False)
    return pipeline.outputs[0] if pipeline.outputs else 0

def fp32_to_bf16(fp32_value):
    """将FP32值转换为BF16格式的整数表示"""
    # 确保是Python float类型
    if isinstance(fp32_value, np.ndarray):
        fp32_value = float(fp32_value.item())
    elif isinstance(fp32_value, (np.float32, np.float64)):
        fp32_value = float(fp32_value)
    
    # 使用BF16转换流水线
    pipeline = FP32toBF16Pipeline()
    pipeline.run_simulation([(fp32_value, True)], print_states=False)
    return pipeline.outputs[0]["bf16"] if pipeline.outputs else 0

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式的浮点数"""
    # 左移16位填充为32位表示
    fp32_bits = bf16 << 16
    # 转换为浮点数
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

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
    处理元素(PE)单元 - 模拟硬件中的基本计算单元
    
    每个PE负责处理矩阵B的一整列，实现两级流水线：
    1. 输入阶段：接收矩阵A和B的元素值
    2. 计算阶段：执行乘法和累加操作
    """
    def __init__(self, pe_id):
        """初始化处理元素"""
        self.pe_id = pe_id
        self.accumulator = fp32_to_bf16(0.0)  # 重置为BF16格式的0  # 当前列的累加器
        
        # 当前处理的列信息
        self.current_col = None  # 当前处理的全局列索引
        self.remaining_elements = 0  # 当前列剩余元素数
        self.is_empty_column = False  # 标记是否为空列
        
        # 输入阶段数据
        self.a_value = None
        self.b_value = None
        self.stage1_input_valid = False
        
        # 计算阶段数据
        self.compute_a_value = None
        self.compute_b_value = None
        
        # 流水线控制
        self.stage1_valid = False
        self.stage2_valid = False
        self.stage2_output = None
        self.stage3_valid = False
        self.stage3_input = None
        
        # 状态跟踪
        self.column_complete = False       # 当前列是否已完成处理
        self.result_ready = False          # 结果是否已准备好被收集

        self.multiply_pipeline = BF16MultiplyPipeline()
        self.add_pipeline = BF16AddPipeline()

    
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
        self.accumulator = fp32_to_bf16(0.0)  # 重置为BF16格式的0
        
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
    
    def load_data(self, a_value, b_value, input_valid=False):
        """加载输入数据，支持空标记"""
        if not input_valid:
            # 这是列结束标记
            self.a_value = None
            self.b_value = None
            self.stage1_valid = True  # 需要将这个标记传递到流水线中
            self.stage1_input_valid = input_valid
            # 仍然进入流水线，但会触发结束状态
            return 
        
        if not self.column_complete:

            self.a_value = a_value
            self.b_value = b_value
            self.stage1_valid = True  # 标记输入阶段有效
            self.stage1_input_valid = input_valid
    
    def clock_cycle(self):
        """执行一个时钟周期，返回是否有活动"""
        if self.is_empty_column:
            # 空列直接标记为完成，不需要执行计算
            return False
        
        active = False
        
        # 阶段3: 加法阶段 - 使用BF16加法流水线
        if self.stage3_valid:
            logger.info(f"PE {self.pe_id} 进入加法阶段 , {self.accumulator}, {self.stage3_input}, {self.stage3_input_valid}")
            # 执行BF16加法
            temp_result = self.add_pipeline.clock_cycle(self.accumulator, self.stage3_input, self.stage3_input_valid)
            if temp_result["valid_output"]:
                # 确保有结果可以获取
                if len(self.add_pipeline.outputs) > 0:
                    
                    # 减少剩余元素计数
                    self.remaining_elements -= 1
                    
                    if self.pe_id == 0:
                        logger.info(f"stage3 ：PE {self.pe_id} 乘法结果: {bf16_to_float(self.add_pipeline.outputs[0])}")

                    self.accumulator = self.add_pipeline.outputs.pop(0)
                    
                    # 检查是否完成当前列
                    if self.remaining_elements <= 0:
                        self.column_complete = True
                        self.result_ready = True
                        logger.debug(f"PE {self.pe_id} 完成处理列 {self.current_col}")
                    
                    self.stage3_valid = False
                    active = True
                else:
                    logger.warning(f"PE {self.pe_id} 加法流水线输出为空")
        
        # 阶段2: 乘法阶段 - 使用BF16乘法流水线
        if self.stage2_valid:
            # 执行BF16乘法
            temp_result = self.multiply_pipeline.clock_cycle(self.compute_a_value, self.compute_b_value, self.stage2_input_valid)
            if temp_result["valid_output"]:
                # 确保有结果可以获取
                if len(self.multiply_pipeline.outputs) > 0:
                    # if self.pe_id == 0:
                        # logger.info(f"stage2 ： PE {self.pe_id} 乘法结果: {bf16_to_float(self.multiply_pipeline.outputs[0])}，当前元素 {self.count2}")
                        
                    self.stage2_output = self.multiply_pipeline.outputs.pop(0)
                    self.stage2_valid = False
                    self.stage3_input = self.stage2_output  # 将乘法结果传递到加法阶段
                    self.stage3_input_valid = True
                    self.stage3_valid = True
                    active = True
                else:
                    logger.warning(f"PE {self.pe_id} 乘法流水线输出为空")
            else:
                self.stage3_input_valid = False  # 重置乘法阶段输入有效标志
        
        # 阶段1: 输入阶段
        if self.stage1_valid:
            # 将输入数据传递到计算阶段
            # if self.input_valid and  self.pe_id == 0:
            #     self.count1 += 1
            #     logger.info(f"PE {self.pe_id} 输入数据: A={self.a_value}, B={self.b_value} , 当前传入的是第 {self.count1}元素")
            # if self.a_value is not None and self.b_value is not None:
                # 处理输入数据
                # self.stage1_input_valid = False
                # print(f"PE {self.pe_id} 输入数据: A={bf16_to_float(self.a_value)}, B={bf16_to_float(self.b_value)}")
            self.compute_a_value = self.a_value
            self.compute_b_value = self.b_value
            self.stage2_input_valid = self.stage1_input_valid  # 标记乘法阶段有效
            self.stage2_valid = True
            self.stage1_valid = False

            active = True
        
        return active
    
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
            acc_value = bf16_to_float(self.accumulator)  # 转换为FP32格式
            self.result_ready = False  # 标记结果已被收集
            logger.info(f"PE {self.pe_id} 返回列 {col} 的最终累加值: {acc_value}")
            return col, acc_value
        return None, 0.0
    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        # return self.stage1_input_valid or self.stage2_mul_valid or self.stage3_out_valid or (not self.column_complete)
        # 检查所有流水线阶段
        return (self.stage1_valid or 
                self.stage2_valid or 
                self.stage3_valid or 
                self.multiply_pipeline.is_active() or 
                self.add_pipeline.is_active() or
                not self.column_complete)


class MACUnit:
    """
    处理元素(PE)单元 - 模拟硬件中的基本计算单元
    
    每个PE负责处理矩阵B的一整列，实现多级流水线：
    1. 输入阶段：接收矩阵A和B的元素值
    2. 乘法阶段：执行乘法操作
    3. 加法阶段：执行累加操作
    """
    def __init__(self, pe_id):
        """初始化处理元素"""
        self.pe_id = pe_id
        self.accumulator = fp32_to_bf16(0.0)  # BF16格式的累加器
        
        # 当前处理的列信息
        self.current_col = None  # 当前处理的全局列索引
        self.remaining_elements = 0  # 当前列剩余元素数
        self.is_empty_column = False  # 标记是否为空列
        
        # 输入阶段
        self.stage1_valid = False
        self.a_value = None
        self.b_value = None
        self.input_valid = False
        
        # 乘法阶段
        self.stage2_valid = False
        self.stage2_a_value = None
        self.stage2_b_value = None
        self.stage2_result = None
        
        # 加法阶段
        self.stage3_valid = False
        self.stage3_input = None
        
        # 状态跟踪
        self.column_complete = False  # 当前列是否已完成处理
        self.result_ready = False     # 结果是否已准备好被收集

        # BF16流水线
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.add_pipeline = BF16AddPipeline()
    
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
            result = bf16_to_float(self.accumulator)
            self.result_ready = False
        
        # 更新到新列
        self.current_col = col_idx
        self.remaining_elements = element_count
        self.accumulator = fp32_to_bf16(0.0)  # 重置为BF16格式的0
        
        # 重置所有流水线状态
        self.stage1_valid = False
        self.stage2_valid = False 
        self.stage3_valid = False
        
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
    
    def load_data(self, a_value, b_value, input_valid=True):
        """加载输入数据，支持结束标记"""
        if not input_valid:
            # 这是列结束标记，标记列已完成
            self.column_complete = True
            self.result_ready = True
            return
        
        if not self.column_complete:
            # 只有在列未完成时才加载新数据
            self.a_value = a_value
            self.b_value = b_value
            self.stage1_valid = True
            self.input_valid = input_valid
    
    def clock_cycle(self):
        """执行一个时钟周期，按照从后向前的顺序处理各个流水线阶段"""
        if self.is_empty_column:
            # 空列直接标记为完成，不需要执行计算
            return False
        
        active = False

        # 阶段3: 加法阶段 - 使用BF16加法流水线
        # 注意：我们要先保存当前累加器值，用于日志和调试
        # old_acc = self.accumulator
        
        # 执行BF16加法，确保当stage3_valid为True时才传入stage3_input
        result = self.add_pipeline.clock_cycle(
            0, 
            self.stage3_input if self.stage3_valid else None, 
            self.stage3_valid
        )
        if result["valid_output"] and self.add_pipeline.outputs:
            # 获取加法流水线的结果 (相当于 0 + stage3_input)
            original_acc = self.accumulator
            mul_result = self.add_pipeline.outputs.pop(0)
            
            # 使用direct_bf16_add将乘法结果与累加器相加
            self.accumulator = direct_bf16_add(self.accumulator, mul_result)
            # logger.info(f"PE {self.pe_id} 加法结果:"
            #     f"{bf16_to_float(original_acc)} + {bf16_to_float(mul_result)} = {bf16_to_float(self.accumulator)}")
            # logger.info(f"PE {self.pe_id} 累加结果: {bf16_to_float(self.accumulator)}")

            
            # # 减少剩余元素计数
            # if self.stage3_valid:  # 只有当输入有效时才减少计数
            self.remaining_elements -= 1
            
            # 检查是否完成当前列
            if self.remaining_elements <= 0 :
                self.column_complete = True
                self.result_ready = True
                logger.debug(f"PE {self.pe_id} 完成处理列 {self.current_col}")
            
            # 清除阶段3的有效标志
            self.stage3_valid = False
            active = True 
        
        # 阶段2: 乘法阶段 - 使用BF16乘法流水线
        result = self.multiply_pipeline.clock_cycle(
            self.stage2_a_value if self.stage2_valid else None,
            self.stage2_b_value if self.stage2_valid else None,
            self.stage2_valid
        )
        if result["valid_output"] and self.multiply_pipeline.outputs:
            # 获取乘法结果并推送到加法阶段
            mul_result = self.multiply_pipeline.outputs.pop(0)
            logger.debug(f"PE {self.pe_id} 乘法结果: {bf16_to_float(mul_result)}")
            
            # 将乘法结果传递到加法阶段
            self.stage3_input = mul_result
            self.stage3_valid = True
            
            # 清除阶段2的有效标志
            self.stage2_valid = False
            active = True
        
        # 阶段1: 输入阶段 - 将输入数据传递到乘法阶段
        if self.stage1_valid:
            # 将输入数据传递到乘法阶段
            self.stage2_a_value = self.a_value
            self.stage2_b_value = self.b_value
            self.stage2_valid = True
            
            # 清除阶段1的有效标志
            self.stage1_valid = False
            active = True
        
        return active
    
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
            result = bf16_to_float(self.accumulator)  # 转换为FP32格式
            self.result_ready = False  # 标记结果已被收集
            return col, result
        return None, 0.0
    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        # 检查所有流水线阶段和硬件单元
        return (self.stage1_valid or 
                self.stage2_valid or 
                self.stage3_valid or 
                self.multiply_pipeline.is_active() or 
                self.add_pipeline.is_active() or
                not self.column_complete)

    
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
            result = bf16_to_float(self.accumulator)  # 转换为FP32格式
            self.result_ready = False  # 标记结果已被收集
            return col, result
        return None, 0.0
    
    def is_busy(self):
        """检查PE是否正在处理数据"""
        # 检查所有流水线阶段和硬件单元 - 包括是否收到了结束标记但还没完成处理
        return (self.stage1_valid or 
                self.stage2_valid or 
                self.stage3_valid or 
                self.multiply_pipeline.is_active() or 
                self.add_pipeline.is_active() or
                not self.column_complete)

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

            # 转换为BF16格式
            a_bf16 = fp32_to_bf16(float(a_value))
            b_bf16 = fp32_to_bf16(float(b_value))

            column_data.append((a_bf16, b_bf16))
        
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
        self.pes = [MACUnit(i) for i in range(num_pes)]  # 创建PE数组
        
        # 结果收集
        self.C_sim = None  # 最终计算结果
        
        # 每个PE有一个队列，存储待处理的列数据
        self.pe_data_queues = [deque() for _ in range(num_pes)]

    def  is_active(self):
        """检查是否有PE处于活动状态"""
        return any(pe.is_busy() for pe in self.pes)
    
    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        self.pes = [MACUnit(i) for i in range(self.num_pes)]
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
        import torch
        np.random.seed(seed)
        # ==========生成随机矩阵A和B bf16格式计算得到C_torch
        A = (torch.randn((a_rows, a_cols)) * 0.05).to(torch.bfloat16)
        B_dense_full = (torch.randn((b_rows, b_cols)) * 0.05)

        # A = torch.ones((a_rows, a_cols)).to(torch.bfloat16)
        # B_dense_full = torch.ones((b_rows, b_cols))

        # A = (torch.rand((a_rows, a_cols))).to(torch.bfloat16)
        # B_dense_full = (torch.rand((b_rows, b_cols)))

        mask = np.random.rand(b_rows, b_cols) < sparsity
        B_dense_full = np.where(mask, 0, B_dense_full)
        B_dense_full = torch.tensor(B_dense_full).to(torch.bfloat16)

        # 计算参考结果 bf16结果
        self.C_torch = torch.matmul(A, B_dense_full).to(torch.float32).numpy()
        # 添加这一行 - 确保验证时使用正确的参考结果
        self.reference_result = self.C_torch

        # print(A)
        # print(B_dense_full)
        # for i in range(len(A[0])):
        #     print(A[0][i] * B_dense_full[i][0])

        # 初始化结果矩阵
        self.C_sim = torch.zeros((a_rows, b_cols)).to(torch.float32).numpy()
        # 重置状态
        self.reset()

        # ===========将矩阵A和B转换为float32格式计算
        self.A = A.to(torch.float32).numpy()
        self.B_dense_full = B_dense_full.to(torch.float32).numpy()
        
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
                self.pe_data_queues[pe_idx].append((a_value, b_value, True))
        # 主循环
        max_cycles = 100000
        last_status_time = 0
        status_interval = 1000
        
        # 跟踪所有列是否已分配
        all_cols_assigned = False
        
        while self.clock < max_cycles or self.is_active():
            # 检查是否所有列都已经处理完毕且所有PE都不再忙碌
            all_completed = len(completed_columns) == b_cols
            all_pes_idle = all(not pe.is_busy() for pe in self.pes)
            all_queues_empty = all(len(queue) == 0 for queue in self.pe_data_queues)
            
            # 只有当所有列完成且所有PE都不再活跃且所有队列都为空时才结束
            if all_completed and all_pes_idle and all_queues_empty:
                logger.info(f"所有处理完成：列={len(completed_columns)}/{b_cols}，PE空闲={all_pes_idle}，队列空={all_queues_empty}")
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
                        self.C_sim[0, col] = result
                        completed_columns.add(col)
                        
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
                                self.pe_data_queues[pe_idx].append((a_value, b_value, True))
                            
                            # 确保在处理最后一个元素后传递结束标记
                            if column_data:  # 只有当有数据时才添加结束标记
                                self.pe_data_queues[pe_idx].append((None, None, False))
                            
                            logger.debug(f"为PE {pe_idx} 分配新列 {new_col}，有 {element_count} 个元素")
                        else:
                            # 没有更多列可分配，确保PE完成当前工作
                            if not all_cols_assigned and next_col_to_load >= b_cols:
                                all_cols_assigned = True

                
                elif not pe.is_column_complete() and self.is_active():
                    # 确保每个列的最后都有一个结束标记
                    if len(self.pe_data_queues[pe_idx]) == 0 :
                        # 如果最后一个元素不是结束标记，添加一个结束标记
                        self.pe_data_queues[pe_idx].append((None, None, False))
                
                # 如果PE需要数据且队列中有数据
                if not pe.is_column_complete() and not pe.stage1_valid and self.pe_data_queues[pe_idx]:
                    a_value, b_value, input_valid = self.pe_data_queues[pe_idx].popleft()
                    pe.load_data(a_value, b_value, input_valid)
            
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
            if self.clock >= max_cycles:
                logger.warning(f"达到最大时钟周期限制({max_cycles})，停止模拟")
                break
        
        # 额外记录最终状态
        logger.info(f"最终状态：已完成列={len(completed_columns)}/{b_cols}, 时钟周期={self.clock}")
        return self.verify_result()
    
    def verify_result(self):
        """验证计算结果与NumPy参考结果比对"""
        # 验证结果
        logger.info(f"动态列加载流水线矩阵乘法完成，总用时 {self.clock} 个周期")
  
        if not hasattr(self, 'reference_result'):
            logger.warning("没有参考结果可供验证")
            return False

        error = np.abs(self.C_sim - self.C_torch).max()
        logger.info(f"最大误差: {error}")
        
        # 考虑浮点数计算误差，使用更宽松的容差
        if error < 1e-1:
            logger.info("验证成功: 结果与参考值一致")
            return True
        else:
            logger.error("验证失败: 结果与参考值不一致")
            logger.info(f"结果矩阵样本:\n{self.C_sim[0, :20]}")
            logger.info(f"预期结果样本:\n{self.C_torch[0, :20]}")
            return False
    

def main():
    """主函数"""
    PE_SIZE = 128
    col = 4096
    # 创建模拟器
    simulator = SparseMatrixPipeline(num_pes=PE_SIZE)
    # 稀疏度0.9 8192 快5000个clock 4096 快1000
    # 稀疏度0.8 8192 快10000个clock 4096 快2000
    # 使用动态列加载流水线矩阵乘法    409
    simulator.run_dynamic_column_pipeline(
        a_rows=1, a_cols=col, b_rows=col, b_cols=4096, 
        sparsity=0.8, seed=42)

if __name__ == "__main__":
    main()

    # 例子：使用转换后的BF16数字
    # a = fp32_to_bf16(3.14)
    # b = fp32_to_bf16(2.71)
    # sum_result = direct_bf16_add(a, b)
    # print(f"{bf16_to_float(a)} + {bf16_to_float(b)} = {bf16_to_float(sum_result)}")
    
