import numpy as np
import logging
from collections import deque
import struct
import torch
import sys
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm

# 修复导入路径
try:
    from .bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
except ImportError:
    # 如果相对导入失败，尝试从其他路径导入
    try:
        from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
    except ImportError:
        # 尝试从hbm模块导入
        try:
            from .bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
        except ImportError:
            raise ImportError("无法找到bf16_sim模块")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VecMatProdSim")

def store_csr_in_simple_blocks_fast(csr_matrix, elements_per_block=2):
    values = csr_matrix.data
    col_indices = csr_matrix.indices
    row_pointers = csr_matrix.indptr
    num_rows = csr_matrix.shape[0]
    total_values = len(values)  # 修复：添加缺失的变量

    # 预先生成每个元素的行号
    row_indices = np.zeros_like(values, dtype=np.int32)
    for row in range(num_rows):
        row_indices[row_pointers[row]:row_pointers[row+1]] = row

    num_blocks = (total_values + elements_per_block - 1) // elements_per_block
    blocks = []

    for block_idx in range(num_blocks):
        start_idx = block_idx * elements_per_block
        end_idx = min(start_idx + elements_per_block, total_values)

        block_values = values[start_idx:end_idx].tolist()
        block_col_indices = col_indices[start_idx:end_idx].tolist()
        block_row_indices = row_indices[start_idx:end_idx]

        if len(block_row_indices) == 0:
            continue

        row_start = block_row_indices[0]
        row_end = block_row_indices[-1]
        num_block_rows = row_end - row_start + 1

        # 生成块内row_ptr
        block_row_ptr = [0]
        cur = 0
        for r in range(row_start, row_end + 1):
            # 统计本行在block中的元素数
            count = np.sum(block_row_indices == r)
            cur += count
            block_row_ptr.append(cur)

        block = {
            "values": block_values,
            "col_indices": block_col_indices,
            "row_ptr": block_row_ptr,
            "row_start_index": int(row_start),
        }
        blocks.append(block)
    return blocks

def direct_bf16_add(a, b):
    """直接计算两个BF16数字的和"""
    from bf16_sim import BF16AddPipeline as BF16AddPipelineClass
    pipeline = BF16AddPipelineClass()
    pipeline.run_simulation([(a, b, True)], print_states=False)
    return pipeline.outputs[0] if pipeline.outputs else 0

def fp32_to_bf16(fp32_value):
    """将FP32值转换为BF16格式的整数表示"""
    if isinstance(fp32_value, np.ndarray):
        fp32_value = float(fp32_value.item())
    elif isinstance(fp32_value, (np.float32, np.float64)):
        fp32_value = float(fp32_value)
    
    pipeline = FP32toBF16Pipeline()
    pipeline.run_simulation([(fp32_value, True)], print_states=False)
    return pipeline.outputs[0]["bf16"] if pipeline.outputs else 0

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式的浮点数"""
    if bf16 is None: return 0.0
    fp32_bits = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

class CSRMatrix:
    """稀疏矩阵表示，基于行存储格式"""
    def __init__(self, data, indices, shape, sparsity):
        self.data = data
        self.indices = indices
        self.shape = shape
        self.sparsity = sparsity
        
    @classmethod
    def from_dense(cls, dense_matrix, sparsity=0.8):
        """从稠密矩阵创建稀疏矩阵表示"""
        rows, cols = dense_matrix.shape
        data = []
        indices = []
        
        for i in range(rows):
            row_data = []
            row_indices = []
            for j in range(cols):
                if dense_matrix[i, j] != 0:
                    row_data.append(dense_matrix[i, j])
                    row_indices.append(j)
            data.append(row_data)
            indices.append(row_indices)
        
        return cls(data, indices, dense_matrix.shape, sparsity), dense_matrix

class ProcessingElement:
    """
    处理元素(PE) - 模拟一个乘法-累加单元.
    
    每个PE负责执行乘法和累加操作，使用三级流水线：
    1. 输入阶段：接收a_value, b_value和col_index
    2. 乘法阶段：执行a * b
    3. 加法阶段：将乘法结果累加到PERow向量的对应位置
    """
    def __init__(self, pe_id, perow_id, perow_result_vector):
        self.pe_id = pe_id
        self.perow_id = perow_id
        self.perow_result_vector = perow_result_vector

        self.multiply_pipeline = BF16MultiplyPipeline()
        self.add_pipeline = BF16AddPipeline()
        
        # 流水线阶段
        self.stage1_input = None  # (a_value, b_value, col_index)
        self.stage2_input = None  # (a_value, b_value, col_index)
        self.stage3_input = None  # (mul_result, col_index)
        
        # 控制流水线输入
        self.stage2_new_input = False
        self.stage3_new_input = False
        
        self.tasks_completed = 0

    def load_task(self, a_value, b_value, col_index):
        """加载新任务到PE"""
        if self.is_idle():
            self.stage1_input = (a_value, b_value, col_index)
            return True
        return False

    def clock_cycle(self):
        """执行一个时钟周期"""
        # --- 阶段 3: 加法 ---
        if self.stage3_input:
            mul_result, col_idx = self.stage3_input
            
            # 仅在有新输入时送入流水线
            if self.stage3_new_input:
                current_val = self.perow_result_vector[col_idx]
                add_result = self.add_pipeline.clock_cycle(current_val, mul_result, True)
                self.stage3_new_input = False
            else:
                # 否则, 仅推进流水线
                add_result = self.add_pipeline.clock_cycle(None, None, False)

            if add_result["valid_output"] and self.add_pipeline.outputs:
                self.perow_result_vector[col_idx] = self.add_pipeline.outputs.pop(0)
                self.tasks_completed += 1
                self.stage3_input = None  # 任务完成
        elif self.add_pipeline.is_active():
             self.add_pipeline.clock_cycle(None, None, False)


        # --- 阶段 2: 乘法 ---
        if self.stage2_input:
            a_value, b_value, col_index = self.stage2_input
            
            if self.stage2_new_input:
                mul_res = self.multiply_pipeline.clock_cycle(a_value, b_value, True)
                self.stage2_new_input = False
            else:
                mul_res = self.multiply_pipeline.clock_cycle(None, None, False)
            
            if mul_res["valid_output"] and self.multiply_pipeline.outputs:
                self.stage3_input = (self.multiply_pipeline.outputs.pop(0), col_index)
                self.stage3_new_input = True # 为下一阶段准备新输入
                self.stage2_input = None # 传递到下一阶段
        elif self.multiply_pipeline.is_active():
            self.multiply_pipeline.clock_cycle(None, None, False)

        # --- 阶段 1: 输入 ---
        if self.stage1_input:
            self.stage2_input = self.stage1_input
            self.stage2_new_input = True # 为下一阶段准备新输入
            self.stage1_input = None
            
        return self.is_busy()
        
    def is_busy(self):
        """检查PE是否在处理任务"""
        return self.stage1_input or self.stage2_input or self.stage3_input or \
               self.multiply_pipeline.is_active() or self.add_pipeline.is_active()

    def is_idle(self):
        return not self.is_busy()

class ProcessingElementRow:
    """
    处理元素行(PERow) - 包含一组PE和对应的结果向量部分
    """
    def __init__(self, perow_id, num_pes=128, vector_size=4096):
        self.perow_id = perow_id
        self.num_pes = num_pes
        self.vector_size = vector_size
        self.result_vector = [fp32_to_bf16(0.0) for _ in range(vector_size)]
        self.pes = [ProcessingElement(i, perow_id, self.result_vector) for i in range(num_pes)]
        self.task_queue = deque()
        self.assigned_tasks_count = 0

    def assign_tasks(self, tasks):
        """分配一批任务给此PERow"""
        self.task_queue.extend(tasks)
        self.assigned_tasks_count += len(tasks)

    def clock_cycle(self):
        """执行一个时钟周期"""
        # 为空闲PE分配新任务
        for pe in self.pes:
            if pe.is_idle() and self.task_queue:
                task = self.task_queue.popleft()
                pe.load_task(*task)
        
        # 执行所有PE的时钟周期
        for pe in self.pes:
            pe.clock_cycle()

    def is_busy(self):
        """检查此PERow是否还有任务或其PE是否繁忙"""
        return len(self.task_queue) > 0 or any(pe.is_busy() for pe in self.pes)

    def get_completed_tasks_count(self):
        return sum(pe.tasks_completed for pe in self.pes)

    def get_result_vector(self):
        return self.result_vector

    def reset(self):
        self.result_vector = [fp32_to_bf16(0.0) for _ in range(self.vector_size)]
        self.pes = [ProcessingElement(i, self.perow_id, self.result_vector) for i in range(self.num_pes)]
        self.task_queue.clear()
        self.assigned_tasks_count = 0

class VectorMatrixRowProductSimulatorWithHBM:
    """
    HBM模式下的多batch向量矩阵乘法模拟器.
    支持多batch输入，将perows分组处理不同batch，矩阵B在各组间复用。
    """
    def __init__(self, num_perows=32, pes_per_row=128, vector_size=4096):
        self.num_perows = num_perows
        self.pes_per_row = pes_per_row
        self.vector_size = vector_size
        self.perows = [ProcessingElementRow(i, pes_per_row, vector_size) for i in range(num_perows)]
        self.clock = 0
        self.final_result_matrix = None  # 改为矩阵以支持多batch

    def reset(self, batch_size=1):
        """重置模拟器状态"""
        self.clock = 0
        for perow in self.perows:
            perow.reset()
        self.final_result_matrix = np.zeros((batch_size, self.vector_size), dtype=np.float32)

    def run_simulation(self, A_vectors, B_matrix_sparse, elements_per_block=2048):
        """
        执行基于HBM数据块的多batch向量矩阵乘法模拟.
        
        Args:
            A_vectors: 输入向量矩阵，形状为 (batch_size, vector_dim) 或 (vector_dim,)
            B_matrix_sparse: 稀疏权重矩阵，形状为 (vector_dim, output_dim)
            elements_per_block: 每个HBM块的元素数量
            
        Returns:
            dict: 包含输出矩阵和时钟周期数的字典
        """
        # 处理单batch输入且为（4096，）的
        if A_vectors.ndim == 1:
            A_vectors = A_vectors.reshape(1, -1)
        
        batch_size, vector_dim = A_vectors.shape
        logger.info(f"行积：开始HBM模式多batch模拟. A({batch_size}, {vector_dim}) @ B({B_matrix_sparse.shape})")
        
        # 计算perows分组策略 - 支持batch > num_perows的时间复用
        if batch_size <= self.num_perows:
            # 情况1: batch <= perows，每个batch分配若干perows
            perows_per_batch = self.num_perows // batch_size
            batches_per_perow = 1
            logger.info(f"batch_size: {batch_size}, 每个batch分配 {perows_per_batch} 个perows")
        else:
            # 情况2: batch > perows，每个perow处理多个batch（时间复用）
            perows_per_batch = 1
            batches_per_perow = (batch_size + self.num_perows - 1) // self.num_perows  # ceil division
            logger.info(f"batch_size: {batch_size}, 每个perow处理 {batches_per_perow} 个batch（时间复用）")
        
        self.reset(batch_size)

        # --- 1. 数据准备 ---
        B_csr = csr_matrix(B_matrix_sparse)
        logger.info("将稀疏矩阵B分块...")
        B_blocks = store_csr_in_simple_blocks_fast(B_csr, elements_per_block)
        logger.info(f"矩阵B被分为 {len(B_blocks)} 个块.")
        
        # 转换所有batch的输入向量为BF16
        A_vectors_bf16 = []
        for batch_idx in range(batch_size):
            A_vector_bf16 = [fp32_to_bf16(val) for val in A_vectors[batch_idx]]
            A_vectors_bf16.append(A_vector_bf16)

        # --- 2. 按块处理 ---
        # 需要为每个batch处理所有blocks，支持时间复用
        if batch_size <= self.num_perows:
            # 情况1: 正常模式，所有batch可以并行处理
            num_time_slots = 1
            batches_per_slot = batch_size
        else:
            # 情况2: 时间复用模式，分多个时间片处理
            num_time_slots = batches_per_perow
            batches_per_slot = self.num_perows
        
        block_idx = 0
        total_tasks_generated = 0
        cycle = 0
        num_blocks = len(B_blocks)
        pbar = tqdm(total=num_blocks * num_time_slots + 100, desc="行积：处理HBM多batch数据", unit="cycle")
        
        # 外层循环：时间片
        for time_slot in range(num_time_slots):
            block_idx = 0  # 每个时间片都要处理所有blocks
            
            # 确定当前时间片要处理的batch范围
            start_batch_idx = time_slot * batches_per_slot
            end_batch_idx = min(start_batch_idx + batches_per_slot, batch_size)
            current_batches = list(range(start_batch_idx, end_batch_idx))
            
            logger.info(f"时间片 {time_slot+1}/{num_time_slots}: 处理batch {current_batches}")
            
            # 重置所有PERows（清除上一个时间片的状态）
            if time_slot > 0:
                for perow in self.perows:
                    perow.reset()
            
            # 内层循环：处理所有blocks
            while block_idx < num_blocks or any(perow.is_busy() for perow in self.perows):
                # 注入新的block数据
                if block_idx < num_blocks:
                    block = B_blocks[block_idx]
                    
                    # 为当前时间片的每个batch生成任务
                    for batch_offset, batch_idx in enumerate(current_batches):
                        batch_tasks = []
                        b_values_bf16 = [fp32_to_bf16(v) for v in block["values"]]
                        num_rows_in_block = len(block["row_ptr"]) - 1
                        
                        A_vector_bf16 = A_vectors_bf16[batch_idx]
                        
                        for r_offset in range(num_rows_in_block):
                            start, end = block["row_ptr"][r_offset], block["row_ptr"][r_offset+1]
                            if start == end: 
                                continue
                            
                            # 计算出该行在全局A向量中的真实行号
                            abs_row_idx = block["row_start_index"] + r_offset
                            # 取出A向量对应行的元素
                            a_val = A_vector_bf16[abs_row_idx]
                            
                            # 遍历该行在B矩阵中的非零元素
                            for j in range(start, end):
                                b_val = b_values_bf16[j]
                                target_col_idx = block["col_indices"][j]
                                batch_tasks.append((a_val, b_val, target_col_idx))
                        
                        total_tasks_generated += len(batch_tasks)
                        
                        # 将当前batch的任务分配给对应的perow
                        if batch_size <= self.num_perows:
                            # 情况1: 每个batch分配多个perows
                            start_perow_idx = batch_offset * perows_per_batch
                            end_perow_idx = min(start_perow_idx + perows_per_batch, self.num_perows)
                            batch_perow_count = end_perow_idx - start_perow_idx
                        else:
                            # 情况2: 每个perow处理一个batch
                            start_perow_idx = batch_offset
                            end_perow_idx = batch_offset + 1
                            batch_perow_count = 1
                        
                        # 在该batch对应的perows间均匀分配任务
                        if batch_perow_count > 0 and len(batch_tasks) > 0:
                            tasks_per_perow = len(batch_tasks) // batch_perow_count
                            extra_tasks = len(batch_tasks) % batch_perow_count
                            task_idx = 0
                            
                            for perow_offset in range(batch_perow_count):
                                perow_idx = start_perow_idx + perow_offset
                                num_tasks_for_this_perow = tasks_per_perow + (1 if perow_offset < extra_tasks else 0)
                                if num_tasks_for_this_perow > 0:
                                    tasks_to_assign = batch_tasks[task_idx : task_idx + num_tasks_for_this_perow]
                                    self.perows[perow_idx].assign_tasks(tasks_to_assign)
                                    task_idx += num_tasks_for_this_perow
                    
                    block_idx += 1
                
                # 推进所有perow流水线
                for perow in self.perows:
                    perow.clock_cycle()
                cycle += 1
                pbar.update(1)
            
            # 当前时间片处理完毕，收集结果
            logger.info(f"时间片 {time_slot+1} 处理完毕，收集结果...")
            for batch_offset, batch_idx in enumerate(current_batches):
                if batch_size <= self.num_perows:
                    # 情况1: 每个batch分配多个perows，需要累加
                    start_perow_idx = batch_offset * perows_per_batch
                    end_perow_idx = min(start_perow_idx + perows_per_batch, self.num_perows)
                    
                    for perow_idx in range(start_perow_idx, end_perow_idx):
                        perow_result = self.perows[perow_idx].get_result_vector()
                        for col_idx in range(self.vector_size):
                            self.final_result_matrix[batch_idx, col_idx] += bf16_to_float(perow_result[col_idx])
                else:
                    # 情况2: 每个perow处理一个batch，直接复制
                    perow_idx = batch_offset
                    perow_result = self.perows[perow_idx].get_result_vector()
                    for col_idx in range(self.vector_size):
                        self.final_result_matrix[batch_idx, col_idx] += bf16_to_float(perow_result[col_idx])
            
                pbar.close()
        
        logger.info(f"多batch模拟完成. 总周期: {cycle}, 总任务数: {total_tasks_generated}")
        
        return {
            "output_matrix": self.final_result_matrix,
            "output_vector": self.final_result_matrix[0] if batch_size == 1 else None,  # 兼容单batch情况
            "clock": cycle,
            "batch_size": batch_size
        }

    def verify_result(self, A_vectors, B_matrix_sparse):
        """验证多batch计算结果"""
        logger.info("开始验证多batch结果...")
        
        # 处理输入格式
        if A_vectors.ndim == 1:
            A_vectors = A_vectors.reshape(1, -1)
        
        reference_result = A_vectors @ B_matrix_sparse
        
        error = np.abs(self.final_result_matrix - reference_result).max()
        logger.info(f"与Numpy精确结果的最大误差: {error}")
        
        is_correct = np.allclose(self.final_result_matrix, reference_result, rtol=1e-1, atol=1e-1)
        if is_correct:
            logger.info("✅ 多batch验证成功!")
        else:
            logger.error("❌ 多batch验证失败!")
            logger.error(f"结果矩阵样本:\n{self.final_result_matrix[:3, :10]}")
            logger.error(f"预期结果样本:\n{reference_result[:3, :10]}")
        return is_correct


def main():
    """主测试函数, 测试多batch HBM模拟器"""
    # --- 配置 ---
    vec_dim = 4096
    output_dim = 4096
    sparsity = 0.9
    elements_per_block = 256
    num_perows = 32
    pes_per_row = 128
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("="*60)
    logger.info("测试HBM模式多batch稀疏向量-矩阵乘法模拟器")
    
    # 测试不同的batch配置
    test_cases = [
        # {"batch_size": 1, "desc": "单batch测试"},
        # {"batch_size": 4, "desc": "4-batch测试 (每8个perow处理1个batch)"},
        # {"batch_size": 8, "desc": "8-batch测试 (每4个perow处理1个batch)"},
        # {"batch_size": 32, "desc": "32-batch测试 (每1个perow处理1个batch)"},
        {"batch_size": 64, "desc": "64-batch测试 (时间复用，每个perow处理2个batch)"},
    ]
    
    for test_case in test_cases:
        batch_size = test_case["batch_size"]
        desc = test_case["desc"]
        
            
        logger.info(f"\n--- {desc} ---")
        
        # --- 数据生成 ---
        vec_row = min(1024, vec_dim)  # 确保不超过设定的维度
        A_vectors = np.random.randn(batch_size, vec_row).astype(np.float32) * 0.1
        
        B_dense = torch.randn((vec_row, output_dim)) * 0.01
        mask = torch.rand(vec_row, output_dim) > sparsity
        B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)
        
        # --- 模拟 ---
        simulator = VectorMatrixRowProductSimulatorWithHBM(
            num_perows=num_perows, 
            pes_per_row=pes_per_row, 
            vector_size=output_dim
        )
        
        result = simulator.run_simulation(A_vectors, B_sparse, elements_per_block)
        
        # --- 验证 ---
        is_correct = simulator.verify_result(A_vectors, B_sparse)
        
        logger.info(f"测试结果: {'✅ 通过' if is_correct else '❌ 失败'}")
        logger.info(f"输出形状: {result['output_matrix'].shape}")
        logger.info(f"总周期数: {result['clock']}")
    
    logger.info("="*60)


if __name__ == "__main__":
    main()
