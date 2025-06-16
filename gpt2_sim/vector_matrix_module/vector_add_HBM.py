import numpy as np
import logging
from collections import deque
import struct
import torch
import sys
import os
from scipy.sparse import csr_matrix
from tqdm import tqdm

from vector_matrix_module.bf16_sim import BF16AddPipeline, FP32toBF16Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorAddSim")

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

class ProcessingElement:
    """处理元素(PE) - 负责执行元素加法操作"""
    def __init__(self, pe_id, perow_id, perow_result_vector):
        self.pe_id = pe_id
        self.perow_id = perow_id
        self.perow_result_vector = perow_result_vector
        
        self.add_pipeline = BF16AddPipeline()
        
        # 流水线阶段
        self.stage1_input = None  # (a_value, b_value, col_index)
        self.stage2_input = None  # (a_value, b_value, col_index)
        
        # 控制流水线输入
        self.stage2_new_input = False
        
        self.tasks_completed = 0

    def load_task(self, a_value, b_value, col_index):
        """加载新任务到PE"""
        if self.is_idle():
            self.stage1_input = (a_value, b_value, col_index)
            return True
        return False

    def clock_cycle(self):
        """执行一个时钟周期"""
        # --- 阶段 2: 元素加法 ---
        if self.stage2_input:
            a_value, b_value, col_idx = self.stage2_input
            
            if self.stage2_new_input:
                add_result = self.add_pipeline.clock_cycle(a_value, b_value, True)
                self.stage2_new_input = False
            else:
                add_result = self.add_pipeline.clock_cycle(None, None, False)

            if add_result["valid_output"] and self.add_pipeline.outputs:
                self.perow_result_vector[col_idx] = self.add_pipeline.outputs.pop(0)
                self.tasks_completed += 1
                self.stage2_input = None  # 任务完成
        elif self.add_pipeline.is_active():
            self.add_pipeline.clock_cycle(None, None, False)

        # --- 阶段 1: 输入 ---
        if self.stage1_input:
            self.stage2_input = self.stage1_input
            self.stage2_new_input = True
            self.stage1_input = None
            
        return self.is_busy()
        
    def is_busy(self):
        """检查PE是否在处理任务"""
        return self.stage1_input or self.stage2_input or self.add_pipeline.is_active()

    def is_idle(self):
        return not self.is_busy()

class ProcessingElementRow:
    """处理元素行(PERow) - 包含一组PE和对应的结果向量部分"""
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

class VectorAddSimulatorWithHBM:
    """HBM模式下的向量元素加法模拟器"""
    def __init__(self, num_perows=32, pes_per_row=128, vector_size=4096):
        self.num_perows = num_perows
        self.pes_per_row = pes_per_row
        self.vector_size = vector_size
        self.perows = [ProcessingElementRow(i, pes_per_row, vector_size) for i in range(num_perows)]
        self.clock = 0
        self.final_result_vector = np.zeros(vector_size, dtype=np.float32)

    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        for perow in self.perows:
            perow.reset()
        self.final_result_vector = np.zeros(self.vector_size, dtype=np.float32)

    def run_simulation(self, A_vector, B_vector, elements_per_block=2048):
        """执行基于HBM数据块的向量元素加法模拟"""
        logger.info(f"开始HBM模式模拟. A({A_vector.shape[0]}) + B({B_vector.shape[0]})")
        self.reset()

        # --- 1. 数据准备 ---
        # 转换输入向量为BF16
        A_vector_bf16 = [fp32_to_bf16(val) for val in A_vector]
        B_vector_bf16 = [fp32_to_bf16(val) for val in B_vector]

        # --- 2. 按块处理 ---
        total_tasks_generated = 0
        num_blocks = (self.vector_size + elements_per_block - 1) // elements_per_block

        for block_idx in tqdm(range(num_blocks), desc="处理HBM块"):
            start_idx = block_idx * elements_per_block
            end_idx = min((block_idx + 1) * elements_per_block, self.vector_size)
            
            # 为当前块生成任务
            block_tasks = []
            for i in range(start_idx, end_idx):
                block_tasks.append((A_vector_bf16[i], B_vector_bf16[i], i))
            
            if not block_tasks: continue
            
            total_tasks_generated += len(block_tasks)

            # 将当前块的任务均匀分配给PERows
            tasks_per_perow = len(block_tasks) // self.num_perows
            extra_tasks = len(block_tasks) % self.num_perows
            task_idx = 0
            
            for perow_idx in range(self.num_perows):
                num_tasks_for_this_perow = tasks_per_perow + (1 if perow_idx < extra_tasks else 0)
                if num_tasks_for_this_perow > 0:
                    tasks_to_assign = block_tasks[task_idx : task_idx + num_tasks_for_this_perow]
                    self.perows[perow_idx].assign_tasks(tasks_to_assign)
                    task_idx += num_tasks_for_this_perow

            # 运行模拟直到当前块的任务全部完成
            while any(perow.is_busy() for perow in self.perows):
                for perow in self.perows:
                    perow.clock_cycle()
                self.clock += 1

        # --- 3. 最终累加 ---
        logger.info("所有块处理完毕，开始累加最终结果...")
        for perow in self.perows:
            perow_result = perow.get_result_vector()
            for col_idx in range(self.vector_size):
                self.final_result_vector[col_idx] += bf16_to_float(perow_result[col_idx])
        
        logger.info(f"模拟完成. 总周期: {self.clock}, 总任务数: {total_tasks_generated}")
        
        return {
            "output_vector": self.final_result_vector,
            "clock": self.clock,
        }

    def verify_result(self, A_vector, B_vector):
        """验证计算结果"""
        logger.info("开始验证结果...")
        reference_result = A_vector + B_vector  # NumPy中的+操作符就是元素加法
        
        error = np.abs(self.final_result_vector - reference_result).max()
        logger.info(f"与Numpy精确结果的最大误差: {error}")
        
        is_correct = np.allclose(self.final_result_vector, reference_result, rtol=1e-2, atol=1e-2)
        if is_correct:
            logger.info("✅ 验证成功!")
        else:
            logger.error("❌ 验证失败!")
            logger.error(f"结果向量样本:\n{self.final_result_vector[:10]}")
            logger.error(f"预期结果样本:\n{reference_result[:10]}")
        return is_correct

def main():
    """主测试函数"""
    # --- 配置 ---
    vector_size = 4096
    elements_per_block = 256
    num_perows = 32
    pes_per_row = 128
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logger.info("="*50)
    logger.info("测试HBM模式向量元素加法模拟器")
    
    # --- 数据生成 ---
    logger.info("生成测试数据...")
    A_vector = np.random.randn(vector_size).astype(np.float32) * 0.1
    B_vector = np.random.randn(vector_size).astype(np.float32) * 0.1
    
    # --- 模拟 ---
    simulator = VectorAddSimulatorWithHBM(
        num_perows=num_perows, 
        pes_per_row=pes_per_row, 
        vector_size=vector_size
    )
    simulator.run_simulation(A_vector, B_vector, elements_per_block)
    
    # --- 验证 ---
    simulator.verify_result(A_vector, B_vector)
    logger.info("="*50)

if __name__ == "__main__":
    main()