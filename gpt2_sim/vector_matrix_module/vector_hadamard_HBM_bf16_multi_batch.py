import numpy as np
import logging
from collections import deque
import struct
import torch
import sys
import os
from tqdm import tqdm

from .bf16_sim import BF16MultiplyPipeline, FP32toBF16Pipeline
from .utils import fp32_to_bf16, bf16_to_float

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VecHadamardSimMultiBatch")

class ProcessingElement:
    """处理元素(PE) - 负责执行Hadamard积操作"""
    def __init__(self, pe_id, perow_id, perow_result_vector):
        self.pe_id = pe_id
        self.perow_id = perow_id
        self.perow_result_vector = perow_result_vector
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.stage1_input = None  # (a_value, b_value, col_index)
        self.stage2_input = None  # (a_value, b_value, col_index)
        self.stage2_new_input = False
        self.tasks_completed = 0

    def load_task(self, a_value, b_value, col_index):
        if self.is_idle():
            self.stage1_input = (a_value, b_value, col_index)
            return True
        return False

    def clock_cycle(self):
        if self.stage2_input:
            a_value, b_value, col_idx = self.stage2_input
            if self.stage2_new_input:
                mul_result = self.multiply_pipeline.clock_cycle(a_value, b_value, True)
                self.stage2_new_input = False
            else:
                mul_result = self.multiply_pipeline.clock_cycle(None, None, False)
            if mul_result["valid_output"] and self.multiply_pipeline.outputs:
                self.perow_result_vector[col_idx] = self.multiply_pipeline.outputs.pop(0)
                self.tasks_completed += 1
                self.stage2_input = None
        elif self.multiply_pipeline.is_active():
            self.multiply_pipeline.clock_cycle(None, None, False)
        if self.stage1_input:
            self.stage2_input = self.stage1_input
            self.stage2_new_input = True
            self.stage1_input = None
        return self.is_busy()
    def is_busy(self):
        return self.stage1_input or self.stage2_input or self.multiply_pipeline.is_active()
    def is_idle(self):
        return not self.is_busy()

class ProcessingElementRow:
    def __init__(self, perow_id, num_pes=128, vector_size=4096):
        self.perow_id = perow_id
        self.num_pes = num_pes
        self.vector_size = vector_size
        self.result_vector = [fp32_to_bf16(0.0) for _ in range(vector_size)]
        self.pes = [ProcessingElement(i, perow_id, self.result_vector) for i in range(num_pes)]
        self.task_queue = deque()
        self.assigned_tasks_count = 0
    def assign_tasks(self, tasks):
        self.task_queue.extend(tasks)
        self.assigned_tasks_count += len(tasks)
    def clock_cycle(self):
        for pe in self.pes:
            if pe.is_idle() and self.task_queue:
                task = self.task_queue.popleft()
                pe.load_task(*task)
        for pe in self.pes:
            pe.clock_cycle()
    def is_busy(self):
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

class VectorHadamardSimulatorWithHBMMultiBatch:
    """HBM模式下的多batch向量Hadamard积模拟器"""
    def __init__(self, num_perows=32, pes_per_row=128, vector_size=4096):
        self.num_perows = num_perows
        self.pes_per_row = pes_per_row
        self.vector_size = vector_size
        self.perows = [ProcessingElementRow(i, pes_per_row, vector_size) for i in range(num_perows)]
        self.clock = 0
        self.final_result_matrix = None

    def reset(self, batch_size=1):
        self.clock = 0
        for perow in self.perows:
            perow.reset()
        self.final_result_matrix = np.zeros((batch_size, self.vector_size), dtype=np.float32)

    def run_simulation(self, A_vectors_bf16, B_vectors_bf16, elements_per_block=2048):
        if isinstance(A_vectors_bf16, list):
            A_vectors_bf16 = np.array(A_vectors_bf16)
        if isinstance(B_vectors_bf16, list):
            B_vectors_bf16 = np.array(B_vectors_bf16)
        if A_vectors_bf16.ndim == 1:
            A_vectors_bf16 = A_vectors_bf16.reshape(1, -1)
        if B_vectors_bf16.ndim == 1:
            B_vectors_bf16 = B_vectors_bf16.reshape(1, -1)
        batch_size, vector_size = A_vectors_bf16.shape
        assert B_vectors_bf16.shape == (batch_size, vector_size), "A和B的batch和长度必须一致"
        logger.info(f"多batch Hadamard: A({batch_size}, {vector_size}) ⊙ B({batch_size}, {vector_size})")
        perows_per_batch = self.num_perows // batch_size
        if perows_per_batch == 0:
            raise ValueError(f"batch_size({batch_size}) 不能大于 num_perows({self.num_perows})")
        self.reset(batch_size)
        num_blocks = (self.vector_size + elements_per_block - 1) // elements_per_block
        total_tasks_generated = 0
        for block_idx in tqdm(range(num_blocks), desc="多batch Hadamard 处理HBM块"):
            start_idx = block_idx * elements_per_block
            end_idx = min((block_idx + 1) * elements_per_block, self.vector_size)
            for batch_idx in range(batch_size):
                block_tasks = []
                for i in range(start_idx, end_idx):
                    block_tasks.append((A_vectors_bf16[batch_idx, i], B_vectors_bf16[batch_idx, i], i))
                total_tasks_generated += len(block_tasks)
                start_perow_idx = batch_idx * perows_per_batch
                end_perow_idx = min(start_perow_idx + perows_per_batch, self.num_perows)
                batch_perow_count = end_perow_idx - start_perow_idx
                if batch_perow_count > 0 and len(block_tasks) > 0:
                    tasks_per_perow = len(block_tasks) // batch_perow_count
                    extra_tasks = len(block_tasks) % batch_perow_count
                    task_idx = 0
                    for perow_offset in range(batch_perow_count):
                        perow_idx = start_perow_idx + perow_offset
                        num_tasks_for_this_perow = tasks_per_perow + (1 if perow_offset < extra_tasks else 0)
                        if num_tasks_for_this_perow > 0:
                            tasks_to_assign = block_tasks[task_idx : task_idx + num_tasks_for_this_perow]
                            self.perows[perow_idx].assign_tasks(tasks_to_assign)
                            task_idx += num_tasks_for_this_perow
            # 运行模拟直到所有perow空闲
            while any(perow.is_busy() for perow in self.perows):
                for perow in self.perows:
                    perow.clock_cycle()
                self.clock += 1
        logger.info("所有块处理完毕，开始按batch分组累加最终结果...")
        for batch_idx in range(batch_size):
            start_perow_idx = batch_idx * perows_per_batch
            end_perow_idx = min(start_perow_idx + perows_per_batch, self.num_perows)
            for perow_idx in range(start_perow_idx, end_perow_idx):
                perow_result = self.perows[perow_idx].get_result_vector()
                for col_idx in range(self.vector_size):
                    self.final_result_matrix[batch_idx, col_idx] += perow_result[col_idx]
        logger.info(f"多batch Hadamard模拟完成. 总周期: {self.clock}, 总任务数: {total_tasks_generated}")
        return {
            "output_matrix": self.final_result_matrix,
            "output_vector": self.final_result_matrix[0] if batch_size == 1 else None,
            "clock": self.clock,
            "batch_size": batch_size
        }

    def verify_result(self, A_vectors_bf16, B_vectors_bf16):
        if isinstance(A_vectors_bf16, list):
            A_vectors_bf16 = np.array(A_vectors_bf16)
        if isinstance(B_vectors_bf16, list):
            B_vectors_bf16 = np.array(B_vectors_bf16)
        if A_vectors_bf16.ndim == 1:
            A_vectors_bf16 = A_vectors_bf16.reshape(1, -1)
        if B_vectors_bf16.ndim == 1:
            B_vectors_bf16 = B_vectors_bf16.reshape(1, -1)
        batch_size, vector_size = A_vectors_bf16.shape
        result_float = np.vectorize(bf16_to_float)(self.final_result_matrix)
        A_fp32 = np.vectorize(bf16_to_float)(A_vectors_bf16)
        B_fp32 = np.vectorize(bf16_to_float)(B_vectors_bf16)
        reference_result = A_fp32 * B_fp32
        error = np.abs(result_float - reference_result).max()
        logger.info(f"与Numpy精确结果的最大误差: {error}")
        is_correct = np.allclose(result_float, reference_result, rtol=1e-2, atol=1e-2)
        if is_correct:
            logger.info("✅ 多batch验证成功!")
            logger.info(f"结果矩阵样本:\n{result_float[:3, :10]}")
            logger.info(f"预期结果样本:\n{reference_result[:3, :10]}")
        else:
            logger.error("❌ 多batch验证失败!")
            logger.error(f"结果矩阵样本:\n{result_float[:3, :10]}")
            logger.error(f"预期结果样本:\n{reference_result[:3, :10]}")
        return is_correct

def main():
    vector_size = 4096
    elements_per_block = 256
    num_perows = 32
    pes_per_row = 128
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("="*60)
    logger.info("测试HBM模式多batch向量Hadamard积模拟器")
    test_cases = [
        {"batch_size": 1, "desc": "单batch测试"},
        {"batch_size": 4, "desc": "4-batch测试 (每8个perow处理1个batch)"},
        {"batch_size": 8, "desc": "8-batch测试 (每4个perow处理1个batch)"},
        {"batch_size": 32, "desc": "32-batch测试 (每1个perow处理1个batch)"},
    ]
    for test_case in test_cases:
        batch_size = test_case["batch_size"]
        desc = test_case["desc"]
        if batch_size > num_perows:
            logger.warning(f"跳过测试: {desc} - batch_size({batch_size}) > num_perows({num_perows})")
            continue
        logger.info(f"\n--- {desc} ---")
        A_vectors_fp32 = np.random.randn(batch_size, vector_size).astype(np.float32) * 2
        B_vectors_fp32 = np.random.randn(batch_size, vector_size).astype(np.float32) * 2
        A_vectors = np.array([[fp32_to_bf16(val) for val in row] for row in A_vectors_fp32])
        B_vectors = np.array([[fp32_to_bf16(val) for val in row] for row in B_vectors_fp32])
        simulator = VectorHadamardSimulatorWithHBMMultiBatch(
            num_perows=num_perows,
            pes_per_row=pes_per_row,
            vector_size=vector_size
        )
        result = simulator.run_simulation(A_vectors, B_vectors, elements_per_block)
        is_correct = simulator.verify_result(A_vectors, B_vectors)
        logger.info(f"测试结果: {'✅ 通过' if is_correct else '❌ 失败'}")
        logger.info(f"输出形状: {result['output_matrix'].shape}")
        logger.info(f"总周期数: {result['clock']}")
    logger.info("="*60)

if __name__ == "__main__":
    main() 