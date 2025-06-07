import numpy as np
import logging
from collections import deque
import struct
import torch
import sys
import os

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'gpt2_sim', 'gpt-2'))
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorMatrixRowProductV2")

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
    fp32_bits = bf16 << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

class SparseMatrix:
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
    处理元素(PE) - 完全借鉴MACUnit的设计
    
    每个PE负责执行乘法和累加操作，使用三级流水线：
    1. 输入阶段：接收a_value, b_value和col_index
    2. 乘法阶段：执行a * b
    3. 加法阶段：将乘法结果累加到PERow向量的对应位置
    
    累加操作直接在PE中进行，使用加法流水线
    """
    def __init__(self, pe_id, perow_id, perow_result_vector):
        self.pe_id = pe_id
        self.perow_id = perow_id
        self.perow_result_vector = perow_result_vector  # 绑定到PERow的结果向量
        
        # 为每个PE创建独立的流水线实例，避免状态冲突
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.add_pipeline = BF16AddPipeline()
        
        # 输入数据
        self.a_value = None
        self.b_value = None
        self.col_index = None
        
        # 流水线控制，三级流水线
        self.stage1_valid = False
        self.stage2_valid = False
        self.stage3_valid = False
        
        # 阶段数据
        self.stage2_a_value = None
        self.stage2_b_value = None
        self.stage2_col_index = None
        self.stage2_input_valid = False
        
        self.stage3_input = None
        self.stage3_col_index = None
        self.stage3_input_valid = False
        
        # 任务计数器
        self.tasks_loaded = 0
        self.tasks_completed = 0
        
    def load_data(self, a_value, b_value, col_index):
        """加载输入数据到PE"""
        if not self.is_busy():
            self.a_value = a_value
            self.b_value = b_value
            self.col_index = col_index
            self.stage1_valid = True
            self.tasks_loaded += 1
            return True
        return False
    
    def clock_cycle(self):
        """执行一个时钟周期，三级流水线，完全借鉴MACUnit"""
        active = False
        
        # 阶段3: 加法阶段 - 将乘法结果累加到PERow向量，借鉴MACUnit
        if self.stage3_valid:
            col_idx = self.stage3_col_index
            if 0 <= col_idx < len(self.perow_result_vector):
                # 使用add_pipeline计算 perow_result_vector[col_idx] + stage3_input
                if self.stage3_input_valid:
                    result = self.add_pipeline.clock_cycle(
                        self.perow_result_vector[col_idx], 
                        self.stage3_input, 
                        True
                    )
                    # 标记输入已处理
                    self.stage3_input_valid = False
                else:
                    # 推进加法流水线
                    result = self.add_pipeline.clock_cycle(None, None, False)
                if result["valid_output"] and self.add_pipeline.outputs:
                    # 更新PERow向量对应位置的值
                    old_value = bf16_to_float(self.perow_result_vector[col_idx])
                    new_value_bf16 = self.add_pipeline.outputs.pop(0)
                    self.perow_result_vector[col_idx] = new_value_bf16
                    new_value = bf16_to_float(new_value_bf16)
                    
                    # 任务完成计数
                    self.tasks_completed += 1
                    
                    # 调试信息
                    if self.perow_id == 0 and self.pe_id < 2:
                        logger.debug(f"PE {self.pe_id} 完成累加: col={col_idx}, {old_value:.6f} + {bf16_to_float(self.stage3_input):.6f} = {new_value:.6f}, 完成={self.tasks_completed}/{self.tasks_loaded}")
                    
                    # 清除阶段3的有效标志和输入数据
                    self.stage3_valid = False
                    self.stage3_input = None
                    self.stage3_col_index = None
                    active = True
            else:
                # 无效列索引，直接清除
                self.stage3_valid = False

        # 阶段2: 乘法阶段 - a * b，借鉴MACUnit
        if self.stage2_valid:
            # 只在第一次调用时传入输入数据，后续调用传入None以推进流水线
            if self.stage2_input_valid:
                temp_result = self.multiply_pipeline.clock_cycle(
                    self.stage2_a_value, 
                    self.stage2_b_value, 
                    True
                )
                # 标记输入已处理，后续只推进流水线
                self.stage2_input_valid = False
            else:
                # 推进流水线但不输入新数据
                temp_result = self.multiply_pipeline.clock_cycle(None, None, False)
            
            # 调试乘法阶段
            if self.perow_id == 0 and self.pe_id < 2:
                logger.debug(f"PE {self.pe_id} 乘法阶段: input_valid={self.stage2_input_valid}, valid_output={temp_result.get('valid_output', False)}, outputs_len={len(self.multiply_pipeline.outputs)}")
            
            if temp_result["valid_output"] and len(self.multiply_pipeline.outputs) > 0:
                mul_result = self.multiply_pipeline.outputs.pop(0)
                
                # 调试乘法结果
                if self.perow_id == 0 and self.pe_id < 2:
                    logger.debug(f"PE {self.pe_id} 乘法完成: {bf16_to_float(self.stage2_a_value):.6f} * {bf16_to_float(self.stage2_b_value):.6f} = {bf16_to_float(mul_result):.6f}")
                
                # 将乘法结果传递到加法阶段
                self.stage3_input = mul_result
                self.stage3_col_index = self.stage2_col_index
                self.stage3_input_valid = True
                self.stage3_valid = True
                
                # 清除阶段2状态
                self.stage2_valid = False
                self.stage2_a_value = None
                self.stage2_b_value = None
                self.stage2_col_index = None
                self.stage2_input_valid = False
                active = True
        
        # 阶段1: 输入阶段，借鉴MACUnit
        if self.stage1_valid:
            self.stage2_a_value = self.a_value
            self.stage2_b_value = self.b_value
            self.stage2_col_index = self.col_index
            self.stage2_input_valid = True
            self.stage2_valid = True
            
            # 清除阶段1状态
            self.stage1_valid = False
            self.a_value = None
            self.b_value = None
            self.col_index = None
            active = True
        
        return active
    
    def is_busy(self):
        """检查PE是否繁忙，借鉴MACUnit"""
        # 检查各个阶段和流水线状态
        stage_busy = (self.stage1_valid or self.stage2_valid or self.stage3_valid)
        pipeline_busy = (self.multiply_pipeline.is_active() or self.add_pipeline.is_active())
        
        busy = stage_busy or pipeline_busy
        
        # 如果流水线长期活跃但没有真正的任务在进行，强制清理
        if pipeline_busy and not stage_busy and self.tasks_completed >= self.tasks_loaded:
            # 尝试推进流水线以清空内部状态
            self.multiply_pipeline.clock_cycle(None, None, False)
            self.add_pipeline.clock_cycle(None, None, False)
            
            # 再次检查是否仍然活跃
            pipeline_busy = (self.multiply_pipeline.is_active() or self.add_pipeline.is_active())
            busy = pipeline_busy
        
        # 调试信息 - 仅在有问题时打印
        if self.perow_id == 0 and self.pe_id < 2:
            if busy and self.tasks_completed > 0 and self.tasks_completed % 10 == 0:
                logger.debug(f"PE {self.pe_id} 忙碌状态: stage_busy={stage_busy}(1:{self.stage1_valid}, 2:{self.stage2_valid}, 3:{self.stage3_valid}), pipeline_busy={pipeline_busy}(mul:{self.multiply_pipeline.is_active()}, add:{self.add_pipeline.is_active()}), 完成={self.tasks_completed}")
        
        return busy

class ProcessingElementRow:
    """
    处理元素行(PERow) - 包含128个PE和4096维结果向量
    在PERow的cycle中，只需要为PE分配新任务，执行PE就好了
    """
    def __init__(self, perow_id, num_pes=128, vector_size=4096):
        self.perow_id = perow_id
        self.num_pes = num_pes
        self.vector_size = vector_size
        
        # 4096维结果向量 - 使用BF16格式存储
        self.result_vector = [fp32_to_bf16(0.0) for _ in range(vector_size)]
        
        # 创建PE，并传递result_vector的引用
        self.pes = [ProcessingElement(i, perow_id, self.result_vector) for i in range(num_pes)]
        
        # 任务队列
        self.task_queue = deque()  # [(a_value, b_value, col_index), ...]
        
        # 状态跟踪
        self.elements_processed = 0
        self.elements_assigned = 0
        self.is_idle = True
        
    def assign_tasks(self, tasks):
        """批量分配任务给此PERow"""
        self.task_queue.extend(tasks)
        self.elements_assigned += len(tasks)
        self.elements_processed = 0
        self.is_idle = False
        
        if self.perow_id == 0:  # 调试信息
            logger.info(f"PERow {self.perow_id} 分配了 {len(tasks)} 个任务")
    
    def clock_cycle(self):
        """
        执行一个时钟周期
        在PERow的cycle中，只需要为PE分配新任务，执行PE就好了
        """
        active = False

        
        # PE直接累加到结果向量中，不需要收集输出
        # 只需要跟踪处理进度
        outputs_collected = 0
        
        # 为空闲PE分配新任务
        tasks_assigned = 0
        for pe in self.pes:
            if not pe.is_busy() and self.task_queue:
                a_value, b_value, col_index = self.task_queue.popleft()
                
                if pe.load_data(a_value, b_value, col_index):
                    tasks_assigned += 1
                    if self.perow_id == 0 and tasks_assigned <= 5:  # 只打印前几个
                        logger.debug(f"PERow {self.perow_id} PE {pe.pe_id} 分配任务: col={col_index}")

        
        # 执行所有PE的时钟周期
        pe_active_count = 0
        for pe in self.pes:
            if pe.clock_cycle():
                active = True
                pe_active_count += 1
        
        # 收集所有PE已完成的任务数
        total_completed = sum(pe.tasks_completed for pe in self.pes)
        self.elements_processed = total_completed
        
        # 更新空闲状态 - 基于任务队列和PE状态
        queue_empty = len(self.task_queue) == 0
        all_pes_idle = all(not pe.is_busy() for pe in self.pes)
        all_tasks_completed = self.elements_processed >= self.elements_assigned
        
        # 更详细的调试信息
        if self.perow_id == 0 and self.elements_processed < 50:
            busy_pes = sum(1 for pe in self.pes if pe.is_busy())
            logger.debug(f"PERow {self.perow_id}: 队列={len(self.task_queue)}, 忙碌PE={busy_pes}, 已完成={self.elements_processed}/{self.elements_assigned}")
        
        if queue_empty and all_pes_idle and all_tasks_completed:
            if not self.is_idle:
                if self.perow_id == 0:
                    logger.info(f"PERow {self.perow_id} 完成所有任务，处理了 {self.elements_processed}/{self.elements_assigned} 个元素")
            self.is_idle = True
        
        return active or tasks_assigned > 0 or outputs_collected > 0
    
    def is_processing_complete(self):
        """检查是否完成所有处理"""
        return self.is_idle
    
    def get_result_vector(self):
        """获取结果向量的拷贝（BF16格式）"""
        return self.result_vector.copy()
    
    def reset_vector(self):
        """重置结果向量"""
        self.result_vector = [fp32_to_bf16(0.0) for _ in range(self.vector_size)]
    
    def is_busy(self):
        """检查PERow是否繁忙"""
        return not self.is_idle

class VectorMatrixRowProductSimulatorV2:
    """向量矩阵行积乘法模拟器V2 - 借鉴MACUnit设计"""
    def __init__(self, num_perows=32, pes_per_row=128, vector_size=4096):
        self.num_perows = num_perows
        self.pes_per_row = pes_per_row
        self.vector_size = vector_size
        self.perows = [ProcessingElementRow(i, pes_per_row, vector_size) 
                      for i in range(num_perows)]
        
        # 时钟和状态
        self.clock = 0
        self.final_result_vector = np.zeros(vector_size, dtype=np.float32)
        
    def reset(self):
        """重置模拟器状态"""
        self.clock = 0
        self.perows = [ProcessingElementRow(i, self.pes_per_row, self.vector_size) 
                      for i in range(self.num_perows)]
        self.final_result_vector = np.zeros(self.vector_size, dtype=np.float32)
    
    def run_vector_matrix_multiply(self, a_rows=1, a_cols=4096, b_rows=4096, b_cols=4096, 
                                  sparsity=0.8, seed=42):
        """执行向量矩阵乘法：C = A * B (行积方式)"""
        logger.info(f"开始向量矩阵行积乘法V2 ({a_rows}x{a_cols}) * ({b_rows}x{b_cols}), 稀疏度={sparsity}")
        
        # 生成测试数据
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        A = (torch.randn((a_rows, a_cols)) * 0.1).to(torch.bfloat16)
        B_dense = (torch.randn((b_rows, b_cols)) * 0.1).to(torch.bfloat16)
        
        # 创建稀疏矩阵B
        np.random.seed(seed)
        mask = np.random.rand(b_rows, b_cols) < sparsity
        B_sparse = np.where(mask, 0, B_dense.to(torch.float32).numpy())
        sparse_matrix_b, _ = SparseMatrix.from_dense(B_sparse, sparsity)
        
        # 计算参考结果
        A_bf16 = A.to(torch.bfloat16)
        B_sparse_bf16 = torch.tensor(B_sparse).to(torch.bfloat16)
        self.reference_result = torch.matmul(A_bf16, B_sparse_bf16).to(torch.float32).numpy()
        
        A_np = A.to(torch.float32).numpy()
        
        logger.info(f"生成完成，总计非零元素: {np.count_nonzero(B_sparse)}")
        
        # 重置状态
        self.reset()
        
        # 收集所有乘法任务
        all_tasks = []
        total_elements = 0
        
        for k in range(a_cols):
            a_value = A_np[0, k]
            
            if k < len(sparse_matrix_b.data):
                row_data = sparse_matrix_b.data[k]
                row_indices = sparse_matrix_b.indices[k]
                
                if row_data:
                    for b_value, col_index in zip(row_data, row_indices):
                        a_bf16 = fp32_to_bf16(float(a_value))
                        b_bf16 = fp32_to_bf16(float(b_value))
                        all_tasks.append((a_bf16, b_bf16, col_index))
                        total_elements += 1
        
        logger.info(f"总共生成 {total_elements} 个乘法任务")
        
        # 智能任务分配
        tasks_per_perow = total_elements // self.num_perows
        extra_tasks = total_elements % self.num_perows
        
        task_start = 0
        for perow_idx in range(self.num_perows):
            num_tasks = tasks_per_perow + (1 if perow_idx < extra_tasks else 0)
            
            if num_tasks > 0:
                perow_tasks = all_tasks[task_start:task_start + num_tasks]
                self.perows[perow_idx].assign_tasks(perow_tasks)
                task_start += num_tasks
        
        # 主循环
        max_cycles = max(total_elements // 32, 50000)
        status_interval = max(1000, max_cycles // 50)
        last_status_time = 0
        
        while self.clock < max_cycles:
            # 检查所有任务是否完成
            all_perows_idle = all(not perow.is_busy() for perow in self.perows)
            
            if all_perows_idle:
                logger.info(f"所有处理完成")
                break
            
            # 执行所有PERow的时钟周期
            any_active = False
            for perow in self.perows:
                if perow.clock_cycle():
                    any_active = True
            
            # 定期打印状态
            if self.clock - last_status_time >= status_interval:
                last_status_time = self.clock
                busy_perows = sum(1 for perow in self.perows if perow.is_busy())
                total_processed = sum(perow.elements_processed for perow in self.perows)
                
                logger.info(f"周期 {self.clock}: 已处理 {total_processed}/{total_elements} 任务, "
                          f"活跃PERow={busy_perows}")
            
            self.clock += 1
            
            if self.clock >= max_cycles:
                logger.warning(f"达到最大时钟周期限制({max_cycles})，停止模拟")
                break
        
        # 最终累加：将32个PERow的结果向量累加得到最终的向量C
        for perow_idx in range(self.num_perows):
            perow_result = self.perows[perow_idx].get_result_vector()
            
            # 调试：检查前几个PERow的非零结果
            if perow_idx < 3:
                non_zero_count = sum(1 for x in perow_result if bf16_to_float(x) != 0.0)
                logger.info(f"PERow {perow_idx} 有 {non_zero_count} 个非零元素")
                if non_zero_count > 0:
                    non_zero_values = [(i, bf16_to_float(perow_result[i])) for i in range(min(10, len(perow_result))) if bf16_to_float(perow_result[i]) != 0.0]
                    logger.info(f"PERow {perow_idx} 前几个非零值: {non_zero_values}")
            
            for col_idx in range(self.vector_size):
                if col_idx < len(perow_result):
                    # 将BF16结果累加到最终结果中
                    self.final_result_vector[col_idx] += bf16_to_float(perow_result[col_idx])
        
        total_processed = sum(perow.elements_processed for perow in self.perows)
        logger.info(f"向量矩阵行积乘法V2完成，总用时 {self.clock} 个周期")
        logger.info(f"处理了 {total_processed}/{total_elements} 个任务")
        
        return self.verify_result()
    
    def verify_result(self):
        """验证计算结果"""
        if not hasattr(self, 'reference_result'):
            logger.warning("没有参考结果可供验证")
            return False
        
        # 构建结果矩阵
        result_matrix = self.final_result_vector.reshape(1, -1)
        
        error = np.abs(result_matrix - self.reference_result).max()
        logger.info(f"最大误差: {error}")
        
        # 根据矩阵规模调整容差
        matrix_size = result_matrix.shape[1]
        if matrix_size >= 4096:
            tolerance = 5e-2
        else:
            tolerance = 1e-1
            
        if error < tolerance:
            logger.info(f"验证成功: 结果与参考值一致（容差={tolerance}）")
            return True
        else:
            logger.error(f"验证失败: 结果与参考值不一致（容差={tolerance}）")
            logger.info(f"结果向量样本:\n{result_matrix[0, :20]}")
            logger.info(f"预期结果样本:\n{self.reference_result[0, :20]}")
            return False

def test_small_matrix():
    """测试小规模矩阵乘法的正确性"""
    print("\n" + "="*60)
    print("小规模矩阵乘法测试V2")
    print("="*60)
    
    simulator = VectorMatrixRowProductSimulatorV2(num_perows=2, pes_per_row=4, vector_size=8)
    
    success = simulator.run_vector_matrix_multiply(
        a_rows=1, a_cols=8, b_rows=8, b_cols=8,
        sparsity=0.5, seed=42
    )
    
    if success:
        print("✅ 小规模测试V2成功")
    else:
        print("❌ 小规模测试V2失败")
    
    return success

def main():
    """主函数"""
    # 先测试小规模
    test_small_matrix()
    vector_size = 1024
    # 创建模拟器
    simulator = VectorMatrixRowProductSimulatorV2(num_perows=32, pes_per_row=128, vector_size=vector_size)
    
    # 运行4096x4096矩阵测试
    # success = simulator.run_vector_matrix_multiply(
    #     a_rows=32, a_cols=4096, b_rows=4096, b_cols=4096,
    #     sparsity=0.8, seed=42
    # )
    success = simulator.run_vector_matrix_multiply(
        a_rows=1, a_cols=vector_size, b_rows=vector_size, b_cols=vector_size,
        sparsity=0.8, seed=42
    )
    
    # 计算统计信息
    total_cycles = simulator.clock
    
    print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"总周期数: {total_cycles}")
    
    # 显示部分结果
    print(f"结果向量前10个元素: {simulator.final_result_vector[:10]}")
    if hasattr(simulator, 'reference_result'):
        print(f"参考结果前10个元素: {simulator.reference_result[0, :10]}")
        error = np.abs(simulator.final_result_vector.reshape(1, -1) - simulator.reference_result).max()
        print(f"最大绝对误差: {error:.6f}")
    
    print()
    
    if success:
        print(f"✅ 测试V2成功!")
    else:
        print(f"❌ 测试V2失败")
        
    print("=" * 80)

if __name__ == "__main__":
    main()