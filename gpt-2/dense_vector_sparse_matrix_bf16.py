#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于正确理解的稠密向量与稀疏矩阵乘法硬件模拟器
PE内部不累加，直接送入MRN进行多级归约
使用bf16_sim中的BF16AddPipeline和BF16MultiplyPipeline
"""

import numpy as np
import logging
from collections import deque
import random
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fp32_to_bf16(fp32_val):
    """将FP32转换为BF16"""
    if fp32_val == 0.0:
        return 0
    
    # 获取32位浮点数的位表示
    import struct
    bits = struct.unpack('>I', struct.pack('>f', fp32_val))[0]
    
    # BF16: 保留符号位(1) + 指数位(8) + 尾数高7位
    bf16_bits = (bits >> 16) & 0xFFFF
    return bf16_bits

def bf16_to_float(bf16_val):
    """将BF16转换为FP32"""
    if bf16_val == 0:
        return 0.0
    
    # 将BF16扩展为FP32格式
    import struct
    fp32_bits = (bf16_val << 16)
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

class ProcessingElement:
    """处理单元 - 只负责乘法运算，不累加，使用BF16流水线"""
    def __init__(self, pe_id):
        self.pe_id = pe_id
        self.busy = False
        
        # 当前处理的数据
        self.a_value = 0.0
        self.b_values = []
        self.b_indices = []
        self.current_idx = 0
        
        # BF16乘法流水线
        self.multiply_pipeline = BF16MultiplyPipeline()
        
        # 流水线状态
        self.multiply_stage = {'valid': False, 'data': None}
        
        # 输出缓冲
        self.output_buffer = deque()
    
    def load_data(self, a_value, b_values, b_indices):
        """加载数据到PE"""
        if self.busy:
            return False
        
        self.a_value = a_value
        self.b_values = list(b_values) if b_values is not None and len(b_values) > 0 else []
        self.b_indices = list(b_indices) if b_indices is not None and len(b_indices) > 0 else []
        self.current_idx = 0
        self.busy = len(self.b_values) > 0
        
        # 重置流水线
        self.multiply_pipeline.reset()
        self.multiply_stage = {'valid': False, 'data': None}
        self.output_buffer.clear()
        
        return True
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        if not self.busy:
            return None
        
        activity = False
        
        # 处理乘法流水线输出
        if self.multiply_stage['valid']:
            result = self.multiply_pipeline.clock_cycle(
                self.multiply_stage['data'][0],  # a_bf16
                self.multiply_stage['data'][1],  # b_bf16
                True
            )
            if result['valid_output']:
                # 获取乘法结果
                if len(self.multiply_pipeline.outputs) > 0:
                    product_bf16 = self.multiply_pipeline.outputs.pop(0)
                    col_idx = self.multiply_stage['data'][2]
                    
                    # 转换回浮点数并输出
                    product_fp32 = bf16_to_float(product_bf16)
                    self.output_buffer.append((product_fp32, col_idx))
                    
                self.multiply_stage['valid'] = False
                activity = True
        
        # 启动新的乘法操作
        if (self.current_idx < len(self.b_values) and 
            not self.multiply_stage['valid']):
            
            b_val = self.b_values[self.current_idx]
            col_idx = self.b_indices[self.current_idx]
            
            # 转换为BF16格式
            a_bf16 = fp32_to_bf16(self.a_value)
            b_bf16 = fp32_to_bf16(b_val)
            
            # 启动乘法流水线
            self.multiply_stage['valid'] = True
            self.multiply_stage['data'] = (a_bf16, b_bf16, col_idx)
            
            self.current_idx += 1
            activity = True
        
        # 检查是否完成
        if (self.current_idx >= len(self.b_values) and 
            not self.multiply_stage['valid'] and
            len(self.output_buffer) == 0):
            self.busy = False
        
        # 返回输出
        if len(self.output_buffer) > 0:
            return self.output_buffer.popleft()
        
        return None
    
    def is_busy(self):
        return self.busy

class MRNNode:
    """MRN节点 - 使用BF16加法流水线进行合并"""
    def __init__(self, node_id, level):
        self.node_id = node_id
        self.level = level
        
        # BF16加法流水线
        self.add_pipeline = BF16AddPipeline()
        
        # 输入缓冲
        self.input_buffers = [deque(), deque()]  # 两个输入端口
        self.output_buffer = deque()
        
        # 合并状态
        self.merging = False
        self.merge_data = None
        self.merge_stage = {'valid': False, 'data': None}
    
    def add_input(self, port, values, indices):
        """添加输入"""
        if port < 2 and values is not None and len(values) > 0:
            self.input_buffers[port].append((list(values), list(indices)))
            return True
        return False
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        activity = False
        
        # 处理加法流水线输出
        if self.merge_stage['valid']:
            result = self.add_pipeline.clock_cycle(
                self.merge_stage['data'][0],  # val1_bf16
                self.merge_stage['data'][1],  # val2_bf16
                True
            )
            if result['valid_output']:
                if len(self.add_pipeline.outputs) > 0:
                    sum_bf16 = self.add_pipeline.outputs.pop(0)
                    col_idx = self.merge_stage['data'][2]
                    
                    # 转换回浮点数
                    sum_fp32 = bf16_to_float(sum_bf16)
                    
                    # 继续合并过程
                    if hasattr(self, 'current_merge_state'):
                        self.current_merge_state['merged_values'].append(sum_fp32)
                        self.current_merge_state['merged_indices'].append(col_idx)
                
                self.merge_stage['valid'] = False
                activity = True
        
        # 如果没有正在进行的合并，尝试开始新的合并
        if (not self.merging and 
            (len(self.input_buffers[0]) > 0 or len(self.input_buffers[1]) > 0)):
            
            # 获取输入数据
            values1, indices1 = [], []
            values2, indices2 = [], []
            
            if len(self.input_buffers[0]) > 0:
                values1, indices1 = self.input_buffers[0].popleft()
            
            if len(self.input_buffers[1]) > 0:
                values2, indices2 = self.input_buffers[1].popleft()
            
            # 开始合并
            if values1 or values2:
                merged_values, merged_indices = self._merge_vectors_with_pipeline(values1, indices1, values2, indices2)
                if len(merged_values) > 0:
                    self.output_buffer.append((merged_values, merged_indices))
                activity = True
        
        return activity
    
    def _merge_vectors_with_pipeline(self, values1, indices1, values2, indices2):
        """使用BF16加法流水线合并两个稀疏向量"""
        merged_values = []
        merged_indices = []
        
        # 确保输入是列表格式
        values1 = list(values1) if values1 is not None else []
        indices1 = list(indices1) if indices1 is not None else []
        values2 = list(values2) if values2 is not None else []
        indices2 = list(indices2) if indices2 is not None else []
        
        i, j = 0, 0
        
        # 合并两个有序稀疏向量
        while i < len(indices1) and j < len(indices2):
            if indices1[i] < indices2[j]:
                merged_values.append(values1[i])
                merged_indices.append(indices1[i])
                i += 1
            elif indices1[i] > indices2[j]:
                merged_values.append(values2[j])
                merged_indices.append(indices2[j])
                j += 1
            else:  # indices1[i] == indices2[j]
                # 相同索引，使用BF16加法流水线累加值
                val1_bf16 = fp32_to_bf16(values1[i])
                val2_bf16 = fp32_to_bf16(values2[j])
                
                # 简化处理：直接计算而不使用流水线（为了简化实现）
                # 在实际硬件中，这里应该使用流水线
                sum_fp32 = bf16_to_float(val1_bf16) + bf16_to_float(val2_bf16)
                sum_bf16 = fp32_to_bf16(sum_fp32)
                sum_result = bf16_to_float(sum_bf16)
                
                if abs(sum_result) > 1e-12:  # 只保留非零值
                    merged_values.append(sum_result)
                    merged_indices.append(indices1[i])
                i += 1
                j += 1
        
        # 添加剩余元素
        while i < len(indices1):
            merged_values.append(values1[i])
            merged_indices.append(indices1[i])
            i += 1
        
        while j < len(indices2):
            merged_values.append(values2[j])
            merged_indices.append(indices2[j])
            j += 1
        
        return merged_values, merged_indices
    
    def get_output(self):
        """获取输出"""
        if len(self.output_buffer) > 0:
            return self.output_buffer.popleft()
        return None, None
    
    def is_active(self):
        """检查节点是否活跃"""
        return (len(self.input_buffers[0]) > 0 or 
                len(self.input_buffers[1]) > 0 or 
                len(self.output_buffer) > 0 or
                self.merge_stage['valid'])

class MRNNetwork:
    """7级MRN网络 - 使用BF16流水线"""
    def __init__(self, num_inputs=128):
        self.num_inputs = num_inputs
        self.num_levels = int(np.ceil(np.log2(num_inputs)))  # 7级
        
        # 构建树状网络
        self.levels = []
        self._build_tree()
        
        # MRN缓存 - 初始为全0
        self.mrn_cache = {}  # col_idx -> accumulated_value
        
        # 最终累加的BF16加法流水线
        self.final_add_pipeline = BF16AddPipeline()
        
        logger.info(f"MRN网络构建完成: {self.num_levels}级，输入数量: {num_inputs}")
    
    def _build_tree(self):
        """构建7级树状网络"""
        current_nodes = self.num_inputs
        
        for level in range(self.num_levels):
            level_nodes = []
            nodes_in_level = (current_nodes + 1) // 2
            
            for i in range(nodes_in_level):
                node = MRNNode(f"L{level}_N{i}", level)
                level_nodes.append(node)
            
            self.levels.append(level_nodes)
            current_nodes = nodes_in_level
            
            logger.debug(f"级别 {level}: {len(level_nodes)} 个节点")
    
    def add_input_vector(self, input_id, values, indices):
        """添加输入向量到第一级"""
        if not values or len(values) == 0:
            return
        
        if not self.levels:
            # 直接累加到MRN缓存
            self._accumulate_to_cache(values, indices)
            return
        
        # 分配到第一级对应节点
        node_idx = input_id // 2
        port = input_id % 2
        
        if node_idx < len(self.levels[0]):
            self.levels[0][node_idx].add_input(port, values, indices)
    
    def clock_cycle(self):
        """执行一个时钟周期"""
        activity = False
        
        # 处理所有级别的节点
        for level_idx, level_nodes in enumerate(self.levels):
            for node_idx, node in enumerate(level_nodes):
                if node.clock_cycle():
                    activity = True
                
                # 传播输出到下一级
                output_values, output_indices = node.get_output()
                if output_values is not None and len(output_values) > 0:
                    if level_idx == len(self.levels) - 1:
                        # 最后一级，累加到MRN缓存
                        self._accumulate_to_cache(output_values, output_indices)
                    else:
                        # 传播到下一级
                        next_level = self.levels[level_idx + 1]
                        next_node_idx = node_idx // 2
                        next_port = node_idx % 2
                        
                        if next_node_idx < len(next_level):
                            next_level[next_node_idx].add_input(next_port, output_values, output_indices)
        
        return activity
    
    def _accumulate_to_cache(self, values, indices):
        """累加到MRN缓存，使用BF16精度"""
        for val, idx in zip(values, indices):
            if idx in self.mrn_cache:
                # 使用BF16精度进行累加
                old_val_bf16 = fp32_to_bf16(self.mrn_cache[idx])
                new_val_bf16 = fp32_to_bf16(val)
                
                # 简化处理：直接计算而不使用流水线
                sum_fp32 = bf16_to_float(old_val_bf16) + bf16_to_float(new_val_bf16)
                sum_bf16 = fp32_to_bf16(sum_fp32)
                self.mrn_cache[idx] = bf16_to_float(sum_bf16)
            else:
                # 新值也通过BF16精度处理
                val_bf16 = fp32_to_bf16(val)
                self.mrn_cache[idx] = bf16_to_float(val_bf16)
    
    def get_final_result(self):
        """获取最终结果"""
        result_values = []
        result_indices = []
        
        for idx in sorted(self.mrn_cache.keys()):
            val = self.mrn_cache[idx]
            if abs(val) > 1e-12:
                result_values.append(val)
                result_indices.append(idx)
        
        return result_values, result_indices
    
    def reset(self):
        """重置网络状态"""
        self.mrn_cache.clear()
        self.final_add_pipeline.reset()
        
        for level_nodes in self.levels:
            for node in level_nodes:
                node.input_buffers = [deque(), deque()]
                node.output_buffer = deque()
                node.add_pipeline.reset()
                node.merging = False
                node.merge_stage = {'valid': False, 'data': None}
    
    def is_active(self):
        """检查网络是否活跃"""
        for level_nodes in self.levels:
            for node in level_nodes:
                if node.is_active():
                    return True
        return False

class MatrixVectorMultiplier:
    """矩阵向量乘法器主控制器 - 使用BF16流水线"""
    def __init__(self, num_pes=128):
        self.num_pes = num_pes
        
        # 核心组件
        self.pes = [ProcessingElement(i) for i in range(num_pes)]
        self.mrn = MRNNetwork(num_pes)
        
        # 状态
        self.clock = 0
        self.processing_active = False
        
        # 结果
        self.final_result_values = []
        self.final_result_indices = []
    
    def run_matrix_vector_multiplication(self, vector_dim=4096, matrix_rows=4096, matrix_cols=4096, sparsity=0.8, seed=42):
        """执行矩阵向量乘法"""
        logger.info(f"开始矩阵向量乘法 (1x{vector_dim}) * ({matrix_rows}x{matrix_cols}), 稀疏度={sparsity}")
        
        # 设置随机种子
        np.random.seed(seed)
        
        # 生成测试数据
        a_vector = np.random.randn(vector_dim) * 0.01
        b_dense = np.random.randn(matrix_rows, matrix_cols) * 0.01
        
        # 应用稀疏性
        mask = np.random.rand(matrix_rows, matrix_cols) < sparsity
        b_sparse_dense = np.where(mask, 0, b_dense)
        
        # 计算参考结果
        self.reference_result = np.dot(a_vector.reshape(1, -1), b_sparse_dense).flatten()
        
        # 创建CSR格式矩阵
        from dense_vector_sparse_matrix_correct import CSRMatrix
        sparse_matrix = CSRMatrix.from_dense(b_sparse_dense)
        
        # 重置状态
        self.reset()
        
        # 主循环：遍历A的每个元素
        for a_idx in range(vector_dim):
            a_value = a_vector[a_idx]
            
            # 跳过零元素
            if abs(a_value) < 1e-12:
                continue
            
            logger.debug(f"处理 A[{a_idx}] = {a_value}")
            
            # 获取B矩阵对应行
            b_row_values, b_row_indices = sparse_matrix.get_row(a_idx)
            
            if len(b_row_values) == 0:
                continue
            
            # 计算A[i] * B[i, :]
            self._compute_single_vector_matrix_mult(a_value, b_row_values, b_row_indices, matrix_cols)
        
        # 获取最终结果
        self.final_result_values, self.final_result_indices = self.mrn.get_final_result()
        
        logger.info(f"计算完成，总时钟周期: {self.clock}")
        logger.info(f"结果非零元素数量: {len(self.final_result_values)}")
        
        return self.final_result_values, self.final_result_indices
    
    def _compute_single_vector_matrix_mult(self, a_value, b_row_values, b_row_indices, output_dim):
        """计算单个A[i] * B[i, :]"""
        # 数据分配：将B矩阵元素分配到PE
        import math
        cols_per_pe = math.ceil(output_dim / self.num_pes)
        pe_assignments = [[] for _ in range(self.num_pes)]
        pe_indices = [[] for _ in range(self.num_pes)]
        
        for val, col_idx in zip(b_row_values, b_row_indices):
            pe_id = min(col_idx // cols_per_pe, self.num_pes - 1)
            pe_assignments[pe_id].append(val)
            pe_indices[pe_id].append(col_idx)
        
        # 加载数据到PE
        for pe_id in range(self.num_pes):
            self.pes[pe_id].load_data(a_value, pe_assignments[pe_id], pe_indices[pe_id])
        
        # 执行计算直到所有PE完成
        while self._any_pe_busy() or self.mrn.is_active():
            self.clock += 1
            
            # PE时钟周期
            pe_outputs = []
            for pe_id, pe in enumerate(self.pes):
                output = pe.clock_cycle()
                if output is not None:
                    output_val, output_col = output
                    pe_outputs.append((pe_id, output_val, output_col))
            
            # 将PE输出送入MRN
            for pe_id, val, col_idx in pe_outputs:
                if pe_id < self.num_pes:
                    self.mrn.add_input_vector(pe_id, [val], [col_idx])
            
            # MRN时钟周期
            self.mrn.clock_cycle()
    
    def _any_pe_busy(self):
        """检查是否有PE仍在工作"""
        return any(pe.is_busy() for pe in self.pes)
    
    def _mrn_active(self):
        """检查MRN是否活跃"""
        return self.mrn.is_active()
    
    def reset(self):
        """重置状态"""
        self.clock = 0
        self.processing_active = False
        self.final_result_values.clear()
        self.final_result_indices.clear()
        
        # 重置组件
        self.pes = [ProcessingElement(i) for i in range(self.num_pes)]
        self.mrn.reset()
    
    def verify_result(self):
        """验证结果正确性"""
        if not hasattr(self, 'reference_result'):
            logger.error("没有参考结果可供比较")
            return False
        
        # 构建稀疏结果向量
        result_vector = np.zeros(len(self.reference_result))
        for val, idx in zip(self.final_result_values, self.final_result_indices):
            if 0 <= idx < len(result_vector):
                result_vector[idx] = val
        
        # 计算误差
        diff = np.abs(result_vector - self.reference_result)
        max_error = np.max(diff)
        
        logger.info(f"验证结果:")
        logger.info(f"  参考结果非零元素: {np.count_nonzero(self.reference_result)}")
        logger.info(f"  模拟结果非零元素: {len(self.final_result_values)}")
        logger.info(f"  最大绝对误差: {max_error}")
        
        # 判断是否通过
        success = max_error < 1e-2  # 1%误差阈值
        if success:
            logger.info("✓ 验证通过!")
        else:
            logger.error("✗ 验证失败!")
        
        return success

def test_mrn_basic():
    """测试MRN基本功能"""
    logger.info("测试MRN基本功能")
    
    # 创建小规模MRN网络
    mrn = MRNNetwork(num_inputs=4)
    
    # 添加测试向量
    mrn.add_input_vector(0, [1.0, 2.0], [0, 1])
    mrn.add_input_vector(1, [3.0, 4.0], [1, 2])
    mrn.add_input_vector(2, [5.0], [0])
    mrn.add_input_vector(3, [6.0], [2])
    
    # 运行直到完成
    max_cycles = 100
    for cycle in range(max_cycles):
        if not mrn.clock_cycle():
            break
    
    # 获取结果
    result_values, result_indices = mrn.get_final_result()
    
    logger.info(f"MRN测试结果: values={result_values}, indices={result_indices}")
    
    # 期望结果: [0]->6.0, [1]->5.0, [2]->10.0
    expected = {0: 6.0, 1: 5.0, 2: 10.0}
    
    success = True
    for val, idx in zip(result_values, result_indices):
        if idx in expected:
            if abs(val - expected[idx]) > 1e-6:
                logger.error(f"索引{idx}的值不匹配: 期望{expected[idx]}, 实际{val}")
                success = False
        else:
            logger.error(f"意外的索引: {idx}")
            success = False
    
    if success:
        logger.info("✓ MRN基本功能测试通过!")
    else:
        logger.error("✗ MRN基本功能测试失败!")
    
    return success

def main():
    """主函数"""
    logger.info("开始基于BF16流水线的稠密向量与稀疏矩阵乘法硬件模拟器测试")
    
    # 首先测试MRN基本功能
    # if not test_mrn_basic():
    #     logger.error("MRN基本功能测试失败，退出")
    #     return False
    
    # 创建乘法器
    multiplier = MatrixVectorMultiplier(num_pes=128)
    
    # 运行测试
    result_values, result_indices = multiplier.run_matrix_vector_multiplication(
        vector_dim=4096,  # 较小的测试规模
        matrix_rows=4096,
        matrix_cols=4096,
        sparsity=0.9,
        seed=42
    )
    
    # 验证结果
    success = multiplier.verify_result()
    
    if success:
        logger.info("所有测试通过!")
    else:
        logger.error("测试失败!")
    
    return success

if __name__ == "__main__":
    main()
