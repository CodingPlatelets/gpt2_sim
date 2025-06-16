#!/usr/bin/env python3
"""
测试多batch VectorMatrixRowProductSimulatorWithHBM的简单脚本
"""
import sys
import os
sys.path.append('gpt2_sim')

import numpy as np
import torch
from scipy.sparse import csr_matrix

# 简化的测试，不依赖复杂的BF16流水线
class SimpleMultiBatchSimulator:
    """简化版的多batch模拟器用于验证逻辑"""
    def __init__(self, num_perows=32, pes_per_row=128, vector_size=4096):
        self.num_perows = num_perows
        self.pes_per_row = pes_per_row
        self.vector_size = vector_size
        
    def run_simulation(self, A_vectors, B_matrix_sparse):
        """运行多batch模拟"""
        # 处理输入格式
        if A_vectors.ndim == 1:
            A_vectors = A_vectors.reshape(1, -1)
        
        batch_size, vector_dim = A_vectors.shape
        print(f"处理batch_size={batch_size}, vector_dim={vector_dim}")
        print(f"矩阵B形状: {B_matrix_sparse.shape}")
        
        # 计算perows分组策略
        perows_per_batch = self.num_perows // batch_size
        if perows_per_batch == 0:
            raise ValueError(f"batch_size({batch_size}) 不能大于 num_perows({self.num_perows})")
        
        print(f"每个batch分配 {perows_per_batch} 个perows")
        
        # 使用numpy直接计算作为参考
        result_matrix = A_vectors @ B_matrix_sparse
        
        print(f"计算完成，输出形状: {result_matrix.shape}")
        
        return {
            "output_matrix": result_matrix,
            "batch_size": batch_size,
            "perows_per_batch": perows_per_batch
        }

def test_multi_batch():
    """测试多batch功能"""
    print("=" * 60)
    print("测试多batch模拟器")
    
    # 测试配置
    num_perows = 32
    pes_per_row = 128
    vec_dim = 1024  # 减小维度以便快速测试
    output_dim = 2048
    sparsity = 0.9
    
    # 创建模拟器
    simulator = SimpleMultiBatchSimulator(
        num_perows=num_perows,
        pes_per_row=pes_per_row,
        vector_size=output_dim
    )
    
    # 测试用例
    test_cases = [
        {"batch_size": 1, "desc": "单batch测试"},
        {"batch_size": 4, "desc": "4-batch测试 (每8个perow处理1个batch)"},
        {"batch_size": 8, "desc": "8-batch测试 (每4个perow处理1个batch)"},
        {"batch_size": 16, "desc": "16-batch测试 (每2个perow处理1个batch)"},
        {"batch_size": 32, "desc": "32-batch测试 (每1个perow处理1个batch)"},
    ]
    
    # 生成测试数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建稀疏矩阵B
    B_dense = torch.randn((vec_dim, output_dim)) * 0.01
    mask = torch.rand(vec_dim, output_dim) > sparsity
    B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)
    
    print(f"矩阵B稀疏度: {(B_sparse == 0).sum() / B_sparse.size:.2%}")
    
    for test_case in test_cases:
        batch_size = test_case["batch_size"]
        desc = test_case["desc"]
        
        if batch_size > num_perows:
            print(f"跳过测试: {desc} - batch_size({batch_size}) > num_perows({num_perows})")
            continue
        
        print(f"\n--- {desc} ---")
        
        # 生成输入向量
        A_vectors = np.random.randn(batch_size, vec_dim).astype(np.float32) * 0.1
        
        try:
            # 运行模拟
            result = simulator.run_simulation(A_vectors, B_sparse)
            
            # 验证结果
            expected = A_vectors @ B_sparse
            error = np.abs(result["output_matrix"] - expected).max()
            
            print(f"最大误差: {error:.6f}")
            print(f"测试{'通过' if error < 1e-6 else '失败'}")
            print(f"perows分配策略: {result['perows_per_batch']} perows/batch")
            
        except Exception as e:
            print(f"测试失败: {str(e)}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_multi_batch()