#!/usr/bin/env python3
"""
Trapezoid模块测试
可以用 uv run gpt2_sim/trapezoid_module/test_trapezoid.py 运行
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from gpt2_sim.trapezoid_module.trapezoid_sim import TrapezoidPipeline
from gpt2_sim.trapezoid_module.utils import naive_matmul
from gpt2_sim.hbm.csr_hbm_values_base import store_csr_in_simple_blocks
from scipy.sparse import csr_matrix


def test_simple_case():
    """测试简单的矩阵乘法案例"""
    print("\n===== 测试简单矩阵乘法 =====")

    # 定义简单的测试矩阵
    A = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
    B = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])

    M, K = A.shape
    _, N = B.shape

    # 计算正确结果作为参考
    expected_C = naive_matmul(A, B)
    print("预期结果矩阵：")
    print(expected_C)

    # 创建TrapezoidPipeline实例
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=4)

    # 运行流水线
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_with_bf16([(A, B)], print_states=True)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")
    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])

def test_hbm_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=100000, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])

def test_hbm_small_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 128, 128

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=32)

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 32)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=8000)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])


def test_hbm_multi_small_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)

    num_trapezoids = 128
    trapezoid_list = []

    for i in range(num_trapezoids):
        trap = TrapezoidPipeline(M, K, N, 128)
        trapezoid_list.append(trap)

    main_trap = trapezoid_list[0]

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = main_trap.run_pipeline_hbm_multi_with_bf16([A], hbm_data_lists, trapezoid_list, max_cycles=100000, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["combined_c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["combined_c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["combined_c_matrix"])

def test_sparse_matrices():
    """测试稀疏矩阵乘法"""
    print("\n===== 测试稀疏矩阵乘法 =====")

    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(A)
    print(B)

    # 计算参考结果
    expected_C = naive_matmul(A, B)

    # 创建TrapezoidPipeline实例
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    # 运行流水线
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_with_bf16([(A, B)], max_cycles=100000, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])


def test_multiple_matrices():
    """测试多个矩阵乘法"""
    print("\n===== 测试多个矩阵乘法 =====")

    # 准备测试数据
    test_cases = []
    for i in range(3):
        size = 4 + i
        A = np.random.randint(0, 3, size=(2, size))
        B = np.random.randint(0, 3, size=(size, 3))
        test_cases.append((A, B))

    # 计算参考结果
    expected_results = []
    for A, B in test_cases:
        expected_results.append(naive_matmul(A, B))

    # 创建TrapezoidPipeline实例
    max_M = max(A.shape[0] for A, B in test_cases)
    max_K = max(A.shape[1] for A, B in test_cases)
    max_N = max(B.shape[1] for A, B in test_cases)

    pipeline = TrapezoidPipeline(M=max_M, K=max_K, N=max_N, PE_num=8)

    # 运行流水线
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_with_bf16(test_cases, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    # 验证结果
    print("\n结果验证：")
    all_correct = True
    for i, (expected, actual) in enumerate(zip(expected_results, result["c_matrices"])):
        if np.allclose(expected, actual, rtol=1e-2, atol=1e-2):
            print(f"✓ 矩阵 {i}: 正确")
        else:
            print(f"✗ 矩阵 {i}: 错误")
            all_correct = False

    if all_correct:
        print("✓ 所有矩阵计算正确！")
    else:
        print("✗ 有矩阵计算错误！")


if __name__ == "__main__":
    print("🚀 开始Trapezoid模块测试")
    
    try:
        # 运行简单测试
        test_simple_case()
        
        print("\n✅ 简单测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
