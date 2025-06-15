#!/usr/bin/env python3
"""
MFIU快速测试
可以用 uv run gpt2_sim/trapezoid_module/test/test_mfiu_quick.py 运行
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import time
from scipy.sparse import csr_matrix
from gpt2_sim.trapezoid_module.module.mfiu_sim import MFIUPipeline
from gpt2_sim.trapezoid_module.module.mfiu_sim_dense import MFIUPipelineDenseA
from gpt2_sim.trapezoid_module.utils import get_values_offset_mask

def quick_performance_test():
    """快速性能测试"""
    print("🚀 MFIU快速性能测试")
    print("-" * 40)
    
    # 测试参数
    M, K, N = 1, 5, 5
    
    # 生成测试数据
    print(f"生成测试矩阵 ({M}x{K}) × ({K}x{N})...")
    np.random.seed(42)
    A = np.ones((M, K), dtype=int)  # 稠密A
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])  # 稀疏B
    
    print(f"A矩阵稠密度: {np.count_nonzero(A) / A.size * 100:.1f}%")
    print(f"B矩阵稠密度: {np.count_nonzero(B) / B.size * 100:.1f}%")
    
    # 准备数据
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)
    values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    values_B, offset_B, masks_B = get_values_offset_mask(csr_B)
    
    width = M * N
    bit_width = K
    
    # 测试1: 原始MFIU
    print("\n1️⃣ 测试原始MFIU...")
    original_mfiu = MFIUPipeline(width, bit_width)
    
    start_time = time.time()
    original_results = original_mfiu.run_pipeline(
        [masks_A], [masks_B], [offset_A], [offset_B], 
        len(values_A), len(values_B), max_cycles=30, print_states=False
    )
    original_time = time.time() - start_time
    
    original_valid_cycles = sum(1 for r in original_results if r["valid"])
    print(f"✅ 原始MFIU: {original_time*1000:.1f}ms, {original_valid_cycles}个有效周期")
    
    # 测试2: 稠密A优化MFIU
    print("\n2️⃣ 测试稠密A优化MFIU...")
    dense_mfiu = MFIUPipelineDenseA(width, bit_width)
    
    start_time = time.time()
    dense_results = dense_mfiu.run_pipeline(
        mask_B_cols=[masks_B], 
        offset_B_cols=[offset_B], 
        len_values_A=len(values_A), 
        len_values_B=len(values_B), 
        max_cycles=30,
        print_states=False
    )
    dense_time = time.time() - start_time
    
    dense_valid_cycles = sum(1 for r in dense_results if r["valid"])
    print(f"✅ 稠密A优化: {dense_time*1000:.1f}ms, {dense_valid_cycles}个有效周期")
    
    # 性能对比
    speedup = original_time / dense_time if dense_time > 0 else float('inf')
    
    print(f"\n📊 性能对比:")
    print(f"  时间提升: {speedup:.2f}x")
    print(f"  时间节省: {(original_time - dense_time)*1000:.1f}ms")
    print(f"  有效周期对比: 原始{original_valid_cycles} vs 优化{dense_valid_cycles}")
    
    # 验证结果一致性
    original_output = None
    dense_output = None
    
    for res in original_results:
        if res["valid"]:
            original_output = res["output"]
            break
    
    for res in dense_results:
        if res["valid"]:
            dense_output = res["output"]
            break
    
    if original_output and dense_output:
        print(original_output)
        print(dense_output)
        if (np.array_equal(original_output[0], dense_output[0]) and 
            np.array_equal(original_output[1], dense_output[1])):
            print(f"  结果验证: ✅ 输出完全一致")
        else:
            print(f"  结果验证: ❌ 输出不一致")
    else:
        print(f"  结果验证: ⚠️ 缺少有效输出")
    
    return {
        "speedup": speedup,
        "original_time": original_time,
        "dense_time": dense_time,
        "results_match": original_output and dense_output and 
                        np.array_equal(original_output[0], dense_output[0])
    }

def test_mfiu_quick_fix():
    """快速测试修复后的MFIU稠密A优化版本"""
    print("🔧 测试MFIU稠密A优化版本修复")
    print("=" * 40)
    
    # 简单测试数据
    M, K, N = 1, 4, 4
    bit_width = 4
    width = M * N  # 1 * 4 = 4
    
    # 创建稠密A矩阵（全1）
    A = np.ones((M, K), dtype=int)
    # 创建简单B矩阵
    B = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1], 
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    
    print(f"测试矩阵:")
    print(f"A (稠密): \n{A}")
    print(f"B: \n{B}")
    
    # 转换为CSR格式
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)
    
    # 获取mask和offset
    values_A, offset_A, mask_A = get_values_offset_mask(csr_A)
    values_B, offset_B, mask_B = get_values_offset_mask(csr_B)
    
    print(f"\n预处理结果:")
    print(f"A mask: {mask_A}")
    print(f"A offset: {offset_A}")
    print(f"B mask: {mask_B}")
    print(f"B offset: {offset_B}")
    
    # 测试原始版本
    print(f"\n🔄 运行原始MFIU...")
    original_mfiu = MFIUPipeline(width=width, bit_width=bit_width)
    start_time = time.time()
    original_results = original_mfiu.run_pipeline(
        mask_A_rows=[mask_A],
        mask_B_cols=[mask_B],
        offset_A_rows=[offset_A],
        offset_B_cols=[offset_B],
        len_values_A=len(values_A),
        len_values_B=len(values_B),
        max_cycles=20,
        print_states=False
    )
    original_time = time.time() - start_time
    
    # 测试优化版本
    print(f"🚀 运行稠密A优化MFIU...")
    dense_mfiu = MFIUPipelineDenseA(width=width, bit_width=bit_width)
    start_time = time.time()
    dense_results = dense_mfiu.run_pipeline(
        mask_B_cols=[mask_B],
        offset_B_cols=[offset_B],
        len_values_A=len(values_A),
        len_values_B=len(values_B),
        max_cycles=20,
        print_states=False
    )
    dense_time = time.time() - start_time
    
    # 比较结果
    print(f"\n📊 结果比较:")
    print(f"原始版本周期数: {len(original_results)}")
    print(f"优化版本周期数: {len(dense_results)}")
    print(f"原始版本时间: {original_time:.6f}s")
    print(f"优化版本时间: {dense_time:.6f}s")
    print(f"性能提升: {original_time/dense_time:.2f}x" if dense_time > 0 else "N/A")
    
    # 验证输出一致性
    original_valid_outputs = [r for r in original_results if r['valid']]
    dense_valid_outputs = [r for r in dense_results if r['valid']]
    
    print(f"\n有效输出数量:")
    print(f"原始版本: {len(original_valid_outputs)}")
    print(f"优化版本: {len(dense_valid_outputs)}")
    
    if len(original_valid_outputs) == len(dense_valid_outputs):
        outputs_match = True
        for i, (orig, dense) in enumerate(zip(original_valid_outputs, dense_valid_outputs)):
            if orig['output'] != dense['output']:
                outputs_match = False
                print(f"⚠️  周期 {i} 输出不匹配:")
                print(f"  原始: {orig['output']}")
                print(f"  优化: {dense['output']}")
                break
        
        if outputs_match:
            print("✅ 结果验证: 输出完全一致!")
        else:
            print("❌ 结果验证: 输出不一致")
    else:
        print("❌ 结果验证: 有效输出数量不同")
    
    return original_time, dense_time

if __name__ == "__main__":
    result = quick_performance_test()
    if result["speedup"] > 1.5:
        print(f"\n🎉 优化成功！获得 {result['speedup']:.2f}x 性能提升")
    elif result["speedup"] > 1.0:
        print(f"\n👍 有一定优化效果，获得 {result['speedup']:.2f}x 性能提升")
    else:
        print(f"\n🤔 优化效果不明显，可能需要进一步调整")

    test_mfiu_quick_fix() 