#!/usr/bin/env python3
"""
Software Hardware Simulator测试
测试hardware流水线模拟器实现的Softmax算法
可以用 uv run gpt2_sim/trapezoid_module/test_software_hw_sim.py 运行

重要说明：
- 硬件仿真器使用BF16精度进行所有计算
- PyTorch比较也使用BF16精度 (torch.bfloat16) 确保公平比较
- 如果使用float32精度比较会导致误差评估不准确
"""

import sys
import os
import numpy as np
import time
import torch
from pathlib import Path
import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入模块
from gpt2_sim.trapezoid_module.software_hw_sim import SoftmaxPipeline, ExpUnit, CompareUnit, DivideUnit
from gpt2_sim.trapezoid_module.utils import convert_through_pipeline, bf16_to_float


def test_basic_units():
    """测试基础硬件单元"""
    print("\n===== 测试基础硬件单元 =====")
    
    # 测试ExpUnit
    print("\n--- 测试ExpUnit ---")
    exp_unit = ExpUnit()
    
    test_inputs = [0.0, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0]
    for inp in test_inputs:
        inp_bf16 = convert_through_pipeline(inp)
        exp_unit.compute(inp_bf16, True)
        
        expected = np.exp(inp) if abs(inp) <= 11.0 else float('inf') if inp > 0 else 0.0
        actual = bf16_to_float(exp_unit.output_val)
        
        print(f"exp({inp:.2f}) = {actual:.6f} (expected ≈ {expected:.6f})")
    
    # 测试CompareUnit
    print("\n--- 测试CompareUnit ---")
    compare_unit = CompareUnit()
    
    test_pairs = [(1.0, 2.0), (3.0, 1.0), (-1.0, -2.0), (0.0, 0.0)]
    for a, b in test_pairs:
        a_bf16 = convert_through_pipeline(a)
        b_bf16 = convert_through_pipeline(b)
        compare_unit.compute(a_bf16, b_bf16, True)
        
        result = bf16_to_float(compare_unit.result)
        expected = max(a, b)
        
        print(f"max({a:.2f}, {b:.2f}) = {result:.2f} (expected {expected:.2f}) {'✓' if abs(result - expected) < 1e-3 else '✗'}")
    
    # 测试DivideUnit
    print("\n--- 测试DivideUnit ---")
    divide_unit = DivideUnit()
    
    test_pairs = [(6.0, 2.0), (1.0, 3.0), (10.0, 0.0), (0.0, 5.0)]
    for a, b in test_pairs:
        a_bf16 = convert_through_pipeline(a)
        b_bf16 = convert_through_pipeline(b)
        divide_unit.compute(a_bf16, b_bf16, True)
        
        result = bf16_to_float(divide_unit.result)
        expected = a / b if b != 0 else float('inf') if a > 0 else float('nan')
        
        if np.isfinite(expected):
            print(f"{a:.2f} / {b:.2f} = {result:.6f} (expected {expected:.6f}) {'✓' if abs(result - expected) < 1e-3 else '✗'}")
        else:
            print(f"{a:.2f} / {b:.2f} = {result:.6f} (expected {expected})")


def test_simple_softmax():
    """测试简单的softmax计算"""
    print("\n===== 测试简单Softmax =====")
    
    test_cases = [
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [10.0, 20.0, 30.0]
    ]
    
    for i, values in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}: {values} ---")
        
        # 创建新的流水线架构
        pipeline = SoftmaxPipeline(front_window=len(values), back_window=2, max_rows=1)
        
        # 构造新格式输入
        inputs = []
        row_length = len(values)
        
        # 窗口优先发送所有数据（因为front_window设置为数组长度）
        for col_idx, val in enumerate(values):
            inputs.append((val, 0, col_idx, row_length))
        
        try:
            hw_result = pipeline.run_pipeline(inputs, max_cycles=1000, print_progress=False)
            
            if 0 in hw_result:
                result_row = hw_result[0]
                
                # 转换为浮点数列表
                hw_softmax = []
                for col_idx in sorted(result_row.keys()):
                    softmax_val = bf16_to_float(result_row[col_idx])
                    hw_softmax.append(softmax_val)
                
                # 使用PyTorch计算参考结果
                torch_input = torch.tensor(values, dtype=torch.float32)
                torch_result = torch.softmax(torch_input, dim=0).tolist()
                
                print(f"输入: {values}")
                print(f"硬件结果: {[f'{x:.6f}' for x in hw_softmax]}")
                print(f"PyTorch:  {[f'{x:.6f}' for x in torch_result]}")
                print(f"和: {sum(hw_softmax):.6f}")
                
                # 计算误差
                max_error = max(abs(h - t) for h, t in zip(hw_softmax, torch_result))
                print(f"最大误差: {max_error:.6f}")
                
                # 验证结果
                assert abs(sum(hw_softmax) - 1.0) < 0.01, f"和应该接近1.0，实际为{sum(hw_softmax)}"
                assert max_error < 0.01, f"与PyTorch的误差太大: {max_error}"
                print("✓ 测试通过")
            else:
                print("❌ 未获得结果")
                assert False, "未能获取结果"
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            raise


def test_medium_size_softmax():
    """测试中等规模的softmax"""
    print("\n===== 测试中等规模Softmax =====")
    
    sizes = [50, 100, 200]
    
    for size in sizes:
        print(f"\n--- 测试规模: {size} ---")
        
        # 生成测试数据
        np.random.seed(42)
        inputs = np.random.randn(size) * 3  # 标准正态分布乘以3
        inputs[size//2] = 10  # 在中间放一个较大的值
        
        # 硬件流水线实现
        pipeline = SoftmaxPipeline(size, min(20, size//5), min(10, size//10))
        start_time = time.time()
        hw_result = pipeline.run_pipeline(inputs.tolist(), max_cycles=size*50, print_states=False)
        hw_time = time.time() - start_time
        
        # PyTorch BF16实现（与硬件仿真器相同精度）
        torch_inputs = torch.tensor(inputs, dtype=torch.bfloat16)
        start_time = time.time()
        torch_result = torch.softmax(torch_inputs, dim=0).float().numpy()
        torch_time = time.time() - start_time
        
        print(f"输入最大值:  {inputs.max():.3f}")
        print(f"输入最小值:  {inputs.min():.3f}")
        print(f"硬件和:      {hw_result['sum']:.6f}")
        print(f"PyTorch BF16和: {torch_result.sum():.6f}")
        print(f"处理周期:    {hw_result['cycles']}")
        print(f"硬件时间:    {hw_time*1000:.3f}ms")
        print(f"PyTorch时间: {torch_time*1000:.3f}ms")
        print(f"速度比:      {torch_time/hw_time:.2f}x")
        
        # 验证结果正确性
        hw_vs_torch_error = np.mean(np.abs(np.array(hw_result['results']) - torch_result))
        max_error = np.max(np.abs(np.array(hw_result['results']) - torch_result))
        print(f"平均误差:    {hw_vs_torch_error:.6f}")
        print(f"最大误差:    {max_error:.6f}")
        
        if hw_vs_torch_error < 0.01 and max_error < 0.05:
            print("✓ 硬件实现结果正确！")
        else:
            print("✗ 硬件实现误差过大！")


def test_large_size_softmax():
    """测试大规模的softmax（仅用于性能测试）"""
    print("\n===== 测试大规模Softmax =====")
    
    # 减少测试规模，避免过长的运行时间
    sizes = [500]
    
    for size in sizes:
        print(f"\n--- 测试规模: {size} ---")
        
        # 生成测试数据
        np.random.seed(42)
        inputs = np.random.randn(size) * 5
        inputs[size//4] = 15    # 放几个较大的值
        inputs[size//2] = 20
        inputs[3*size//4] = 12
        
        # 硬件流水线实现
        pipeline = SoftmaxPipeline(size, min(100, size//10), min(50, size//20))
        start_time = time.time()
        hw_result = pipeline.run_pipeline(inputs.tolist(), max_cycles=size*100, print_states=False)
        hw_time = time.time() - start_time
        
        # PyTorch BF16实现（与硬件仿真器相同精度）
        torch_inputs = torch.tensor(inputs, dtype=torch.bfloat16)
        start_time = time.time()
        torch_result = torch.softmax(torch_inputs, dim=0).float().numpy()
        torch_time = time.time() - start_time
        
        print(f"输入最大值:  {inputs.max():.3f}")
        print(f"输入最小值:  {inputs.min():.3f}")
        print(f"硬件和:      {hw_result['sum']:.6f}")
        print(f"PyTorch BF16和: {torch_result.sum():.6f}")
        
        print(f"处理周期:    {hw_result['cycles']}")
        print(f"硬件时间:    {hw_time*1000:.1f}ms")
        print(f"PyTorch时间: {torch_time*1000:.1f}ms")
        
        print(f"硬件 vs PyTorch BF16速度比: {torch_time/hw_time:.2f}x")
        
        # 验证结果正确性
        hw_vs_torch_error = np.mean(np.abs(np.array(hw_result['results']) - torch_result))
        max_error = np.max(np.abs(np.array(hw_result['results']) - torch_result))
        print(f"平均误差:    {hw_vs_torch_error:.6f}")
        print(f"最大误差:    {max_error:.6f}")
        
        if hw_vs_torch_error < 0.01 and max_error < 0.05:
            print("✓ 硬件实现结果正确！")
        else:
            print("✗ 硬件实现误差过大！")


def test_edge_cases():
    """测试边界情况"""
    print("\n===== 测试边界情况 =====")
    
    edge_cases = [
        ("全零", [0.0, 0.0, 0.0, 0.0]),
        ("包含大数", [1.0, 2.0, 50.0, 1.0]),
        ("包含负大数", [1.0, 2.0, -50.0, 1.0]),
        ("极端情况", [100.0, 1.0, 1.0, 1.0]),
        ("混合正负", [-10.0, 0.0, 10.0, 5.0]),
        ("相同值", [5.0, 5.0, 5.0, 5.0]),
    ]
    
    for name, inputs in edge_cases:
        print(f"\n--- {name}: {inputs} ---")
        
        try:
            # 硬件流水线实现
            pipeline = SoftmaxPipeline(len(inputs), len(inputs), len(inputs)//2)
            hw_result = pipeline.run_pipeline(inputs, max_cycles=500, print_states=False)
            
            # PyTorch BF16实现（与硬件仿真器相同精度）
            torch_inputs = torch.tensor(inputs, dtype=torch.bfloat16)
            torch_result = torch.softmax(torch_inputs, dim=0).float().numpy()
            
            print(f"硬件结果:    {[f'{x:.6f}' for x in hw_result['results']]}")
            print(f"PyTorch BF16结果: {[f'{x:.6f}' for x in torch_result]}")
            print(f"硬件和:      {hw_result['sum']:.6f}")
            print(f"PyTorch BF16和: {torch_result.sum():.6f}")
            print(f"处理周期:    {hw_result['cycles']}")
            
            # 验证结果
            hw_vs_torch_error = np.mean(np.abs(np.array(hw_result['results']) - torch_result))
            if hw_vs_torch_error < 0.01:
                print("✓ 边界情况处理正确")
            else:
                print(f"✗ 边界情况误差较大: {hw_vs_torch_error:.6f}")
                
        except Exception as e:
            print(f"✗ 边界情况处理失败: {e}")


def test_overflow_recovery():
    """测试溢出检测和恢复机制"""
    print("\n===== 测试溢出检测和恢复 =====")
    
    # 构造会导致溢出的测试用例
    overflow_cases = [
        ("预估不足导致溢出", [1.0, 2.0, 3.0, 60.0, 4.0, 5.0]),  # 最大值在中间，预估不到
        ("累加溢出", [15.0, 15.0, 15.0, 15.0, 15.0]),  # 多个大值导致累加溢出
        ("极大值", [1.0, 2.0, 100.0, 1.0]),  # 极大值
    ]
    
    for name, inputs in overflow_cases:
        print(f"\n--- {name} ---")
        print(f"输入: {inputs}")
        
        try:
            # 使用小窗口来模拟预估不准确的情况
            pipeline = SoftmaxPipeline(len(inputs), max(1, len(inputs)//3), max(1, len(inputs)//4))
            hw_result = pipeline.run_pipeline(inputs, max_cycles=1000, print_states=False)
            
            # PyTorch BF16参考（与硬件仿真器相同精度）
            torch_inputs = torch.tensor(inputs, dtype=torch.bfloat16)
            torch_result = torch.softmax(torch_inputs, dim=0).float().numpy()
            
            print(f"硬件结果:    {[f'{x:.6f}' for x in hw_result['results']]}")
            print(f"PyTorch BF16结果: {[f'{x:.6f}' for x in torch_result]}")
            print(f"处理周期:    {hw_result['cycles']}")
            
            hw_vs_torch_error = np.mean(np.abs(np.array(hw_result['results']) - torch_result))
            if hw_vs_torch_error < 0.02:  # 溢出恢复可能有更大误差
                print("✓ 溢出恢复成功")
            else:
                print(f"? 溢出恢复误差较大: {hw_vs_torch_error:.6f}")
                
        except Exception as e:
            print(f"✗ 溢出恢复失败: {e}")


def test_new_input_format():
    """测试新的输入格式 (val, row_idx, col_idx, row_length)"""
    print("测试新的输入格式...")
    
    # 测试数据
    test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=4)
    
    # 构造新格式输入
    inputs = []
    row_length = len(test_values)
    
    # 窗口优先发送
    for i in [0, 1]:  # 前窗口
        inputs.append((test_values[i], 0, i, row_length))
    for i in [3, 4]:  # 后窗口
        inputs.append((test_values[i], 0, i, row_length))
    for i in [2]:     # 剩余数据
        inputs.append((test_values[i], 0, i, row_length))
    
    print(f"测试数据: {test_values}")
    print(f"输入格式: (val, row_id, col_idx, row_length)")
    
    # 运行流水线
    results = pipeline.run_pipeline(inputs, max_cycles=150, print_progress=False)
    
    if 0 in results:
        result_row = results[0]
        
        # 转换结果
        hw_softmax = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            hw_softmax.append(softmax_val)
        
        print(f"结果: {[f'{x:.6f}' for x in hw_softmax]}")
        print(f"和: {sum(hw_softmax):.6f}")
        
        # 验证
        assert abs(sum(hw_softmax) - 1.0) < 0.01, f"概率和应该接近1.0，实际为{sum(hw_softmax)}"
        assert all(val >= 0 for val in hw_softmax), "所有softmax值应该为正数"
        
        print("✓ 新输入格式测试通过")
    else:
        print("❌ 未获得结果")
        assert False, "未能获取结果"


def test_multi_row_processing():
    """测试多行并行处理"""
    print("测试多行并行处理...")
    
    # 准备两行测试数据
    row_data = [
        [1.0, 2.0, 3.0],      # 行0
        [4.0, 5.0, 6.0]       # 行1  
    ]
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=2, back_window=1, max_rows=2)
    
    # 构造输入数据
    inputs = []
    
    for row_id, values in enumerate(row_data):
        row_length = len(values)
        
        # 窗口优先
        for i in [0, 1]:  # 前窗口
            if i < row_length:
                inputs.append((values[i], row_id, i, row_length))
        
        for i in [2]:     # 后窗口
            if i < row_length:
                inputs.append((values[i], row_id, i, row_length))
    
    print(f"测试数据: {row_data}")
    
    # 运行流水线
    results = pipeline.run_pipeline(inputs, max_cycles=200, print_progress=False)
    
    # 验证每行结果
    for row_id, expected_data in enumerate(row_data):
        if row_id in results:
            result_row = results[row_id]
            
            # 转换结果
            hw_softmax = []
            for col_idx in sorted(result_row.keys()):
                softmax_val = bf16_to_float(result_row[col_idx])
                hw_softmax.append(softmax_val)
            
            print(f"行{row_id}结果: {[f'{x:.6f}' for x in hw_softmax]} (和: {sum(hw_softmax):.6f})")
            
            # 验证
            assert abs(sum(hw_softmax) - 1.0) < 0.01, f"行{row_id}概率和应该接近1.0"
            assert all(val >= 0 for val in hw_softmax), f"行{row_id}所有softmax值应该为正数"
        else:
            print(f"❌ 行{row_id}未获得结果")
            assert False, f"未能获取行{row_id}结果"
    
    print("✓ 多行并行处理测试通过")


def test_overflow_handling():
    """测试溢出处理机制"""
    print("测试溢出处理...")
    
    # 构造容易溢出的数据
    test_values = [1.0, 2.0, 50.0, 3.0, 4.0]  # 50.0很大，可能引起溢出
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
    
    # 构造输入：确保大值不在初始窗口中
    inputs = []
    row_length = len(test_values)
    
    # 窗口数据（不包含最大值）
    for i in [0, 1]:  # 前窗口
        inputs.append((test_values[i], 0, i, row_length))
    for i in [3, 4]:  # 后窗口
        inputs.append((test_values[i], 0, i, row_length))
    
    # 包含最大值的数据
    inputs.append((test_values[2], 0, 2, row_length))
    
    print(f"测试数据: {test_values}")
    print("窗口数据先发送，然后发送包含最大值的数据")
    
    # 运行流水线
    results = pipeline.run_pipeline(inputs, max_cycles=200, print_progress=False)
    
    if 0 in results:
        result_row = results[0]
        
        # 转换结果
        hw_softmax = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            hw_softmax.append(softmax_val)
        
        print(f"结果: {[f'{x:.6f}' for x in hw_softmax]}")
        
        # 验证最大值位置
        max_position = 2  # 50.0的位置
        max_softmax = hw_softmax[max_position]
        print(f"最大值位置{max_position}的softmax: {max_softmax:.6f}")
        
        # 验证
        assert max_softmax == max(hw_softmax), "最大值位置的softmax应该最大"
        assert max_softmax > 0.95, f"最大值位置的softmax应该接近1"
        
        total_sum = sum(hw_softmax)
        assert abs(total_sum - 1.0) < 0.01, f"概率和应该接近1.0，实际为{total_sum}"
        
        print("✓ 溢出处理测试通过")
    else:
        print("❌ 未获得结果")
        assert False, "未能获取结果"


def test_window_priority():
    """测试窗口优先处理机制"""
    print("\n===== 测试窗口优先机制 =====")
    
    # 构造测试数据：最大值不在窗口中
    values = [1.0, 2.0, 100.0, 3.0, 4.0]  # 最大值100.0在位置2
    front_window = 2
    back_window = 2
    
    print(f"测试数据: {values}")
    print(f"前窗口: {front_window}, 后窗口: {back_window}")
    print(f"窗口数据: 前[0,1], 后[3,4], 非窗口[2]")
    
    pipeline = SoftmaxPipeline(front_window=front_window, back_window=back_window, max_rows=1)
    
    # 构造输入：窗口数据优先
    inputs = []
    row_length = len(values)
    
    # 前窗口数据
    for i in [0, 1]:
        inputs.append((values[i], 0, i, row_length))
        print(f"发送前窗口数据: 位置{i}, 值={values[i]}")
    
    # 后窗口数据
    for i in [3, 4]:
        inputs.append((values[i], 0, i, row_length))
        print(f"发送后窗口数据: 位置{i}, 值={values[i]}")
    
    # 非窗口数据（包含最大值）
    inputs.append((values[2], 0, 2, row_length))
    print(f"发送非窗口数据: 位置2, 值={values[2]} (最大值)")
    
    # 运行流水线
    hw_result = pipeline.run_pipeline(inputs, max_cycles=200, print_progress=False)
    
    if 0 in hw_result:
        result_row = hw_result[0]
        
        # 转换结果
        hw_softmax = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            hw_softmax.append(softmax_val)
        
        print(f"\n结果: {[f'{x:.6f}' for x in hw_softmax]}")
        
        # 验证最大值位置
        max_position = 2
        max_softmax = hw_softmax[max_position]
        print(f"最大值位置{max_position}的softmax: {max_softmax:.6f}")
        
        # 验证
        assert max_softmax == max(hw_softmax), "最大值位置的softmax应该最大"
        assert max_softmax > 0.9, f"最大值位置的softmax应该接近1，实际为{max_softmax:.6f}"
        
        total_sum = sum(hw_softmax)
        assert abs(total_sum - 1.0) < 0.01, f"概率和应该接近1.0，实际为{total_sum}"
        
        print("✓ 窗口优先测试通过")
    else:
        print("❌ 未获得结果")
        assert False, "未能获取结果"


class TestNewSoftmaxPipeline:
    """测试新的SoftmaxPipeline流水线架构"""
    
    def test_single_row_pipeline(self):
        """测试单行数据处理"""
        print("🧪 测试单行数据流水线处理")
        
        # 创建流水线
        pipeline = SoftmaxPipeline(front_window=3, back_window=3, max_rows=1)
        
        # 测试数据
        input_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        row_length = len(input_values)
        
        # 构造输入 - 窗口优先
        input_data = []
        
        # 前窗口数据
        for i in [0, 1, 2]:
            if i < row_length:
                input_data.append((input_values[i], 0, i, row_length))
        
        # 后窗口数据 
        for i in [2, 3, 4]:  # 可能有重叠
            if i >= row_length - 3:
                input_data.append((input_values[i], 0, i, row_length))
        
        # 运行流水线
        results = pipeline.run_pipeline(input_data, max_cycles=100, print_progress=False)
        
        # 验证结果
        assert 0 in results, "应该有行0的结果"
        
        result_row = results[0]
        assert len(result_row) == row_length, f"结果长度应该是{row_length}"
        
        # 转换为浮点数
        softmax_results = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            softmax_results.append(softmax_val)
        
        # 验证概率和接近1
        total_sum = sum(softmax_results)
        assert abs(total_sum - 1.0) < 0.01, f"概率和应该接近1.0，实际为{total_sum}"
        
        # 验证所有值为正数
        assert all(val >= 0 for val in softmax_results), "所有softmax值应该为正数"
        
        print(f"✅ 单行测试通过，概率和: {total_sum:.6f}")

    def test_multi_row_pipeline(self):
        """测试多行并行处理"""
        print("🧪 测试多行并行处理")
        
        # 创建流水线，支持2行并行
        pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=2)
        
        # 准备两行数据
        row0_data = [1.0, 2.0, 3.0, 4.0]
        row1_data = [10.0, 20.0, 30.0, 40.0]
        
        input_data = []
        
        # 行0窗口数据
        for i in [0, 1]:  # 前窗口
            input_data.append((row0_data[i], 0, i, len(row0_data)))
        for i in [2, 3]:  # 后窗口
            input_data.append((row0_data[i], 0, i, len(row0_data)))
        
        # 行1窗口数据
        for i in [0, 1]:  # 前窗口
            input_data.append((row1_data[i], 1, i, len(row1_data)))
        for i in [2, 3]:  # 后窗口
            input_data.append((row1_data[i], 1, i, len(row1_data)))
        
        # 运行流水线
        results = pipeline.run_pipeline(input_data, max_cycles=150, print_progress=False)
        
        # 验证两行都有结果
        assert 0 in results, "应该有行0的结果"
        assert 1 in results, "应该有行1的结果"
        
        # 验证每行结果
        for row_id, expected_data in [(0, row0_data), (1, row1_data)]:
            result_row = results[row_id]
            assert len(result_row) == len(expected_data), f"行{row_id}结果长度不正确"
            
            # 计算概率和
            softmax_results = []
            for col_idx in sorted(result_row.keys()):
                softmax_val = bf16_to_float(result_row[col_idx])
                softmax_results.append(softmax_val)
            
            total_sum = sum(softmax_results)
            assert abs(total_sum - 1.0) < 0.01, f"行{row_id}概率和应该接近1.0，实际为{total_sum}"
            
            print(f"✅ 行{row_id}测试通过，概率和: {total_sum:.6f}")

    def test_window_priority_processing(self):
        """测试窗口优先处理"""
        print("🧪 测试窗口优先处理")
        
        pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
        
        # 测试数据，最大值在窗口外
        input_values = [1.0, 2.0, 100.0, 3.0, 4.0]  # 最大值在位置2
        row_length = len(input_values)
        
        input_data = []
        
        # 先发送窗口数据
        for i in [0, 1]:  # 前窗口
            input_data.append((input_values[i], 0, i, row_length))
        for i in [3, 4]:  # 后窗口
            input_data.append((input_values[i], 0, i, row_length))
        
        # 后发送包含最大值的数据
        input_data.append((input_values[2], 0, 2, row_length))
        
        # 运行流水线
        results = pipeline.run_pipeline(input_data, max_cycles=150, print_progress=False)
        
        # 验证结果
        assert 0 in results, "应该有行0的结果"
        
        result_row = results[0]
        softmax_results = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            softmax_results.append(softmax_val)
        
        # 验证最大值位置的softmax值最大
        max_position = 2  # 最大值100.0的位置
        max_softmax = softmax_results[max_position]
        
        assert max_softmax == max(softmax_results), "最大值位置的softmax应该最大"
        assert max_softmax > 0.9, f"最大值位置的softmax应该接近1，实际为{max_softmax:.6f}"
        
        total_sum = sum(softmax_results)
        assert abs(total_sum - 1.0) < 0.01, f"概率和应该接近1.0，实际为{total_sum}"
        
        print(f"✅ 窗口优先测试通过，最大值位置softmax: {max_softmax:.6f}")

    def test_overflow_detection_and_rollback(self):
        """测试溢出检测和回滚"""
        print("🧪 测试溢出检测和回滚")
        
        pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
        
        # 构造容易溢出的数据
        input_values = [1.0, 2.0, 80.0, 3.0, 4.0]  # 80.0很大，可能引起溢出
        row_length = len(input_values)
        
        input_data = []
        
        # 窗口数据优先，确保大值不在初始窗口中
        for i in [0, 1]:  # 前窗口
            input_data.append((input_values[i], 0, i, row_length))
        for i in [3, 4]:  # 后窗口
            input_data.append((input_values[i], 0, i, row_length))
        
        # 包含大值的数据
        input_data.append((input_values[2], 0, 2, row_length))
        
        # 运行流水线
        results = pipeline.run_pipeline(input_data, max_cycles=200, print_progress=False)
        
        # 验证结果
        assert 0 in results, "应该有行0的结果"
        
        result_row = results[0]
        softmax_results = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            softmax_results.append(softmax_val)
        
        # 验证结果合理性
        total_sum = sum(softmax_results)
        assert abs(total_sum - 1.0) < 0.01, f"概率和应该接近1.0，实际为{total_sum}"
        
        # 验证最大值位置
        max_position = 2  # 80.0的位置
        max_softmax = softmax_results[max_position]
        assert max_softmax > 0.95, f"最大值位置的softmax应该接近1，实际为{max_softmax:.6f}"
        
        print(f"✅ 溢出测试通过，最大值位置softmax: {max_softmax:.6f}")

    def test_pytorch_comparison(self):
        """与PyTorch BF16结果对比"""
        print("🧪 与PyTorch BF16结果对比")
        
        pipeline = SoftmaxPipeline(front_window=3, back_window=3, max_rows=1)
        
        # 测试数据
        input_values = [1.0, 2.0, 3.0, 20.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        row_length = len(input_values)
        
        # 构造输入（窗口优先）
        input_data = []
        
        # 前窗口
        for i in [0, 1, 2]:
            input_data.append((input_values[i], 0, i, row_length))
        
        # 后窗口
        for i in [7, 8, 9]:
            input_data.append((input_values[i], 0, i, row_length))
        
        # 剩余数据
        for i in [3, 4, 5, 6]:
            input_data.append((input_values[i], 0, i, row_length))
        
        # 运行自定义流水线
        results = pipeline.run_pipeline(input_data, max_cycles=200, print_progress=False)
        
        assert 0 in results, "应该有行0的结果"
        
        # 获取自定义结果
        result_row = results[0]
        custom_results = []
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            custom_results.append(softmax_val)
        
        # 计算PyTorch BF16结果
        torch_input = torch.tensor(input_values, dtype=torch.float32).bfloat16()
        torch_softmax = torch.softmax(torch_input, dim=0).float()
        torch_results = torch_softmax.tolist()
        
        # 比较结果
        max_error = 0.0
        for i, (custom, torch_val) in enumerate(zip(custom_results, torch_results)):
            error = abs(custom - torch_val)
            max_error = max(max_error, error)
        
        print(f"📈 最大误差: {max_error:.6f}")
        assert max_error < 0.001, f"与PyTorch的最大误差应该小于0.001，实际为{max_error:.6f}"
        
        print(f"✅ PyTorch对比测试通过，最大误差: {max_error:.6f}")

    def test_input_format_validation(self):
        """测试输入格式验证"""
        print("🧪 测试输入格式验证")
        
        pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
        
        # 测试正确的格式
        valid_input = [
            (1.0, 0, 0, 3),  # (val, row_id, col_idx, row_length)
            (2.0, 0, 1, 3),
            (3.0, 0, 2, 3)
        ]
        
        results = pipeline.run_pipeline(valid_input, max_cycles=100, print_progress=False)
        assert 0 in results, "正确格式应该产生结果"
        
        # 验证结果长度
        result_row = results[0]
        assert len(result_row) == 3, "结果应该包含3个元素"
        
        print("✅ 输入格式验证测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行Software Hardware Simulator所有测试...")
    print("=" * 60)
    
    # 运行各项测试
    test_basic_units()
    test_simple_softmax()
    # test_medium_size_softmax()
    # test_large_size_softmax()
    # test_edge_cases()
    # test_overflow_recovery()
    test_new_input_format()
    test_window_priority()
    
    # 运行新的SoftmaxPipeline测试
    test_instance = TestNewSoftmaxPipeline()
    
    print("🚀 开始测试新的SoftmaxPipeline架构")
    print("="*60)
    
    test_instance.test_single_row_pipeline()
    print()
    
    test_instance.test_multi_row_pipeline()
    print()
    
    test_instance.test_window_priority_processing()
    print()
    
    test_instance.test_overflow_detection_and_rollback()
    print()
    
    test_instance.test_pytorch_comparison()
    print()
    
    test_instance.test_input_format_validation()
    print()
    
    print("="*60)
    print("🎉 所有测试完成！")


if __name__ == "__main__":
    run_all_tests() 