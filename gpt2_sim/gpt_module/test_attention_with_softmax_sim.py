"""
Attention 模块与 Softmax 硬件模拟器集成测试
"""
import numpy as np
import time
from .attention_sim import Attention, convert_matrix_to_bf16
from .test import generate_x_wq_wk_xt, generate_matrix
from ..vector_matrix_module.softmax import Softmax


def test_attention_with_softmax_sim():
    """测试集成了 Softmax 硬件模拟器的 Attention 模块"""
    print("=" * 60)
    print("测试 Attention 模块 + Softmax 硬件模拟器")
    print("=" * 60)
    
    # 测试配置
    vector_size = 1024  # 减小向量大小以便快速测试
    PE_num = 128
    PE_rows = 32
    data_num_per_cycle = 256
    
    print(f"配置参数:")
    print(f"  向量维度: {vector_size}")
    print(f"  PE 数量: {PE_num}")
    print(f"  PE 行数: {PE_rows}")
    print(f"  每周期数据量: {data_num_per_cycle}")
    print()
    
    # 创建 Attention 模块
    attention = Attention(PE_num, PE_rows, data_num_per_cycle)
    
    # 生成测试数据
    print("生成测试数据...")
    X, wq, wk, xt = generate_x_wq_wk_xt(
        past_token_length=255, 
        channel=vector_size, 
        sparse_ratio=0.95
    )
    past_token_num = xt.shape[0]
    wv = generate_matrix(past_token_num, vector_size, 0.9)
    
    print(f"数据形状:")
    print(f"  X: {X.shape}")
    print(f"  Wq: {wq.shape}")
    print(f"  Wk: {wk.shape}")
    print(f"  Xt: {xt.shape}")
    print(f"  Wv: {wv.shape}")
    print()
    
    # 参考实现（使用简单的 Softmax）
    print("计算参考结果...")
    start_time = time.time()
    reference_result = (Softmax().forward(((X @ wq) @ wk.T) @ xt.T)) @ wv
    reference_time = time.time() - start_time
    print(f"参考计算耗时: {reference_time*1000:.2f}ms")
    
    # 加载权重到硬件模拟器
    print("加载权重到硬件模拟器...")
    attention.Wq.load_from_hbm(wq)
    attention.Wk.load_from_hbm(wk.T)
    attention.XT.load_from_hbm(xt.T)
    attention.Wv.load_from_hbm(wv)
    
    # 转换输入为 BF16
    x_bf16 = convert_matrix_to_bf16(X)
    
    # 运行硬件模拟
    print("运行硬件模拟...")
    start_time = time.time()
    softmax_out, sim_result, softmax_sim_info = attention.forward(x_bf16, past_token_num)
    sim_time = time.time() - start_time
    print(f"模拟计算耗时: {sim_time*1000:.2f}ms")
    print()
    
    # 打印 Softmax 硬件模拟详细信息
    print("=== Softmax 硬件模拟详细信息 ===")
    print(f"总周期数: {softmax_sim_info['total_cycles']}")
    print(f"PE 效率: {softmax_sim_info['pe_efficiency']:.2%}")
    print(f"模拟执行时间: {softmax_sim_info['execution_time']*1000:.2f}ms")
    print(f"总处理元素数: {softmax_sim_info['total_elements']}")
    
    print("\n周期分解:")
    for stage, cycles in softmax_sim_info['cycles_breakdown'].items():
        print(f"  {stage}: {cycles} 周期")
    
    print(f"\n总 Attention 周期数: {attention.cycles}")
    print("=" * 40)
    
    # 结果验证
    print("\n=== 结果验证 ===")
    error = np.max(np.abs(reference_result - sim_result))
    print(f"最大误差: {error:.6f}")
    
    # 验证 softmax 输出的性质
    softmax_row_sums = np.sum(softmax_out, axis=-1)
    print(f"Softmax 行和 (应该接近1): {softmax_row_sums}")
    
    # 性能对比
    print(f"\n=== 性能对比 ===")
    print(f"参考实现耗时: {reference_time*1000:.2f}ms")
    print(f"硬件模拟耗时: {sim_time*1000:.2f}ms")
    if sim_time > 0:
        speedup = reference_time / sim_time
        print(f"性能比率: {speedup:.2f}x {'(模拟更快)' if speedup > 1 else '(参考更快)'}")
    
    # 测试结果判断
    print(f"\n=== 测试结果 ===")
    if error < 1e-2:
        print("✓ 精度测试通过 - 误差在可接受范围内")
    else:
        print("✗ 精度测试失败 - 误差过大")
        print(f"参考结果样本: {reference_result.flatten()[:5]}")
        print(f"模拟结果样本: {sim_result.flatten()[:5]}")
    
    if np.allclose(softmax_row_sums, 1.0, rtol=1e-2, atol=1e-2):
        print("✓ Softmax 属性验证通过 - 行和接近1")
    else:
        print("✗ Softmax 属性验证失败 - 行和不接近1")
    
    if softmax_sim_info['pe_efficiency'] > 0.1:  # 至少10%的PE利用率
        print("✓ PE 利用率测试通过")
    else:
        print("✗ PE 利用率过低")
    
    print("=" * 60)
    return {
        'error': error,
        'softmax_sim_info': softmax_sim_info,
        'attention_cycles': attention.cycles,
        'performance_ratio': reference_time / sim_time if sim_time > 0 else 0
    }


def test_different_input_sizes():
    """测试不同输入大小下的性能"""
    print("\n" + "=" * 60)
    print("测试不同输入大小下的性能")
    print("=" * 60)
    
    test_configs = [
        {'vector_size': 256, 'past_tokens': 128, 'desc': '小规模'},
        {'vector_size': 512, 'past_tokens': 256, 'desc': '中等规模'},
        {'vector_size': 1024, 'past_tokens': 512, 'desc': '大规模'},
    ]
    
    attention = Attention(128, 32, 256)
    
    for config in test_configs:
        print(f"\n--- {config['desc']} ({config['vector_size']}D, {config['past_tokens']} tokens) ---")
        
        # 生成数据
        X, wq, wk, xt = generate_x_wq_wk_xt(
            config['past_tokens'], 
            config['vector_size'], 
            0.95
        )
        wv = generate_matrix(xt.shape[0], config['vector_size'], 0.9)
        
        # 加载权重
        attention.Wq.load_from_hbm(wq)
        attention.Wk.load_from_hbm(wk.T)
        attention.XT.load_from_hbm(xt.T)
        attention.Wv.load_from_hbm(wv)
        
        x_bf16 = convert_matrix_to_bf16(X)
        
        # 运行测试
        start_time = time.time()
        _, _, sim_info = attention.forward(x_bf16, xt.shape[0])
        elapsed_time = time.time() - start_time
        
        print(f"执行时间: {elapsed_time*1000:.2f}ms")
        print(f"Softmax 周期: {sim_info['total_cycles']}")
        print(f"PE 效率: {sim_info['pe_efficiency']:.2%}")
        print(f"处理元素数: {sim_info['total_elements']}")


def benchmark_softmax_implementation():
    """对比不同 Softmax 实现的性能"""
    print("\n" + "=" * 60)
    print("Softmax 实现性能对比")
    print("=" * 60)
    
    from ..vector_matrix_module.softmax_sim import SoftmaxSim
    
    # 测试数据
    test_data = [
        np.random.randn(1, 128) * 0.1,
        np.random.randn(1, 256) * 0.1,
        np.random.randn(1, 512) * 0.1,
        np.random.randn(1, 1024) * 0.1,
    ]
    
    # 创建实例
    simple_softmax = Softmax()
    sim_softmax = SoftmaxSim(128, 32, 256)
    
    for i, data in enumerate(test_data):
        seq_len = data.shape[1]
        print(f"\n--- 序列长度: {seq_len} ---")
        
        # 简单实现
        start_time = time.time()
        simple_result = simple_softmax.forward(data)
        simple_time = time.time() - start_time
        
        # 硬件模拟实现
        start_time = time.time()
        sim_result, sim_info = sim_softmax.forward(data, use_bf16=True)
        sim_time = time.time() - start_time
        
        # 结果对比
        error = np.max(np.abs(simple_result - sim_result))
        
        print(f"简单实现耗时: {simple_time*1000:.3f}ms")
        print(f"硬件模拟耗时: {sim_time*1000:.3f}ms")
        print(f"精度误差: {error:.6f}")
        print(f"硬件周期数: {sim_info['total_cycles']}")
        print(f"PE 效率: {sim_info['pe_efficiency']:.2%}")


if __name__ == "__main__":
    # 运行主要测试
    test_result = test_attention_with_softmax_sim()
    
    # 运行额外测试
    test_different_input_sizes()
    benchmark_softmax_implementation()
    
    print(f"\n{'='*60}")
    print("所有测试完成!")
    print(f"主测试误差: {test_result['error']:.6f}")
    print(f"总 Attention 周期数: {test_result['attention_cycles']}")
    print(f"Softmax PE 效率: {test_result['softmax_sim_info']['pe_efficiency']:.2%}")
    print(f"{'='*60}")