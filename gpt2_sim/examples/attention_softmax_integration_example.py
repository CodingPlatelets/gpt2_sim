"""
Attention + Softmax 硬件模拟器集成使用示例

这个示例展示了如何使用集成了硬件模拟版本 Softmax 的 Attention 模块，
以及如何分析硬件性能指标。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gpt_module.attention_sim import Attention, convert_matrix_to_bf16
from gpt_module.test import generate_x_wq_wk_xt, generate_matrix


def simple_attention_example():
    """简单的 Attention 使用示例"""
    print("=" * 50)
    print("简单 Attention + Softmax 硬件模拟示例")
    print("=" * 50)
    
    # 步骤1: 创建 Attention 模拟器
    print("1. 创建 Attention 硬件模拟器...")
    attention = Attention(
        PE_num=128,          # 128个处理单元
        PE_rows=32,          # 32行PE
        data_num_per_cycle=256  # 每周期处理256个数据
    )
    print("   ✓ Attention 模拟器创建完成")
    
    # 步骤2: 准备测试数据
    print("\n2. 生成测试数据...")
    vector_dim = 512     # 向量维度
    past_tokens = 128    # 过去的token数量
    
    X, Wq, Wk, Xt = generate_x_wq_wk_xt(
        past_token_length=past_tokens,
        channel=vector_dim,
        sparse_ratio=0.95  # 95%稀疏度
    )
    Wv = generate_matrix(Xt.shape[0], vector_dim, 0.9)
    
    print(f"   输入 X 形状: {X.shape}")
    print(f"   查询权重 Wq 形状: {Wq.shape}")  
    print(f"   键权重 Wk 形状: {Wk.shape}")
    print(f"   过去tokens Xt 形状: {Xt.shape}")
    print(f"   值权重 Wv 形状: {Wv.shape}")
    print("   ✓ 测试数据生成完成")
    
    # 步骤3: 加载权重到硬件模拟器
    print("\n3. 加载权重到硬件模拟器...")
    attention.Wq.load_from_hbm(Wq)
    attention.Wk.load_from_hbm(Wk.T)  # 转置
    attention.XT.load_from_hbm(Xt.T)  # 转置
    attention.Wv.load_from_hbm(Wv)
    print("   ✓ 权重加载完成")
    
    # 步骤4: 转换输入为BF16格式
    print("\n4. 转换输入为 BF16 格式...")
    X_bf16 = convert_matrix_to_bf16(X)
    print("   ✓ BF16 转换完成")
    
    # 步骤5: 运行硬件模拟
    print("\n5. 运行硬件模拟...")
    softmax_output, attention_output, softmax_sim_info = attention.forward(
        X_bf16, 
        past_token_num=Xt.shape[0]
    )
    print("   ✓ 硬件模拟完成")
    
    # 步骤6: 分析结果
    print("\n6. 分析模拟结果...")
    print(f"   Softmax 输出形状: {softmax_output.shape}")
    print(f"   Attention 输出形状: {attention_output.shape}")
    print(f"   Softmax 行和检查: {np.sum(softmax_output, axis=-1)} (应该接近1)")
    
    # 步骤7: 硬件性能分析
    print("\n7. 硬件性能分析...")
    print("   === Softmax 硬件性能 ===")
    print(f"   总周期数: {softmax_sim_info['total_cycles']}")
    print(f"   PE 利用效率: {softmax_sim_info['pe_efficiency']:.1%}")
    print(f"   模拟执行时间: {softmax_sim_info['execution_time']*1000:.2f}ms")
    print(f"   处理元素总数: {softmax_sim_info['total_elements']}")
    
    print("\n   === 周期分解 ===")
    for stage, cycles in softmax_sim_info['cycles_breakdown'].items():
        print(f"   {stage}: {cycles} 周期")
    
    print(f"\n   === 整体 Attention 性能 ===")
    print(f"   总周期数: {attention.cycles}")
    
    print("\n" + "=" * 50)
    print("示例运行完成！")
    return softmax_output, attention_output, softmax_sim_info


def performance_analysis_example():
    """性能分析示例"""
    print("\n" + "=" * 50)
    print("性能分析示例")
    print("=" * 50)
    
    # 测试不同配置下的性能
    configs = [
        {'PE_num': 64, 'PE_rows': 16, 'desc': '小规模'},
        {'PE_num': 128, 'PE_rows': 32, 'desc': '中规模'}, 
        {'PE_num': 256, 'PE_rows': 64, 'desc': '大规模'},
    ]
    
    vector_dim = 512
    past_tokens = 256
    
    # 生成一次测试数据，所有配置使用相同数据
    X, Wq, Wk, Xt = generate_x_wq_wk_xt(past_tokens, vector_dim, 0.95)
    Wv = generate_matrix(Xt.shape[0], vector_dim, 0.9)
    X_bf16 = convert_matrix_to_bf16(X)
    
    results = []
    
    for config in configs:
        print(f"\n--- {config['desc']} 配置 (PE_num={config['PE_num']}, PE_rows={config['PE_rows']}) ---")
        
        # 创建对应配置的模拟器
        attention = Attention(config['PE_num'], config['PE_rows'], 256)
        
        # 加载权重
        attention.Wq.load_from_hbm(Wq)
        attention.Wk.load_from_hbm(Wk.T)
        attention.XT.load_from_hbm(Xt.T)
        attention.Wv.load_from_hbm(Wv)
        
        # 运行模拟并计时
        import time
        start_time = time.time()
        _, _, sim_info = attention.forward(X_bf16, Xt.shape[0])
        elapsed_time = time.time() - start_time
        
        # 记录结果
        result = {
            'config': config['desc'],
            'pe_num': config['PE_num'],
            'pe_rows': config['PE_rows'],
            'softmax_cycles': sim_info['total_cycles'],
            'pe_efficiency': sim_info['pe_efficiency'],
            'total_attention_cycles': attention.cycles,
            'wall_time': elapsed_time
        }
        results.append(result)
        
        print(f"Softmax 周期数: {result['softmax_cycles']}")
        print(f"PE 效率: {result['pe_efficiency']:.1%}")
        print(f"总 Attention 周期: {result['total_attention_cycles']}")
        print(f"实际执行时间: {result['wall_time']*1000:.2f}ms")
    
    # 性能对比分析
    print(f"\n=== 性能对比分析 ===")
    print(f"{'配置':<8} {'PE数量':<8} {'Softmax周期':<12} {'PE效率':<10} {'总周期':<10} {'实际时间':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config']:<8} {result['pe_num']:<8} {result['softmax_cycles']:<12} "
              f"{result['pe_efficiency']:.1%:<10} {result['total_attention_cycles']:<10} "
              f"{result['wall_time']*1000:.1f}ms")
    
    return results


def optimization_suggestions():
    """基于模拟结果提供优化建议"""
    print("\n" + "=" * 50)
    print("硬件优化建议")
    print("=" * 50)
    
    # 运行一个基准测试
    attention = Attention(128, 32, 256)
    X, Wq, Wk, Xt = generate_x_wq_wk_xt(256, 512, 0.95)
    Wv = generate_matrix(Xt.shape[0], 512, 0.9)
    
    attention.Wq.load_from_hbm(Wq)
    attention.Wk.load_from_hbm(Wk.T)
    attention.XT.load_from_hbm(Xt.T)
    attention.Wv.load_from_hbm(Wv)
    
    X_bf16 = convert_matrix_to_bf16(X)
    _, _, sim_info = attention.forward(X_bf16, Xt.shape[0])
    
    print("基于当前模拟结果的优化建议：")
    print()
    
    # PE 利用率分析
    pe_eff = sim_info['pe_efficiency']
    if pe_eff < 0.5:
        print("🔧 PE 利用率优化:")
        print(f"   当前 PE 效率: {pe_eff:.1%} (较低)")
        print("   建议: 减少 PE 数量或增加数据并行度")
        print("   建议: 优化数据分块策略")
    elif pe_eff > 0.8:
        print("✅ PE 利用率优化:")
        print(f"   当前 PE 效率: {pe_eff:.1%} (良好)")
        print("   建议: 可以考虑增加 PE 数量来提高吞吐量")
    else:
        print("⚖️ PE 利用率优化:")
        print(f"   当前 PE 效率: {pe_eff:.1%} (中等)")
        print("   建议: 当前配置基本合理")
    
    print()
    
    # 周期分解分析
    print("🔍 计算阶段分析:")
    total_cycles = sim_info['total_cycles']
    for stage, cycles in sim_info['cycles_breakdown'].items():
        percentage = cycles / total_cycles * 100 if total_cycles > 0 else 0
        print(f"   {stage}: {cycles} 周期 ({percentage:.1f}%)")
        
        if stage == 'exp_computation' and percentage > 50:
            print("     → 指数计算占比较高，建议优化指数运算单元")
        elif stage == 'division_computation' and percentage > 30:
            print("     → 除法计算占比较高，建议使用近似除法或查表法")
    
    print()
    print("📊 数据流优化:")
    elements = sim_info['total_elements']
    print(f"   处理元素数: {elements}")
    print(f"   平均每周期处理: {elements/total_cycles:.1f} 元素" if total_cycles > 0 else "N/A")
    print("   建议: 优化内存访问模式和数据预取策略")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # 运行示例
    print("开始运行 Attention + Softmax 硬件模拟器集成示例\n")
    
    try:
        # 基本使用示例
        softmax_out, attn_out, sim_info = simple_attention_example()
        
        # 性能分析示例
        perf_results = performance_analysis_example()
        
        # 优化建议
        optimization_suggestions()
        
        print("\n🎉 所有示例运行完成！")
        print("\n💡 主要收获:")
        print("1. 成功集成了硬件模拟版本的 Softmax 到 Attention 模块")
        print("2. 可以详细分析 Softmax 计算的硬件性能指标")
        print("3. 支持不同硬件配置下的性能对比")
        print("4. 提供了基于模拟结果的硬件优化建议")
        
    except Exception as e:
        print(f"❌ 示例运行出现错误: {e}")
        print("请检查依赖模块是否正确导入")