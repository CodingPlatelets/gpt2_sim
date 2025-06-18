"""
简单的 Attention + Softmax 硬件模拟器使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gpt_module.attention_sim import Attention, convert_matrix_to_bf16
from gpt_module.test import generate_x_wq_wk_xt, generate_matrix


def main():
    """主要示例函数"""
    print("=" * 50)
    print("Attention + Softmax 硬件模拟器示例")
    print("=" * 50)
    
    # 创建 Attention 模拟器
    print("1. 创建 Attention 硬件模拟器...")
    attention = Attention(PE_num=128, PE_rows=32, data_num_per_cycle=256)
    print("   ✓ 创建完成")
    
    # 生成测试数据
    print("\n2. 生成测试数据...")
    vector_dim = 512
    past_tokens = 128
    
    X, Wq, Wk, Xt = generate_x_wq_wk_xt(past_tokens, vector_dim, 0.95)
    Wv = generate_matrix(Xt.shape[0], vector_dim, 0.9)
    
    print(f"   输入形状: X{X.shape}, Wq{Wq.shape}, Wk{Wk.shape}")
    print("   ✓ 数据生成完成")
    
    # 加载权重
    print("\n3. 加载权重...")
    attention.Wq.load_from_hbm(Wq)
    attention.Wk.load_from_hbm(Wk.T)
    attention.XT.load_from_hbm(Xt.T)
    attention.Wv.load_from_hbm(Wv)
    print("   ✓ 权重加载完成")
    
    # 运行硬件模拟
    print("\n4. 运行硬件模拟...")
    X_bf16 = convert_matrix_to_bf16(X)
    softmax_out, attn_out, sim_info = attention.forward(X_bf16, Xt.shape[0])
    print("   ✓ 模拟完成")
    
    # 分析结果
    print("\n5. 结果分析...")
    print(f"   Softmax 输出形状: {softmax_out.shape}")
    print(f"   Attention 输出形状: {attn_out.shape}")
    print(f"   Softmax 行和: {np.sum(softmax_out, axis=-1):.4f} (应该接近1)")
    
    print("\n6. 硬件性能分析...")
    print(f"   总周期数: {sim_info['total_cycles']}")
    print(f"   PE 效率: {sim_info['pe_efficiency']:.1%}")
    print(f"   执行时间: {sim_info['execution_time']*1000:.2f}ms")
    
    print("\n   周期分解:")
    for stage, cycles in sim_info['cycles_breakdown'].items():
        print(f"     {stage}: {cycles} 周期")
    
    print("\n" + "=" * 50)
    print("✓ 示例运行完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()