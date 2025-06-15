from ..module import MacUnit
from ...bf16_module.utils import convert_through_pipeline
import struct
def bf16_to_float(bf16):
    """将BF16转换为浮点数"""
    fp32_bits = bf16 << 16
    return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

def test_mac_unit_accumulation():
    """测试MacUnit的累加功能"""
    print("\n" + "=" * 80)
    print("测试MacUnit累加功能")
    print("=" * 80)
    
    # 模拟向量点积计算: [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
    vector_a = [1.0, 2.0, 3.0]
    vector_b = [4.0, 5.0, 6.0]
    
    input_data = list(zip(vector_a, vector_b))
    initial_acc = 0.0
    bf16_initial_acc = convert_through_pipeline(initial_acc)
    
    print("计算向量点积:")
    print(f"向量A: {vector_a}")
    print(f"向量B: {vector_b}")
    print("预期结果: 1*4 + 2*5 + 3*6 = 32")
    print(f"初始累加器: {initial_acc}")
    
    # 创建MAC单元并运行
    mac_unit = MacUnit()
    results = mac_unit.run_pipeline_with_bf16(input_data, bf16_initial_acc, max_cycles=50, print_states=False)
    
    # 获取最终结果
    final_result = None
    for result in reversed(results):  # 从后往前找最后一个非None结果
        if result is not None:
            final_result = bf16_to_float(result)
            break
    
    print(f"\n最终累加结果: {final_result:.6f}")
    print(f"预期结果: 32.0")
    if final_result is not None:
        error = abs(final_result - 32.0)
        print(f"误差: {error:.6f}")

test_mac_unit_accumulation()