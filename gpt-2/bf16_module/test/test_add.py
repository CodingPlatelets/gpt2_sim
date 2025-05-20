import struct
import random
from ..module import BF16AddPipeline
from ..utils import convert_through_pipeline

def test_bf16add():
    # 创建流水线实例
    pipeline = BF16AddPipeline()

    import torch
    import numpy as np

    # 将浮点数转换为BF16表示
    def float_to_bf16(value):
        # 将float转换为32位整数表示
        fp32_bits = struct.unpack(">I", struct.pack(">f", value))[0]
        # 取高16位作为BF16
        bf16 = (fp32_bits >> 16) & 0xFFFF
        return bf16

    # 将BF16转换为浮点数表示
    def bf16_to_float(bf16):
        # 左移16位填充为32位表示
        fp32_bits = bf16 << 16
        # 转换为浮点数
        return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

    def test_cases_bf16_to_float_add_to_bf16_to_float(test_cases):
        results = []
        for case in test_cases:
            results.append(bf16_to_float(case[0]) + bf16_to_float(case[1]))
        for i in range(len(results)):
            results[i] = convert_through_pipeline(results[i])
            results[i] = bf16_to_float(results[i])
        return results

    # 准备常规测试用例
    regular_cases = [
        (1.5, 2.25, True),  # 简单加法: 1.5 + 2.25 = 3.75
        (3.14159, -1.5, True),  # 异号加法: 3.14159 + (-1.5) ≈ 1.64159
        (-3.0, -2.0, True),  # 负数加法: (-3.0) + (-2.0) = -5.0
        (100.0, 0.001, True),  # 量级差异大: 100.0 + 0.001 ≈ 100.001
        (0.0, 0.0, True),  # 零加零: 0.0 + 0.0 = 0.0
        (1e4, 1e4, True),  # 大数: 10000 + 10000 = 20000
        (0.1, 0.2, True),  # 小数: 0.1 + 0.2 = 0.3 (注意精度问题)
        (1.0, -1.0, True),  # 正好抵消: 1.0 + (-1.0) = 0.0
    ]

    for _ in range(10):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        regular_cases.append((a, b, True))

    subnormal_cases = [
        (1e-38, 1e-38, True),  # 极小数 + 极小数
        (1e-38, -1e-38, True),  # 极小数 + 负极小数
        (1e-38, 0.0, True),  # 极小数 + 0
        (1e-38, 1e-20, True),  # 极小数 + 更大极小数
        (1e-38, -1e-20, True),  # 极小数 + 负极小数
        (1e-38, 1.0, True),  # 极小数 + 1
        (1e-38, -1.0, True),  # 极小数 + -1
        (1e-45, 1e-45, True),  # 更极小数 + 更极小数
        (1e-45, 0.0, True),  # 更极小数 + 0
        (1e-45, 1e-38, True),  # 更极小数 + 极小数
    ]

    # 准备特殊情况测试用例
    # 定义特殊值
    inf_pos = float("inf")  # 正无穷大
    inf_neg = float("-inf")  # 负无穷大
    nan = float("nan")  # NaN

    special_cases = [
        (0.0, -0.0, True),  # 正零 + 负零 = 正零
        (-0.0, -0.0, True),  # 负零 + 负零 = 负零
        (inf_pos, 1.0, True),  # 正无穷大 + 任意数 = 正无穷大
        (inf_neg, 1.0, True),  # 负无穷大 + 任意数 = 负无穷大
        (inf_pos, inf_pos, True),  # 正无穷大 + 正无穷大 = 正无穷大
        (inf_pos, inf_neg, True),  # 正无穷大 + 负无穷大 = NaN
        (nan, 1.0, True),  # NaN + 任意数 = NaN
        (nan, nan, True),  # NaN + NaN = NaN
    ]

    # 合并所有测试用例
    test_cases = regular_cases + subnormal_cases + special_cases + [(0.0, 1.0, False)]

    # 转换测试用例为BF16格式
    bf16_test_cases = [
        (convert_through_pipeline(a), convert_through_pipeline(b), valid)
        for a, b, valid in test_cases
    ]

    sim_cases = test_cases_bf16_to_float_add_to_bf16_to_float(bf16_test_cases)

    print("Starting BF16 Addition Pipeline Simulation")
    print("=" * 80)

    # 运行模拟
    results = pipeline.run_simulation(bf16_test_cases, print_states=True)

    print("=" * 80)
    print("Final Results:")
    print(
        "{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15} {:<15}".format(
            "Test",
            "Input A",
            "Input B",
            "Expected Sum",
            "Custom BF16 Result",
            "PyTorch Result",
            "Error vs PyTorch",
            "Sim pytorch add",
        )
    )
    print("-" * 80)

    for i, output in enumerate(pipeline.outputs):
        if i < len(test_cases) and test_cases[i][2]:  # 只检查有效输入
            a, b, _ = test_cases[i]
            c = sim_cases[i]
            # 自定义实现的结果
            custom_result = bf16_to_float(output)

            # 获取PyTorch的BF16计算结果
            try:
                torch_a = torch.tensor(a, dtype=torch.float32).bfloat16()
                torch_b = torch.tensor(b, dtype=torch.float32).bfloat16()
                torch_sum = (torch_a + torch_b).float().item()

                # 计算与PyTorch结果的差异
                if np.isnan(custom_result) and np.isnan(torch_sum):
                    error = "N/A (Both NaN)"
                elif (
                    np.isinf(custom_result)
                    and np.isinf(torch_sum)
                    and np.sign(custom_result) == np.sign(torch_sum)
                ):
                    error = "N/A (Both Inf)"
                else:
                    error = abs(custom_result - torch_sum)
            except:
                torch_sum = "N/A"
                error = "N/A"

            # 预期结果
            if a == 0.0 and b == -0.0 or a == -0.0 and b == 0.0:
                expected = 0.0  # 预期输出正零
            else:
                expected = a + b

            print(
                "{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15} {:<15}".format(
                    i,
                    f"{a:.6g}",
                    f"{b:.6g}",
                    f"{expected:.6g}" if not np.isnan(expected) else "NaN",
                    (
                        f"{custom_result:.6g} ({hex(output)})"
                        if not np.isnan(custom_result)
                        else f"NaN ({hex(output)})"
                    ),
                    f"{torch_sum:.6g}" if torch_sum != "N/A" else torch_sum,
                    (
                        f"{error:.6g}"
                        if error != "N/A (Both NaN)"
                        and error != "N/A (Both Inf)"
                        and error != "N/A"
                        else error
                    ),
                    f"{c:.6g}",
                )
            )

    print("\nDetailed Analysis of Special Cases:")
    special_start = len(regular_cases)
    special_end = len(test_cases) - 1  # 除去无效输入

    for i in range(special_start, special_end):
        if i < len(pipeline.outputs):
            a, b, _ = test_cases[i]
            output = pipeline.outputs[i]
            custom_result = bf16_to_float(output)

            print(f"\nTest {i}: {a} + {b}")
            print(f"  BF16 Result: {hex(output)} -> {custom_result}")

            # 特殊情况的分析
            if np.isnan(a) or np.isnan(b):
                print("  Analysis: Input contains NaN, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) and np.isinf(b) and np.sign(a) != np.sign(b):
                print("  Analysis: Infinity - Infinity, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) or np.isinf(b):
                expected_sign = np.sign(a) if np.isinf(a) else np.sign(b)
                print(
                    f"  Analysis: One input is Infinity, result should be {expected_sign}Infinity"
                )
                print(
                    f"  Correct: {np.isinf(custom_result) and np.sign(custom_result) == expected_sign}"
                )
            elif a == 0.0 and b == 0.0:
                if (a == 0.0 and b == -0.0) or (a == -0.0 and b == 0.0):
                    print(
                        "  Analysis: Positive zero + Negative zero, result should be +0.0"
                    )
                    print(
                        f"  Correct: {custom_result == 0.0 and not np.signbit(custom_result)}"
                    )
                else:
                    expected_sign = np.signbit(a)
                    print(
                        f"  Analysis: Zero + Zero with same sign, result should maintain sign"
                    )
                    print(
                        f"  Correct: {custom_result == 0.0 and np.signbit(custom_result) == expected_sign}"
                    )

test_bf16add()