import struct
import random
from ..module import BF16MultiplyPipeline
from ..utils import convert_through_pipeline

def test_bf16multiply():
    # 创建流水线实例
    pipeline = BF16MultiplyPipeline()

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

    # 准备常规测试用例
    regular_cases = [
        (2.0, 3.0, True),  # 简单乘法: 2.0 * 3.0 = 6.0
        (3.124, 2.249, True),
        (1.5, 2.25, True),  # 小数乘法: 1.5 * 2.25 = 3.375
        (3.14159, -1.5, True),  # 异号乘法: 3.14159 * (-1.5) ≈ -4.71
        (-3.0, -2.0, True),  # 负数乘法: (-3.0) * (-2.0) = 6.0
        (100.0, 0.01, True),  # 量级差异大: 100.0 * 0.01 = 1.0
        (0.0, 5.0, True),  # 零乘任何数: 0.0 * 5.0 = 0.0
        (1e-2, 1e2, True),  # 指数抵消: 0.01 * 100 = 1.0
        (0.1, 0.1, True),  # 小数平方: 0.1 * 0.1 = 0.01
    ]

    for _ in range(10):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        regular_cases.append((a, b, True))

    # 新增：下溢/非规格化测试用例
    subnormal_cases = [
        (1e-38, 1.0, True),  # 极小数 * 1
        (1e-38, 2.0, True),  # 极小数 * 2
        (1e-38, 0.5, True),  # 极小数 * 0.5
        (1e-38, 1e-2, True),  # 极小数 * 极小数
        (1e-38, -1.0, True),  # 极小数 * -1
        (1e-38, 1e38, True),  # 极小数 * 极大数
        (1e-38, 1e-38, True),  # 极小数 * 极小数
        (1e-20, 1e-20, True),  # 仍然很小
    ]

    # 准备特殊情况测试用例
    # 定义特殊值
    inf_pos = float("inf")  # 正无穷大
    inf_neg = float("-inf")  # 负无穷大
    nan = float("nan")  # NaN

    special_cases = [
        (0.0, 0.0, True),  # 零乘零: 0.0 * 0.0 = 0.0
        (inf_pos, 0.0, True),  # 无穷大 * 零 = NaN
        (inf_pos, 2.0, True),  # 正无穷大 * 正数 = 正无穷大
        (inf_pos, -2.0, True),  # 正无穷大 * 负数 = 负无穷大
        (inf_neg, inf_pos, True),  # 负无穷大 * 正无穷大 = 负无穷大
        (inf_pos, inf_pos, True),  # 正无穷大 * 正无穷大 = 正无穷大
        (nan, 1.0, True),  # NaN * 任意数 = NaN
        (nan, nan, True),  # NaN * NaN = NaN
    ]

    # 合并所有测试用例
    test_cases = regular_cases + subnormal_cases + special_cases + [(0.0, 1.0, False)]

    # 转换测试用例为BF16格式
    bf16_test_cases = [
        (convert_through_pipeline(a), convert_through_pipeline(b), valid)
        for a, b, valid in test_cases
    ]

    print("Starting BF16 Multiplication Pipeline Simulation")
    print("=" * 80)

    # 运行模拟
    results = pipeline.run_simulation(bf16_test_cases, print_states=True)

    print("=" * 80)
    print("Final Results:")
    print(
        "{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
            "Test",
            "Input A",
            "Input B",
            "Expected Product",
            "Custom BF16 Result",
            "PyTorch Result",
            "Error vs PyTorch",
        )
    )
    print("-" * 80)

    for i, output in enumerate(pipeline.outputs):
        if i < len(test_cases) and test_cases[i][2]:  # 只检查有效输入
            a, b, _ = test_cases[i]

            # 自定义实现的结果
            custom_result = bf16_to_float(output)

            # 获取PyTorch的BF16计算结果
            try:
                torch_a = torch.tensor(a, dtype=torch.float32).bfloat16()
                torch_b = torch.tensor(b, dtype=torch.float32).bfloat16()
                torch_product = (torch_a * torch_b).float().item()

                # 计算与PyTorch结果的差异
                if np.isnan(custom_result) and np.isnan(torch_product):
                    error = "N/A (Both NaN)"
                elif (
                    np.isinf(custom_result)
                    and np.isinf(torch_product)
                    and np.sign(custom_result) == np.sign(torch_product)
                ):
                    error = "N/A (Both Inf)"
                else:
                    # 对于非常小的结果，使用相对误差
                    if abs(torch_product) > 1e-10:
                        error = abs((custom_result - torch_product) / torch_product)
                    else:
                        error = abs(custom_result - torch_product)
            except:
                torch_product = "N/A"
                error = "N/A"

            # 预期结果
            expected = a * b

            print(
                "{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
                    i,
                    f"{a:.6g}",
                    f"{b:.6g}",
                    f"{expected:.6g}" if not np.isnan(expected) else "NaN",
                    (
                        f"{custom_result:.6g} ({hex(output)})"
                        if not np.isnan(custom_result)
                        else f"NaN ({hex(output)})"
                    ),
                    f"{torch_product:.6g}" if torch_product != "N/A" else torch_product,
                    (
                        f"{error:.6g}"
                        if error != "N/A (Both NaN)"
                        and error != "N/A (Both Inf)"
                        and error != "N/A"
                        else error
                    ),
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

            print(f"\nTest {i}: {a} * {b}")
            print(f"  BF16 Result: {hex(output)} -> {custom_result}")

            # 特殊情况的分析
            if np.isnan(a) or np.isnan(b):
                print("  Analysis: Input contains NaN, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif (np.isinf(a) and b == 0.0) or (np.isinf(b) and a == 0.0):
                print("  Analysis: Infinity * Zero, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) or np.isinf(b):
                # 计算符号
                expected_sign = np.sign(a) * np.sign(b)
                print(
                    f"  Analysis: One input is Infinity, result should be {expected_sign}Infinity"
                )
                print(
                    f"  Correct: {np.isinf(custom_result) and np.sign(custom_result) == expected_sign}"
                )
            elif a == 0.0 or b == 0.0:
                print("  Analysis: Multiplication by zero, result should be zero")
                print(f"  Correct: {custom_result == 0.0}")

                # 检查零的符号（虽然通常乘法中零的符号并不那么重要）
                expected_sign = np.sign(a) * np.sign(b)
                if expected_sign == -0.0:
                    print(f"  Sign check: Result should be -0.0")
                    print(f"  Correct sign: {np.signbit(custom_result)}")
                else:
                    print(f"  Sign check: Result should be +0.0")
                    print(f"  Correct sign: {not np.signbit(custom_result)}")

test_bf16multiply()