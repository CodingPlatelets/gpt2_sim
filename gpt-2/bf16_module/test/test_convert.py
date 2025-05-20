import struct
import random
from ..module import FP32toBF16Pipeline

def test_fp32_to_bf16():
    # 创建流水线实例
    pipeline = FP32toBF16Pipeline()

    # 准备测试数据：[(FP32值, 是否有效), ...]

    # 可能溢出数据(fp32)

    # 0 | 1 1 1 || 1 1 1 1 || 1 | 0 0 0|| 0 .... || 0 0 0 0
    #   7F8F FFFF
    # 0 1 1 1 1 1 1 1

    data_1 = struct.unpack(">f", struct.pack(">I", 0x7F8FFFFF))[0]
    data_2 = struct.unpack(">f", struct.pack(">I", 0x7F800000))[0]
    data_3 = struct.unpack(">f", struct.pack(">I", 0x7F800001))[0]
    data_4 = struct.unpack(">f", struct.pack(">I", 0xFFFFFFFF))[0]

    test_data = [
        (3.14159, True),  # Pi，十六进制表示为 0x40490FDB
        (1.5, True),  # 1.5，十六进制表示为 0x3FC00000
        (-2.25, True),  # -2.25，十六进制表示为 0xC0100000
        (0.0, True),  # 无效输入
        (65504.0, True),  # BF16能表示的最大正数
        (1e-20, True),  # 非常小的数
        (data_1, True),  # NAN
        (data_2, True),  # inf
        (data_3, True),
        (data_4, True),
    ]
    for _ in range(10):
        a = random.uniform(-10, 10)
        test_data.append((a, True))

    print("Starting FP32 to BF16 Pipeline Simulation")
    print("=" * 50)

    # 运行模拟
    results = pipeline.run_simulation(test_data)
    import math

    print("=" * 50)
    print("Final Outputs:")
    for i, output in enumerate(pipeline.outputs):
        bf16_value = output["bf16"]
        # 将BF16转回FP32表示进行验证
        bf16_as_fp32 = struct.unpack(">f", struct.pack(">I", bf16_value << 16))[0]

        # 找到对应的原始输入，通过FP32位级表示比较
        original_input = None
        for fp32, valid in test_data:
            if valid:
                fp32_bits = struct.unpack(">I", struct.pack(">f", fp32))[0]
                if fp32_bits == output["original_fp32"]:
                    original_input = fp32
                    break

        print(f"Output {i}: BF16: {hex(bf16_value)} -> As FP32: {bf16_as_fp32}")
        if original_input is not None:
            print(f"  Original input: {original_input}")

    print("\nVerification:")
    valid_inputs = [(fp32, i) for i, (fp32, valid) in enumerate(test_data) if valid]

    for i, (fp32, orig_idx) in enumerate(valid_inputs):
        if i < len(pipeline.outputs):
            # 验证模拟结果与直接计算结果是否一致
            direct_result = pipeline.fp32_to_bf16(fp32)
            pipeline_result = pipeline.outputs[i]["bf16"]
            print(f"Input {orig_idx}: {fp32}")
            print(f"  Direct conversion: {hex(direct_result)}")
            print(f"  Pipeline output:   {hex(pipeline_result)}")
            print(f"  Match: {direct_result == pipeline_result}")

test_fp32_to_bf16()