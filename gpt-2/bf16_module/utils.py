from .module.convert_sim import FP32toBF16Pipeline
from .module.add_sim import BF16AddPipeline
from .module.multiply_sim import BF16MultiplyPipeline
import struct

def convert_through_pipeline(value):
    """通过完整流水线模拟转换FP32到BF16"""
    temp_pipeline = FP32toBF16Pipeline()
    temp_pipeline.run_simulation([(value, True)], print_states=False)
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0


def bf16_add(bf16_a, bf16_b):
    sim = BF16AddPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], False)
    return sim.outputs[0]


def bf16_mul(bf16_a, bf16_b):
    sim = BF16MultiplyPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], False)
    return sim.outputs[0]

def bf16_to_float(bf16):
    # 左移16位填充为32位表示
    fp32_bits = bf16 << 16
    # 转换为浮点数
    return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
