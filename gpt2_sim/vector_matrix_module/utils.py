from bf16_module.utils import convert_through_pipeline
import numpy as np
import struct
from vector_matrix_module.bf16_sim import BF16AddPipeline, FP32toBF16Pipeline


def generate_matrix(M, N, sparse_ratio):
    
    return np.random.choice([0, 0.1], size=(M, N), p = [sparse_ratio, 1 - sparse_ratio])

def convert_matrix_to_bf16(A):
    A_bf16 = np.zeros_like(A,dtype=np.uint16)
    if A.ndim == 1:
        for i in range(A.shape[0]):
            A_bf16[i] = convert_through_pipeline(float(A[i]))
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
    return A_bf16

def fp32_to_bf16(fp32_value):
    """将FP32值转换为BF16格式的整数表示"""
    if isinstance(fp32_value, np.ndarray):
        fp32_value = float(fp32_value.item())
    elif isinstance(fp32_value, (np.float32, np.float64)):
        fp32_value = float(fp32_value)
    
    pipeline = FP32toBF16Pipeline()
    pipeline.run_simulation([(fp32_value, True)], print_states=False)
    return pipeline.outputs[0]["bf16"] if pipeline.outputs else 0

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式的浮点数"""
    if bf16 is None: return 0.0
    fp32_bits = int(bf16) << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]