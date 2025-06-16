import numpy as np
import math
import struct
from bf16_module import BF16AddPipeline, FP32toBF16Pipeline
from collections import Counter
from scipy.sparse import csr_matrix

def bf16_add(bf16_a, bf16_b):
    sim = BF16AddPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], print_states=False)
    return sim.outputs[0]

def bf16_add_list(bf16_a_list, bf16_b_list):
    sim = BF16AddPipeline()
    result = []
    inputs = []
    for a, b in zip(bf16_a_list, bf16_b_list):
        inputs.append((a, b, True))
    sim.run_simulation(inputs, False)
    assert len(bf16_a_list) == len(bf16_b_list)
    for i in range(len(bf16_a_list)):
        result.append(sim.outputs[i])
    return result


def naive_matmul(A, B):
    """使用朴素方法计算矩阵乘法，用于结果验证"""
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

    return C


def convert_through_pipeline(value):
    """通过完整流水线模拟转换FP32到BF16"""
    temp_pipeline = FP32toBF16Pipeline()
    temp_pipeline.run_simulation([(value, True)], print_states=False)
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0


def min_bits_needed(bit_width):
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))


def bf16_to_float(bf16):
    # 左移16位填充为32位表示
    fp32_bits = bf16 << 16
    # 转换为浮点数
    return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]


def find_singles(lst):
    # 统计每个元素出现的次数
    counts = Counter(lst)
    # 返回只出现一次的元素
    return [item for item, count in counts.items() if count == 1]


def get_values_offset_mask(matrix: csr_matrix):
    values = matrix.data
    col_indices = matrix.indices
    row_ptr = matrix.indptr

    # 获取矩阵大小
    num_rows = len(row_ptr) - 1
    num_cols = matrix.shape[1]

    # 创建每行的二进制掩码
    masks = []

    for row in range(num_rows):
        start = row_ptr[row]
        end = row_ptr[row + 1]
        row_mask = 0

        for i in range(start, end):
            col = int(col_indices[i])
            row_mask |= 1 << (num_cols - 1 - col)
        masks.append(row_mask)

    return values, row_ptr, masks

def get_values_offset_mask_direct(values_input, col_indices_input, row_ptr_input, num_cols):
    num_rows = len(row_ptr_input) - 1

    # 创建每行的二进制掩码
    masks = []

    for row in range(num_rows):
        start = row_ptr_input[row]
        end = row_ptr_input[row + 1]
        row_mask = 0

        for i in range(start, end):
            col = int(col_indices_input[i])
            row_mask |= 1 << (num_cols - 1 - col)
        masks.append(row_mask)

    return values_input, row_ptr_input, masks, num_rows



def test():
    x = convert_through_pipeline(288)
    y = bf16_to_float(x)

test()