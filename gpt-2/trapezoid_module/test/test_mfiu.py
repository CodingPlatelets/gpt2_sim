import numpy as np
from scipy.sparse import csr_matrix
from ..module import MFIUPipeline
from ..utils import get_values_offset_mask

def test_MFIU_unit(A: np.array, B: np.array):
    print("\n===== 测试 MFIU Pipeline =====")
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    width = M * N
    bit_width = K

    mfiu_pipeline = MFIUPipeline(width, bit_width)

    values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    values_B, offset_B, masks_B = get_values_offset_mask(csr_B)

    results = mfiu_pipeline.run_pipeline(
        [masks_A, masks_A], [masks_B, masks_B], [offset_A, offset_A], [offset_B, offset_B], len(values_A), len(values_B), 20
    )
    for i, res in enumerate(results):
        if res["valid"]:
            print(f"周期 {i+1} 输出: {res['output']}")


def test_MFIU():
    # 运行测试
    print("\n===== 测试用例1：简单矩阵 =====")
    A1 = np.array([[1, 1, 1, 1]])

    B1 = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 0]])
    test_MFIU_unit(A1, B1)

    #np.random.seed(42)
    #A3 = np.random.choice([0, 1], size=(1, 64), p=[0, 1])
    #B3 = np.random.choice([0, 1], size=(64, 10), p=[0.9, 0.1])
    #test_MFIU_unit(A3, B3)

test_MFIU()