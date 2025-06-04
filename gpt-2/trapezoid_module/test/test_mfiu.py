import numpy as np
from scipy.sparse import csr_matrix
from ..module import MFIUPipeline, MFIUPipelineDenseA
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

def test_dense_MFIU_unit(A: np.array, B: np.array):
    print("\n===== 测试 MFIU Pipeline =====")
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    width = M * N
    bit_width = K

    mfiu_pipeline = MFIUPipelineDenseA(width, bit_width)

    values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    values_B, offset_B, masks_B = get_values_offset_mask(csr_B)

    results = mfiu_pipeline.run_pipeline(
       [masks_B, masks_B], [offset_B, offset_B], len(values_A), len(values_B), 20
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

def test_dense_MFIU():
    # 运行测试
    print("\n===== 测试用例2：简单矩阵 dense(A) =====")
    A1 = np.array([[1, 1, 1, 1]])

    B1 = np.array([[1, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 0]])
    test_dense_MFIU_unit(A1, B1)

    #np.random.seed(42)
    #A3 = np.random.choice([0, 1], size=(1, 64), p=[0, 1])
    #B3 = np.random.choice([0, 1], size=(64, 10), p=[0.9, 0.1])
    #test_MFIU_unit(A3, B3)

def test_mfiu_performance():
    """运行MFIU性能测试"""
    print("\n===== MFIU性能测试 =====")
    
    # 快速测试
    from .test_mfiu_quick import quick_performance_test
    result = quick_performance_test()
    if result["speedup"] > 1.5:
        print(f"\n🎉 优化成功！获得 {result['speedup']:.2f}x 性能提升")
    elif result["speedup"] > 1.0:
        print(f"\n👍 有一定优化效果，获得 {result['speedup']:.2f}x 性能提升")
    else:
        print(f"\n🤔 优化效果不明显，可能需要进一步调整") 
    

#test_MFIU()
#test_dense_MFIU()
test_mfiu_performance()