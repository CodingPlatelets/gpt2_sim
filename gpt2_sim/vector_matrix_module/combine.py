from trapezoid_module.trapezoid_sim import TrapezoidPipeline
from trapezoid_module.utils import naive_matmul
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix
from vector_matrix.store_csr_in_simple_blocks import store_csr_in_simple_blocks_fast
from vector_matrix.vector_matrix_row_product_HBM import VectorMatrixRowProductSimulatorWithHBM
from vector_matrix.softmax import softmax

def test_combine():
# 创建稀疏矩阵
    np.random.seed(42)
    vec_dim = 4096
    sparsity = 0.9

    M, K, N = 1, vec_dim, vec_dim

    # 随机生成稀疏矩阵

    A = np.random.randn(vec_dim).astype(np.float32) * 0.1
    A = A.reshape(1, vec_dim)  
    
    B_dense = np.random.randn(vec_dim, vec_dim).astype(np.float32) * 0.01
    mask = np.random.rand(vec_dim, vec_dim) > sparsity
    B = np.where(mask, B_dense, 0).astype(np.float32)

    expected_C = naive_matmul(A, B)

    num_trapezoids = 32
    trapezoid_list = []

    for i in range(num_trapezoids):
        trap =  TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)
        trapezoid_list.append(trap)

    main_trap = trapezoid_list[0]

    hbm_data_lists = store_csr_in_simple_blocks_fast(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()


    result = main_trap.run_pipeline_hbm_multi_with_bf16([A], hbm_data_lists, trapezoid_list, max_cycles=100000, print_states=False)
    tgx_cycle = result['cycles']
    tgx_output = result["combined_c_matrix"]
    end_time = time.time()
    
    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")
        # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["combined_c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["combined_c_matrix"])

    output = softmax(tgx_output)

    
    B_dense = torch.randn((vec_dim, vec_dim)) * 0.01
    mask = torch.rand(vec_dim, vec_dim) > sparsity
    B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)


    simulator = VectorMatrixRowProductSimulatorWithHBM(num_perows=32,  pes_per_row=128,  vector_size=4096 )
    
    A_vector = np.asarray(output).flatten()
    result = simulator.run_simulation(A_vector, B_sparse, 256)

    simulator.verify_result(tgx_output, B_sparse)
    ly_cycle = result["clock"]

    total_cycle = tgx_cycle + ly_cycle

    print(f"总周期数: {total_cycle}")


if __name__ == "__main__":
    test_combine()


