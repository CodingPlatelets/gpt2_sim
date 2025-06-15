from trapezoid_module.trapezoid_sim import TrapezoidPipeline
from trapezoid_module.utils import naive_matmul
import time
import numpy as np
import torch
from scipy.sparse import csr_matrix
from zb_idea.store_csr_in_simple_blocks import store_csr_in_simple_blocks_fast
from zb_idea.vector_matrix_row_product_HBM import VectorMatrixRowProductSimulatorWithHBM


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

    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    hbm_data_lists = store_csr_in_simple_blocks_fast(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=100000, print_states=False)
    tgx_cycle = result['cycles']
    tgx_output = result["c_matrix"]
    end_time = time.time()
    
    print(tgx_output)
    print(tgx_output.shape)
    print(tgx_output.dtype)

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    
    sparsity = 0.9
    B_dense = torch.randn((vec_dim, vec_dim)) * 0.01
    mask = torch.rand(vec_dim, vec_dim) > sparsity
    B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)

    simulator = VectorMatrixRowProductSimulatorWithHBM(
        num_perows=32, 
        pes_per_row=128, 
        vector_size=4096
        )
    A_vector = np.asarray(tgx_output).flatten()
    result = simulator.run_simulation(A_vector, B_sparse, 256)

    simulator.verify_result(tgx_output, B_sparse)
    ly_output = result["output_vector"]
    ly_cycle = result["clock"]

    total_cycle = tgx_cycle + ly_cycle
    print(ly_output)
    print(ly_output.shape)
    print(ly_output.dtype)

    print(f"总周期数: {total_cycle}")
    print(f"TGX周期数: {tgx_cycle}")
    print(f"LY周期数: {ly_cycle}")

if __name__ == "__main__":
    test_combine()


