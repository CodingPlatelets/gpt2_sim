import numpy as np
from scipy.sparse import csr_matrix
from trapezoid_module.trapezoid_sim import TrapezoidPipeline
from hbm import store_csr_in_simple_blocks_fast
from bf16_module.utils import convert_through_pipeline, bf16_to_float
from .matmul_sim import Matmul


def generate_matrix(M, N, sparse_ratio):
    
    return np.random.choice([0, 0.01], size=(M, N), p = [sparse_ratio, 1 - sparse_ratio])

def generate_matrix_batch(B, M, N, sparse_ratio):
    matrix_batch = []
    for b in range(0, B):
        matrix = np.random.choice([0, 1], size=(M, N), p = [sparse_ratio, 1 - sparse_ratio])
        matrix_batch.append(matrix)
    matrix_batch = np.array(matrix_batch).reshape(B, M, N)
    return matrix_batch 


def generate_x_wq_wk_xt(past_token_length, channel, sparse_ratio):
    
    x = generate_matrix(1, channel, 0)
    wq = generate_matrix(channel, channel, sparse_ratio)
    wk = generate_matrix(channel, channel, sparse_ratio)
    xt = generate_matrix(past_token_length + 1, channel, sparse_ratio)

    return x, wq, wk, xt

def generate_x_wq_wk_xt_batch(past_token_length, channel, sparse_ratio, batch):
    x = generate_matrix_batch(batch, 1, channel, 0)
    wq = generate_matrix(channel, channel, sparse_ratio)
    wk = generate_matrix(channel, channel, sparse_ratio)
    xt = generate_matrix_batch(batch, past_token_length + 1, channel, sparse_ratio)

    return x, wq, wk, xt

def generate_trapezoid_rows(PE_num, PE_rows):
    
    trapezoid_rows = []
    for i in range(PE_rows):
        trap = TrapezoidPipeline(1, 1, 1, PE_num)
        trapezoid_rows.append(trap)

    return trapezoid_rows

def convert_matrix_to_bf16(A):
    A_bf16 = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
    return A_bf16

def convert_batch_matrix_to_bf16(A_batch_matrices):
    bf16_A_batch_matrices = np.zeros_like(A_batch_matrices)
    for i in range(bf16_A_batch_matrices.shape[0]):
        for j in range(bf16_A_batch_matrices.shape[1]):
            for k in range(bf16_A_batch_matrices.shape[2]):
                if A_batch_matrices[i, j, k] != 0:
                    bf16_A_batch_matrices[i, j, k] = convert_through_pipeline(float(A_batch_matrices[i, j, k]))
    bf16_A_batch_matrices = np.array(bf16_A_batch_matrices).reshape(A_batch_matrices.shape[0], A_batch_matrices.shape[1], A_batch_matrices.shape[2])
    return bf16_A_batch_matrices

def convert_hbm_data_to_bf16(B_data_list):
    bf16_B_data_list = []
    for B_data in B_data_list:
        values_B = B_data.get("values", [])
        bf16_values_B = []
        for val in values_B:
            if val != 0:
                bf16_values_B.append(convert_through_pipeline(float(val)))
            else:
                bf16_values_B.append(0)
        bf16_B_data = {
            "values": bf16_values_B,
            "col_indices": B_data.get("col_indices", []),
            "row_ptr": B_data.get("row_ptr", []),
            "row_start_index": B_data.get("row_start_index", 0)
        }
        bf16_B_data_list.append(bf16_B_data)
    return bf16_B_data_list


def x_wq_wk_xt_multiply(x, wq, wk, xt, trapezoid_rows:list[TrapezoidPipeline], data_num_per_cycle):

    main_trap = trapezoid_rows[0]
    cycles = 0
    #x * wq
    expected_x_wq = np.matmul(x, wq)

    wq_from_hbm = store_csr_in_simple_blocks_fast(csr_matrix(wq.T), data_num_per_cycle)
    print("x * wq...")
    for trape in trapezoid_rows:
        trape.reset(1, wq.shape[1], wq.shape[0])
    result = main_trap.run_pipeline_hbm_multi_with_bf16([x], wq_from_hbm, trapezoid_rows, -1)
    cycles += result['cycles']

    print("test result x * wq: ")
    if np.allclose(expected_x_wq, result["combined_c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ True!")
    else:
        print("✗ False!")
        print("diff: ")
        print(expected_x_wq - result["combined_c_matrix"])
    xwq = result["combined_c_matrix_bf16"]

    expected_x_wq_k = np.matmul(expected_x_wq, wk.T) 
    wk_from_hbm = store_csr_in_simple_blocks_fast(csr_matrix(wk), data_num_per_cycle)
    wk_from_hbm_bf16 = convert_hbm_data_to_bf16(wk_from_hbm)
    print("wk...")
    for trape in trapezoid_rows:
        trape.reset(1, wk.shape[1], wk.shape[0])
    
    
    result = main_trap.run_pipeline_hbm_multi([xwq], wk_from_hbm_bf16, trapezoid_rows, -1)


    print("test result x * wq * wk: ")
    if np.allclose(expected_x_wq_k, result["combined_c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ True!")
    else:
        print("✗ False!")
        print("diff: ")
        print(expected_x_wq_k - result["combined_c_matrix"])

    

def test():
    #x, wq, wk, xt = generate_x_wq_wk_xt(4095, 4096, 0.9)
    #trapezoid_rows = generate_trapezoid_rows(128, 32)
    #x_wq_wk_xt_multiply(x, wq, wk, xt, trapezoid_rows, 256)

    matmul = Matmul(128, 32, 256)
    x, wq, wk, xt = generate_x_wq_wk_xt(100, 256, 0.9)
    x_bf16 = convert_matrix_to_bf16(x)
    matmul.load_from_hbm(wq)
    expected_xwq = np.matmul(x, wq)
    xwq = matmul.forward(x_bf16, x.shape[0], x.shape[1], wq.shape[1])
    
    matmul.load_from_hbm(wk.T)
    expected_xwqk = np.matmul(expected_xwq, wk.T)
    xwqk = matmul.forward(xwq, x.shape[0], x.shape[1], wk.shape[0], test=False)

    expected_xwqkt = np.matmul(expected_xwqk, xt.T)
    matmul.load_from_hbm(xt.T)
    xwqkt = matmul.forward(xwqk, x.shape[0], x.shape[1], xt.shape[0], test=True)



    if np.allclose(expected_xwqkt, xwqkt, rtol=1e-2, atol=1e-2):
        print("✓ True!")
    else:
        print("✗ False!")
        print("diff: ")
        print(expected_xwqkt - xwqkt)

    print()
    print(expected_xwqkt)
    print()
    print(xwqkt)

    big_num = 12000.0
    big_num_convert = bf16_to_float(convert_through_pipeline(big_num))
    print(big_num_convert)

def test_batch():
    matmul = Matmul(128, 32, 256)
    x, wq, wk, xt = generate_x_wq_wk_xt_batch(100, 256, 0.9, 4)
    #print(x.shape)
    x_bf16 = convert_batch_matrix_to_bf16(x)

    matmul.load_from_hbm(wq)
    expected_xwq = np.matmul(x, wq)
    xwq = matmul.forward(x_bf16, x.shape[1], x.shape[2], wq.shape[1], True)

    if np.allclose(expected_xwq, xwq, rtol=1e-2, atol=1e-2):
        print("✓ True!")
    else:
        print("✗ False!")
        print("diff: ")
        print(expected_xwq - xwq)


test_batch()

    

    
    

    
