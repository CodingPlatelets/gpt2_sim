import numpy as np
import time
from .trapezoid_sim import TrapezoidPipeline
from .utils import naive_matmul
from hbm.csr_hbm_values_base import store_csr_in_simple_blocks
from scipy.sparse import csr_matrix


def test_simple_case():
    """测试简单的矩阵乘法案例"""
    print("\n===== 测试简单矩阵乘法 =====")

    # 定义简单的测试矩阵
    A = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
    B = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])

    M, K = A.shape
    _, N = B.shape

    # 计算正确结果作为参考
    expected_C = naive_matmul(A, B)
    print("预期结果矩阵：")
    print(expected_C)

    # 创建TrapezoidPipeline实例
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=4)

    # 运行流水线
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_with_bf16([(A, B)], print_states=True)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")
    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])

def test_hbm_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=100000, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])

def test_hbm_small_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 128, 128

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=32)

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 32)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=8000)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["c_matrix"])


def test_hbm_multi_small_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)

    num_trapezoids = 128
    trapezoid_list = []

    for i in range(num_trapezoids):
        trap = TrapezoidPipeline(M, K, N, 128)
        trapezoid_list.append(trap)

    main_trap = trapezoid_list[0]

    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = main_trap.run_pipeline_hbm_multi_with_bf16([A], hbm_data_lists, trapezoid_list, max_cycles=100000, print_states=False)
    end_time = time.time()

    # 打印结果
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")

    print("\n预期结果矩阵：")
    print(expected_C)

    print("\n实际结果矩阵：")
    print(result["combined_c_matrix"])

    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["combined_c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵：")
        print(expected_C - result["combined_c_matrix"])

def test_sparse_matrices():
    """测试稀疏矩阵乘法"""
    print("\n===== 测试稀疏矩阵乘法 =====")

    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(A)
    print(B)

    # 计算参考结果
    expected_C = naive_matmul(A, B)

    # 创建TrapezoidPipeline实例
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    # 运行流水线
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_with_bf16([(A, B)], max_cycles=100000 ,print_states=False)
    end_time = time.time()

    # 打印结果摘要
    print(f"\n流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {result['cycles']}")
    print(
        f"A矩阵非零元素: {np.count_nonzero(A)} / {A.size} ({np.count_nonzero(A)/A.size*100:.1f}%)"
    )
    print(
        f"B矩阵非零元素: {np.count_nonzero(B)} / {B.size} ({np.count_nonzero(B)/B.size*100:.1f}%)"
    )
    print(
        f"C矩阵非零元素: {np.count_nonzero(result['c_matrix'])} / {M*N} ({np.count_nonzero(result['c_matrix'])/(M*N)*100:.1f}%)"
    )
    print(expected_C)
    print(result["c_matrix"])
    # 验证结果
    print("\n结果验证：")
    if np.allclose(expected_C, result["c_matrix"], rtol=1e-2, atol=1e-2):
        print("✓ 结果正确！")

        # 显示一些结果样本
        print("\n结果矩阵样本（前5x5）：")
        sample_size = min(5, M, N)
        print("期望值:")
        print(expected_C[:sample_size, :sample_size])
        print("实际值:")
        print(result["c_matrix"][:sample_size, :sample_size])
    else:
        print("✗ 结果不匹配！")
        print("差异矩阵样本（前5x5）：")
        sample_size = min(5, M, N)
        diff = expected_C - result["c_matrix"]
        print(diff[:sample_size, :sample_size])
        max_diff = np.max(np.abs(diff))
        print(f"最大差异: {max_diff}")


def test_multiple_matrices():
    """测试多个形状相同的矩阵乘法"""
    print("\n===== 测试多矩阵批处理 =====")

    # 定义矩阵形状
    M, K, N = 4, 32, 32

    # 创建多组测试矩阵
    num_matrices = 5
    test_cases = []
    np.random.seed(42)  # 固定随机种子以便结果可复现

    print(f"生成 {num_matrices} 组测试矩阵 (形状: {M}x{K} * {K}x{N})...")

    # 生成不同稀疏度的矩阵
    for i in range(num_matrices):
        # 随机生成不同稀疏度的矩阵
        sparsity_A = 0.8 - (i * 0.15)  # A矩阵从80%稀疏度递减
        sparsity_B = 0.9 - (i * 0.1)  # B矩阵从90%稀疏度递减

        sparsity_A = max(0.1, min(0.9, sparsity_A))  # 限制在10%-90%范围内
        sparsity_B = max(0.1, min(0.9, sparsity_B))  # 限制在10%-90%范围内

        A = np.random.choice([0, 1], size=(M, K), p=[sparsity_A, 1 - sparsity_A])
        B = np.random.choice([0, 1], size=(K, N), p=[sparsity_B, 1 - sparsity_B])

        # 确保每个矩阵至少有一些非零元素
        if np.count_nonzero(A) == 0:
            A[0, 0] = 1
        if np.count_nonzero(B) == 0:
            B[0, 0] = 1

        test_cases.append((A, B))

        print(f"矩阵组 #{i+1}:")
        print(
            f"  A非零元素: {np.count_nonzero(A)}/{A.size} ({np.count_nonzero(A)/A.size*100:.1f}%)"
        )
        print(
            f"  B非零元素: {np.count_nonzero(B)}/{B.size} ({np.count_nonzero(B)/B.size*100:.1f}%)"
        )

    # 创建TrapezoidPipeline实例
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=16)

    # 运行批处理流水线
    print("\n运行批处理流水线...")
    start_time = time.time()
    results = pipeline.run_pipeline_with_bf16(test_cases, print_states=True)
    end_time = time.time()

    # 打印总体结果
    print(f"\n流水线运行完成，总耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"总周期数: {results['cycles']}")
    print(f"平均每矩阵耗时: {(end_time - start_time)*1000/num_matrices:.2f}ms")

    # 验证所有结果
    all_correct = True
    total_flops = 0

    print("\n验证矩阵乘法结果:")
    for i, ((A, B), result) in enumerate(zip(test_cases, results["results"])):
        # 计算正确结果
        expected_C = naive_matmul(A, B)

        # 计算此次乘法的理论FLOPs
        # 对于矩阵乘法，FLOPs = 2*M*N*K (每个元素需要K次乘法和K-1次加法)
        flops = 2 * M * N * K
        total_flops += flops

        # 验证结果
        is_correct = np.allclose(expected_C, results["c_matrix"], rtol=1e-2, atol=1e-2)
        all_correct = all_correct and is_correct

        print(f"\n矩阵组 #{i+1}:")
        print(f"  结果验证: {'✓ 正确' if is_correct else '✗ 错误'}")
        print(
            f"  非零元素: {np.count_nonzero(results['c_matrix'])}/{M*N} ({np.count_nonzero(results['c_matrix'])/(M*N)*100:.1f}%)"
        )

        if not is_correct:
            # 打印差异信息
            diff = expected_C - results["c_matrix"]
            max_diff = np.max(np.abs(diff))
            print(f"  最大差异: {max_diff}")
            print("  差异矩阵样本（前3x3）:")
            sample_size = min(3, M, N)
            print(diff[:sample_size, :sample_size])

    # 计算性能指标
    total_time_ms = (end_time - start_time) * 1000
    flops_per_second = total_flops / (total_time_ms / 1000)

    print("\n总体性能指标:")
    print(f"  总计算量: {total_flops/1e6:.2f} MFLOPs")
    print(f"  计算性能: {flops_per_second/1e6:.2f} MFLOPs/s")
    print(f"  总体结果: {'全部正确 ✓' if all_correct else '存在错误 ✗'}")

    return results


#test_sparse_matrices()
# test_multiple_matrices()
#test_hbm_matrices()
#test_hbm_small_matrices()
test_hbm_multi_small_matrices()
