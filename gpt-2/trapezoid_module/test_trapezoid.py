import numpy as np
import time
from .trapezoid_sim import TrapezoidPipeline
from .utils import naive_matmul, convert_through_pipeline
from hbm.csr_hbm_values_base import store_csr_in_simple_blocks, store_csr_in_simple_blocks_fast
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

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1]) * 0.1
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1]) * 0.1
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)
    pipeline = TrapezoidPipeline(M=M, K=K, N=N, PE_num=128)

    hbm_data_lists = store_csr_in_simple_blocks_fast(csr_matrix(B.T), 256)
    #print(hbm_data_lists)
    print("\n运行流水线...")
    start_time = time.time()
    result = pipeline.run_pipeline_hbm_with_bf16([A], hbm_data_lists, max_cycles=-1, print_states=False)
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

def test_hbm_multi_batch_for_weight():
    """测试批处理权重共享HBM多流水线处理"""
    print("\n===== 测试批处理权重共享HBM处理 =====")
    
    # 基础参数设置
    np.random.seed(42)
    M, K, N = 1, 1024, 1024
    num_trapezoids = 32  # pe_row
    
    # 生成共享的权重矩阵B
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 256)
    
    print(f"基础设置: M={M}, K={K}, N={N}, pe_row={num_trapezoids}")
    print(f"权重矩阵B非零元素: {np.count_nonzero(B)}/{B.size} ({np.count_nonzero(B)/B.size*100:.1f}%)")
    print(f"HBM数据块数量: {len(hbm_data_lists)}")

    # 测试场景配置
    test_scenarios = [
        {
            "name": "情况1: batch=4, pe_row=32 (完全均匀分组)",
            "batch_size": 4,
            "description": "pe_row % batch == 0, 每个batch分配8个trapezoid"
        },
        {
            "name": "情况2: batch=8, pe_row=32 (完全均匀分组)", 
            "batch_size": 8,
            "description": "pe_row % batch == 0, 每个batch分配4个trapezoid"
        },
        {
            "name": "情况3: batch=16, pe_row=32 (完全均匀分组)",
            "batch_size": 16,
            "description": "pe_row % batch == 0, 每个batch分配2个trapezoid"
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']}")
        print(f"📝 {scenario['description']}")
        
        batch_size = scenario["batch_size"]
        
        # 生成批处理A矩阵 - 每个batch使用不同的A矩阵
        A_batch_matrices = []
        expected_results = []
        
        A_batch_matrices = []
        for i in range(batch_size):
            A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
            A_batch_matrices.append(A)
        A_batch_matrices = np.array(A_batch_matrices).reshape(batch_size, M, K)

        expected_results = [naive_matmul(A_batch_matrices[i], B) for i in range(batch_size)]
        
        print(f"生成了{batch_size}个不同的A矩阵作为batch输入")
        
        # 🔧 新增：将A矩阵转换为BF16格式
        print("正在转换A矩阵为BF16格式...")
        
        bf16_A_batch_matrices = np.zeros_like(A_batch_matrices)
        for i in range(bf16_A_batch_matrices.shape[0]):
            for j in range(bf16_A_batch_matrices.shape[1]):
                for k in range(bf16_A_batch_matrices.shape[2]):
                    if A_batch_matrices[i, j, k] != 0:
                        bf16_A_batch_matrices[i, j, k] = convert_through_pipeline(float(A_batch_matrices[i, j, k]))
        bf16_A_batch_matrices = np.array(bf16_A_batch_matrices).reshape(batch_size, M, K)

        # 🔧 新增：将B数据列表转换为BF16格式
        print("正在转换B数据为BF16格式...")
        bf16_hbm_data_lists = []
        
        for data_idx, B_data in enumerate(hbm_data_lists):
            # 转换B矩阵的values
            values_B = B_data.get("values", [])
            bf16_values_B = []
            for val in values_B:
                if val != 0:
                    bf16_values_B.append(convert_through_pipeline(float(val)))
                else:
                    bf16_values_B.append(0)

            # 创建新的B数据字典，保持col_indices和row_ptr不变
            bf16_B_data = {
                "values": bf16_values_B,
                "col_indices": B_data.get("col_indices", []),
                "row_ptr": B_data.get("row_ptr", []),
                "row_start_index": B_data.get("row_start_index", 0)
            }
            bf16_hbm_data_lists.append(bf16_B_data)
            
            if data_idx % 100 == 0 or data_idx == len(hbm_data_lists) - 1:
                print(f"  B数据块转换进度: {data_idx + 1}/{len(hbm_data_lists)}")
        
        print(f"✅ BF16转换完成！")
        print(f"   A矩阵batch数: {bf16_A_batch_matrices.shape[0]}")
        print(f"   B数据块数: {len(bf16_hbm_data_lists)}")
        
        # 创建trapezoid列表
        trapezoid_list = []
        for i in range(num_trapezoids):
            trap = TrapezoidPipeline(M, K, N, 128)
            trapezoid_list.append(trap)
        
        main_trap = trapezoid_list[0]
        
        # 🔧 修改：使用BF16格式的数据运行批处理权重共享流水线
        print("运行批处理权重共享流水线（BF16模式）...")
        start_time = time.time()
        result = main_trap.run_pipeline_hbm_multi_batch_for_weight_fast(
            bf16_A_batch_matrices,  # 使用BF16格式的A矩阵
            bf16_hbm_data_lists,    # 使用BF16格式的B数据
            trapezoid_list, 
            max_cycles=-1, 
            print_states=False
        )
        end_time = time.time()
        
        # 打印基本结果信息
        print(f"✅ 流水线运行完成，耗时: {(end_time - start_time)*1000:.2f}ms")
        print(f"总周期数: {result['cycles']}")
        print(f"处理的batch数: {result['num_batches']}")
        print(f"每batch分配的PE数: {result['pe_row_per_batch']}")
        
        print("\n📊 结果验证:")
        print(f"生成了 {len(result['individual_batch_results'])} 个batch结果")
        
        # 验证每个batch的独立结果
        all_correct = True
        for batch_idx in range(batch_size):
            batch_key = f"batch_{batch_idx}"
            if batch_key in result['individual_batch_results']:
                actual_result = result['individual_batch_results'][batch_key]['c_matrix']
                expected_result = expected_results[batch_idx]
                
                # 🔧 修改：由于使用BF16，需要放宽精度要求
                is_correct = np.allclose(expected_result, actual_result, rtol=5e-2, atol=5e-2)
                
                if is_correct:
                    print(f"  Batch {batch_idx}: ✓ 正确")
                else:
                    print(f"  Batch {batch_idx}: ✗ 错误")
                    max_diff = np.max(np.abs(expected_result - actual_result))
                    mean_expected = np.mean(np.abs(expected_result))
                    print(f"    最大差异: {max_diff}")
                    print(f"    相对误差: {max_diff/mean_expected*100:.2f}%")
                    print(f"    期望结果前5个元素: {expected_result.flatten()[:5]}")
                    print(f"    实际结果前5个元素: {actual_result.flatten()[:5]}")
                    all_correct = False
            else:
                print(f"  Batch {batch_idx}: ⚠️  未找到结果")
                all_correct = False
        
        if all_correct:
            print("✅ 所有batch结果验证通过！")
        else:
            print("❌ 部分batch结果验证失败！")
            # 🔧 新增：详细的调试信息
            print("\n🔍 调试信息:")
            for batch_idx in range(min(2, batch_size)):  # 只打印前2个batch的详细信息
                batch_key = f"batch_{batch_idx}"
                if batch_key in result['individual_batch_results']:
                    actual_result = result['individual_batch_results'][batch_key]['c_matrix']
                    print(f"  Batch {batch_idx}:")
                    print(f"    实际结果统计: 非零={np.count_nonzero(actual_result)}, "
                          f"最大值={np.max(actual_result):.6f}, "
                          f"最小值={np.min(actual_result):.6f}")
                    print(f"    期望结果统计: 非零={np.count_nonzero(expected_results[batch_idx])}, "
                          f"最大值={np.max(expected_results[batch_idx]):.6f}, "
                          f"最小值={np.min(expected_results[batch_idx]):.6f}")
        
        # 验证combined_c_matrix的形状
        combined_shape = result['combined_c_matrix'].shape
        expected_shape = (batch_size, M, N)
        print(f"\n📐 形状验证:")
        print(f"组合矩阵形状 - 期望: {expected_shape}, 实际: {combined_shape}")
        if combined_shape == expected_shape:
            print("✅ 组合矩阵形状正确！")
        else:
            print("❌ 组合矩阵形状不符合要求！")
        
        # 计算总体统计信息
        total_nonzero_actual = np.count_nonzero(result['combined_c_matrix'])
        expected_total_nonzero = sum(np.count_nonzero(expected_results[i]) 
                                    for i in range(batch_size))
        
        print(f"\n📈 统计信息:")
        print(f"总非零元素 - 期望: {expected_total_nonzero}, 实际: {total_nonzero_actual}")
        
        # 性能指标
        total_ops = batch_size * 2 * M * N * K  # 每个batch的FLOPs
        ops_per_second = total_ops / ((end_time - start_time) + 1e-9)  # 避免除零
        
        print(f"\n📈 性能指标:")
        print(f"总计算量: {total_ops/1e6:.2f} MFLOPs")
        print(f"计算性能: {ops_per_second/1e6:.2f} MFLOPs/s")
        print(f"平均每batch周期数: {result['cycles']/batch_size:.1f}")
        
        print("-" * 80)


def test_hbm_multi_batch_for_weight_edge_cases():
    """测试批处理权重共享HBM的边界情况"""
    print("\n===== 测试批处理权重共享HBM边界情况 =====")
    
    # 基础参数设置
    np.random.seed(42)
    M, K, N = 1, 512, 512
    num_trapezoids = 16  # 较小的pe_row用于测试
    
    # 生成共享的权重矩阵B
    B = np.random.choice([0, 1], size=(K, N), p=[0.95, 0.05])  # 更稀疏的矩阵
    hbm_data_lists = store_csr_in_simple_blocks(csr_matrix(B.T), 128)
    
    print(f"基础设置: M={M}, K={K}, N={N}, pe_row={num_trapezoids}")
    print(f"权重矩阵B非零元素: {np.count_nonzero(B)}/{B.size} ({np.count_nonzero(B)/B.size*100:.1f}%)")
    
    # 测试错误情况
    print("\n🔍 测试错误输入验证:")
    
    # 创建trapezoid列表
    trapezoid_list = []
    for i in range(num_trapezoids):
        trap = TrapezoidPipeline(M, K, N, 64)
        trapezoid_list.append(trap)
    
    main_trap = trapezoid_list[0]
    
    # 测试1: batch不是2的幂次方
    try:
        A_batch_invalid = [np.random.random((M, K)) for _ in range(3)]  # 3不是2的幂次方
        result = main_trap.run_pipeline_hbm_multi_batch_for_weight(
            A_batch_invalid, hbm_data_lists, trapezoid_list, max_cycles=1000
        )
        print("❌ 应该抛出batch必须是2的幂次方的错误")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")
    
    # 测试2: batch > pe_row
    try:
        A_batch_too_large = [np.random.random((M, K)) for _ in range(32)]  # 32 > 16
        result = main_trap.run_pipeline_hbm_multi_batch_for_weight(
            A_batch_too_large, hbm_data_lists, trapezoid_list, max_cycles=1000
        )
        print("❌ 应该抛出batch > pe_row的错误")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")
    
    # 测试3: pe_row % batch != 0的情况
    # 修改trapezoid数量为奇数
    trapezoid_list_odd = trapezoid_list[:15]  # 15个trapezoid
    try:
        A_batch_valid = [np.random.random((M, K)) for _ in range(4)]  # 4是2的幂次方
        result = main_trap.run_pipeline_hbm_multi_batch_for_weight(
            A_batch_valid, hbm_data_lists, trapezoid_list_odd, max_cycles=1000
        )
        print("❌ 应该抛出pe_row % batch != 0的错误")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")
    
    # 测试4: 正常情况 - batch=1的特殊情况
    print("\n✅ 测试正常情况 - batch=1:")
    A_batch_single = [np.random.choice([0, 1], size=(M, K), p=[0.3, 0.7]) * 0.1]
    expected_single = naive_matmul(A_batch_single[0], B)
    
    start_time = time.time()
    result = main_trap.run_pipeline_hbm_multi_batch_for_weight(
        A_batch_single, hbm_data_lists, trapezoid_list, max_cycles=50000
    )
    end_time = time.time()
    
    print(f"单batch处理耗时: {(end_time - start_time)*1000:.2f}ms")
    print(f"组合矩阵形状: {result['combined_c_matrix'].shape}")
    
    # 验证单batch结果
    actual_single = result['combined_c_matrix'][0]  # 取第一个batch的结果
    is_correct = np.allclose(expected_single, actual_single, rtol=1e-2, atol=1e-2)
    print(f"单batch结果验证: {'✅ 正确' if is_correct else '❌ 错误'}")
    
    print("-" * 80)


def test_hbm_multi_matrices():
    # 创建稀疏矩阵
    np.random.seed(42)
    M, K, N = 1, 4096, 4096

    # 随机生成稀疏矩阵

    A = np.random.choice([0, 1], size=(M, K), p=[0, 1])
    B = np.random.choice([0, 1], size=(K, N), p=[0.9, 0.1])
    print(B)
    #M, K, N = 1, 4, 3
    expected_C = naive_matmul(A, B)

    num_trapezoids = 32
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
            f"  非零元素: {np.count_nonzero(results["c_matrix"])}/{M*N} ({np.count_nonzero(results["c_matrix"])/(M*N)*100:.1f}%)"
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
#test_hbm_multi_matrices()
test_hbm_multi_batch_for_weight()
#test_hbm_multi_batch_for_weight_edge_cases()

