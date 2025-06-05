import numpy as np
import time
from scipy.sparse import csr_matrix
from tqdm import tqdm
from ..module.mfiu_sim import MFIUPipeline
from ..module.mfiu_sim_dense import MFIUPipelineDenseA
from ..utils import get_values_offset_mask

def test_large_matrix_performance():
    """测试大矩阵的性能对比"""
    print("🚀 大矩阵MFIU性能测试")
    print("=" * 60)
    
    # 测试不同大小的矩阵
    test_cases = [
        {"M": 1, "K": 32, "N": 32, "name": "中等矩阵 (1x32x32)", "sparsity": 0.8},
        {"M": 1, "K": 64, "N": 64, "name": "大矩阵 (1x64x64)", "sparsity": 0.85},
        {"M": 1, "K": 128, "N": 128, "name": "超大矩阵 (1x128x128)", "sparsity": 0.9},
        {"M": 1, "K": 256, "N": 256, "name": "巨型矩阵 (1x256x256)", "sparsity": 0.95},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n{'='*20} {case['name']} {'='*20}")
        M, K, N = case["M"], case["K"], case["N"]
        sparsity = case["sparsity"]
        
        # 生成测试数据
        print(f"📊 生成测试数据...")
        np.random.seed(42)
        A = np.ones((M, K), dtype=int)  # 稠密A矩阵
        B = np.random.choice([0, 1], size=(K, N), p=[sparsity, 1-sparsity])  # 稀疏B矩阵
        
        print(f"  A矩阵: {M}×{K}, 稠密度 100.0%")
        print(f"  B矩阵: {K}×{N}, 稠密度 {np.count_nonzero(B) / B.size * 100:.1f}%")
        print(f"  总元素数: {M*K + K*N:,}")
        
        # 预处理数据
        print(f"🔄 预处理数据...")
        csr_A = csr_matrix(A)
        csr_B = csr_matrix(B.T)
        values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
        values_B, offset_B, masks_B = get_values_offset_mask(csr_B)
        
        width = M * N
        bit_width = K
        
        print(f"  Width: {width}, Bit_width: {bit_width}")
        print(f"  A values: {len(values_A)}, B values: {len(values_B)}")
        
        # 测试原始MFIU
        print(f"🔄 测试原始MFIU...")
        original_mfiu = MFIUPipeline(width, bit_width)
        
        start_time = time.time()
        try:
            original_results = original_mfiu.run_pipeline(
                [masks_A], [masks_B], [offset_A], [offset_B], 
                len(values_A), len(values_B), max_cycles=50
            )
            original_time = time.time() - start_time
            original_valid_cycles = sum(1 for r in original_results if r["valid"])
            original_success = True
        except Exception as e:
            print(f"  ❌ 原始MFIU失败: {e}")
            original_time = float('inf')
            original_valid_cycles = 0
            original_success = False
        
        if original_success:
            print(f"  ✅ 完成: {original_time:.3f}s, {original_valid_cycles}个有效周期")
        
        # 测试稠密A优化MFIU
        print(f"🚀 测试稠密A优化MFIU...")
        dense_mfiu = MFIUPipelineDenseA(width, bit_width)
        
        start_time = time.time()
        try:
            dense_results = dense_mfiu.run_pipeline(
                mask_B_cols=[masks_B], 
                offset_B_cols=[offset_B], 
                len_values_A=len(values_A), 
                len_values_B=len(values_B), 
                max_cycles=50
            )
            dense_time = time.time() - start_time
            dense_valid_cycles = sum(1 for r in dense_results if r["valid"])
            dense_success = True
        except Exception as e:
            print(f"  ❌ 稠密A优化失败: {e}")
            dense_time = float('inf')
            dense_valid_cycles = 0
            dense_success = False
        
        if dense_success:
            print(f"  ✅ 完成: {dense_time:.3f}s, {dense_valid_cycles}个有效周期")
        
        # 结果对比
        if original_success and dense_success:
            speedup = original_time / dense_time if dense_time > 0 else float('inf')
            time_saved = (original_time - dense_time) * 1000
            
            print(f"\n📊 性能对比:")
            print(f"  原始MFIU: {original_time:.3f}s")
            print(f"  稠密A优化: {dense_time:.3f}s")
            print(f"  性能提升: {speedup:.2f}x")
            print(f"  时间节省: {time_saved:.1f}ms")
            print(f"  有效周期: 原始{original_valid_cycles} vs 优化{dense_valid_cycles}")
            
            # 简单验证结果一致性（只检查第一个有效输出）
            original_output = None
            dense_output = None
            
            for res in original_results:
                if res["valid"]:
                    original_output = res["output"]
                    break
            
            for res in dense_results:
                if res["valid"]:
                    dense_output = res["output"]
                    break
            
            results_match = False
            if original_output and dense_output:
                if (np.array_equal(original_output[0], dense_output[0]) and 
                    np.array_equal(original_output[1], dense_output[1])):
                    print(f"  结果验证: ✅ 输出一致")
                    results_match = True
                else:
                    print(f"  结果验证: ❌ 输出不一致")
            else:
                print(f"  结果验证: ⚠️ 缺少有效输出")
            
            results.append({
                "name": case["name"],
                "size": f"{M}×{K}×{N}",
                "elements": M*K + K*N,
                "original_time": original_time,
                "dense_time": dense_time,
                "speedup": speedup,
                "time_saved_ms": time_saved,
                "results_match": results_match,
                "original_cycles": original_valid_cycles,
                "dense_cycles": dense_valid_cycles,
            })
        else:
            print(f"  ⚠️ 测试失败，跳过对比")
    
    # 总结报告
    print(f"\n{'='*20} 测试总结 {'='*20}")
    print(f"{'矩阵规模':<20} {'性能提升':<10} {'时间节省':<12} {'结果正确':<10}")
    print("-" * 60)
    
    for result in results:
        match_icon = "✅" if result["results_match"] else "❌"
        print(f"{result['size']:<20} {result['speedup']:.2f}x{'':<6} "
              f"{result['time_saved_ms']:.1f}ms{'':<6} {match_icon}")
    
    # 计算平均性能提升
    if results:
        avg_speedup = np.mean([r["speedup"] for r in results if r["speedup"] != float('inf')])
        total_time_saved = sum([r["time_saved_ms"] for r in results])
        success_rate = sum([1 for r in results if r["results_match"]]) / len(results) * 100
        
        print("-" * 60)
        print(f"{'平均性能提升:':<20} {avg_speedup:.2f}x")
        print(f"{'总时间节省:':<20} {total_time_saved:.1f}ms")
        print(f"{'结果正确率:':<20} {success_rate:.1f}%")
    
    return results

def test_extreme_large_matrix():
    """测试极大矩阵（谨慎使用）"""
    print("\n🔥 极大矩阵测试 (谨慎运行)")
    print("=" * 40)
    
    # 极大矩阵参数
    M, K, N = 1, 4096, 4096
    sparsity = 0.9  # 非常稀疏
    
    print(f"⚠️  即将测试 {M}×{K}×{N} 矩阵")
    print(f"⚠️  总元素数: {M*K + K*N:,}")
    print(f"⚠️  B矩阵稀疏度: {sparsity*100:.1f}%")
    
    # 询问是否继续
    try:
        user_input = input("是否继续？(y/N): ").lower().strip()
        if user_input not in ['y', 'yes']:
            print("❌ 用户取消测试")
            return None
    except:
        print("❌ 无法获取用户输入，取消测试")
        return None
    
    print(f"🚀 开始极大矩阵测试...")
    
    # 生成数据
    print(f"📊 生成测试数据...")
    np.random.seed(42)
    A = np.ones((M, K), dtype=int)
    B = np.random.choice([0, 1], size=(K, N), p=[sparsity, 1-sparsity])
    
    print(f"  实际B稠密度: {np.count_nonzero(B) / B.size * 100:.2f}%")
    
    # 预处理
    print(f"🔄 预处理数据...")
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)
    values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    values_B, offset_B, masks_B = get_values_offset_mask(csr_B)
    
    width = M * N
    bit_width = K
    
    print(f"  Width: {width}, Bit_width: {bit_width}")
    print(f"  预计内存使用: ~{(width * bit_width * 8) / 1024 / 1024:.1f}MB")
    
    # 只测试稠密A优化版本（原始版本可能太慢）
    print(f"🚀 测试稠密A优化MFIU...")
    dense_mfiu = MFIUPipelineDenseA(width, bit_width)
    
    start_time = time.time()
    try:
        with tqdm(total=100, desc="MFIU处理") as pbar:
            dense_results = dense_mfiu.run_pipeline(
                mask_B_cols=[masks_B], 
                offset_B_cols=[offset_B], 
                len_values_A=len(values_A), 
                len_values_B=len(values_B), 
                max_cycles=100
            )
            pbar.update(100)
        
        dense_time = time.time() - start_time
        dense_valid_cycles = sum(1 for r in dense_results if r["valid"])
        
        print(f"✅ 完成极大矩阵测试!")
        print(f"  处理时间: {dense_time:.2f}s")
        print(f"  有效周期: {dense_valid_cycles}")
        print(f"  每周期平均时间: {dense_time/len(dense_results)*1000:.2f}ms")
        
        return {
            "size": f"{M}×{K}×{N}",
            "time": dense_time,
            "cycles": dense_valid_cycles,
            "avg_cycle_time": dense_time/len(dense_results)
        }
        
    except Exception as e:
        print(f"❌ 极大矩阵测试失败: {e}")
        return None

if __name__ == "__main__":
    # 运行大矩阵测试
    results = test_large_matrix_performance()
    
    # 可选：运行极大矩阵测试
    print(f"\n" + "="*60)
    extreme_result = test_extreme_large_matrix()
    if extreme_result:
        print(f"🎉 极大矩阵测试成功完成！") 