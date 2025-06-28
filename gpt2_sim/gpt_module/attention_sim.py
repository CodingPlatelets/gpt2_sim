from .matmul_sim import Matmul
# 使用 software_hw_sim.py 中的硬件模拟版本
from softmax_module.software_hw_sim import SoftmaxPipeline
from .test import generate_x_wq_wk_xt, generate_matrix
from .test_tgx import generate_x_wq_wk_xt_batch
from vector_matrix_module.row_product_module import RowProduct
from vector_matrix_module.row_product_multibatch_module import RowProduct as RowProductMultiBatch
from bf16_module.utils import convert_through_pipeline
import numpy as np

class Attention:

    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        self.data_num_per_cycle = data_num_per_cycle
        self.cycles = 0
        self.hbm_data = {}
        
        self.Wq = Matmul(PE_num, PE_rows, data_num_per_cycle)
        self.Wk = Matmul(PE_num, PE_rows, data_num_per_cycle)
        self.XT = Matmul(PE_num, PE_rows, data_num_per_cycle)
        self.Wv = RowProduct(PE_num, PE_rows, data_num_per_cycle)
        self.Wv_multi_batch = RowProductMultiBatch(PE_num, PE_rows, data_num_per_cycle)

        # 使用 software_hw_sim.py 中的 SoftmaxPipeline
        # 窗口大小可以根据 PE 配置自适应调整
        front_window = min(8, PE_rows // 4)  # 前窗口大小
        back_window = min(8, PE_rows // 4)   # 后窗口大小
        max_rows = min(4, PE_rows // 8)      # 最大并行行数
        
        self.softmax = SoftmaxPipeline(
            front_window=front_window,
            back_window=back_window, 
            max_rows=max_rows
        )
        
        # 存储 softmax 相关的模拟信息
        self.softmax_sim_info = {}

    def forward(self, x, past_token_num):
        # 对于 Wq、Wk 来说，它们都是 (K, K) 的方阵，其列数应与输入向量的维度 K 相同。

        if x.ndim == 3:
            k_dim = x.shape[2]
            M = x.shape[1]
        else:
            k_dim = x.shape[1]
            M = x.shape[0]

        # 1) X @ Wq
        
        xw = self.Wq.forward(x, M, k_dim, k_dim, test=False)

        # 2) (XWq) @ Wk^T
        xww = self.Wk.forward(xw, M, k_dim, k_dim, test=False)

        # 3) ((XWq)Wk^T) @ XT^T
        xwwx = self.XT.forward(xww, M, k_dim, past_token_num, test=True)

        self.cycles += self.Wq.cycles + self.Wk.cycles + self.XT.cycles

        # 4) softmax - 使用硬件流水线模拟版本
        softmax_out, softmax_sim_info = self._run_softmax_pipeline(xwwx)

        if x.ndim == 3:
            xwwxt = self.Wv_multi_batch.forward(softmax_out)
            self.cycles += self.Wv_multi_batch.cycles
        else:
            xwwxt = self.Wv.forward(softmax_out)
            self.cycles += self.Wv.cycles
         
        # 累计周期数（包含 softmax 的周期）
        self.cycles += softmax_sim_info.get('total_cycles', 0)
        self.softmax_sim_info = softmax_sim_info
        
        # 返回结果和模拟信息
        return softmax_out, xwwxt, softmax_sim_info

    def _run_softmax_pipeline(self, input_matrix):
        """
        运行 softmax 硬件流水线模拟
        
        Args:
            input_matrix: 输入矩阵 (batch_size, seq_len) 或 (batch_size, 1, seq_len)
            
        Returns:
            tuple: (softmax_output, simulation_info)
        """
        import time
        start_time = time.time()
        
        # 确保输入是二维的
        if input_matrix.ndim == 1:
            input_matrix = input_matrix.reshape(1, -1)
        # 新增：支持 (batch, 1, length) 自动 squeeze
        if input_matrix.ndim == 3 and input_matrix.shape[1] == 1:
            input_matrix = input_matrix.squeeze(1)

        batch_size, seq_len = input_matrix.shape
        
        # 准备流水线输入数据: (val, row_id, col_idx, row_length)
        pipeline_input = []
        
        for row_id in range(batch_size):
            # 按照窗口优先的顺序准备数据
            row_data = input_matrix[row_id]
            
            # 前窗口数据
            front_window = self.softmax.front_window
            for col_idx in range(min(front_window, seq_len)):
                val = float(row_data[col_idx])
                pipeline_input.append((val, row_id, col_idx, seq_len))
            
            # 后窗口数据
            back_window = self.softmax.back_window
            for col_idx in range(max(0, seq_len - back_window), seq_len):
                if col_idx >= front_window:  # 避免重复
                    val = float(row_data[col_idx])
                    pipeline_input.append((val, row_id, col_idx, seq_len))
            
            # 剩余数据
            for col_idx in range(front_window, max(front_window, seq_len - back_window)):
                val = float(row_data[col_idx])
                pipeline_input.append((val, row_id, col_idx, seq_len))
        
        # 运行流水线
        results = self.softmax.run_pipeline(
            pipeline_input, 
            max_cycles=2000,  # 增加最大周期数以适应大矩阵
            print_progress=False  # 在 attention 中不打印详细进度
        )
        
        # 重构输出矩阵
        output_matrix = np.zeros_like(input_matrix)

        for row_id in range(batch_size):
            if row_id in results:
                row_results = results[row_id]
                for col_idx in sorted(row_results.keys()):
                    # 从 BF16 转换为 float
                    from softmax_module.software_hw_sim import bf16_to_float
                    softmax_val = bf16_to_float(row_results[col_idx])
                    output_matrix[row_id, col_idx] = softmax_val
        
        # 计算模拟信息
        execution_time = time.time() - start_time
        total_cycles = self.softmax.cycle_count
        
        simulation_info = {
            'total_cycles': total_cycles,
            'execution_time': execution_time,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'total_elements': batch_size * seq_len,
            'pipeline_input_count': len(pipeline_input),
            'front_window': self.softmax.front_window,
            'back_window': self.softmax.back_window,
            'max_rows': self.softmax.max_rows,
            'pe_efficiency': self._calculate_pe_efficiency(total_cycles, batch_size * seq_len),
            'cycles_breakdown': {
                'data_input': len(pipeline_input),
                'pipeline_processing': total_cycles - len(pipeline_input),
                'total': total_cycles
            }
        }
        
        return output_matrix, simulation_info
    
    def _calculate_pe_efficiency(self, cycles, total_elements):
        """计算 PE 利用效率"""
        if cycles == 0:
            return 0.0
        
        # 估算理论最少周期数 (假设完美并行)
        theoretical_min_cycles = max(1, total_elements // self.PE_num)
        return theoretical_min_cycles / cycles if cycles > 0 else 0.0


def convert_matrix_to_bf16(A):
    A_bf16 = np.zeros_like(A)
    if A.ndim == 1:
        for i in range(A.shape[0]):
            A_bf16[i] = convert_through_pipeline(float(A[i]))
    else:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
    return A_bf16




def test():
    vector_size = 1024  # 适中的向量大小用于测试
    attention = Attention(128, 32, 256)

    
    X, wq, wk, xt = generate_x_wq_wk_xt(past_token_length=127, channel=vector_size, sparse_ratio=0.95)  # 减small past_token_length
    past_token_num = xt.shape[0]

    wv = generate_matrix(past_token_num , vector_size, 0.9)

    # 更新测试代码以使用简单的 Softmax 进行对比
    from vector_matrix_module.softmax import Softmax
    expected_xwqkx =  (Softmax().forward( ( (X @ wq) @ wk.T ) @ xt.T ) ) @ wv

    attention.Wq.load_from_hbm(wq)
    attention.Wk.load_from_hbm(wk.T)
    attention.XT.load_from_hbm(xt.T)
    attention.Wv.load_from_hbm(wv)
    
    x_bf16 = convert_matrix_to_bf16(X)

    print("开始运行 Attention 硬件模拟...")
    import time
    start_time = time.time()
    
    # 更新调用方式以接收额外的模拟信息
    out, xwwxt, softmax_sim_info = attention.forward(x_bf16, past_token_num= past_token_num)
    
    total_time = time.time() - start_time

    # 打印详细的模拟信息
    print("=" * 50)
    print("=== Softmax 硬件流水线模拟信息 ===")
    print(f"输入矩阵形状: {X.shape} -> softmax输入: {out.shape}")
    print(f"序列长度: {softmax_sim_info['seq_len']}")
    print(f"批次大小: {softmax_sim_info['batch_size']}")
    print(f"总处理元素: {softmax_sim_info['total_elements']}")
    print()
    print(f"=== 流水线配置 ===")
    print(f"前窗口大小: {softmax_sim_info['front_window']}")
    print(f"后窗口大小: {softmax_sim_info['back_window']}")
    print(f"最大并行行数: {softmax_sim_info['max_rows']}")
    print(f"流水线输入数据点: {softmax_sim_info['pipeline_input_count']}")
    print()
    print(f"=== 性能指标 ===")
    print(f"总周期数: {softmax_sim_info['total_cycles']}")
    print(f"PE 利用效率: {softmax_sim_info['pe_efficiency']:.2%}")
    print(f"模拟执行时间: {softmax_sim_info['execution_time']*1000:.2f}ms")
    print(f"总执行时间: {total_time*1000:.2f}ms")
    print()
    print(f"=== 周期分解 ===")
    for stage, cycles in softmax_sim_info['cycles_breakdown'].items():
        percentage = cycles / softmax_sim_info['total_cycles'] * 100 if softmax_sim_info['total_cycles'] > 0 else 0
        print(f"{stage}: {cycles} 周期 ({percentage:.1f}%)")
    print()
    print(f"=== 整体 Attention 性能 ===")
    print(f"Attention 总周期数: {attention.cycles}")
    print("=" * 50)
    
    # 验证 softmax 输出的性质
    print(f"\n=== Softmax 输出验证 ===")
    row_sums = np.sum(out, axis=-1)
    print(f"Softmax 行和: {row_sums} (应该接近1)")
    print(f"行和检查通过: {np.allclose(row_sums, 1.0, rtol=1e-2, atol=1e-2)}")
    
    # 准确性验证
    print(f"\n=== 准确性验证 ===")
    if np.allclose(expected_xwqkx, xwwxt, rtol=1e-2, atol=1e-2) :
        print("✓ Attention 整体结果验证通过!")
        max_error = np.max(np.abs(expected_xwqkx - xwwxt))
        print(f"最大误差: {max_error:.6f}")
    else:
        print("✗ Attention 整体结果验证失败!")
        print("期望结果样本:", expected_xwqkx.flatten()[:5])
        print("实际结果样本:", xwwxt.flatten()[:5])
        max_error = np.max(np.abs(expected_xwqkx - xwwxt))
        print(f"最大误差: {max_error:.6f}")

def test_batch():
    vector_size = 256  # 为了演示，batch测试用较小向量
    batch_size = 32
    attention = Attention(128, 8, 256)

    # 生成batch测试数据
    from .test_tgx import generate_x_wq_wk_xt_batch, convert_batch_matrix_to_bf16
    from vector_matrix_module.softmax import Softmax
    X, wq, wk, xt = generate_x_wq_wk_xt_batch(past_token_length=63, channel=vector_size, sparse_ratio=0.95, batch=batch_size)
    past_token_num = xt.shape[1]  # batch, seq, channel
    wv = generate_matrix(past_token_num , vector_size, 0.9)

    print("xt.shape", xt.shape)

    # 计算 batch 版 numpy 参考结果
    expected_xwqkx = []
    softmax = Softmax()
    for i in range(batch_size):
        xwq = X[i] @ wq
        xwqwk = xwq @ wk.T
        xwqwkxt = xwqwk @ xt[i].T
        softmax_out = softmax.forward(xwqwkxt)
        ref = softmax_out @ wv
        expected_xwqkx.append(ref)
    expected_xwqkx = np.array(expected_xwqkx)

    attention.Wq.load_from_hbm(wq)
    attention.Wk.load_from_hbm(wk.T)
    attention.XT.load_from_hbm_batch(xt.transpose(0, 2, 1))  
    attention.Wv_multi_batch.load_from_hbm(wv)
    x_bf16 = convert_batch_matrix_to_bf16(X)


    print("开始运行 Batch Attention 硬件模拟...")
    import time
    start_time = time.time()

    # 更新调用方式以接收额外的模拟信息
    out, xwwxt, softmax_sim_info = attention.forward(x_bf16, past_token_num= past_token_num)
    
    total_time = time.time() - start_time


    print("=" * 50)
    print("=== Softmax 硬件流水线模拟信息 ===")
    print(f"输入矩阵形状: {X.shape} -> softmax输入: {out.shape}")
    print(f"序列长度: {softmax_sim_info['seq_len']}")
    print(f"批次大小: {softmax_sim_info['batch_size']}")
    print(f"总处理元素: {softmax_sim_info['total_elements']}")
    print()
    print(f"=== 流水线配置 ===")
    print(f"前窗口大小: {softmax_sim_info['front_window']}")
    print(f"后窗口大小: {softmax_sim_info['back_window']}")
    print(f"最大并行行数: {softmax_sim_info['max_rows']}")
    print(f"流水线输入数据点: {softmax_sim_info['pipeline_input_count']}")
    print()
    print(f"=== 性能指标 ===")
    print(f"总周期数: {softmax_sim_info['total_cycles']}")
    print(f"PE 利用效率: {softmax_sim_info['pe_efficiency']:.2%}")
    print(f"模拟执行时间: {softmax_sim_info['execution_time']*1000:.2f}ms")
    print(f"总执行时间: {total_time*1000:.2f}ms")
    print()
    print(f"=== 周期分解 ===")
    for stage, cycles in softmax_sim_info['cycles_breakdown'].items():
        percentage = cycles / softmax_sim_info['total_cycles'] * 100 if softmax_sim_info['total_cycles'] > 0 else 0
        print(f"{stage}: {cycles} 周期 ({percentage:.1f}%)")
    print()
    print(f"=== 整体 Attention 性能 ===")
    print(f"Attention 总周期数: {attention.cycles}")
    print("=" * 50)
    
    # 验证 softmax 输出的性质
    print(f"\n=== Softmax 输出验证 ===")
    row_sums = np.sum(out, axis=-1)
    print(f"Softmax 行和: {row_sums} (应该接近1)")
    print(f"行和检查通过: {np.allclose(row_sums, 1.0, rtol=1e-2, atol=1e-2)}")
    
    # 准确性验证
    print(f"\n=== 准确性验证 ===")
    if np.allclose(expected_xwqkx, xwwxt, rtol=1e-2, atol=1e-2) :
        print("✓ Attention 整体结果验证通过!")
        max_error = np.max(np.abs(expected_xwqkx - xwwxt))
        print(f"最大误差: {max_error:.6f}")
    else:
        print("✗ Attention 整体结果验证失败!")
        print("期望结果样本:", expected_xwqkx.flatten()[:5])
        print("实际结果样本:", xwwxt.flatten()[:5])
        max_error = np.max(np.abs(expected_xwqkx - xwwxt))
        print(f"最大误差: {max_error:.6f}")

# 运行测试
if __name__ == "__main__":
    #test()
    test_batch()