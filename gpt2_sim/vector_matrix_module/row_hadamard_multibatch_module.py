from .vector_hadamard_HBM_bf16_multi_batch import VectorHadamardSimulatorWithHBMMultiBatch
import torch
import numpy as np
from .utils import convert_matrix_to_bf16, generate_matrix, fp32_to_bf16,bf16_to_float

class RowHadamardMultiBatch:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        self.data_num_per_cycle = data_num_per_cycle
        self.cycles = 0
        self.hbm_data = None
        self.simulator = VectorHadamardSimulatorWithHBMMultiBatch(PE_rows, PE_num)

    def load_from_hbm(self, vector):
        vector_bf16 = convert_matrix_to_bf16(vector)
        self.hbm_data = vector_bf16

    def forward(self, X):
        return_1d = False
        if isinstance(X, list):
            X = np.array(X)
        if X.ndim == 1:
            return_1d = True
            X = X.reshape(1, -1)
        elif X.ndim == 2:
            pass
        else:
            raise ValueError(f"输入维度不支持: {X.ndim}")
        if self.hbm_data is None:
            raise ValueError("请先调用 load_from_hbm 加载向量")
        if isinstance(self.hbm_data, list):
            self.hbm_data = np.array(self.hbm_data)
        if self.hbm_data.ndim == 1:
            hbm = self.hbm_data.reshape(1, -1)
        else:
            hbm = self.hbm_data
        batch_size = X.shape[0]
        if batch_size > self.PE_rows:
            raise ValueError(f"batch_size({batch_size}) 不能大于 PE_rows({self.PE_rows})")
        assert X.shape[1] == hbm.shape[1], f"输入长度 {X.shape[1]} 与HBM向量长度 {hbm.shape[1]} 不一致"
        if hbm.shape[0] == 1 and batch_size > 1:
            hbm = np.repeat(hbm, batch_size, axis=0)
        elif hbm.shape[0] != batch_size:
            raise ValueError(f"HBM向量batch数 {hbm.shape[0]} 与输入batch数 {batch_size} 不一致")
        self.simulator.vector_size = hbm.shape[1]
        sim_res = self.simulator.run_simulation(X, hbm, self.data_num_per_cycle)
        self.cycles = sim_res["clock"]
        hadamard_out = np.uint16(sim_res["output_matrix"])
        self.simulator.verify_result(X, hbm)
        if return_1d:
            return hadamard_out[0]
        return hadamard_out

if __name__ == "__main__":
    row_hadamard = RowHadamardMultiBatch(128, 32, 256)
    vector_size = 4096
    np.random.seed(42)
    # 单batch测试
    A_vector_fp32 = generate_matrix(1, vector_size, 0)
    B_vector_fp32 = generate_matrix(1, vector_size, 0)
    A_vector_bf16 = convert_matrix_to_bf16(A_vector_fp32)
    B_vector_bf16 = convert_matrix_to_bf16(B_vector_fp32)
    row_hadamard.load_from_hbm(B_vector_fp32)
    result = row_hadamard.forward(A_vector_bf16)
    print("单batch输出形状:", result.shape)
    # 多batch测试
    for batch_size in [4, 8, 16, 32]:
        print(f"\n--- batch_size={batch_size} ---")
        A_multi = generate_matrix(batch_size, vector_size, 0)
        B_multi = generate_matrix(batch_size, vector_size, 0)
        A_multi_bf16 = convert_matrix_to_bf16(A_multi)
        B_multi_bf16 = convert_matrix_to_bf16(B_multi)
        row_hadamard.load_from_hbm(B_multi)
        result_multi = row_hadamard.forward(A_multi_bf16)
        result_multi = np.vectorize(bf16_to_float)(result_multi)
        print(f"输入形状: {A_multi_bf16.shape}, 输出形状: {result_multi.shape}")
        expected = np.vectorize(lambda x, y: x * y)(A_multi, B_multi)
        error = np.abs(result_multi - expected).max()
        print(f"最大误差: {error:.6f}")
        print(f"测试结果: {'✅ 通过' if error < 1e-1 else '❌ 失败'}") 