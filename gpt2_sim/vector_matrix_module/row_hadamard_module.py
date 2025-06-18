from .vector_hadamard_HBM_bf16 import VectorHadamardSimulatorWithHBM
import torch
import numpy as np
from .utils import convert_matrix_to_bf16, generate_matrix, fp32_to_bf16

class RowHadamard:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        # 也就是 elements_per_block
        self.data_num_per_cycle = data_num_per_cycle
        self.cycles = 0
        self.hbm_data = None
        self.simulator = VectorHadamardSimulatorWithHBM(PE_rows, PE_num)
        
    def load_from_hbm(self, vector):
        """加载向量到HBM"""
        # 将fp32矩阵转换成bf16
        vector_bf16 = convert_matrix_to_bf16(vector)
        # vector_bf16 = np.array([fp32_to_bf16(val) for val in vector.flatten()])
        self.hbm_data = vector_bf16
    
    def forward(self, X):
        # 若输入是 (1, N) 的二维矩阵，转换为一维向量
        if X.ndim == 2 and X.shape[0] == 1:
            X = X.flatten()

        if self.hbm_data.ndim == 2 and self.hbm_data.shape[0] == 1:
            self.hbm_data = self.hbm_data.flatten()

        # 确保输入向量长度与HBM中的向量长度一致
        assert self.hbm_data is not None, "请先调用 load_from_hbm 加载向量"
        assert X.shape[0] == self.hbm_data.shape[0], (
            f"输入长度 {X.shape[0]} 与HBM向量长度 {self.hbm_data.shape[0]} 不一致")

        self.simulator.vector_size = self.hbm_data.shape[0]

        # 执行Hadamard积
        sim_res = self.simulator.run_simulation(X, self.hbm_data, self.data_num_per_cycle)

        hadamard_out = np.array(sim_res["output_vector"]).reshape(1,-1)

        # 可选：验证结果正确性（调试阶段保留）
        self.simulator.verify_result(X, self.hbm_data)

        return hadamard_out
    

if __name__ == "__main__":
    row_hadamard = RowHadamard(128, 32, 256)
    vector_size = 4096

    # 生成测试矩阵
    A_vector_fp32 = generate_matrix(1, vector_size, 0)
    B_vector_fp32 = generate_matrix(1, vector_size, 0)

    # 将矩阵转换为BF16格式
    A_vector_bf16 = convert_matrix_to_bf16(A_vector_fp32)
    # 加载B向量到HBM
    row_hadamard.load_from_hbm(B_vector_fp32)
    
    # 执行Hadamard积
    result = row_hadamard.forward(A_vector_bf16) 