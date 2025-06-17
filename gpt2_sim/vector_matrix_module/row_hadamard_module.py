from .vector_hadamard_HBM import VectorHadamardSimulatorWithHBM
import torch
import numpy as np


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
        self.hbm_data = vector
    
    def forward(self, X):
        # 接收float32格式的数据，内部会转为bf16进行计算
        # 返回结果也是float32格式

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

        # 可选：验证结果正确性（调试阶段保留）
        self.simulator.verify_result(X, self.hbm_data)

        return sim_res["output_vector"]
    

def generate_matrix(M, N, sparse_ratio):
    
    return np.random.choice([0, 0.01], size=(M, N), p = [sparse_ratio, 1 - sparse_ratio])
if __name__ == "__main__":
    row_hadamard = RowHadamard(128, 32, 256)
    vector_size = 4096

    # A_vector = np.random.randn(vector_size).astype(np.float32) * 0.1
    # B_vector = np.random.randn(vector_size).astype(np.float32) * 0.1

    A_vector = generate_matrix(1,vector_size,0)
    B_vector = generate_matrix(1,vector_size,0)

    row_hadamard.load_from_hbm(B_vector)
    result = row_hadamard.forward(A_vector) 