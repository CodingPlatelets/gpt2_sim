from .vector_matrix_row_product_HBM import VectorMatrixRowProductSimulatorWithHBM
from .vector_matrix_row_product_HBM import CSRMatrix
import torch
import numpy as np


class RowProduct:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        # 也就是 elements_per_block
        self.data_num_per_cycle = data_num_per_cycle
        self.cycles = 0
        self.hbm_data = None
        self.simulator = VectorMatrixRowProductSimulatorWithHBM(PE_rows, PE_num )
        
    
    def load_from_hbm(self, matrix):

        self.hbm_data = matrix
    
    def forward(self, X):
        # 接收float32格式的数据，内部会转为bf16进行计算
        # 返回结果也是float32格式

        # 若输入是 (1, N) 的二维矩阵，转换为一维向量
        if X.ndim == 2 and X.shape[0] == 1:
            X = X.flatten()

        # 结果向量长度 = 权重矩阵列数
        assert self.hbm_data is not None, "请先调用 load_from_hbm 加载权重矩阵"
        assert X.shape[0] == self.hbm_data.shape[0], (
            f"输入长度 {X.shape[0]} 与权重行数 {self.hbm_data.shape[0]} 不一致")

        self.simulator.vector_size = self.hbm_data.shape[1]

        sim_res = self.simulator.run_simulation(X, self.hbm_data, self.data_num_per_cycle)

        # 可选：验证结果正确性（调试阶段保留）
        # self.simulator.verify_result(X, self.hbm_data)

        return sim_res["output_vector"]
    

if __name__ == "__main__":
    row_product = RowProduct(128, 32, 256,)
    vec_dim = 4096
    sparsity = 0.95
    row = 256

    A_vector = np.random.randn(row).astype(np.float32) * 0.1

    B_dense = torch.randn((row, vec_dim)) * 0.01
    mask = torch.rand(row, vec_dim) > sparsity
    B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)

    row_product.load_from_hbm(B_sparse)
    row_product.forward(A_vector)
    

