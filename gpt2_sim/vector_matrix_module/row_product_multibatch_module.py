from .vector_matrix_row_product_HBM_multi_batch import VectorMatrixRowProductSimulatorWithHBM
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
        self.simulator = VectorMatrixRowProductSimulatorWithHBM(PE_rows, PE_num)

    def load_from_hbm(self, matrix):
        self.hbm_data = matrix

    def forward(self, X):
        """
        支持多batch的前向传播
        
        Args:
            X: 输入数据，可以是：
               - 1D数组 (vector_dim,) - 单个向量
               - 2D数组 (1, vector_dim) - 单batch  
               - 2D数组 (batch_size, vector_dim) - 多batch
        
        Returns:
            输出结果：
            - 原始输入为1D时：返回1D数组 (output_dim,)
            - 原始输入为2D时：返回2D数组 (batch_size, output_dim)
        """
        # 接收float32格式的数据，内部会转为bf16进行计算
        # 返回结果也是float32格式

        # 记录原始输入是否为单向量
        return_1d = False
        
        # 处理输入格式
        if X.ndim == 1:
            # 单向量输入，需要转换为2D并记录返回格式
            return_1d = True
            X = X.reshape(1, -1)
        elif X.ndim == 2:
            # 2D输入，保持不变
            pass
        else:
            raise ValueError(f"输入维度不支持: {X.ndim}，只支持1D或2D")

        # 基本检查
        assert self.hbm_data is not None, "请先调用 load_from_hbm 加载权重矩阵"
        
        batch_size, vector_dim = X.shape
        assert vector_dim == self.hbm_data.shape[0], (
            f"输入向量维度 {vector_dim} 与权重矩阵行数 {self.hbm_data.shape[0]} 不一致")
        
        # 检查batch_size是否超过PE资源限制
        if batch_size > self.PE_rows:
            raise ValueError(f"batch_size({batch_size}) 不能大于 PE_rows({self.PE_rows})")

        # 设置输出向量大小
        self.simulator.vector_size = self.hbm_data.shape[1]

        # 运行模拟
        sim_res = self.simulator.run_simulation(X, self.hbm_data, self.data_num_per_cycle)
        self.cycles = sim_res["clock"]

        # 根据原始输入格式返回相应的输出
        if return_1d:
            # 原始输入是1D向量，返回1D数组（与原代码行为一致）
            if "output_vector" in sim_res and sim_res["output_vector"] is not None:
                return sim_res["output_vector"]
            else:
                return sim_res["output_matrix"][0]
        else:
            # 原始输入是2D数组，返回2D数组
            return sim_res["output_matrix"]

    def verify_result(self, X):
        """验证计算结果的正确性"""
        return self.simulator.verify_result(X, self.hbm_data)


if __name__ == "__main__":
    # 兼容你原有的测试代码，同时增加多batch测试
    print("=" * 60)
    print("测试RowProduct多batch功能")
    
    row_product = RowProduct(128, 32, 256)
    vec_dim = 4096
    sparsity = 0.95
    row = 256

    # 生成测试数据
    np.random.seed(42)
    A_vector = np.random.randn(row).astype(np.float32) * 0.1
    B_dense = torch.randn((row, vec_dim)) * 0.01
    mask = torch.rand(row, vec_dim) > sparsity
    B_sparse = np.where(mask, B_dense.numpy(), 0).astype(np.float32)

    row_product.load_from_hbm(B_sparse)
    
    print(f"权重矩阵形状: {B_sparse.shape}")
    print(f"稀疏度: {(B_sparse == 0).sum() / B_sparse.size:.2%}")


    # 测试2: 单batch输入 (1, vector_dim)
    print("\n--- 测试2: 单batch输入 ---")
    A_single_batch = A_vector.reshape(1, -1)
    result2 = row_product.forward(A_single_batch)
    expected2 = A_single_batch @ B_sparse
    print(f"输入形状: {A_single_batch.shape}")
    print(f"输出形状: {result2.shape}")
    error2 = np.abs(result2 - expected2).max()
    print(f"最大误差: {error2:.6f}")
    print(f"测试结果: {'✅ 通过' if error2 < 1e-1 else '❌ 失败'}")

    # 测试3: 多batch输入
    print("\n--- 测试3: 多batch输入 ---")
    batch_sizes = [4, 16]
    
    for batch_size in batch_sizes:
        if batch_size <= 32:  # PE_rows限制
            print(f"\n  batch_size={batch_size}:")
            A_multi_batch = np.random.randn(batch_size, row).astype(np.float32) * 0.1
            result3 = row_product.forward(A_multi_batch)
            expected3 = A_multi_batch @ B_sparse
            print(f"  输入形状: {A_multi_batch.shape}")
            print(f"  输出形状: {result3.shape}")
            error3 = np.abs(result3 - expected3).max()
            print(f"  最大误差: {error3:.6f}")
            print(f"  测试结果: {'✅ 通过' if error3 < 1e-1 else '❌ 失败'}")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("\n使用说明:")
    print("1. 原有的单向量输入方式保持不变")
    print("2. 新增支持多batch输入: forward(batch_matrix)")
    print("3. batch_size不能超过PE_rows(32)")