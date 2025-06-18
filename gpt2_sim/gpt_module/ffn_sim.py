from .matmul_sim import Matmul
from vector_matrix_module.row_product_module import RowProduct  # 可能后续做 residual 用，这里预留
from vector_matrix_module.softmax import Softmax  # 备用，未使用
from bf16_module.utils import convert_through_pipeline
from .test import generate_matrix
import numpy as np

class FFN:
    """简化版前馈网络：Y = X @ W1 @ W2，无激活。所有矩阵乘法均复用 Matmul。"""

    def __init__(self, PE_num: int, PE_rows: int, data_num_per_cycle: int = 256):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        self.data_num_per_cycle = data_num_per_cycle

        self.W1 = Matmul(PE_num, PE_rows, data_num_per_cycle)
        self.W2 = Matmul(PE_num, PE_rows, data_num_per_cycle)

        # 记录权重形状，便于 forward 中确定列数
        self._hidden_dim = None  # W1 列数 / W2 行数
        self._out_dim = None     # W2 列数

    def load_weights(self, w1: np.ndarray, w2: np.ndarray):
        """加载两层权重到 HBM，并记录维度。w1.shape=(K, H), w2.shape=(H, D)"""
        assert w1.shape[1] == w2.shape[0], "W1 列数必须等于 W2 行数"
        self._hidden_dim = w1.shape[1]
        self._out_dim = w2.shape[1]
        self.W1.load_from_hbm(w1)
        self.W2.load_from_hbm(w2.T)  # 注意 Matmul 内部使用 B.T

    def forward(self, x_bf16: np.ndarray):
        """x_bf16 应为 bf16 格式 (1, K)。返回 float32 结果矩阵 (1, D)。"""
        # ------- 第一层 -------
        k_dim = x_bf16.shape[1]
        h_dim = self._hidden_dim
        out1_bf16 = self.W1.forward(x_bf16, x_bf16.shape[0], k_dim, h_dim, test=False)

        # ------- 第二层 -------
        d_dim = self._out_dim
        out2 = self.W2.forward(out1_bf16, out1_bf16.shape[0], h_dim, d_dim, test=False)
        return out2

# ======================= 单元测试 =======================

def convert_matrix_to_bf16(A: np.ndarray):
    A_bf16 = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
    return A_bf16

def test_ffn():
    """简单验证 FFN 输出与 NumPy 结果一致"""
    print("11111111111111111111111111")
    vector_size = 256
    hidden_dim = 4 * vector_size  # 常见 FFN 隐层 = 4 * d_model

    ffn = FFN(128, 32, 256)

    # 生成稠密权重
    W1 = generate_matrix(vector_size, hidden_dim, 0.95)  # 全稠密
    W2 = generate_matrix(hidden_dim, vector_size, 0.95)

    ffn.load_weights(W1, W2)

    # 生成输入
    X = generate_matrix(1, vector_size, 0.0)
    X_bf16 = convert_matrix_to_bf16(X)

    # 计算
    out_sim = ffn.forward(X_bf16)
    out_np = X @ W1 @ W2

    if np.allclose(out_np, out_sim, rtol=1e-2, atol=1e-2):
        print("✓ FFN 仿真结果正确！")
    else:
        print("✗ FFN 结果不一致！diff max:", np.max(np.abs(out_np - out_sim)))
        # 打印前几个差值
        print(out_np[:, :10] - out_sim[:, :10])

if __name__ == "__main__":
    test_ffn() 