from .matmul_sim import Matmul
from vector_matrix_module.softmax import Softmax
from .test import generate_x_wq_wk_xt, generate_matrix
from vector_matrix_module.row_product_module import RowProduct
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

        self.softmax = Softmax()

    def forward(self, x, past_token_num):
        # 对于 Wq、Wk 来说，它们都是 (K, K) 的方阵，其列数应与输入向量的维度 K 相同。

        k_dim = x.shape[1]  # 输入向量的维度 K

        # 1) X @ Wq
        xw = self.Wq.forward(x, x.shape[0], k_dim, k_dim, test=False)

        # 2) (XWq) @ Wk^T
        xww = self.Wk.forward(xw, xw.shape[0], k_dim, k_dim, test=False)

        # 3) ((XWq)Wk^T) @ XT^T
        xwwx = self.XT.forward(xww, xww.shape[0], k_dim, past_token_num, test=True)

        # 4) softmax
        out = self.softmax.forward(xwwx)

        # 5) (softmax) @ Wv
        xwwxt = self.Wv.forward(out)
         
        return out, xwwxt

       

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
    vector_size = 4096
    attention = Attention(128, 32, 256)

    
    X, wq, wk, xt = generate_x_wq_wk_xt(past_token_length=255, channel=vector_size, sparse_ratio=0.95)
    past_token_num = xt.shape[0]

    wv = generate_matrix(past_token_num , vector_size, 0.9)

    expected_xwqkx =  (Softmax().forward( ( (X @ wq) @ wk.T ) @ xt.T ) ) @ wv

    attention.Wq.load_from_hbm(wq)
    attention.Wk.load_from_hbm(wk.T)
    attention.XT.load_from_hbm(xt.T)
    attention.Wv.load_from_hbm(wv)
    
    x_bf16 = convert_matrix_to_bf16(X)

    out, xwwxt = attention.forward(x_bf16, past_token_num= past_token_num)

    
    if np.allclose(expected_xwqkx, xwwxt, rtol=1e-2, atol=1e-2) :
        print("✓ True!")
    else:
        print("✗ False!")
        print("diff: ")
        print(expected_xwqkx - xwwxt)

# test()