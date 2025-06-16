import numpy as np
from .attention_sim import Attention, convert_matrix_to_bf16
from .ffn_sim import FFN, convert_matrix_to_bf16 as convert_bf16_ffn
from .test import generate_x_wq_wk_xt, generate_matrix
from vector_matrix_module.softmax import Softmax

class AttnFFNSim:
    """简单组合：Attention 输出接 FFN"""

    def __init__(self, PE_num=128, PE_rows=32, data_per_cycle=256):
        self.attn = Attention(PE_num, PE_rows, data_per_cycle)
        self.ffn = FFN(PE_num, PE_rows, data_per_cycle)

    def load_attention_weights(self, wq, wk, xt, wv):
        """加载注意力所需权重"""
        self.attn.Wq.load_from_hbm(wq)
        self.attn.Wk.load_from_hbm(wk.T)
        self.attn.XT.load_from_hbm(xt.T)
        self.attn.Wv.load_from_hbm(wv)

    def load_ffn_weights(self, w1, w2):
        self.ffn.load_weights(w1, w2)

    def forward(self, x_bf16, past_token_num):
        # Attention
        attn_out, _ = self.attn.forward(x_bf16, past_token_num)
        # 转 bf16 供 FFN
        attn_out_bf16 = convert_matrix_to_bf16(attn_out)
        # FFN
        ffn_out = self.ffn.forward(attn_out_bf16)
        return ffn_out

# =================== 测试 ===================

def test_attn_ffn():
    vec_dim = 256
    hidden_dim = 4 * vec_dim
    past_len = 300  # past token 已有 255，函数内部会生成 xt 行 = past_len+1

    # 生成数据
    X, wq, wk, xt = generate_x_wq_wk_xt(past_len, vec_dim, 0.95)
    wv_rows = xt.shape[0]  # = past_len + 1，与 Attention 中使用的 past_token_num 对齐
    wv = generate_matrix(wv_rows, vec_dim, 0.95)
    W1 = generate_matrix(vec_dim, hidden_dim, 0.95)
    W2 = generate_matrix(hidden_dim, vec_dim, 0.95)

    # NumPy 参考
    attn_scores = Softmax().forward(((X @ wq) @ wk.T) @ xt.T)  # (1, wv_rows)
    attn_out_np = attn_scores @ wv  # (1, vec_dim)
    expected = attn_out_np @ W1 @ W2  # (1, vec_dim)

    # 模拟器
    sim = AttnFFNSim()
    sim.load_attention_weights(wq, wk, xt, wv)
    sim.load_ffn_weights(W1, W2)

    X_bf16 = convert_matrix_to_bf16(X)
    out_sim = sim.forward(X_bf16, wv_rows)

    if np.allclose(expected, out_sim, rtol=1e-2, atol=1e-2):
        print("✓ Attn+FFN 仿真结果正确！")
    else:
        print("✗ Attn+FFN 结果不一致！diff max:", np.max(np.abs(expected - out_sim)))

if __name__ == "__main__":
    test_attn_ffn() 