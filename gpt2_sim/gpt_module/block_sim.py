from .add_and_layernorm_sim import LayerNorm_Sim,Residual_Sim
from vector_matrix_module.softmax import Softmax
from .attention_sim import Attention
from .ffn_sim import FFN
import numpy as np
import struct
from bf16_module.utils import convert_through_pipeline

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式的浮点数"""
    if bf16 is None: return 0.0
    fp32_bits = int(bf16) << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
class Block_Sim:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        """
        初始化Block模拟器
        
        Args:
            PE_num: 每个PE行中的PE数量
            PE_rows: PE行的数量
            data_num_per_cycle: 每个周期处理的数据量
        """
        # 初始化各个模块
        self.ln_1 = LayerNorm_Sim(PE_num, PE_rows, data_num_per_cycle)
        self.attn = Attention(PE_num, PE_rows, data_num_per_cycle)
        self.res_1 = Residual_Sim(PE_num, PE_rows, data_num_per_cycle)
        self.ln_2 = LayerNorm_Sim(PE_num, PE_rows, data_num_per_cycle)
        self.ffn = FFN(PE_num, PE_rows, data_num_per_cycle)
        self.res_2 = Residual_Sim(PE_num, PE_rows, data_num_per_cycle)
        
    def forward(self, x, past_token_num):
        """
        执行Block的前向传播
        
        Args:
            x: 输入张量，形状为(1, vector_size)
            past_token_num: 过去的token数量
            
        Returns:
            处理后的张量
        """
        # 1. 第一个LayerNorm
        norm1_out = self.ln_1.forward(x)
        
        # 2. Attention
        attn_out, _ = self.attn.forward(norm1_out, past_token_num)
        
        # 将Attention输出从BF16转换为FP32
        x_fp32 = np.vectorize(bf16_to_float)(x)
        
        # 3. 第一个残差连接
        self.res_1.row_add_res.load_from_hbm(x_fp32)
        residual1_out = self.res_1.forward(attn_out)
        
        # 4. 第二个LayerNorm
        norm2_out = self.ln_2.forward(residual1_out)
        
        # 5. FFN
        ffn_out = self.ffn.forward(norm2_out)
        
        # 将FFN输出从BF16转换为FP32
        residual1_out_fp32 = np.vectorize(bf16_to_float)(residual1_out)
        
        # 6. 第二个残差连接
        self.res_2.row_add_res.load_from_hbm(residual1_out_fp32)
        final_out = self.res_2.forward(ffn_out)
        
        return final_out

    def verify_result(self, x, ln1_weight, ln1_bias, ln2_weight, ln2_bias, 
                     wq, wk, xt, wv, w1, w2, past_token_num):
        """
        验证硬件模拟结果与PyTorch实现的结果
        
        Args:
            x: 输入张量
            ln1_weight, ln1_bias: 第一个LayerNorm的权重和偏置
            ln2_weight, ln2_bias: 第二个LayerNorm的权重和偏置
            wq, wk, xt, wv: Attention模块的权重
            w1, w2: FFN模块的权重
            past_token_num: 过去的token数量
            
        Returns:
            bool: 结果是否匹配
        """
        # 加载权重到各个模块
        self.ln_1.row_hadamard.load_from_hbm(ln1_weight)
        self.ln_1.row_add_bais.load_from_hbm(ln1_bias)
        
        self.attn.Wq.load_from_hbm(wq)
        self.attn.Wk.load_from_hbm(wk.T)
        self.attn.XT.load_from_hbm(xt.T)
        self.attn.Wv.load_from_hbm(wv)
        
        self.ln_2.row_hadamard.load_from_hbm(ln2_weight)
        self.ln_2.row_add_bais.load_from_hbm(ln2_bias)
        
        self.ffn.load_weights(w1, w2)
        
        # 硬件模拟结果
        hw_out = self.forward(x, past_token_num)

        hw_out_fp32 = np.vectorize(bf16_to_float)(hw_out)
        
        # 转换为NumPy数组
        x_np = x if isinstance(x, np.ndarray) else np.array(x)
        
        # 将输入转换为FP32进行计算
        x = np.vectorize(bf16_to_float)(x_np)
        # 1. 第一个LayerNorm
        mean1 = np.mean(x, axis=-1, keepdims=True)
        var1 = np.mean((x - mean1) ** 2, axis=-1, keepdims=True)
        norm1_out = (x - mean1) / np.sqrt(var1 + 1e-12)
        norm1_out = ln1_weight * norm1_out + ln1_bias
        
        # 2. Attention
        xw = norm1_out @ wq
        xww = xw @ wk.T
        xwwx = xww @ xt.T

        attn_out = Softmax().forward(xwwx)@wv
        
        # 3. 第一个残差连接
        residual1_out = x + attn_out
        
        # 4. 第二个LayerNorm
        mean2 = np.mean(residual1_out, axis=-1, keepdims=True)
        var2 = np.mean((residual1_out - mean2) ** 2, axis=-1, keepdims=True)
        norm2_out = (residual1_out - mean2) / np.sqrt(var2 + 1e-12)
        norm2_out = ln2_weight * norm2_out + ln2_bias
        
        # 5. FFN
        ffn_out = norm2_out @ w1 @ w2
        
        # 6. 第二个残差连接
        torch_out = residual1_out + ffn_out

        # # 将结果转换回BF16
        # np_out_bf16 = np.zeros_like(torch_out)
        # for i in range(torch_out.shape[0]):
        #     for j in range(torch_out.shape[1]):
        #         np_out_bf16[i, j] = convert_through_pipeline(float(torch_out[i, j]))
        
        
        # 计算误差
        error = np.abs(hw_out_fp32 - torch_out).max()
        print(f"最大误差: {error}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out_fp32, torch_out, rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与Numpy实现匹配")
            print(f"Numpy输出示例:\n{torch_out[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out_fp32[0, :10]}")
        else:
            print("❌ 验证失败: 硬件模拟结果与Numpy实现不匹配")
            print(f"Numpy输出示例:\n{torch_out[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out_fp32[0, :10]}")
        
        return is_correct

def generate_matrix(M, N, sparse_ratio):
    return np.random.choice([0, 0.01], size=(M, N), p=[sparse_ratio, 1 - sparse_ratio])

def generate_x_wq_wk_xt(past_token_length, channel, sparse_ratio):
    
    x = generate_matrix(1, channel, 0)
    wq = generate_matrix(channel, channel, sparse_ratio)
    wk = generate_matrix(channel, channel, sparse_ratio)
    xt = generate_matrix(past_token_length + 1, channel, sparse_ratio)

    return x, wq, wk, xt

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

def test_block():
    """测试Block模拟器"""
    vector_size = 256
    hidden_dim = 4 * vector_size
    past_token_num = 255
    
    # 创建Block模拟器
    block = Block_Sim(128, 32, 256)

    X, wq, wk, xt = generate_x_wq_wk_xt(past_token_length=255, channel=vector_size, sparse_ratio=0.95)
    past_token_num = xt.shape[0]

    wv = generate_matrix(past_token_num , vector_size, 0.9)
    
    x_bf16 = convert_matrix_to_bf16(X)
    # 生成LayerNorm权重和偏置

    ln1_weight = generate_matrix(1, vector_size, 0)
    ln1_bias = generate_matrix(1, vector_size, 0)
    ln2_weight = generate_matrix(1, vector_size, 0)
    ln2_bias = generate_matrix(1, vector_size, 0)

    
    # 生成FFN权重
    w1 = generate_matrix(vector_size, hidden_dim, 0.95)  # 全稠密
    w2 = generate_matrix(hidden_dim, vector_size, 0.95)
    
    # 验证结果
    block.verify_result(x_bf16, ln1_weight, ln1_bias, ln2_weight, ln2_bias,
                       wq, wk, xt, wv, w1, w2, past_token_num)

if __name__ == "__main__":
    test_block() 