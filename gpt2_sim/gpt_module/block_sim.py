from .layernorm_sim import LayerNorm_Sim,LayerNormCoreVerify
from .residual_sim import Residual_Sim2
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
        self.res_1 = Residual_Sim2(PE_num, PE_rows, data_num_per_cycle)
        self.ln_2 = LayerNorm_Sim(PE_num, PE_rows, data_num_per_cycle)
        self.ffn = FFN(PE_num, PE_rows, data_num_per_cycle)
        self.res_2 = Residual_Sim2(PE_num, PE_rows, data_num_per_cycle)

    def load_attention_weights(self, wq, wk, xt, wv):
        """加载注意力所需权重"""
        self.attn.Wq.load_from_hbm(wq)
        self.attn.Wk.load_from_hbm(wk.T)
        self.attn.XT.load_from_hbm(xt.T)
        self.attn.Wv.load_from_hbm(wv)

    def load_ffn_weights(self, w1, w2):
        self.ffn.load_weights(w1, w2)

    def load_ln_weights(self, ln1_weight, ln1_bias, ln2_weight, ln2_bias):
        self.ln_1.load_ln_weights(ln1_weight,ln1_bias)
        self.ln_2.load_ln_weights(ln2_weight,ln2_bias)
    
    def load_weight(self,ln1_weight, ln1_bias, ln2_weight, ln2_bias, 
                     wq, wk, xt, wv, w1, w2):
        self.load_attention_weights(wq, wk, xt, wv)
        self.load_ffn_weights(w1, w2)
        self.load_ln_weights(ln1_weight, ln1_bias, ln2_weight, ln2_bias)

        
    def forward(self, x_bf16, past_token_num):
        """
        执行Block的前向传播
        
        Args:
            x: 输入张量，形状为(1, vector_size)
            past_token_num: 过去的token数量
            
        Returns:
            处理后的张量
        """
        # 1. 第一个LayerNorm
        norm1_out_bf16 = self.ln_1.forward(x_bf16)
        
        # 2. Attention
        attn_out, _ = self.attn.forward(norm1_out_bf16, past_token_num)
        # print(f"hw attn_out输出示例:\n{attn_out[0, :10]}")
        # 转 bf16 供 FFN
        attn_out_bf16 = convert_matrix_to_bf16(attn_out)
        
        # 3. 第一个残差连接
        residual1_out = self.res_1.forward(attn_out_bf16,x_bf16)
        # print(f"hw res1_out输出示例:\n{residual1_out[0, :10]}")
        
        # 4. 第二个LayerNorm
        norm2_out = self.ln_2.forward(residual1_out)
        
        # 5. FFN
        ffn_out = self.ffn.forward(norm2_out)
        
        # 6. 第二个残差连接
        final_out = self.res_2.forward(ffn_out,residual1_out)

        
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
        self.load_weight(ln1_weight, ln1_bias, ln2_weight, ln2_bias, 
                     wq, wk, xt, wv, w1, w2)
        
        # 硬件模拟结果
        hw_out = self.forward(x, past_token_num)

        hw_out_fp32 = np.vectorize(bf16_to_float)(hw_out)
        
        # 转换为NumPy数组
        x_np = x if isinstance(x, np.ndarray) else np.array(x)
        
        # 将输入转换为FP32进行计算
        x_fp32 = np.vectorize(bf16_to_float)(x_np)
        # 1. 第一个LayerNorm
        norm1_out = LayerNormCoreVerify().forward(x_fp32)
        # 2. Attention
        xw = norm1_out @ wq
        xww = xw @ wk.T
        xwwx = xww @ xt.T

        attn_out = Softmax().forward(xwwx)@wv
        attn_out_bf16 = convert_matrix_to_bf16(attn_out)
        # print(f"np attn_out输出示例:\n{attn_out[0, :10]}")
        attn_out = np.vectorize(bf16_to_float)(attn_out_bf16)
        
        # 3. 第一个残差连接
        residual1_out = x_fp32 + attn_out
        # print(f"np res1_out输出示例:\n{residual1_out[0, :10]}")
        
        # 4. 第二个LayerNorm
        norm2_out = LayerNormCoreVerify().forward(residual1_out)
        # 5. FFN
        ffn_out = norm2_out @ w1 @ w2
        ffn_out_bf16 = convert_matrix_to_bf16(ffn_out)
        # print(f"np ffn_out输出示例:\n{ffn_out[0, :10]}")
        ffn_out = np.vectorize(bf16_to_float)(ffn_out_bf16)
        # 6. 第二个残差连接
        torch_out = residual1_out + ffn_out

        
        
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
    
    x = generate_matrix(1, channel, 0.4)
    wq = generate_matrix(channel, channel, sparse_ratio)
    wk = generate_matrix(channel, channel, sparse_ratio)
    xt = generate_matrix(past_token_length + 1, channel, sparse_ratio)

    return x, wq, wk, xt

def convert_matrix_to_bf16(A):
    A_bf16 = np.zeros_like(A,dtype=np.int32)
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