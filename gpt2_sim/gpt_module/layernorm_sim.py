import torch
import numpy as np
import struct
from vector_matrix_module.row_add_module import RowAdd,RowAdd2
from vector_matrix_module.row_hadamard_module import RowHadamard
from bf16_module.utils import convert_through_pipeline
from vector_matrix_module.utils import convert_matrix_to_bf16,generate_matrix,bf16_to_float


class LayerNormCoreSW:
    """LayerNorm的核心计算模块，负责计算均值和方差，并进行归一化"""
    def __init__(self, eps=1e-12):
        self.eps = eps
        
    def forward(self, x):
        # 确保输入是numpy数组
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            
        # 确保输入是2D数组
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        # 计算均值 (沿着最后一个维度)
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # 计算方差
        var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        
        # 归一化
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        return x_norm

class LayerNorm_Sim:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        """
        初始化LayerNorm模拟器
        
        Args:
            PE_num: 每个PE行中的PE数量
            PE_rows: PE行的数量
            data_num_per_cycle: 每个周期处理的数据量
        """
        # 初始化各个硬件模拟器
        self.row_hadamard = RowHadamard(PE_num, PE_rows, data_num_per_cycle)
        self.row_add_bais = RowAdd(PE_num, PE_rows, data_num_per_cycle)
        self.norm_core = LayerNormCoreSW()
        
    def load_ln_weights(self, ln_weight, ln_bias):
        self.row_hadamard.load_from_hbm(ln_weight)
        self.row_add_bais.load_from_hbm(ln_bias)
        
    
    def forward(self, x):
        """
        执行LayerNorm操作
        
        Args:
            x: 输入张量，形状为(1, vector_size)，BF16格式
            
        Returns:
            处理后的张量，BF16格式
        """
        # 将输入从BF16转换为FP32
        x_fp32 = np.vectorize(bf16_to_float)(x)
        # 1. LayerNorm核心计算
        norm_out = self.norm_core.forward(x_fp32)
        # 从fp32 转换bf16
        norm_bf16 = convert_matrix_to_bf16(norm_out)
        
        # 2. 应用权重（Hadamard积）
        weighted_out = self.row_hadamard.forward(norm_bf16)
        
        # 3. 应用偏置（加法）
        final_out = self.row_add_bais.forward(weighted_out)

        
        return final_out

    def verify_result(self, x_bf16, weight, bias):
        """
        验证硬件模拟结果与NumPy实现的结果（BF16格式）
        
        Args:
            x_bf16: 输入张量（BF16格式）
            weight: 权重
            bias: 偏置
            
        Returns:
            bool: 结果是否匹配
        """
        # 转换为NumPy数组
        x_bf16_np = x_bf16 if isinstance(x_bf16, np.ndarray) else np.array(x_bf16)
        weight_np = weight if isinstance(weight, np.ndarray) else np.array(weight)
        bias_np = bias if isinstance(bias, np.ndarray) else np.array(bias)
        
        # 将输入转换为FP32进行计算
        x_np_fp32 = np.vectorize(bf16_to_float)(x_bf16_np)
        
        # NumPy实现（FP32计算）
        # 1. LayerNorm
        print(f"NumPyNormout输入示例:\n{x_np_fp32[0, :10]}")
        norm_out = LayerNormCoreSW().forward(x_np_fp32)

        print(f"NumPyNormout输出示例:\n{norm_out[0, :10]}")
        # 2. 应用权重和偏置
        np_out_fp32 = weight_np * norm_out + bias_np

        
        # 硬件模拟结果
        self.load_ln_weights(weight,bias)
        hw_out = self.forward(x_bf16)
        hw_out = np.vectorize(bf16_to_float)(hw_out)
        # 计算误差
        error = np.abs(hw_out - np_out_fp32).max()
        print(f"最大误差: {error}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out, np_out_fp32, rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与NumPy实现匹配")
            print(f"NumPy输出示例:\n{np_out_fp32[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out[0, :10]}")
        else:
            print("❌ 验证失败: 硬件模拟结果与NumPy实现不匹配")
            print(f"NumPy输出示例:\n{np_out_fp32[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out[0, :10]}")
        
        return is_correct

if __name__ == "__main__":
    # 测试代码
    vector_size = 4096
    PE_num = 128
    PE_rows = 32
    data_num_per_cycle = 256

    # 生成测试数据
    weight = generate_matrix(1, vector_size, 0)
    bias = generate_matrix(1, vector_size, 0)
    residual = generate_matrix(1, vector_size, 0)
    x = generate_matrix(1, vector_size, 0.5)

    x_bf16 = convert_matrix_to_bf16(x)

    # 创建layernorm模拟器
    norm_sim = LayerNorm_Sim(
        PE_num=PE_num,
        PE_rows=PE_rows,
        data_num_per_cycle=data_num_per_cycle
    )


    norm_sim.verify_result(x_bf16,weight,bias)