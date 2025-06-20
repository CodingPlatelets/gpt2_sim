import torch
import numpy as np
import struct
from vector_matrix_module.row_add_multibatch_module import RowAddMultiBatch, RowAdd2MultiBatch
from bf16_module.utils import convert_through_pipeline
from vector_matrix_module.utils import convert_matrix_to_bf16, generate_matrix

def bf16_to_float(bf16):
    """将BF16值转换为FP32格式的浮点数"""
    if bf16 is None: return 0.0
    fp32_bits = int(bf16) << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]


class Residual_Sim:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        """
        初始化Residual模拟器
        
        Args:
            PE_num: 每个PE行中的PE数量
            PE_rows: PE行的数量
            data_num_per_cycle: 每个周期处理的数据量
        """
        # 初始化各个硬件模拟器
        self.row_add_res = RowAddMultiBatch(PE_num, PE_rows, data_num_per_cycle)
    
    def load_residual(self, residual):
        self.row_add_res.load_from_hbm(residual)
        
    def forward(self, x_bf16):
        """
        执行Residual操作
        
        Args:
            x: 输入张量，形状为(1, vector_size)
            
        Returns:
            处理后的张量
        """
        # 1. 残差连接
        residual_out = self.row_add_res.forward(x_bf16)

        return residual_out

    def verify_result(self, x_bf16, residual):
        """
        验证硬件模拟结果与NumPy实现的结果（BF16格式）
        
        Args:
            x_bf16: 输入张量（BF16格式）
            residual: 残差连接张量
            
        Returns:
            bool: 结果是否匹配
        """
        # 转换为NumPy数组
        x_np = x_bf16 if isinstance(x_bf16, np.ndarray) else np.array(x_bf16)
        residual_np = residual if isinstance(residual, np.ndarray) else np.array(residual)
        
        # 将输入转换为FP32进行计算
        x_np_fp32 = np.vectorize(bf16_to_float)(x_np)
        
        # NumPy实现（FP32计算）
        np_out_fp32 = x_np_fp32 + residual_np
        
        # 硬件模拟结果
        self.load_residual(residual)
        hw_out = self.forward(x_bf16)

        hw_out = np.vectorize(bf16_to_float)(hw_out)
        
        # 计算误差
        error = np.abs(hw_out - np_out_fp32).max()
        print(f"最大误差: {error}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out, np_out_fp32, rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与NumPy实现匹配")
            # print(f"NumPy输出示例:\n{np_out_fp32[0, :10]}")
            # print(f"硬件模拟输出示例:\n{hw_out[0, :10]}")
        else:
            print("❌ 验证失败: 硬件模拟结果与NumPy实现不匹配")
            print(f"NumPy输出示例:\n{np_out_fp32[:3, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out[:3, :10]}")
        
        return is_correct

class Residual_Sim2:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        """
        初始化Residual模拟器
        
        Args:
            PE_num: 每个PE行中的PE数量
            PE_rows: PE行的数量
            data_num_per_cycle: 每个周期处理的数据量
        """
        # 初始化各个硬件模拟器
        self.row_add_res = RowAdd2MultiBatch(PE_num, PE_rows, data_num_per_cycle)
        
    def forward(self, x_bf16,residual_bf16):
        """
        执行Residual操作
        
        Args:
            x: 输入张量，形状为(1, vector_size)
            residual ,
        Returns:
            处理后的张量
        """
        # 1. 残差连接
        residual_out = self.row_add_res.forward(x_bf16,residual_bf16)

        return residual_out
    

    def verify_result(self, x_bf16, residual_bf16):
        """
        验证硬件模拟结果与NumPy实现的结果（BF16格式）
        
        Args:
            x_bf16: 输入张量（BF16格式）
            residual: 残差连接张量
            
        Returns:
            bool: 结果是否匹配
        """
        # 转换为NumPy数组
        x_np = x_bf16 if isinstance(x_bf16, np.ndarray) else np.array(x_bf16)
        residual_np = residual_bf16 if isinstance(residual_bf16, np.ndarray) else np.array(residual_bf16)
        
        # 将输入转换为FP32进行计算
        x_np_fp32 = np.vectorize(bf16_to_float)(x_np)
        res_np_fp32 = np.vectorize(bf16_to_float)(residual_np)
        
        # NumPy实现（FP32计算）
        np_out_fp32 = x_np_fp32 + res_np_fp32
        
        # 硬件模拟结果
        hw_out = self.forward(x_bf16,residual_bf16)

        hw_out = np.vectorize(bf16_to_float)(hw_out)
        
        # 计算误差
        error = np.abs(hw_out - np_out_fp32).max()
        print(f"最大误差: {error}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out, np_out_fp32, rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与NumPy实现匹配")
            # print(f"NumPy输出示例:\n{np_out_fp32[0, :10]}")
            # print(f"硬件模拟输出示例:\n{hw_out[0, :10]}")
        else:
            print("❌ 验证失败: 硬件模拟结果与NumPy实现不匹配")
            print(f"NumPy输出示例:\n{np_out_fp32[:3, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out[:3, :10]}")
        
        return is_correct


if __name__ == "__main__":
    # 测试代码
    vector_size = 4096
    PE_num = 128
    PE_rows = 32
    data_num_per_cycle = 256
    np.random.seed(42)
    # 单batch测试
    print("\n===== 单batch测试 =====")
    # residual = generate_matrix(1, vector_size, 0)
    # x = generate_matrix(1, vector_size, 0)
    # x_bf16 = convert_matrix_to_bf16(x)
    # residual_bf16 = convert_matrix_to_bf16(residual)
    # add_sim = Residual_Sim(PE_num, PE_rows, data_num_per_cycle)
    # add_sim2 = Residual_Sim2(PE_num, PE_rows, data_num_per_cycle)
    # print("Residual_Sim:")
    # add_sim.verify_result(x_bf16, residual)
    # print("Residual_Sim2:")
    # add_sim2.verify_result(x_bf16, residual_bf16)
    # 多batch测试
    for batch_size in [4, 8, 16, 32]:
        print(f"\n===== 多batch测试 batch_size={batch_size} =====")
        x_multi = generate_matrix(batch_size, vector_size, 0)
        residual_multi = generate_matrix(batch_size, vector_size, 0)
        x_multi_bf16 = convert_matrix_to_bf16(x_multi)
        residual_multi_bf16 = convert_matrix_to_bf16(residual_multi)
        add_sim = Residual_Sim(PE_num, PE_rows, data_num_per_cycle)
        add_sim2 = Residual_Sim2(PE_num, PE_rows, data_num_per_cycle)
        print("Residual_Sim:")
        add_sim.verify_result(x_multi_bf16, residual_multi)
        print("Residual_Sim2:")
        add_sim2.verify_result(x_multi_bf16, residual_multi_bf16)
