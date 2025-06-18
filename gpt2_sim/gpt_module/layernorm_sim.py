import torch
import numpy as np
import struct
from vector_matrix_module.row_add_module import RowAdd,RowAdd2
from vector_matrix_module.row_hadamard_module import RowHadamard
from bf16_module.utils import convert_through_pipeline
from vector_matrix_module.utils import convert_matrix_to_bf16,generate_matrix,bf16_to_float

class LayerNormCoreHW:
    """LayerNorm的硬件模拟模块，使用BF16格式，模拟时钟延迟"""
    """模拟实现硬件，输入输出为bf16,中间过程硬件模拟，操作为fp32，但模拟时钟延迟"""
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.cycles = 0
        self.mean_cycles = 0
        self.var_cycles = 0
        self.norm_cycles = 0
        
    def clock_cycle(self):
        """模拟一个时钟周期的操作"""
        self.cycles += 1
        return self.cycles  # 实际计算在forward中完成
        
    def forward(self, x_bf16):
        """
        执行LayerNorm操作，使用BF16格式
        
        Args:
            x_bf16: 输入张量，BF16格式
        Returns:
            处理后的张量，BF16格式
        """
        # 重置周期计数
        self.cycles = 0
        self.mean_cycles = 0
        self.var_cycles = 0
        self.norm_cycles = 0
        
        # 确保输入是2D数组
        if x_bf16.ndim == 1:
            x_bf16 = x_bf16.reshape(1, -1)
            
        # 转换为FP32进行计算
        x_fp32 = np.vectorize(bf16_to_float)(x_bf16)
            
        # 1. 计算均值
        mean_fp32 = np.zeros((x_fp32.shape[0], 1), dtype=np.float32)
        for i in range(x_fp32.shape[0]):
            # 累加
            sum_fp32 = 0
            for j in range(x_fp32.shape[1]):
                sum_fp32 += x_fp32[i,j]  # 加法操作
                self.clock_cycle()  # 模拟加法延迟
            # 除法
            mean_fp32[i,0] = sum_fp32 / x_fp32.shape[1]  # 除法操作
            self.clock_cycle()  # 模拟除法延迟
            
        self.mean_cycles = self.cycles
        print(f"计算均值完成，消耗周期数: {self.mean_cycles}")
            
        # 2. 计算方差
        var_fp32 = np.zeros((x_fp32.shape[0], 1), dtype=np.float32)
        for i in range(x_fp32.shape[0]):
            # 平方和累加
            sum_sq_fp32 = 0
            for j in range(x_fp32.shape[1]):
                # 减法
                diff_fp32 = x_fp32[i,j] - mean_fp32[i,0]  # 减法操作
                self.clock_cycle()  # 模拟减法延迟
                # 平方
                sq_fp32 = diff_fp32 * diff_fp32  # 乘法操作
                self.clock_cycle()  # 模拟乘法延迟
                # 累加
                sum_sq_fp32 += sq_fp32  # 加法操作
                self.clock_cycle()  # 模拟加法延迟
            # 除法
            var_fp32[i,0] = sum_sq_fp32 / x_fp32.shape[1]  # 除法操作
            self.clock_cycle()  # 模拟除法延迟
            
        self.var_cycles = self.cycles - self.mean_cycles
        print(f"计算方差完成，消耗周期数: {self.var_cycles}")
            
        # 3. 归一化
        x_norm_fp32 = np.zeros_like(x_fp32)
        for i in range(x_fp32.shape[0]):
            # 计算sqrt(var + eps)
            var_eps_fp32 = var_fp32[i,0] + self.eps  # 加法操作
            self.clock_cycle()  # 模拟加法延迟
            std_fp32 = np.sqrt(var_eps_fp32)  # 开方操作
            self.clock_cycle()  # 模拟开方延迟
            
            # 对每个元素进行归一化
            for j in range(x_fp32.shape[1]):
                # 减均值
                diff_fp32 = x_fp32[i,j] - mean_fp32[i,0]  # 减法操作
                self.clock_cycle()  # 模拟减法延迟
                # 除以标准差
                x_norm_fp32[i,j] = diff_fp32 / std_fp32  # 除法操作
                self.clock_cycle()  # 模拟除法延迟
        
        self.norm_cycles = self.cycles - self.mean_cycles - self.var_cycles
        print(f"归一化完成，消耗周期数: {self.norm_cycles}")
        print(f"总周期数: {self.cycles}")
        
        # 将结果转换回BF16格式
        x_norm_bf16 = convert_matrix_to_bf16(x_norm_fp32)
        return x_norm_bf16
        
    def verify_result(self, x_bf16):
        """验证硬件模拟结果与NumPy实现的结果"""
        # 转换为NumPy数组
        x_np = x_bf16 if isinstance(x_bf16, np.ndarray) else np.array(x_bf16)
        
        # 将输入转换为FP32进行计算
        x_np_fp32 = np.vectorize(bf16_to_float)(x_np)
        
        # NumPy实现（FP32计算）
        mean = np.mean(x_np_fp32, axis=-1, keepdims=True)
        var = np.mean((x_np_fp32 - mean) ** 2, axis=-1, keepdims=True)
        np_out_fp32 = (x_np_fp32 - mean) / np.sqrt(var + self.eps)
        
        # 硬件模拟结果
        hw_out = self.forward(x_bf16)
        hw_out_fp32 = np.vectorize(bf16_to_float)(hw_out)
        
        # 计算误差
        error = np.abs(hw_out_fp32 - np_out_fp32).max()
        print(f"最大误差: {error}")
        print(f"总时钟周期: {self.cycles}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out_fp32, np_out_fp32, rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与NumPy实现匹配")
        else:
            print("❌ 验证失败: 硬件模拟结果与NumPy实现不匹配")
            print(f"NumPy输出示例:\n{np_out_fp32[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out_fp32[0, :10]}")
        
        return is_correct

class LayerNormCoreSW:
    """LayerNorm的核心计算模块，负责计算均值和方差，并进行归一化"""
    """软件模拟计算，输入输出为bf16,中间过程为np完成的fp32"""
    def __init__(self, eps=1e-12):
        self.eps = eps
        
    def forward(self, x):
        # 将输入从BF16转换为FP32
        x = np.vectorize(bf16_to_float)(x)
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
        
        # 从fp32 转换bf16
        norm_bf16 = convert_matrix_to_bf16(x_norm)
        return norm_bf16

class LayerNormCoreVerify:
    """LayerNorm的核心计算模块，负责计算均值和方差，并进行归一化"""
    """用于验证，输入输出均为fp32"""
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
        # self.norm_core = LayerNormCoreSW()
        self.norm_core = LayerNormCoreHW()
        
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
        
        # 1. LayerNorm核心计算
        norm_out = self.norm_core.forward(x)
        
        # 2. 应用权重（Hadamard积）
        weighted_out = self.row_hadamard.forward(norm_out)
        
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
        norm_out = LayerNormCoreVerify().forward(x_np_fp32)
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


def test_layerNormCoreHW():
    # 测试代码
    vector_size = 4096
    batch_size = 1

    # 生成测试数据
    x = generate_matrix(batch_size, vector_size, 0.5)
    x_bf16 = convert_matrix_to_bf16(x)

    # 测试软件实现
    print("测试软件实现...")
    norm_sw = LayerNormCoreSW()
    sw_result = norm_sw.forward(x_bf16)
    print("软件实现完成")

    # 测试硬件模拟实现
    print("\n测试硬件模拟实现...")
    norm_hw = LayerNormCoreHW()
    hw_result = norm_hw.forward(x_bf16)
    print(f"硬件模拟完成，总时钟周期: {norm_hw.cycles}")

    # 验证结果
    print("\n验证结果...")
    norm_hw.verify_result(x_bf16)

    # 打印一些示例数据
    print("\n结果示例:")
    print("输入数据 (前5个元素):")
    print(np.vectorize(bf16_to_float)(x_bf16[0, :5]))
    print("\n软件实现结果 (前5个元素):")
    print(sw_result[0, :5])
    print("\n硬件模拟结果 (前5个元素):")
    print(np.vectorize(bf16_to_float)(hw_result[0, :5]))

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

    # test_layerNormCoreHW()