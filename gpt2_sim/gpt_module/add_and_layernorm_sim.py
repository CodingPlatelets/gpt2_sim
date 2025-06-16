import torch
import numpy as np
from ..vector_matrix_module.row_add_module import RowAdd
from ..vector_matrix_module.row_hadamard_module import RowHadamard


class LayerNormCore:
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


class AddAndLayerNorm_Sim:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256):
        """
        初始化Add&LayerNorm模拟器
        
        Args:
            PE_num: 每个PE行中的PE数量
            PE_rows: PE行的数量
            data_num_per_cycle: 每个周期处理的数据量
        """
        # 初始化各个硬件模拟器
        self.row_add_res = RowAdd(PE_num, PE_rows, data_num_per_cycle)
        self.row_hadamard = RowHadamard(PE_num, PE_rows, data_num_per_cycle)
        self.row_add_bais = RowAdd(PE_num, PE_rows, data_num_per_cycle)
        self.norm_core = LayerNormCore()
        
    def forward(self, x):
        """
        执行Add&LayerNorm操作
        
        Args:
            x: 输入张量，形状为(1, vector_size)
            
        Returns:
            处理后的张量
        """
        # 1. 残差连接
        residual_out = self.row_add_res.forward(x)
        
        # 2. LayerNorm核心计算
        norm_out = self.norm_core.forward(residual_out)
        
        # 3. 应用权重（Hadamard积）
        weighted_out = self.row_hadamard.forward(norm_out)
        
        # 4. 应用偏置（加法）
        final_out = self.row_add_bais.forward(weighted_out)
        
        return final_out

    def verify_result(self, x, residual, weight, bias):
        """
        验证硬件模拟结果与PyTorch实现的结果
        
        Args:
            x: 输入张量
            residual: 残差连接张量
            weight: 权重
            bias: 偏置
            
        Returns:
            bool: 结果是否匹配
        """
        # 转换为PyTorch张量
        x_torch = torch.from_numpy(x).float()
        residual_torch = torch.from_numpy(residual).float()
        weight_torch = torch.from_numpy(weight).float()
        bias_torch = torch.from_numpy(bias).float()
        
        # PyTorch实现
        # 1. 残差连接
        residual_out = x_torch + residual_torch
        
        # 2. LayerNorm
        mean = residual_out.mean(-1, keepdim=True)
        var = ((residual_out - mean) ** 2).mean(-1, keepdim=True)
        norm_out = (residual_out - mean) / torch.sqrt(var + 1e-12)
        
        # 3. 应用权重和偏置
        torch_out = weight_torch * norm_out + bias_torch
        
        # 硬件模拟结果
        self.row_add_res.load_from_hbm(residual)
        self.row_hadamard.load_from_hbm(weight)
        self.row_add_bais.load_from_hbm(bias)
        hw_out = self.forward(x)
        
        # 计算误差
        error = np.abs(hw_out - torch_out.numpy()).max()
        print(f"最大误差: {error}")
        
        # 检查结果是否匹配
        is_correct = np.allclose(hw_out, torch_out.numpy(), rtol=1e-2, atol=1e-2)
        if is_correct:
            print("✅ 验证成功: 硬件模拟结果与PyTorch实现匹配")
        else:
            print("❌ 验证失败: 硬件模拟结果与PyTorch实现不匹配")
            print(f"PyTorch输出示例:\n{torch_out[0, :10]}")
            print(f"硬件模拟输出示例:\n{hw_out[0, :10]}")
        
        return is_correct


def generate_matrix(M, N, sparse_ratio):
    return np.random.choice([0, 0.01], size=(M, N), p=[sparse_ratio, 1 - sparse_ratio])


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
    x = generate_matrix(1, vector_size, 0)

    # 创建模拟器
    add_norm_sim = AddAndLayerNorm_Sim(
        PE_num=PE_num,
        PE_rows=PE_rows,
        data_num_per_cycle=data_num_per_cycle
    )
    add_norm_sim.row_add_res.load_from_hbm(residual)
    add_norm_sim.row_hadamard.load_from_hbm(weight)
    add_norm_sim.row_add_bais.load_from_hbm(bias)
    hw_out = add_norm_sim.forward(x)
    # 验证结果
    # add_norm_sim.verify_result(x, residual, weight, bias)