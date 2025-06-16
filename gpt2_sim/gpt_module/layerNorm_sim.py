from ..vector_matrix_module.vector_add_HBM import VectorAddSimulatorWithHBM
from ..vector_matrix_module.row_hadamard_module import RowMult
import numpy as np
import torch

class LayerNorm:
    def __init__(self, PE_num, PE_rows, data_num_per_cycle=256, eps=1e-12):
        self.PE_num = PE_num
        self.PE_rows = PE_rows
        self.data_num_per_cycle = data_num_per_cycle
        self.eps = eps
        self.cycles = 0
        
        # 用于计算均值和方差的加法器
        self.mean_adder = VectorAddSimulatorWithHBM(PE_rows, PE_num)
        self.var_adder = VectorAddSimulatorWithHBM(PE_rows, PE_num)
        
        # 用于缩放和平移的乘法器和加法器
        self.scale_multiplier = RowMult(PE_num, PE_rows, data_num_per_cycle)
        self.bias_adder = VectorAddSimulatorWithHBM(PE_rows, PE_num)
        
        # 存储权重和偏置
        self.weight = None
        self.bias = None
        
    def load_parameters(self, weight, bias):
        """加载LayerNorm的权重和偏置参数"""
        self.weight = weight  # shape: (hidden_size,)
        self.bias = bias      # shape: (hidden_size,)
        self.scale_multiplier.load_from_hbm(weight)
        
    def forward(self, x):
        """
        执行LayerNorm操作
        x: 输入张量，形状为 (batch_size, hidden_size)
        """
        # 1. 计算均值
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # 2. 计算方差
        x_centered = x - mean
        var = np.mean(x_centered ** 2, axis=-1, keepdims=True)
        
        # 3. 标准化
        x_norm = x_centered / np.sqrt(var + self.eps)
        
        # 4. 缩放和平移 (weight * x + bias)
        if self.weight is not None and self.bias is not None:
            # 使用RowMult进行逐元素乘法 (weight * x)
            scaled = self.scale_multiplier.forward(x_norm)
            # 使用硬件模拟器添加偏置 (weight * x + bias)
            result = self.bias_adder.run_simulation(scaled, self.bias, self.data_num_per_cycle)["output_vector"]
        else:
            result = x_norm
            
        return result

def test():
    """测试LayerNorm模块"""
    # 配置参数
    hidden_size = 4096
    batch_size = 1
    PE_num = 128
    PE_rows = 32
    data_num_per_cycle = 256
    
    # 创建LayerNorm模块
    layer_norm = LayerNorm(PE_num, PE_rows, data_num_per_cycle)
    
    # 生成随机权重和偏置
    weight = np.random.randn(hidden_size).astype(np.float32) * 0.1
    bias = np.random.randn(hidden_size).astype(np.float32) * 0.1
    
    # 加载参数
    layer_norm.load_parameters(weight, bias)
    
    # 生成随机输入
    x = np.random.randn(batch_size, hidden_size).astype(np.float32)
    
    # 使用PyTorch的LayerNorm作为参考
    torch_layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm.eps)
    torch_layer_norm.weight.data = torch.from_numpy(weight)
    torch_layer_norm.bias.data = torch.from_numpy(bias)
    
    # 计算参考结果
    expected_output = torch_layer_norm(torch.from_numpy(x)).numpy()
    
    # 使用我们的硬件模拟器
    actual_output = layer_norm.forward(x)
    
    # 验证结果
    if np.allclose(expected_output, actual_output, rtol=1e-2, atol=1e-2):
        print("✓ LayerNorm模拟成功!")
    else:
        print("✗ LayerNorm模拟失败!")
        print("最大误差:", np.max(np.abs(expected_output - actual_output)))

if __name__ == "__main__":
    test() 