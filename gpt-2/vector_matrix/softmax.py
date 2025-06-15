import numpy as np
import torch
import torch.nn.functional as F
def softmax(x: np.ndarray) -> np.ndarray:
    """
    计算softmax函数
    
    参数:
        x: 输入向量，numpy数组类型
        
    返回:
        softmax结果，numpy数组类型
    """
    # 确保输入是numpy数组
    x = np.asarray(x)
    
    # 处理一维输入
    if x.ndim == 1:
        # 数值稳定性：减去最大值
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    # 处理二维输入（比如批量数据）
    elif x.ndim == 2:
        # 对每一行分别计算softmax
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    else:
        raise ValueError("输入维度必须为1或2")

def compare_with_torch():
    """
    比较我们的softmax实现与PyTorch的官方实现
    """
    # 测试用例1：正常范围的值
    x1 = np.array([1.0, 2.0, 3.0])
    torch_x1 = torch.tensor(x1, dtype=torch.float32)
    
    our_result1 = softmax(x1)
    torch_result1 = F.softmax(torch_x1, dim=0).numpy()
    
    print("测试用例1 - 正常范围的值:")
    print("输入:", x1)
    print("我们的实现:", our_result1)
    print("PyTorch实现:", torch_result1)
    print("最大差异:", np.max(np.abs(our_result1 - torch_result1)))
    print("平均差异:", np.mean(np.abs(our_result1 - torch_result1)))
    print()
    
    # 测试用例2：大数值
    x2 = np.array([100.0, 200.0, 300.0])
    torch_x2 = torch.tensor(x2, dtype=torch.float32)
    
    our_result2 = softmax(x2)
    torch_result2 = F.softmax(torch_x2, dim=0).numpy()
    
    print("测试用例2 - 大数值:")
    print("输入:", x2)
    print("我们的实现:", our_result2)
    print("PyTorch实现:", torch_result2)
    print("最大差异:", np.max(np.abs(our_result2 - torch_result2)))
    print("平均差异:", np.mean(np.abs(our_result2 - torch_result2)))
    print()
    
    # 测试用例3：小数值
    x3 = np.array([0.001, 0.002, 0.003])
    torch_x3 = torch.tensor(x3, dtype=torch.float32)
    
    our_result3 = softmax(x3)
    torch_result3 = F.softmax(torch_x3, dim=0).numpy()
    
    print("测试用例3 - 小数值:")
    print("输入:", x3)
    print("我们的实现:", our_result3)
    print("PyTorch实现:", torch_result3)
    print("最大差异:", np.max(np.abs(our_result3 - torch_result3)))
    print("平均差异:", np.mean(np.abs(our_result3 - torch_result3)))
    print()
    
    # 测试用例4：二维输入
    x4 = np.array([[1.0, 2.0, 3.0],
                   [2.0, 3.0, 1.0]])
    torch_x4 = torch.tensor(x4, dtype=torch.float32)
    
    our_result4 = softmax(x4)
    torch_result4 = F.softmax(torch_x4, dim=1).numpy()
    
    print("测试用例4 - 二维输入:")
    print("输入:\n", x4)
    print("我们的实现:\n", our_result4)
    print("PyTorch实现:\n", torch_result4)
    print("最大差异:", np.max(np.abs(our_result4 - torch_result4)))
    print("平均差异:", np.mean(np.abs(our_result4 - torch_result4)))

if __name__ == "__main__":
    compare_with_torch() 