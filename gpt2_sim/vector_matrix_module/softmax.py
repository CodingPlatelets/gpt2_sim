import numpy as np
import torch
import torch.nn.functional as F



class Softmax:
    def __init__(self):
        pass
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return softmax(x)
    
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

def test_batch_softmax():
    print("=== Softmax 多batch测试 ===")
    s = Softmax()
    # 1D输入
    x1 = np.array([1.0, 2.0, 3.0])
    print("1D输入:", s.forward(x1))
    print("sum:", np.sum(s.forward(x1)))
    # 2D输入（batch）
    x2 = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]])
    print("2D batch输入:", s.forward(x2))
    print("sum(axis=1):", np.sum(s.forward(x2), axis=1))
    # 3D输入（应报错）
    x3 = np.random.rand(2,3,4)
    try:
        s.forward(x3)
    except ValueError as e:
        print("3D输入报错:", e)
    print("=== Softmax 多batch测试结束 ===")

if __name__ == "__main__":
    test_batch_softmax()