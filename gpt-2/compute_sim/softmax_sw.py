import numpy as np
import torch
import warnings
from typing import Union, Tuple, List
import math

# bf16 的实际限制
BF16_MAX = 65504.0  # bf16 的最大正数
BF16_MIN = 6.103515625e-05  # bf16 的最小正数
BF16_EXP_MAX_INPUT = 11.0  # exp(11) ≈ 59874, 接近bf16上限
BF16_EXP_MIN_INPUT = -17.0  # exp(-17) ≈ 4e-8, 接近bf16下限

def safe_softmax_bf16(x: Union[np.ndarray, torch.Tensor], 
                      estimate_window: Tuple[int, int] = (100, 50)) -> Union[np.ndarray, torch.Tensor]:
    """
    使用 bf16 精度实现 safe softmax，逐个元素处理，支持预估最大值和溢出重计算
    
    Args:
        x: 输入数据，形状为 (batch_size, seq_len) 或 (seq_len,)
        estimate_window: (前N个, 后N个) 用于预估最大值的窗口大小
        
    Returns:
        softmax 计算结果，与输入相同类型和形状
    """
    is_torch = isinstance(x, torch.Tensor)
    original_dtype = x.dtype
    original_device = x.device if is_torch else None
    
    # 处理单行和多行情况
    if x.ndim == 1:
        return _softmax_single_row_streaming(x, estimate_window, is_torch, original_dtype, original_device)
    else:
        # 批处理多行
        results = []
        for i in range(x.shape[0]):
            if is_torch:
                row = x[i]
            else:
                row = x[i]
            row_result = _softmax_single_row_streaming(row, estimate_window, is_torch, original_dtype, original_device)
            results.append(row_result)
        
        if is_torch:
            return torch.stack(results).to(original_dtype)
        else:
            return np.stack(results).astype(original_dtype)

def _softmax_single_row_streaming(x: Union[np.ndarray, torch.Tensor], 
                                 estimate_window: Tuple[int, int],
                                 is_torch: bool,
                                 original_dtype,
                                 original_device) -> Union[np.ndarray, torch.Tensor]:
    """逐个元素处理单行数据的 softmax 计算"""
    
    front_window, back_window = estimate_window
    seq_len = len(x)
    
    # 验证输入长度
    if seq_len < 500 or seq_len > 2000:
        warnings.warn(f"输入长度 {seq_len} 不在建议范围 [500, 2000] 内")
    
    # 转换为list以便逐个处理
    if is_torch:
        x_list = x.cpu().numpy().tolist()
    else:
        x_list = x.tolist()
    
    # 步骤1: 计算estimate_window中的最大值
    estimated_max = _compute_estimate_max_streaming(x_list, front_window, back_window)
    print(f"预估最大值: {estimated_max}")
    
    # 步骤2: 逐个元素计算softmax，检查是否需要调整最大值
    result, final_max = _compute_softmax_streaming_safe(x_list, estimated_max)
    
    # 转换回原始格式
    if is_torch:
        result_tensor = torch.tensor(result, dtype=torch.float32, device=original_device)
        return result_tensor.to(original_dtype)
    else:
        return np.array(result, dtype=original_dtype)

def _compute_estimate_max_streaming(x_list: List[float], 
                                   front_window: int, 
                                   back_window: int) -> float:
    """逐个元素计算estimate_window中的最大值"""
    
    seq_len = len(x_list)
    front_window = min(front_window, seq_len)
    back_window = min(back_window, seq_len)
    
    estimated_max = float('-inf')
    
    # 处理前window个元素
    for i in range(front_window):
        val = _to_bf16_safe(x_list[i])
        if val > estimated_max:
            estimated_max = val
    
    # 处理后window个元素
    if back_window > 0:
        start_idx = max(front_window, seq_len - back_window)
        for i in range(start_idx, seq_len):
            val = _to_bf16_safe(x_list[i])
            if val > estimated_max:
                estimated_max = val
                
    return estimated_max

def _compute_softmax_streaming_safe(x_list: List[float], 
                                   initial_max: float) -> Tuple[List[float], float]:
    """
    安全地逐个元素计算softmax，动态调整最大值以避免bf16溢出
    
    Returns:
        (result_list, final_max_used)
    """
    
    seq_len = len(x_list)
    current_max = initial_max
    
    while True:
        exp_values = []
        running_sum = 0.0
        need_adjust = False
        true_max_seen = current_max
        
        # 第一遍：计算exp值并检测是否需要调整最大值
        for i in range(seq_len):
            val = _to_bf16_safe(x_list[i])
            
            # 更新观察到的真实最大值
            if val > true_max_seen:
                true_max_seen = val
            
            # 计算 val - current_max
            shifted_val = val - current_max
            
            # 检查shifted_val是否会导致exp溢出bf16
            if shifted_val > BF16_EXP_MAX_INPUT:
                print(f"在位置 {i} 检测到会导致bf16溢出: shifted_val = {shifted_val:.3f} > {BF16_EXP_MAX_INPUT}")
                print(f"val = {val:.3f}, current_max = {current_max:.3f}")
                need_adjust = True
                break
                
            # 计算exp并转换为bf16
            exp_val = _exp_bf16_safe(shifted_val)
            exp_values.append(exp_val)
            
            # 累加到running_sum
            running_sum = _to_bf16_safe(running_sum + exp_val)
            
            # 检查running_sum是否接近bf16上限
            if running_sum > BF16_MAX * 0.8:  # 80%的安全阈值
                print(f"在位置 {i} 检测到running_sum接近bf16上限: {running_sum:.0f}")
                need_adjust = True
                break
        
        if need_adjust:
            # 使用观察到的真实最大值重新计算
            print(f"调整最大值从 {current_max:.3f} 到 {true_max_seen:.3f}")
            current_max = true_max_seen
            continue
        else:
            # 成功计算完所有exp值，现在计算softmax概率
            break
    
    # 第二遍：计算softmax概率
    result = []
    for exp_val in exp_values:
        prob = _to_bf16_safe(exp_val / running_sum)
        result.append(prob)
    
    return result, current_max

def _to_bf16_safe(val: float) -> float:
    """安全的bf16转换，模拟真实bf16的行为"""
    if abs(val) > BF16_MAX:
        # 模拟bf16溢出：超过上限会环绕或变成不可预测的值
        # 这里我们返回一个"溢出"的标志值，但在真实环境中这会是垃圾数据
        return float('nan')  # 用nan表示溢出，真实情况下会是随机值
    elif abs(val) < BF16_MIN and val != 0.0:
        return 0.0  # 下溢出变为0
    else:
        # 模拟bf16的精度限制（7位尾数）
        if val == 0.0:
            return 0.0
        
        # 简单的bf16精度模拟
        sign = 1 if val >= 0 else -1
        abs_val = abs(val)
        
        # 对于正常范围的值，进行精度截断
        # bf16有约3-4位十进制精度
        magnitude = int(math.log10(abs_val)) if abs_val >= 1 else 0
        precision = max(0, 3 - magnitude)  # 保留3-4位有效数字
        
        result = sign * round(abs_val, precision)
        return result

def _exp_bf16_safe(val: float) -> float:
    """在bf16精度下安全计算exp"""
    val_bf16 = _to_bf16_safe(val)
    
    # 如果输入本身就溢出了，返回nan
    if math.isnan(val_bf16):
        return float('nan')
    
    # 检查exp的输入范围
    if val_bf16 > BF16_EXP_MAX_INPUT:
        return float('nan')  # 表示会溢出
    elif val_bf16 < BF16_EXP_MIN_INPUT:
        return 0.0  # 下溢出
    
    try:
        result = math.exp(val_bf16)
        return _to_bf16_safe(result)
    except OverflowError:
        return float('nan')

# 测试函数
def test_softmax_bf16():
    """测试函数"""
    print("测试真实bf16限制下的流式 softmax 函数...")
    
    # 生成测试数据 - 创建一些可能导致溢出的数据
    seq_len = 1000
    
    # 测试普通情况
    print("\n=== 测试普通情况 ===")
    x_normal = np.random.randn(seq_len) * 5  # 较小的值，不太会溢出
    result_normal = safe_softmax_bf16(x_normal)
    print(f"普通情况 - 输出和: {sum(result_normal):.6f}")
    
    # 测试bf16边界情况
    print("\n=== 测试bf16边界情况 ===")
    x_boundary = np.random.randn(seq_len) * 30  # 中等大小的值
    x_boundary[500] = 50  # 在中间放一个大值
    result_boundary = safe_softmax_bf16(x_boundary)
    print(f"边界情况 - 输出和: {sum(result_boundary):.6f}")
    
    # 测试极端溢出情况
    print("\n=== 测试极端溢出情况 ===")
    x_extreme = np.random.randn(seq_len) * 10
    x_extreme[300] = 100  # 非常大的值
    x_extreme[700] = 150  # 更大的值
    result_extreme = safe_softmax_bf16(x_extreme)
    print(f"极端情况 - 输出和: {sum(result_extreme):.6f}")

def test_simple_case():
    """测试简单情况"""
    print("\n=== 简单测试 ===")
    x_simple = [1.0, 2.0, 3.0, 20.0, 4.0]  # 放一个相对大的值
    result = safe_softmax_bf16(np.array(x_simple))
    print(f"输入: {x_simple}")
    print(f"输出: {[f'{r:.6f}' for r in result]}")
    print(f"输出和: {sum(result):.6f}")

if __name__ == "__main__":
    test_simple_case()
    test_softmax_bf16()
