import numpy as np

def get_bf16_parts(bf16):
    """将BF16分解为符号位、指数和尾数"""
    sign = (bf16 >> 15) & 0x1
    exponent = (bf16 >> 7) & 0xFF
    mantissa = bf16 & 0x7F
    return sign, exponent, mantissa

def compose_bf16(sign, exponent, mantissa):
    """将符号位、指数和尾数组合成BF16"""
    return (sign << 15) | (exponent << 7) | mantissa

def bf16_to_float(bf16):
    """将BF16转换为float32"""
    sign, exponent, mantissa = get_bf16_parts(bf16)
    
    # 处理特殊情况
    if exponent == 0:
        if mantissa == 0:
            return 0.0 if sign == 0 else -0.0
        # 非规格化数
        exponent = -126
        mantissa = mantissa / 128.0
    elif exponent == 0xFF:
        if mantissa == 0:
            return float('inf') if sign == 0 else float('-inf')
        return float('nan')
    else:
        # 规格化数
        exponent = exponent - 127
        mantissa = (mantissa / 128.0) + 1.0
    
    value = mantissa * (2.0 ** exponent)
    return -value if sign else value

def float_to_bf16(float_val):
    """将float32转换为BF16"""
    if float_val == 0.0:
        return 0
    if float_val == -0.0:
        return 0x8000
    if np.isinf(float_val):
        return 0x7F80 if float_val > 0 else 0xFF80
    if np.isnan(float_val):
        return 0x7FC0
    
    # 获取符号位
    sign = 1 if float_val < 0 else 0
    float_val = abs(float_val)
    
    # 计算指数和尾数
    exponent = 0
    mantissa = float_val
    
    # 规格化
    if mantissa >= 2.0:
        while mantissa >= 2.0:
            mantissa /= 2.0
            exponent += 1
    elif mantissa < 1.0 and mantissa > 0:
        while mantissa < 1.0:
            mantissa *= 2.0
            exponent -= 1
    
    # 调整指数偏移
    exponent += 127
    
    # 处理非规格化数
    if exponent < 0:
        return compose_bf16(sign, 0, 0)
    
    # 处理溢出
    if exponent > 255:
        return compose_bf16(sign, 0xFF, 0)
    
    # 提取尾数
    mantissa = (mantissa - 1.0) * 128
    mantissa = int(round(mantissa))
    
    return compose_bf16(sign, exponent, mantissa)

def bf16_add(a, b):
    """BF16加法操作"""
    # 转换为float进行计算
    a_float = bf16_to_float(a)
    b_float = bf16_to_float(b)
    result_float = a_float + b_float
    return float_to_bf16(result_float)

def bf16_subtract(a, b):
    """BF16减法操作"""
    a_sign, a_exp, a_mant = get_bf16_parts(a)
    b_sign, b_exp, b_mant = get_bf16_parts(b)
    
    # 处理特殊情况
    if a_exp == 0xFF or b_exp == 0xFF:
        return compose_bf16(a_sign, 0xFF, 0)
    
    # 计算符号
    result_sign = a_sign ^ b_sign
    
    # 计算指数
    result_exp = max(a_exp, b_exp)
    
    # 计算尾数
    a_mant_float = 1.0 + (a_mant / 128.0) if a_exp != 0 else a_mant / 128.0
    b_mant_float = 1.0 + (b_mant / 128.0) if b_exp != 0 else b_mant / 128.0
    
    # 对齐尾数
    if a_exp < b_exp:
        a_mant_float /= (2.0 ** (b_exp - a_exp))
    elif b_exp < a_exp:
        b_mant_float /= (2.0 ** (a_exp - b_exp))
    
    # 执行减法
    result_mant_float = a_mant_float - b_mant_float
    
    # 规格化结果
    if result_mant_float < 0:
        result_mant_float = -result_mant_float
        result_sign = 1 - result_sign
    
    if result_mant_float >= 2.0:
        result_mant_float /= 2.0
        result_exp += 1
    elif result_mant_float < 1.0 and result_mant_float > 0:
        while result_mant_float < 1.0:
            result_mant_float *= 2.0
            result_exp -= 1
    
    # 处理溢出
    if result_exp > 255:
        return compose_bf16(result_sign, 0xFF, 0)
    
    # 处理下溢
    if result_exp < 0:
        return compose_bf16(result_sign, 0, 0)
    
    # 提取尾数
    result_mant = int(round((result_mant_float - 1.0) * 128))
    
    return compose_bf16(result_sign, result_exp, result_mant)

def bf16_multiply(a, b):
    """BF16乘法操作"""
    a_sign, a_exp, a_mant = get_bf16_parts(a)
    b_sign, b_exp, b_mant = get_bf16_parts(b)
    
    # 处理特殊情况
    if a_exp == 0xFF or b_exp == 0xFF:
        return compose_bf16(a_sign ^ b_sign, 0xFF, 0)
    
    # 计算符号
    result_sign = a_sign ^ b_sign
    
    # 计算指数
    result_exp = a_exp + b_exp - 127
    
    # 计算尾数
    a_mant_float = 1.0 + (a_mant / 128.0) if a_exp != 0 else a_mant / 128.0
    b_mant_float = 1.0 + (b_mant / 128.0) if b_exp != 0 else b_mant / 128.0
    result_mant_float = a_mant_float * b_mant_float
    
    # 规格化结果
    if result_mant_float >= 2.0:
        result_mant_float /= 2.0
        result_exp += 1
    
    # 处理溢出
    if result_exp > 255:
        return compose_bf16(result_sign, 0xFF, 0)
    
    # 处理下溢
    if result_exp < 0:
        return compose_bf16(result_sign, 0, 0)
    
    # 提取尾数
    result_mant = int(round((result_mant_float - 1.0) * 128))
    
    return compose_bf16(result_sign, result_exp, result_mant)

def bf16_divide(a, b):
    """BF16除法操作"""
    a_sign, a_exp, a_mant = get_bf16_parts(a)
    b_sign, b_exp, b_mant = get_bf16_parts(b)
    
    # 处理特殊情况
    if b_exp == 0xFF:
        return compose_bf16(a_sign ^ b_sign, 0, 0)
    if a_exp == 0xFF:
        return compose_bf16(a_sign ^ b_sign, 0xFF, 0)
    
    # 计算符号
    result_sign = a_sign ^ b_sign
    
    # 计算指数
    result_exp = a_exp - b_exp + 127
    
    # 计算尾数
    a_mant_float = 1.0 + (a_mant / 128.0) if a_exp != 0 else a_mant / 128.0
    b_mant_float = 1.0 + (b_mant / 128.0) if b_exp != 0 else b_mant / 128.0
    result_mant_float = a_mant_float / b_mant_float
    
    # 规格化结果
    if result_mant_float < 1.0:
        result_mant_float *= 2.0
        result_exp -= 1
    
    # 处理溢出
    if result_exp > 255:
        return compose_bf16(result_sign, 0xFF, 0)
    
    # 处理下溢
    if result_exp < 0:
        return compose_bf16(result_sign, 0, 0)
    
    # 提取尾数
    result_mant = int(round((result_mant_float - 1.0) * 128))
    
    return compose_bf16(result_sign, result_exp, result_mant)

def bf16_sqrt(a):
    """BF16开方操作"""
    sign, exp, mant = get_bf16_parts(a)
    
    # 处理特殊情况
    if sign == 1:
        return compose_bf16(0, 0xFF, 0x40)  # NaN
    if exp == 0xFF:
        if mant == 0:
            return a
        return compose_bf16(0, 0xFF, 0x40)
    if exp == 0 and mant == 0:
        return compose_bf16(0, 0, 0)
    
    # 合成定点数
    unbiased_exp = exp - 127
    if exp == 0:
        # 非规格化数
        leading_zeros = 0
        temp_mant = mant
        while temp_mant and not (temp_mant & 0x80):
            temp_mant <<= 1
            leading_zeros += 1
        mant = mant << leading_zeros
        unbiased_exp = 1 - leading_zeros
    else:
        mant = (mant | 0x80) << 7  # 左移7位，因为BF16尾数是7位
    
    # 调整奇数指数
    if unbiased_exp & 1:
        mant = mant << 1
        unbiased_exp += 1
    
    # 计算新的指数
    result_exp = (unbiased_exp >> 1) + 127
    
    # 牛顿迭代
    a = mant
    x = 1 << ((unbiased_exp + 1) // 2)  # 初始估计值
    for _ in range(6):
        if x == 0:
            break
        a_div_x = a // x
        x = (x + a_div_x) >> 1
        if x == 0:
            x = 1
    
    # 规范化结果
    if x == 0:
        return compose_bf16(0, 0, 0)
    
    # 规范化到[0x80, 0x100)范围
    while x >= 0x100 and result_exp < 254:
        x >>= 1
        result_exp += 1
    while x < 0x80 and result_exp > 0:
        x <<= 1
        result_exp -= 1
    
    # 提取尾数（去掉隐含的1）
    result_mant = x & 0x7F
    
    return compose_bf16(0, result_exp, result_mant)

def test_bf16_sqrt():
    """测试BF16开方操作"""
    print("\n" + "="*60)
    print("BF16开方操作测试")
    print("="*60)
    
    def float_to_bf16(value):
        """将float转换为BF16格式"""
        if value == 0.0:
            return 0
        if value == -0.0:
            return 0x8000
        if np.isinf(value):
            return 0x7F80 if value > 0 else 0xFF80
        if np.isnan(value):
            return 0x7FC0
        
        # 获取符号位
        sign = 1 if value < 0 else 0
        value = abs(value)
        
        # 计算指数和尾数
        exponent = 0
        mantissa = value
        
        # 规格化
        if mantissa >= 2.0:
            while mantissa >= 2.0:
                mantissa /= 2.0
                exponent += 1
        elif mantissa < 1.0 and mantissa > 0:
            while mantissa < 1.0:
                mantissa *= 2.0
                exponent -= 1
        
        # 调整指数偏移
        exponent += 127
        
        # 处理非规格化数
        if exponent < 0:
            return compose_bf16(sign, 0, 0)
        
        # 处理溢出
        if exponent > 255:
            return compose_bf16(sign, 0xFF, 0)
        
        # 提取尾数
        mantissa = (mantissa - 1.0) * 128
        mantissa = int(round(mantissa))
        
        return compose_bf16(sign, exponent, mantissa)
    
    def bf16_to_float(bf16):
        """将BF16转换为float"""
        sign, exponent, mantissa = get_bf16_parts(bf16)
        
        # 处理特殊情况
        if exponent == 0:
            if mantissa == 0:
                return 0.0 if sign == 0 else -0.0
            # 非规格化数
            exponent = -126
            mantissa = mantissa / 128.0
        elif exponent == 0xFF:
            if mantissa == 0:
                return float('inf') if sign == 0 else float('-inf')
            return float('nan')
        else:
            # 规格化数
            exponent = exponent - 127
            mantissa = (mantissa / 128.0) + 1.0
        
        value = mantissa * (2.0 ** exponent)
        return -value if sign else value
    
    def test_case(value, expected=None):
        """测试单个用例"""
        bf16_value = float_to_bf16(value)
        result_bf16 = bf16_sqrt(bf16_value)
        result_float = bf16_to_float(result_bf16)
        
        if expected is None:
            expected = np.sqrt(value)
        
        print(f"输入: {value}")
        print(f"BF16输入: 0x{bf16_value:04X}")
        print(f"BF16结果: 0x{result_bf16:04X}")
        print(f"浮点结果: {result_float}")
        print(f"预期结果: {expected}")
        
        if np.isnan(expected):
            assert np.isnan(result_float), f"期望NaN，得到{result_float}"
        elif np.isinf(expected):
            assert np.isinf(result_float), f"期望Inf，得到{result_float}"
        else:
            error = abs(result_float - expected) / abs(expected) if expected != 0 else abs(result_float)
            print(f"相对误差: {error:.6f}")
            assert error < 0.1, f"误差过大: {error}"
        print()
    
    # 测试用例
    test_cases = [
        # 正常情况
        (4.0, 2.0),      # 简单开方
        (9.0, 3.0),      # 整数开方
        (2.0, None),     # 无理数开方
        (0.25, 0.5),     # 小数开方
        (0.01, 0.1),     # 小数值
        (100.0, 10.0),   # 大数值
        
        # 边界情况
        (0.0, 0.0),      # 零
        (1.0, 1.0),      # 一
        
        # 特殊情况
        (-1.0, None),    # 负数
        (float('inf'), float('inf')),  # 无穷大
        (float('nan'), float('nan')),  # NaN
    ]
    
    # 运行测试
    for value, expected in test_cases:
        try:
            test_case(value, expected)
            print("✅ 测试通过")
        except AssertionError as e:
            print(f"❌ 测试失败: {e}")
        print("-"*60)
    
    print("\n所有测试完成")

if __name__ == "__main__":
    test_bf16_sqrt() 

