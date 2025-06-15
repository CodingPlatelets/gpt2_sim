import torch
import struct

class BF16:
    def __init__(self, value=0):
        """
        初始化 BF16 类，支持从 FP32 转换为 BF16。
        """
        if isinstance(value, float):
            self.value = self.fp32_to_bf16(value)
        elif isinstance(value, int):
            self.value = value
        else:
            raise ValueError("Unsupported value type. Must be float or int.")

    @staticmethod
    def fp32_to_bf16(fp32):
        """
        将 FP32 转换为 BF16，使用临近偶数位舍入。
        """

        # 0X7F80 : 0 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 正无穷大
        # 0XFF80 : 1 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 负无穷大
        # 0XFF90 : 1 1 1 1 | 1 1 1 1 | 1 0 0 1 | 0 0 0 0 NAN 
        # 0X7F7F : 0 1 1 1 | 1 1 1 1 | 0 1 1 1 | 1 1 1 1 最大正数
        # 0XFFFF : 1 1 1 1 | 1 1 1 1 |
        
        # 0 1 1 1 | 1 1 1 1| 0 1 1 1 | 1 1 1 1  
        
        # 将 FP32 转换为 32 位整数表示
        fp32_bits = struct.unpack('>I', struct.pack('>f', fp32))[0]

        def decompose_fp32(fp32):
            sign = (fp32 >> 31) & 0x1
            exponent = (fp32 >> 23) & 0xFF
            mantissa = fp32 & 0x7FFFFF
            return sign, exponent, mantissa
        
        
        fp32_sign, fp32_exponent, fp32_mantissa = decompose_fp32(fp32)
        
        # 提取高 16 位作为 BF16
        bf16_bits = (fp32_bits >> 16) & 0xFFFF
        
        #exp全为1，表示inf or nan  
        if fp32_exponent == 0xFF:
            #nan_bit =
            if fp32_mantissa == 0:
                return bf16_bits
            return (fp32_sign << 15) | (fp32_exponent << 7) | (fp32_mantissa >> 16) | 0x1
         
                 
        # 提取低 16 位用于舍入
        lower_bits = fp32_bits & 0xFFFF

        # 临近偶数位舍入
        if lower_bits > 0x8000 or (lower_bits == 0x8000 and (bf16_bits & 1) == 1):
            bf16_bits += 1

        # 返回 BF16 的 16 位整数表示
        return bf16_bits

    @staticmethod
    def bf16_to_fp32(bf16):
        """
        将 BF16 转换为 FP32。
        """
        import struct

        # 将 BF16 转换为 32 位整数表示
        fp32_bits = bf16 << 16

        # 转换为 FP32 浮点数
        return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

    @staticmethod
    def add(bf16_a, bf16_b):
        """
        实现 BF16 加法器。
        """
        # 转换为 FP32 进行计算
        fp32_a = BF16.bf16_to_fp32(bf16_a)
        fp32_b = BF16.bf16_to_fp32(bf16_b)
        result_fp32 = fp32_a + fp32_b

        # 转换回 BF16
        return BF16.fp32_to_bf16(result_fp32)
    
    @staticmethod
    def bf16_bitwise_add(bf16_a, bf16_b):
        """
        实现 BF16 的按位加法。
        """
        # 0X7F80 : 0 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 正无穷大
        # 0XFF80 : 1 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 负无穷大
        # 0XFF90 : 1 1 1 1 | 1 1 1 1 | 1 0 0 1 | 0 0 0 0 NAN 
        # 0XFFFF : 1 1 1 1 | 1 1 1 1 |
        
        # 0  0 1 1 1 1 1 0 0  0 1 0 0 0 0 0
        # 15 14            7  6           0
        #          8bit             7bit
        # value = (-1)^sign * 2^(e - 127) * 1.frac
        # 提取符号位、指数位和尾数位

        
        def decompose_bf16(bf16):
            sign = (bf16 >> 15) & 0x1  # 符号位
            exponent = (bf16 >> 7) & 0xFF  # 指数位
            mantissa = bf16 & 0x7F  # 尾数位
            return sign, exponent, mantissa

        # 重新组合 BF16
        def compose_bf16(sign, exponent, mantissa):
            return (sign << 15) | (exponent << 7) | (mantissa & 0x7F)
        
        # 常量定义
        POS_INF = 0x7F80  # 正无穷大：0 11111111 0000000
        NEG_INF = 0xFF80  # 负无穷大：1 11111111 0000000
        NAN = 0x7FC0      # NaN：0 11111111 1000000 

        # 分解两个 BF16 数值
        sign_a, exp_a, mant_a = decompose_bf16(bf16_a)
        sign_b, exp_b, mant_b = decompose_bf16(bf16_b)

        # 特殊情况处理
    
        # 1. 处理 NaN
        if ((exp_a == 0xFF and mant_a != 0) or (exp_b == 0xFF and mant_b != 0)):
            return NAN  # 返回 NaN
        
        # 2. 处理无穷大
        if exp_a == 0xFF:  # a 是无穷大
            if exp_b == 0xFF and sign_a != sign_b:  # a 和 b 是符号相反的无穷大
                return NAN  # 无穷大 - 无穷大 = NaN
            return compose_bf16(sign_a, 0xFF, 0)  # 返回 a 的无穷大
        
        if exp_b == 0xFF:  # b 是无穷大
            return compose_bf16(sign_b, 0xFF, 0)  # 返回 b 的无穷大

        # 3. 处理零值
        if (exp_a == 0 and mant_a == 0) and (exp_b == 0 and mant_b == 0):
            # 两个都是零，如果符号相同，返回该符号的零；如果符号不同，返回正零
            if sign_a == sign_b:
                return compose_bf16(sign_a, 0, 0)
            else:
                return compose_bf16(0, 0, 0)  # 正零
        
        if (exp_a == 0 and mant_a == 0):  # a 是零
            return bf16_b  # 返回 b                      
    
        if (exp_b == 0 and mant_b == 0):  # b 是零
            return bf16_a  # 返回 a
        
        
        # 添加隐含的最高位（1），非零指数时，当指数为0时代表非常小数，指数为 1111 1111 表示非常大数（若为-则非常小）
        # 当指数为0时 value = (-1)^sign * 2^(1 - bias) * (0.mantissa)



        if exp_a != 0:
            mant_a |= 0x80  # 添加隐含的1
        else:
            # 非规格化处理
            exp_a = 1  # 调整指数

        if exp_b != 0:
            mant_b |= 0x80  # 添加隐含的1
        else:
            # 非规格化处理
            exp_b = 1  # 调整指数

        # 对齐指数
        if exp_a > exp_b:
            shift = exp_a - exp_b
            mant_b >>= shift
            exp_b = exp_a
        elif exp_b > exp_a:
            shift = exp_b - exp_a
            mant_a >>= shift
            exp_a = exp_b

        # 按符号位执行加减法
        if sign_a == sign_b:
            # 同号，直接相加
            mant_result = mant_a + mant_b
            sign_result = sign_a
        else:
            # 异号，执行减法
            if mant_a >= mant_b:
                mant_result = mant_a - mant_b
                sign_result = sign_a
            else:
                mant_result = mant_b - mant_a
                sign_result = sign_b
                
        # 0X7F80 : 0 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 正无穷大
        # 0XFF80 : 1 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 负无穷大
        # 0XFF90 : 1 1 1 1 | 1 1 1 1 | 1 0 0 1 | 0 0 0 0 NAN 
        # 0XFFFF : 1 1 1 1 | 1 1 1 1 |
        
        # 0 1 1 1 | 1 1 1 1| 0 1 1 1 | 1 1 1 1

        if mant_result == 0:
            return compose_bf16(0, 0, 0)  # 返回正零

        # 归一化结果
        if mant_result & 0x100:  # 尾数溢出，需要右移
            # 提取最低位，用于舍入
            round_bit = mant_result & 0x1  # 当前最低位

            # 右移尾数
            mant_result >>= 1
            exp_a += 1

            # 临近偶数舍入
            # 临近偶数舍入
        if round_bit and (mant_result & 0x1):
            mant_result += 1
            # 检查舍入后是否再次溢出
            if mant_result & 0x100:
                mant_result >>= 1
                exp_a += 1

        while mant_result and not (mant_result & 0x80):  # 尾数不足，需要左移(隐藏位为0了)
            mant_result <<= 1
            exp_a -= 1

        # 去掉隐含的最高位
        mant_result &= 0x7F

        # 处理特殊情况（零、溢出）
        if exp_a >= 0xFF:  # 指数溢出，返回无穷大
            return compose_bf16(sign_result, 0xFF, 0)
        if exp_a <= 0:  # 指数不足，返回零
            return compose_bf16(0, 0, 0)

        # 组合结果
        return compose_bf16(sign_result, exp_a, mant_result)
    
    @staticmethod
    def bf16_bitwise_multiply(bf16_a, bf16_b):
        """
        实现 BF16 的按位乘法。
        """
        # 提取符号位、指数位和尾数位
        def decompose_bf16(bf16):
            sign = (bf16 >> 15) & 0x1  # 符号位
            exponent = (bf16 >> 7) & 0xFF  # 指数位
            mantissa = bf16 & 0x7F  # 尾数位
            return sign, exponent, mantissa

        # 重新组合 BF16
        def compose_bf16(sign, exponent, mantissa):
            return (sign << 15) | (exponent << 7) | (mantissa & 0x7F)

        # 分解两个 BF16 数值
        sign_a, exp_a, mant_a = decompose_bf16(bf16_a)
        sign_b, exp_b, mant_b = decompose_bf16(bf16_b)

        # 符号位：相乘后的符号位是两个符号位的异或
        sign_result = sign_a ^ sign_b

        # 处理特殊情况：NaN 的处理
        if ((exp_a == 0xFF and mant_a != 0) or (exp_b == 0xFF and mant_b != 0)):
            return compose_bf16(0, 0xFF, 0x40)  # 返回 NaN
            
        # 处理特殊情况：零乘以任何数 = 零
        if (exp_a == 0 and mant_a == 0) or (exp_b == 0 and mant_b == 0):
            return compose_bf16(sign_result, 0, 0)  # 零
        
        # 处理特殊情况：无穷大乘以非零数 = 无穷大
        if exp_a == 0xFF or exp_b == 0xFF:
            return compose_bf16(sign_result, 0xFF, 0)  # 无穷大

        # 添加隐含的最高位，处理规格化和非规格化数
        if exp_a != 0:
            mant_a |= 0x80  # 添加隐含的1
        else:
            # 非规格化处理
            exp_a = 1  # 调整指数
            
        if exp_b != 0:
            mant_b |= 0x80  # 添加隐含的1
        else:
            # 非规格化处理
            exp_b = 1  # 调整指数

        # 指数相加（在乘法中，指数是相加的），并减去偏移值（127）
        exp_result = exp_a + exp_b - 127

        # 尾数相乘（带隐含的最高位）
        mant_result = mant_a * mant_b  # 最多可达到16位

                # 找到最高有效位位置
        if mant_result == 0:
            return compose_bf16(sign_result, 0, 0)  # 结果为0
            
        msb_pos = mant_result.bit_length() - 1  # 使用内置方法获取最高位位置（从0开始计数）

        # 直接调整尾数到最终的7位表示
        # 目标是将msb移动到第7位(对应最终结果的隐含位)
        target_pos = 7
        shift = msb_pos - target_pos

        # 1 1 1 1 0 1  shift = 3
        # round bits = 101
        # (1 << shift) = 1000 - 1 = 111
        if shift > 0:  # 需要右移
            # 舍入位处理
            round_bits = mant_result & ((1 << shift) - 1)
            mant_result >>= shift
            exp_result += shift
            
            # 临近偶数舍入
            half_point = 1 << (shift - 1)
            if round_bits > half_point or (round_bits == half_point and (mant_result & 0x1)):
                mant_result += 1
                
                # 如果舍入导致进位(超出7位)
                if mant_result & 0x100:
                    mant_result >>= 1
                    exp_result += 1

        elif shift < 0:  # 需要左移
            mant_result <<= -shift
            exp_result += shift  # 注意这里是减小指数(shift是负的)

        # 去掉隐含的最高位，直接获取7位尾数
        mant_result &= 0x7F

        # 处理特殊情况：下溢和上溢
        if exp_result <= 0:  # 下溢，返回零或非规格化数
            if exp_result < -7:  # 太小，直接返回零
                return compose_bf16(sign_result, 0, 0)
            else:  # 尝试以非规格化形式表示
                mant_result = (mant_result | 0x80) >> (1 - exp_result)
                return compose_bf16(sign_result, 0, mant_result & 0x7F)
                
        if exp_result >= 0xFF:  # 上溢，返回无穷大
            return compose_bf16(sign_result, 0xFF, 0)

        # 组合结果
        return compose_bf16(sign_result, exp_result, mant_result)


    @staticmethod
    def multiply(bf16_a, bf16_b):
        """
        实现 BF16 乘法器。
        """
        # 转换为 FP32 进行计算
        fp32_a = BF16.bf16_to_fp32(bf16_a)
        fp32_b = BF16.bf16_to_fp32(bf16_b)
        result_fp32 = fp32_a * fp32_b

        # 转换回 BF16
        return BF16.fp32_to_bf16(result_fp32)

    def to_fp32(self):
        """
        将当前 BF16 值转换为 FP32。
        """
        return self.bf16_to_fp32(self.value)

    def __add__(self, other):
        if isinstance(other, BF16):
            return BF16(self.add(self.value, other.value))
        else:
            raise ValueError("Addition only supported between BF16 instances.")

    def __mul__(self, other):
        if isinstance(other, BF16):
            return BF16(self.multiply(self.value, other.value))
        else:
            raise ValueError("Multiplication only supported between BF16 instances.")

    def __repr__(self):
        return f"BF16(value={self.value}, fp32={self.to_fp32()})"
    



    

def pytorch_fp32_to_bf16(fp32):
    """
    使用 PyTorch 验证 FP32 转 BF16。
    """
    # 将 FP32 转换为 PyTorch 的 bfloat16
    tensor_fp32 = torch.tensor(fp32, dtype=torch.float32)
    tensor_bf16 = tensor_fp32.to(dtype=torch.bfloat16)
    
    # 转换为 float32 再提取二进制表示
    tensor_bf16_as_fp32 = tensor_bf16.to(dtype=torch.float32)
    bf16_bits = struct.unpack('>H', struct.pack('>f', tensor_bf16_as_fp32.item())[0:2])[0]
    return bf16_bits

def verify_fp32_to_bf16(fp32_value):
    """
    验证自定义 FP32 转 BF16 的实现是否正确。
    """
    # 使用自定义实现
    custom_bf16 = BF16.fp32_to_bf16(fp32_value)

    # 使用 PyTorch 验证
    pytorch_bf16 = pytorch_fp32_to_bf16(fp32_value)

    # 打印结果
    print(f"FP32 Value: {fp32_value}")
    print(f"Custom BF16: {hex(custom_bf16)}")
    print(f"PyTorch BF16: {hex(pytorch_bf16)}")

    # 验证是否一致
    assert custom_bf16 == pytorch_bf16, "Custom implementation does not match PyTorch!"
    print("Custom implementation matches PyTorch!")

def verify_bf16_bitwise_add():
    """
    使用 PyTorch 验证 bf16_bitwise_add 函数的实现是否正确。
    """
    import torch
    import struct
    
    # 测试用例列表，每个用例包含两个浮点数
    test_cases = [
        (1.5, 2.0),      # 简单的相加
        (-1.5, 2.0),     # 带符号的加法
        (0.0, 3.14),     # 零和非零
        (1e-10, 1e10),   # 极端数值
        (65504.0, 0.1),  # 接近 BF16 最大值
        (-0.0, 0.0),     # 正零和负零
    ]
    
    for a, b in test_cases:
        # 1. 将浮点数转换为 BF16 表示
        tensor_a = torch.tensor(a, dtype=torch.float32)
        tensor_b = torch.tensor(b, dtype=torch.float32)
        bf16_a = tensor_a.to(dtype=torch.bfloat16)
        bf16_b = tensor_b.to(dtype=torch.bfloat16)
        
        # 2. 使用 PyTorch 执行 BF16 加法
        pytorch_result = (bf16_a + bf16_b).to(dtype=torch.float32).item()
        
        # 3. 转换为 BF16 整数表示
        def float_to_bf16_int(f):
            tensor = torch.tensor(f, dtype=torch.float32)
            bf16_tensor = tensor.to(dtype=torch.bfloat16)
            # 将 bf16 转回 float32 然后获取二进制表示
            bf16_as_f32 = bf16_tensor.to(dtype=torch.float32)
            # 提取高16位
            bits = struct.unpack('>I', struct.pack('>f', bf16_as_f32.item()))[0]
            return (bits >> 16) & 0xFFFF
        
        bf16_int_a = float_to_bf16_int(a)
        bf16_int_b = float_to_bf16_int(b)
        
        # 4. 使用自定义函数执行加法
        custom_result_int = BF16.bf16_bitwise_add(bf16_int_a, bf16_int_b)
        
        # 5. 将自定义结果转回浮点数
        custom_result = BF16.bf16_to_fp32(custom_result_int)
        
        # 6. 比较结果
        print(f"Test case: {a} + {b}")
        print(f"PyTorch result: {pytorch_result}")
        print(f"Custom result: {custom_result}")
        print(f"Equal: {abs(pytorch_result - custom_result) < 1e-5}")
        print("---")
        
        # 7. 补充打印调试信息
        print(f"BF16 hex - a: {hex(bf16_int_a)}, b: {hex(bf16_int_b)}, result: {hex(custom_result_int)}")
        print(f"BF16 bits - a: {bin(bf16_int_a)}, b: {bin(bf16_int_b)}, result: {bin(custom_result_int)}")
        print("\n")

# 测试代码
if __name__ == "__main__":
    # 测试 FP32 转 BF16
    fp32_value = 3.1415926
    bf16_value = BF16(fp32_value)
    print(f"FP32: {fp32_value} -> BF16: {bf16_value}")

    # 测试加法器
    bf16_a = BF16(1.5)
    bf16_b = BF16(2.25)
    bf16_sum = bf16_a + bf16_b
    print(f"BF16 Add: {bf16_a} + {bf16_b} = {bf16_sum}")

    # 测试乘法器
    bf16_mul = bf16_a * bf16_b
    print(f"BF16 Multiply: {bf16_a} * {bf16_b} = {bf16_mul}")
    
    test_values = [3.1415926, 1.5, 2.25, -1.0, 0.0, 65504.0]  # 包括一些边界值
    for value in test_values:
        verify_fp32_to_bf16(value)
        
    verify_bf16_bitwise_add()