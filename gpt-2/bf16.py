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

        # 将 FP32 转换为 32 位整数表示
        fp32_bits = struct.unpack('>I', struct.pack('>f', fp32))[0]

        # 提取高 16 位作为 BF16
        bf16_bits = (fp32_bits >> 16) & 0xFFFF

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

        # 添加隐含的最高位（1），非零指数时
        if exp_a != 0:
            mant_a |= 0x80
        if exp_b != 0:
            mant_b |= 0x80

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

        # 归一化结果
        if mant_result & 0x100:  # 尾数溢出，需要右移
            mant_result >>= 1
            exp_a += 1
        while mant_result and not (mant_result & 0x80):  # 尾数不足，需要左移
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
        
    binary_data = struct.pack('>f', 3.14)
    print(binary_data) 
    
    # 示例 BF16 数值（以整数形式表示）
    bf16_a = 0x3FC0  # 1.5 in BF16
    bf16_b = 0x4000  # 2.0 in BF16

    # 按位加法
    bf16_result = bf16_bitwise_add(bf16_a, bf16_b)
    print(f"BF16 Add: {hex(bf16_a)} + {hex(bf16_b)} = {hex(bf16_result)}")