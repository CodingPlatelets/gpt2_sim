import struct

class FP32toBF16Pipeline:
    def __init__(self):
        """初始化流水线寄存器和状态"""
        # 流水线寄存器
        self.stage1_fp32 = 0
        self.stage1_valid = False
        
        self.stage2_fp32 = 0
        self.stage2_fp32_sign = 0
        self.stage2_fp32_exponent = 0
        self.stage2_fp32_mantissa = 0
        
        self.stage2_high_bits = 0
        self.stage2_low_bits = 0
        self.stage2_valid = False
        
        self.stage3_bf16 = 0  # 合并后的阶段3（原来的阶段4）
        self.stage3_valid = False
        
        # 输出缓冲区
        self.outputs = []
        self.cycle_count = 0
    
    def reset(self):
        """重置流水线状态"""
        self.__init__()
        
    def _decompose_fp32(self, fp32):
        sign = (fp32 >> 31) & 0x1
        exponent = (fp32 >> 23) & 0xFF
        mantissa = fp32 & 0x7FFFFF
        return sign, exponent, mantissa
    
    def clock_cycle(self, new_fp32=None, new_valid=False):
        """
        模拟一个时钟周期，推进流水线
        
        Args:
            new_fp32: 新的FP32输入值（可选）
            new_valid: 输入是否有效
        """
        # 更新周期计数
        self.cycle_count += 1
        
        # 阶段3（舍入阶段）- 直接从阶段2获取数据并执行舍入
        
        # 0X7F80 : 0 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 正无穷大
        # 0XFF80 : 1 1 1 1 | 1 1 1 1 | 1 0 0 0 | 0 0 0 0 负无穷大
        # 0XFF90 : 1 1 1 1 | 1 1 1 1 | 1 0 0 1 | 0 0 0 0 NAN 
        # 0X7F7F : 0 1 1 1 | 1 1 1 1 | 0 1 1 1 | 1 1 1 1 最大正数
        # 0XFFFF : 1 1 1 1 | 1 1 1 1 |
        
        # 0 1 1 1 | 1 1 1 1| 0 1 1 1 | 1 1 1 1  
        if self.stage2_valid:
            
            if self.stage2_fp32_exponent == 0xFF:
                if self.stage2_fp32_mantissa == 0:
                    self.stage3_bf16 = self.stage2_high_bits
                else:
                    self.stage3_bf16 = (self.stage2_fp32_sign << 15) | (self.stage2_fp32_exponent << 7) | (self.stage2_fp32_mantissa >> 16) | 0x1
                    
            else:
                # 临近偶数位舍入
                if (self.stage2_low_bits > 0x8000 or 
                    (self.stage2_low_bits == 0x8000 and self.stage2_high_bits & 1)):
                    self.stage3_bf16 = (self.stage2_high_bits + 1) & 0xFFFF
                else:
                    self.stage3_bf16 = self.stage2_high_bits
                
            # 如果是有效输出，添加到输出缓冲区
            self.outputs.append({
                "bf16": self.stage3_bf16,
                "original_fp32": self.stage2_fp32  # 存储原始FP32值用于验证
            })
        
        self.stage3_valid = self.stage2_valid
        
        # 阶段2（提取阶段）- 从阶段1获取数据
        self.stage2_fp32 = self.stage1_fp32
        self.stage2_high_bits = (self.stage1_fp32 >> 16) & 0xFFFF
        self.stage2_low_bits = self.stage1_fp32 & 0xFFFF
        self.stage2_fp32_sign, self.stage2_fp32_exponent, self.stage2_fp32_mantissa = self._decompose_fp32(self.stage1_fp32)
        self.stage2_valid = self.stage1_valid
        
        # 阶段1（输入阶段）- 获取新输入
        if new_fp32 is not None and new_valid:
            # 如果提供了浮点数，则转换为32位整数表示
            if isinstance(new_fp32, float):
                self.stage1_fp32 = struct.unpack('>I', struct.pack('>f', new_fp32))[0]
            else:
                # 假设已是整数表示
                self.stage1_fp32 = new_fp32
        else:
            self.stage1_fp32 = 0
        
        self.stage1_valid = new_valid
        
        # 返回当前周期的输出
        return {
            "cycle": self.cycle_count,
            "valid_out": self.stage3_valid,
            "bf16_out": self.stage3_bf16 if self.stage3_valid else None,
            "pipeline_state": self.get_pipeline_state()
        }
    
    def get_pipeline_state(self):
        """返回流水线的当前状态，用于可视化"""
        return {
            "stage1": {
                "fp32": hex(self.stage1_fp32) if self.stage1_valid else "invalid",
                "valid": self.stage1_valid
            },
            "stage2": {
                "fp32": hex(self.stage2_fp32) if self.stage2_valid else "invalid",
                "high_bits": hex(self.stage2_high_bits) if self.stage2_valid else "invalid",
                "low_bits": hex(self.stage2_low_bits) if self.stage2_valid else "invalid",
                "valid": self.stage2_valid
            },
            "stage3": {
                "bf16": hex(self.stage3_bf16) if self.stage3_valid else "invalid",
                "valid": self.stage3_valid
            }
        }

    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - FP32: {state['stage1']['fp32']}")
        print(f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - High: {state['stage2']['high_bits']}, Low: {state['stage2']['low_bits']}")
        print(f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - BF16: {state['stage3']['bf16']}")
        if state['stage3']['valid']:
            print(f"  Output: BF16 = {state['stage3']['bf16']}")
        print()

    @staticmethod
    def fp32_to_bf16(fp32_value):
        """标准转换功能，用于验证"""
        # 将FP32转换为32位整数表示
        if isinstance(fp32_value, float):
            fp32_bits = struct.unpack('>I', struct.pack('>f', fp32_value))[0]
        else:
            fp32_bits = fp32_value
            
        # 提取高16位作为BF16
        bf16_bits = (fp32_bits >> 16) & 0xFFFF
        
        # 提取低16位用于舍入
        lower_bits = fp32_bits & 0xFFFF
        
        # 临近偶数位舍入
        if lower_bits > 0x8000 or (lower_bits == 0x8000 and (bf16_bits & 1) == 1):
            bf16_bits += 1
            
        return bf16_bits
    
    def run_simulation(self, inputs, print_states=True):
        """
        运行流水线模拟
        
        Args:
            inputs: 输入数据列表，每个元素是(fp32值, 有效标志)元组
            print_states: 是否打印每个周期的状态
        """
        self.reset()
        results = []
        
        # 确保输入列表足够长，不足部分用(0, False)填充
        extended_inputs = list(inputs) + [(0, False)] * 3  # 修改为3个周期（原来是4）
        
        # 运行流水线
        for i in range(len(extended_inputs)):
            fp32_value, valid = extended_inputs[i]
            result = self.clock_cycle(fp32_value, valid)
            results.append(result)
            
            if print_states:
                self.print_pipeline_state()
        
        return results
    
class BF16AddPipeline:
    def __init__(self):
        """初始化流水线寄存器和状态"""
        # 阶段1: 获取输入
        self.stage1_valid = False
        self.stage1_bf16_a = 0
        self.stage1_bf16_b = 0
        
        # 阶段2: 分解提取和准备阶段
        self.stage2_valid = False
        self.stage2_sign_a = 0
        self.stage2_exp_a = 0
        self.stage2_mant_a = 0
        self.stage2_sign_b = 0
        self.stage2_exp_b = 0
        self.stage2_mant_b = 0
        
        # 阶段3: 对齐阶段
        self.stage3_valid = False
        self.stage3_sign_a = 0
        self.stage3_sign_b = 0
        self.stage3_exp_result = 0
        self.stage3_mant_a = 0
        self.stage3_mant_b = 0
        
        # 阶段4: 计算阶段
        self.stage4_valid = False
        self.stage4_sign_result = 0
        self.stage4_exp_result = 0
        self.stage4_mant_result = 0
        
        # 输出缓冲区
        self.outputs = []
        self.cycle_count = 0
    
    def reset(self):
        """重置流水线状态"""
        self.__init__()
    
    def decompose_bf16(self, bf16):
        """分解BF16为符号位、指数位和尾数位"""
        sign = (bf16 >> 15) & 0x1
        exponent = (bf16 >> 7) & 0xFF
        mantissa = bf16 & 0x7F
        return sign, exponent, mantissa

    def compose_bf16(self, sign, exponent, mantissa):
        """组合符号位、指数位和尾数位为BF16"""
        return (sign << 15) | (exponent << 7) | (mantissa & 0x7F)
    
    def clock_cycle(self, bf16_a=None, bf16_b=None, valid=False):
        """
        模拟一个时钟周期，推进流水线
        
        Args:
            bf16_a, bf16_b: 输入的两个BF16值
            valid: 输入是否有效
        """
        # 更新周期计数
        self.cycle_count += 1
        
        # 阶段4: 归一化和组合阶段 - 处理阶段3的输出
        if self.stage4_valid:
            # 归一化处理
            mant_result = self.stage4_mant_result
            exp_result = self.stage4_exp_result
            sign_result = self.stage4_sign_result
            
            # 处理溢出情况
            if mant_result & 0x100:  # 尾数溢出，需要右移
                round_bit = mant_result & 0x1  # 当前最低位
                mant_result >>= 1
                exp_result += 1
                # 临近偶数舍入
                if round_bit and (mant_result & 0x1):
                    mant_result += 1
            
            # 处理尾数不足的情况
            while mant_result and not (mant_result & 0x80):  # 尾数不足，需要左移
                mant_result <<= 1
                exp_result -= 1
            
            # 去掉隐含的最高位
            mant_result &= 0x7F
            
            # 处理特殊情况（零、溢出）
            result_bf16 = 0
            if exp_result >= 0xFF:  # 指数溢出
                result_bf16 = self.compose_bf16(sign_result, 0xFF, 0)  # 无穷大
            elif exp_result <= 0:  # 指数不足
                result_bf16 = self.compose_bf16(0, 0, 0)  # 零
            else:
                result_bf16 = self.compose_bf16(sign_result, exp_result, mant_result)
            
            # 将结果添加到输出
            self.outputs.append(result_bf16)
        
        # 阶段3 -> 阶段4: 加减法阶段
        self.stage4_valid = self.stage3_valid
        if self.stage3_valid:
            # 按符号位执行加减法
            if self.stage3_sign_a == self.stage3_sign_b:
                # 同号，直接相加
                self.stage4_mant_result = self.stage3_mant_a + self.stage3_mant_b
                self.stage4_sign_result = self.stage3_sign_a
            else:
                # 异号，执行减法
                if self.stage3_mant_a >= self.stage3_mant_b:
                    self.stage4_mant_result = self.stage3_mant_a - self.stage3_mant_b
                    self.stage4_sign_result = self.stage3_sign_a
                else:
                    self.stage4_mant_result = self.stage3_mant_b - self.stage3_mant_a
                    self.stage4_sign_result = self.stage3_sign_b
            
            self.stage4_exp_result = self.stage3_exp_result
        
        # 阶段2 -> 阶段3: 对齐指数
        self.stage3_valid = self.stage2_valid
        if self.stage2_valid:
            # 获取值
            sign_a = self.stage2_sign_a
            exp_a = self.stage2_exp_a
            mant_a = self.stage2_mant_a
            sign_b = self.stage2_sign_b
            exp_b = self.stage2_exp_b
            mant_b = self.stage2_mant_b
            
            # 对齐指数
            if exp_a > exp_b:
                shift = exp_a - exp_b
                mant_b >>= shift
                exp_result = exp_a
            elif exp_b > exp_a:
                shift = exp_b - exp_a
                mant_a >>= shift
                exp_result = exp_b
            else:
                exp_result = exp_a  # exp_a == exp_b
            
            # 传递到下一阶段
            self.stage3_sign_a = sign_a
            self.stage3_sign_b = sign_b
            self.stage3_exp_result = exp_result
            self.stage3_mant_a = mant_a
            self.stage3_mant_b = mant_b
        
        # 阶段1 -> 阶段2: 分解和准备
        self.stage2_valid = self.stage1_valid
        if self.stage1_valid:
            # 分解两个BF16数值
            sign_a, exp_a, mant_a = self.decompose_bf16(self.stage1_bf16_a)
            sign_b, exp_b, mant_b = self.decompose_bf16(self.stage1_bf16_b)
            
            # 添加隐含的最高位
            if exp_a != 0:
                mant_a |= 0x80  # 添加隐含的1
            if exp_b != 0:
                mant_b |= 0x80  # 添加隐含的1
            
            # 传递到下一阶段
            self.stage2_sign_a = sign_a
            self.stage2_exp_a = exp_a
            self.stage2_mant_a = mant_a
            self.stage2_sign_b = sign_b
            self.stage2_exp_b = exp_b
            self.stage2_mant_b = mant_b
        
        # 阶段1: 获取输入
        if valid and bf16_a is not None and bf16_b is not None:
            self.stage1_bf16_a = bf16_a
            self.stage1_bf16_b = bf16_b
            self.stage1_valid = True
        else:
            self.stage1_valid = False
            self.stage1_bf16_a = 0
            self.stage1_bf16_b = 0
        
        # 返回当前周期的状态
        return {
            "cycle": self.cycle_count,
            "valid_output": self.stage4_valid,
            "pipeline_state": self.get_pipeline_state()
        }
    
    def get_pipeline_state(self):
        """返回流水线的当前状态，用于可视化"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "bf16_a": hex(self.stage1_bf16_a) if self.stage1_valid else "invalid",
                "bf16_b": hex(self.stage1_bf16_b) if self.stage1_valid else "invalid"
            },
            "stage2": {
                "valid": self.stage2_valid,
                "sign_a": self.stage2_sign_a if self.stage2_valid else "invalid",
                "exp_a": self.stage2_exp_a if self.stage2_valid else "invalid",
                "mant_a": hex(self.stage2_mant_a) if self.stage2_valid else "invalid",
                "sign_b": self.stage2_sign_b if self.stage2_valid else "invalid",
                "exp_b": self.stage2_exp_b if self.stage2_valid else "invalid",
                "mant_b": hex(self.stage2_mant_b) if self.stage2_valid else "invalid"
            },
            "stage3": {
                "valid": self.stage3_valid,
                "aligned_exp": self.stage3_exp_result if self.stage3_valid else "invalid",
                "aligned_mant_a": hex(self.stage3_mant_a) if self.stage3_valid else "invalid",
                "aligned_mant_b": hex(self.stage3_mant_b) if self.stage3_valid else "invalid"
            },
            "stage4": {
                "valid": self.stage4_valid,
                "sign": self.stage4_sign_result if self.stage4_valid else "invalid",
                "exp": self.stage4_exp_result if self.stage4_valid else "invalid",
                "mant": hex(self.stage4_mant_result) if self.stage4_valid else "invalid"
            }
        }
    
    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - BF16 A: {state['stage1']['bf16_a']}, BF16 B: {state['stage1']['bf16_b']}")
        print(f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - Prepared operands")
        print(f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - Aligned operands, Exponent: {state['stage3']['aligned_exp']}")
        print(f"  Stage 4: {'Valid' if state['stage4']['valid'] else 'Invalid'} - Sign: {state['stage4']['sign']}, Exp: {state['stage4']['exp']}, Mant: {state['stage4']['mant']}")
        print()
    
    def run_simulation(self, inputs, print_states=True):
        """
        运行流水线模拟
        
        Args:
            inputs: 输入数据列表，每个元素是(bf16_a, bf16_b, valid)元组
            print_states: 是否打印每个周期的状态
        """
        self.reset()
        results = []
        
        # 确保输入列表足够长，不足部分用(0, 0, False)填充
        extended_inputs = list(inputs) + [(0, 0, False)] * 4  # 加4个周期确保流水线清空
        
        # 运行流水线
        for i in range(len(extended_inputs)):
            bf16_a, bf16_b, valid = extended_inputs[i]
            result = self.clock_cycle(bf16_a, bf16_b, valid)
            results.append(result)
            
            if print_states:
                self.print_pipeline_state()
        
        return results
    

def test_fp32_to_bf16():
    # 创建流水线实例
    pipeline = FP32toBF16Pipeline()
    
    # 准备测试数据：[(FP32值, 是否有效), ...]
    
    # 可能溢出数据(fp32)
    
    # 0 | 1 1 1 || 1 1 1 1 || 1 | 0 0 0|| 0 .... || 0 0 0 0
    #   7F8F FFFF            
    # 0 1 1 1 1 1 1 1 
    
    data_1 = struct.unpack('>f', struct.pack('>I', 0x7F8FFFFF))[0]
    data_2 = struct.unpack('>f', struct.pack('>I', 0x7F800000))[0]
    data_3 = struct.unpack('>f', struct.pack('>I', 0x7F800001))[0]
    data_4 = struct.unpack('>f', struct.pack('>I', 0xFFFFFFFF))[0]
    
    test_data = [
        (3.14159, True),  # Pi，十六进制表示为 0x40490FDB
        (1.5, True),      # 1.5，十六进制表示为 0x3FC00000
        (-2.25, True),    # -2.25，十六进制表示为 0xC0100000
        (0.0, True),     # 无效输入
        (65504.0, True),  # BF16能表示的最大正数
        (1e-20, True),    # 非常小的数
        (data_1, True),  #NAN
        (data_2, True), #inf
        (data_3, True),
        (data_4, True)
    ]
    
    print("Starting FP32 to BF16 Pipeline Simulation")
    print("=" * 50)
    
    # 运行模拟
    results = pipeline.run_simulation(test_data)
    import math    
    print("=" * 50)
    print("Final Outputs:")
    for i, output in enumerate(pipeline.outputs):
        bf16_value = output["bf16"]
        # 将BF16转回FP32表示进行验证
        bf16_as_fp32 = struct.unpack('>f', struct.pack('>I', bf16_value << 16))[0]
        
        # 找到对应的原始输入，通过FP32位级表示比较
        original_input = None
        for fp32, valid in test_data:
            if valid:
                fp32_bits = struct.unpack('>I', struct.pack('>f', fp32))[0]
                if fp32_bits == output["original_fp32"]:
                    original_input = fp32
                    break
        
        print(f"Output {i}: BF16: {hex(bf16_value)} -> As FP32: {bf16_as_fp32}")
        if original_input is not None:
            print(f"  Original input: {original_input}")
    
    print("\nVerification:")
    valid_inputs = [(fp32, i) for i, (fp32, valid) in enumerate(test_data) if valid]
    
    for i, (fp32, orig_idx) in enumerate(valid_inputs):
        if i < len(pipeline.outputs):
            # 验证模拟结果与直接计算结果是否一致
            direct_result = pipeline.fp32_to_bf16(fp32)
            pipeline_result = pipeline.outputs[i]["bf16"]
            print(f"Input {orig_idx}: {fp32}")
            print(f"  Direct conversion: {hex(direct_result)}")
            print(f"  Pipeline output:   {hex(pipeline_result)}")
            print(f"  Match: {direct_result == pipeline_result}")
            
def test_bf16add():
    # 创建流水线实例
    pipeline = BF16AddPipeline()
    
    # 将浮点数转换为BF16表示
    def float_to_bf16(value):
        # 将float转换为32位整数表示
        fp32_bits = struct.unpack('>I', struct.pack('>f', value))[0]
        # 取高16位作为BF16
        bf16 = (fp32_bits >> 16) & 0xFFFF
        return bf16
    
    # 将BF16转换为浮点数表示
    def bf16_to_float(bf16):
        # 左移16位填充为32位表示
        fp32_bits = bf16 << 16
        # 转换为浮点数
        return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
    
    # 准备测试用例
    test_cases = [
        (1.5, 2.25, True),        # 简单加法: 1.5 + 2.25 = 3.75
        (3.14159, -1.5, True),    # 异号加法: 3.14159 + (-1.5) ≈ 1.64159
        (-3.0, -2.0, True),       # 负数加法: (-3.0) + (-2.0) = -5.0
        (100.0, 0.001, True),     # 量级差异大: 100.0 + 0.001 ≈ 100.001
        (0.0, 0.0, True),         # 零加零: 0.0 + 0.0 = 0.0
        (0.0, 1.0, False),        # 无效输入
        (1e4, 1e4, True)          # 大数: 10000 + 10000 = 20000
    ]
    
    def convert_through_pipeline(value):
        """通过完整流水线模拟转换FP32到BF16"""
        temp_pipeline = FP32toBF16Pipeline()
        temp_pipeline.run_simulation([(value, True)], print_states=False)
        return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0

    bf16_test_cases = [(convert_through_pipeline(a), 
                        convert_through_pipeline(b), 
                        valid) for a, b, valid in test_cases]
    
    
    # 将浮点数转换为BF16表示
    #bf16_test_cases = [(float_to_bf16(a), float_to_bf16(b), valid) for a, b, valid in test_cases]
    
    print("Starting BF16 Addition Pipeline Simulation")
    print("=" * 50)
    
    # 运行模拟
    results = pipeline.run_simulation(bf16_test_cases)
    
    
    
    print("=" * 50)
    print("Final Results:")
    for i, output in enumerate(pipeline.outputs):
        if i < len(test_cases) and test_cases[i][2]:  # 只检查有效输入
            a, b, _ = test_cases[i]
            expected_sum = a + b
            actual_float = bf16_to_float(output)
            print(f"Test {i}: {a} + {b} = {expected_sum}")
            print(f"  BF16 result: {hex(output)} -> {actual_float}")
            print(f"  Error: {abs(actual_float - expected_sum)}")
            print()

# 测试模拟器
if __name__ == "__main__":
    test_fp32_to_bf16()
    #test_bf16add()
    #t = 0.0
    #return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
    #t_bits = struct.unpack('>I', struct.pack('>f', t))[0]
    #t_bits_float = struct.unpack('>f', struct.pack('>I', t_bits | 0X1))
    #print(t_bits_float)