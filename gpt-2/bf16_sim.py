import struct
import random
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
        # 阶段1: 获取输入并处理特殊情况
        self.stage1_valid = False
        self.stage1_bf16_a = 0
        self.stage1_bf16_b = 0
        self.stage1_special_case = False  # 标记是否是特殊情况
        self.stage1_result = 0  # 特殊情况的直接结果
        
        # 阶段2: 分解提取和准备阶段
        self.stage2_valid = False
        self.stage2_special_case = False
        self.stage2_result = 0
        self.stage2_sign_a = 0
        self.stage2_exp_a = 0
        self.stage2_mant_a = 0
        self.stage2_sign_b = 0
        self.stage2_exp_b = 0
        self.stage2_mant_b = 0
        
        # 阶段3: 对齐阶段
        self.stage3_valid = False
        self.stage3_special_case = False
        self.stage3_result = 0
        self.stage3_sign_a = 0
        self.stage3_sign_b = 0
        self.stage3_exp_result = 0
        self.stage3_mant_a = 0
        self.stage3_mant_b = 0
        
        # 阶段4: 计算阶段
        self.stage4_valid = False
        self.stage4_special_case = False
        self.stage4_result = 0
        self.stage4_sign_result = 0
        self.stage4_exp_result = 0
        self.stage4_mant_result = 0
        
        self.stage5_valid = False
        
        # 常量定义
        self.POS_INF = 0x7F80  # 正无穷大：0 11111111 0000000
        self.NEG_INF = 0xFF80  # 负无穷大：1 11111111 0000000
        self.NAN = 0x7FC0      # NaN：0 11111111 1000000
        
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
    
    def check_special_cases(self, bf16_a, bf16_b):
        """检查并处理特殊情况"""
        sign_a, exp_a, mant_a = self.decompose_bf16(bf16_a)
        sign_b, exp_b, mant_b = self.decompose_bf16(bf16_b)
        
        # 1. 处理 NaN
        if ((exp_a == 0xFF and mant_a != 0) or (exp_b == 0xFF and mant_b != 0)):
            return True, self.NAN
        
        # 2. 处理无穷大
        if exp_a == 0xFF:  # a 是无穷大
            if exp_b == 0xFF and sign_a != sign_b:  # a 和 b 是符号相反的无穷大
                return True, self.NAN
            return True, self.compose_bf16(sign_a, 0xFF, 0)
        
        if exp_b == 0xFF:  # b 是无穷大
            return True, self.compose_bf16(sign_b, 0xFF, 0)
        
        # 3. 处理零值
        if (exp_a == 0 and mant_a == 0) and (exp_b == 0 and mant_b == 0):
            # 两个都是零，如果符号相同，返回该符号的零；如果符号不同，返回正零
            if sign_a == sign_b:
                return True, self.compose_bf16(sign_a, 0, 0)
            else:
                return True, self.compose_bf16(0, 0, 0)
        
        if (exp_a == 0 and mant_a == 0):  # a 是零
            return True, bf16_b
        
        if (exp_b == 0 and mant_b == 0):  # b 是零
            return True, bf16_a
            
        # 不是特殊情况
        return False, 0
    
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
        self.stage5_valid = self.stage4_valid
        if self.stage4_valid:
            result_bf16 = 0
            
            if self.stage4_special_case:
                # 直接输出特殊情况结果
                result_bf16 = self.stage4_result
            else:
                # 归一化处理
                mant_result = self.stage4_mant_result
                exp_result = self.stage4_exp_result
                sign_result = self.stage4_sign_result
                
                # 处理结果为0的情况
                if mant_result == 0:
                    result_bf16 = self.compose_bf16(0, 0, 0)  # 返回正零
                else:
                    # 处理溢出情况
                    if mant_result & 0x100:  # 尾数溢出，需要右移
                        # 提取被丢弃的位，用于舍入
                        round_bit = mant_result & 0x1
                        
                        # 右移尾数
                        mant_result >>= 1
                        exp_result += 1
                        
                        # 临近偶数舍入
                        if round_bit and (mant_result & 0x1):
                            mant_result += 1
                            # 检查舍入是否导致再次溢出
                            if mant_result & 0x100:
                                mant_result >>= 1
                                exp_result += 1
                    
                    # 处理尾数不足的情况
                    while mant_result and not (mant_result & 0x80):
                        mant_result <<= 1
                        exp_result -= 1
                    
                    # 去掉隐含的最高位
                    mant_result &= 0x7F

                    mant_result &= 0x7F

                    # 下溢/非规格化处理
                    if exp_result <= 0:
                        if exp_result < -6:  # 太小，直接返回零
                            result_bf16 = self.compose_bf16(sign_result, 0, 0)
                        else:
                            # 先补隐含1，右移(1-exp_result)位
                            denorm_mant = 0x80 | mant_result
                            shift_amount = 1 - exp_result
                            # 舍入
                            round_bit = (denorm_mant >> (shift_amount - 1)) & 1
                            sticky_bits = (denorm_mant & ((1 << (shift_amount - 1)) - 1)) != 0
                            denorm_mant >>= shift_amount
                            if round_bit and (sticky_bits or (denorm_mant & 1)):
                                denorm_mant += 1
                            result_bf16 = self.compose_bf16(sign_result, 0, denorm_mant & 0x7F)
                    elif exp_result >= 0xFF:
                        result_bf16 = self.compose_bf16(sign_result, 0xFF, 0)
                    else:
                        result_bf16 = self.compose_bf16(sign_result, exp_result, mant_result)
            
            # 将结果添加到输出
            self.outputs.append(result_bf16)
        
        # 阶段3 -> 阶段4: 加减法阶段
        self.stage4_valid = self.stage3_valid
        self.stage4_special_case = self.stage3_special_case
        self.stage4_result = self.stage3_result
        
        if self.stage3_valid and not self.stage3_special_case:
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
        self.stage3_special_case = self.stage2_special_case
        self.stage3_result = self.stage2_result
        
        if self.stage2_valid and not self.stage2_special_case:
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
                # 限制移位数量，避免不必要的大规模移位
                if shift > 24:  # 如果差异过大，直接舍弃小数
                    mant_b = 0
                else:
                    mant_b >>= shift
                exp_result = exp_a
            elif exp_b > exp_a:
                shift = exp_b - exp_a
                if shift > 24:  # 如果差异过大，直接舍弃小数
                    mant_a = 0
                else:
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
        self.stage2_special_case = self.stage1_special_case
        self.stage2_result = self.stage1_result
        
        if self.stage1_valid and not self.stage1_special_case:
            # 分解两个BF16数值
            sign_a, exp_a, mant_a = self.decompose_bf16(self.stage1_bf16_a)
            sign_b, exp_b, mant_b = self.decompose_bf16(self.stage1_bf16_b)
            
            # 处理非规格化数
            if exp_a == 0:
                if mant_a != 0:  # 非规格化数
                    # 找到尾数中的最高位
                    leading_bit = 0
                    temp_mant = mant_a
                    while temp_mant and not (temp_mant & 0x80):  # 0x40 = 0100 0000, 第6位
                        temp_mant <<= 1
                        leading_bit += 1
                    exp_a = 1 - leading_bit
                    mant_a <<= leading_bit  # 左移使隐含位为1
                else:
                    exp_a = 1  # 零的情况，但我们已经在特殊情况中处理了零
            else:
                mant_a |= 0x80  # 添加隐含的1
    
            
            if exp_b == 0:
                if mant_b != 0:  # 非规格化数
                    # 找到尾数中的最高位
                    leading_bit = 0
                    temp_mant = mant_b
                    while temp_mant and not (temp_mant & 0x80):  # 0x40 = 0100 0000, 第6位 
                        temp_mant <<= 1
                        leading_bit += 1
                    exp_b = 1 - leading_bit
                    mant_b <<= leading_bit  # 左移使隐含位为1
                else:
                    exp_b = 1  # 零的情况，但我们已经在特殊情况中处理了零
            else:
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
            
            # 检查特殊情况
            is_special, result = self.check_special_cases(bf16_a, bf16_b)
            self.stage1_special_case = is_special
            self.stage1_result = result
        else:
            self.stage1_valid = False
            self.stage1_bf16_a = 0
            self.stage1_bf16_b = 0
            self.stage1_special_case = False
            self.stage1_result = 0
        
        # 返回当前周期的状态
        return {
            "cycle": self.cycle_count,
            "valid_output": self.stage5_valid,
            "pipeline_state": self.get_pipeline_state()
        }
    
    def get_pipeline_state(self):
        """返回流水线的当前状态，用于可视化"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "bf16_a": hex(self.stage1_bf16_a) if self.stage1_valid else "invalid",
                "bf16_b": hex(self.stage1_bf16_b) if self.stage1_valid else "invalid",
                "special": self.stage1_special_case
            },
            "stage2": {
                "valid": self.stage2_valid,
                "special": self.stage2_special_case,
                "sign_a": self.stage2_sign_a if self.stage2_valid else "invalid",
                "exp_a": self.stage2_exp_a if self.stage2_valid else "invalid",
                "mant_a": hex(self.stage2_mant_a) if self.stage2_valid else "invalid",
                "sign_b": self.stage2_sign_b if self.stage2_valid else "invalid",
                "exp_b": self.stage2_exp_b if self.stage2_valid else "invalid",
                "mant_b": hex(self.stage2_mant_b) if self.stage2_valid else "invalid"
            },
            "stage3": {
                "valid": self.stage3_valid,
                "special": self.stage3_special_case,
                "aligned_exp": self.stage3_exp_result if self.stage3_valid else "invalid",
                "aligned_mant_a": hex(self.stage3_mant_a) if self.stage3_valid else "invalid",
                "aligned_mant_b": hex(self.stage3_mant_b) if self.stage3_valid else "invalid"
            },
            "stage4": {
                "valid": self.stage4_valid,
                "special": self.stage4_special_case,
                "sign": self.stage4_sign_result if self.stage4_valid else "invalid",
                "exp": self.stage4_exp_result if self.stage4_valid else "invalid",
                "mant": hex(self.stage4_mant_result) if self.stage4_valid else "invalid"
            }
        }
    
    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - BF16 A: {state['stage1']['bf16_a']}, BF16 B: {state['stage1']['bf16_b']} {'(Special case)' if state['stage1']['special'] else ''}")
        print(f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - {'Special case' if state['stage2']['special'] else 'Prepared operands'}")
        print(f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - "
            f"{'Special case' if state['stage3']['special'] else 'Aligned operands, Exponent: ' + str(state['stage3']['aligned_exp'])}")
        print(f"  Stage 4: {'Valid' if state['stage4']['valid'] else 'Invalid'} - "
            f"{'Special case' if state['stage4']['special'] else 'Sign: ' + str(state['stage4']['sign']) + ', Exp: ' + str(state['stage4']['exp']) + ', Mant: ' + str(state['stage4']['mant'])}")
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
        
        #print("cycle长度")

        # 运行流水线
        for i in range(len(extended_inputs)):
            bf16_a, bf16_b, valid = extended_inputs[i]
            result = self.clock_cycle(bf16_a, bf16_b, valid)
            results.append(result)
            
            if print_states:
                self.print_pipeline_state()
        
        return results
    
class BF16MultiplyPipeline:
    def __init__(self):
        """初始化流水线寄存器和状态"""
        # 阶段1: 获取输入并处理特殊情况
        self.stage1_valid = False
        self.stage1_bf16_a = 0
        self.stage1_bf16_b = 0
        self.stage1_special_case = False
        self.stage1_result = 0
        
        # 阶段2: 分解提取和指数处理阶段
        self.stage2_valid = False
        self.stage2_special_case = False
        self.stage2_result = 0
        self.stage2_sign_result = 0
        self.stage2_exp_a = 0
        self.stage2_exp_b = 0
        self.stage2_mant_a = 0
        self.stage2_mant_b = 0
        
        # 阶段3: 尾数相乘阶段
        self.stage3_valid = False
        self.stage3_special_case = False
        self.stage3_result = 0
        self.stage3_sign_result = 0
        self.stage3_exp_result = 0
        self.stage3_mant_result = 0
        
        # 阶段4: 规范化和舍入阶段
        self.stage4_valid = False
        self.stage4_special_case = False
        self.stage4_result = 0
        self.stage4_sign_result = 0
        self.stage4_exp_result = 0
        self.stage4_mant_result = 0
        self.stage4_msb_pos = 0
        
        self.stage5_valid = False
        
        # 常量定义
        self.POS_INF = 0x7F80  # 正无穷大：0 11111111 0000000
        self.NEG_INF = 0xFF80  # 负无穷大：1 11111111 0000000
        self.NAN = 0x7FC0      # NaN：0 11111111 1000000
        
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
    
    def check_special_cases(self, bf16_a, bf16_b):
        """检查并处理特殊情况"""
        sign_a, exp_a, mant_a = self.decompose_bf16(bf16_a)
        sign_b, exp_b, mant_b = self.decompose_bf16(bf16_b)
        
        # 符号位：相乘后的符号位是两个符号位的异或
        sign_result = sign_a ^ sign_b
        
        # 1. 处理 NaN
        if ((exp_a == 0xFF and mant_a != 0) or (exp_b == 0xFF and mant_b != 0)):
            return True, self.NAN
        
        # 2. 处理 无穷 × 0 = NaN 的情况
        if ((exp_a == 0xFF and exp_b == 0 and mant_b == 0) or 
            (exp_b == 0xFF and exp_a == 0 and mant_a == 0)):
            return True, self.NAN
        
        # 3. 处理零乘以任何数 = 零（正确保留符号）
        if (exp_a == 0 and mant_a == 0) or (exp_b == 0 and mant_b == 0):
            return True, self.compose_bf16(sign_result, 0, 0)
        
        # 4. 处理无穷大乘以非零数 = 无穷大（带符号）
        if exp_a == 0xFF or exp_b == 0xFF:
            return True, self.compose_bf16(sign_result, 0xFF, 0)
            
        # 不是特殊情况
        return False, 0
    
    def clock_cycle(self, bf16_a=None, bf16_b=None, valid=False):
        """
        模拟一个时钟周期，推进流水线
        
        Args:
            bf16_a, bf16_b: 输入的两个BF16值
            valid: 输入是否有效
        """
        # 更新周期计数
        self.cycle_count += 1
        
        # 阶段5: 输出阶段 - 处理阶段4的规范化和舍入结果
        self.stage5_valid = self.stage4_valid
        
        if self.stage4_valid:
            result_bf16 = 0
            
            if self.stage4_special_case:
                # 直接输出特殊情况结果
                result_bf16 = self.stage4_result
            else:
                # 获取阶段4的计算结果
                sign_result = self.stage4_sign_result
                exp_result = self.stage4_exp_result
                mant_result = self.stage4_mant_result
                
                # 处理结果为0的情况
                if mant_result == 0:
                    result_bf16 = self.compose_bf16(sign_result, 0, 0)
                else:
                    # 处理小数点
                    if mant_result & 0x8000: 
                        exp_result += 1   #右移小数点
                        round_bits = mant_result & 0xFF
                        half_point = 0x80
                        mant_result >>= 8
                    else:
                        round_bits = mant_result & 0x7F
                        half_point = 0x70
                        mant_result >>= 7
                    
                    if round_bits > half_point or (round_bits == half_point and (mant_result & 0x1)):
                        mant_result += 1
                        if mant_result & 0x100:
                            mant_result >>= 1
                            exp_result += 1

                    # 去掉隐含的最高位，获取最终的7位尾数
                    mant_result &= 0x7F
                    
                    # 处理特殊情况：下溢和上溢
                    if exp_result <= 0:  # 下溢，返回零或非规格化数
                        if exp_result < -6:  # 太小，直接返回零
                            result_bf16 = self.compose_bf16(sign_result, 0, 0)
                        else:  # 尝试以非规格化形式表示
                            # 注意: BF16尾数为7位，隐含位需手动处理
                            denorm_mant = 0x80 | mant_result  # 添加隐含位1
                            shift_amount = 1 - exp_result
                            
                            # 保留移位前的低位用于舍入
                            round_bit = (denorm_mant >> (shift_amount - 1)) & 1
                            sticky_bits = (denorm_mant & ((1 << (shift_amount - 1)) - 1)) != 0
                            
                            denorm_mant >>= shift_amount
                            
                            # 应用舍入
                            if round_bit and (sticky_bits or (denorm_mant & 1)):
                                denorm_mant += 1
                                
                            result_bf16 = self.compose_bf16(sign_result, 0, denorm_mant & 0x7F)
                    
                    elif exp_result >= 0xFF:  # 上溢，返回无穷大
                        result_bf16 = self.compose_bf16(sign_result, 0xFF, 0)
                    
                    else:  # 正常情况
                        result_bf16 = self.compose_bf16(sign_result, exp_result, mant_result)
            
            # 将结果添加到输出
            self.outputs.append(result_bf16)
        
        # 阶段4: 规范化和舍入准备阶段 - 处理阶段3的乘法结果
        self.stage4_valid = self.stage3_valid
        self.stage4_special_case = self.stage3_special_case
        self.stage4_result = self.stage3_result
        self.stage4_sign_result = self.stage3_sign_result
        self.stage4_exp_result = self.stage3_exp_result
        
        if self.stage3_valid and not self.stage3_special_case:
            mant_result = self.stage3_mant_result
            
            if mant_result == 0:
                self.stage4_mant_result = 0
            else:
                self.stage4_mant_result = mant_result
                
        # 阶段3: 尾数乘法阶段 - 处理阶段2的处理过的尾数和指数
        self.stage3_valid = self.stage2_valid
        self.stage3_special_case = self.stage2_special_case
        self.stage3_result = self.stage2_result
        self.stage3_sign_result = self.stage2_sign_result
        
        if self.stage2_valid and not self.stage2_special_case:
            # 指数相加（在乘法中，指数是相加的），并减去偏移值（127）
            self.stage3_exp_result = self.stage2_exp_a + self.stage2_exp_b - 127
            
            # 尾数相乘（带隐含的最高位）
            self.stage3_mant_result = self.stage2_mant_a * self.stage2_mant_b
        
        # 阶段2: 分解和准备阶段 - 处理阶段1的输入值
        self.stage2_valid = self.stage1_valid
        self.stage2_special_case = self.stage1_special_case
        self.stage2_result = self.stage1_result
        
        if self.stage1_valid and not self.stage1_special_case:
            # 分解两个BF16数值
            sign_a, exp_a, mant_a = self.decompose_bf16(self.stage1_bf16_a)
            sign_b, exp_b, mant_b = self.decompose_bf16(self.stage1_bf16_b)
            
            # 计算结果符号位
            self.stage2_sign_result = sign_a ^ sign_b
            
            # 处理非规格化数
            if exp_a == 0:
                if mant_a != 0:  # 非规格化数
                    # 找到尾数中的最高位
                    leading_bit = 0
                    temp_mant = mant_a
                    while temp_mant and not (temp_mant & 0x80):  # 0x40 = 0100 0000, 第6位
                        temp_mant <<= 1
                        leading_bit += 1
                    exp_a = 1 - leading_bit
                    mant_a <<= leading_bit  # 左移使隐含位为1
                else:
                    exp_a = 1  # 零的情况，但我们已经在特殊情况中处理了零
            else:
                mant_a |= 0x80  # 添加隐含的1
    
            
            if exp_b == 0:
                if mant_b != 0:  # 非规格化数
                    # 找到尾数中的最高位
                    leading_bit = 0
                    temp_mant = mant_b
                    while temp_mant and not (temp_mant & 0x80):  # 0x40 = 0100 0000, 第6位 
                        temp_mant <<= 1
                        leading_bit += 1
                    exp_b = 1 - leading_bit
                    mant_b <<= leading_bit  # 左移使隐含位为1
                else:
                    exp_b = 1  # 零的情况，但我们已经在特殊情况中处理了零
            else:
                mant_b |= 0x80  # 添加隐含的1
            
            # 传递到下一阶段
            self.stage2_exp_a = exp_a
            self.stage2_exp_b = exp_b
            self.stage2_mant_a = mant_a
            self.stage2_mant_b = mant_b
        
        # 阶段1: 获取输入和特殊情况处理
        if valid and bf16_a is not None and bf16_b is not None:
            self.stage1_bf16_a = bf16_a
            self.stage1_bf16_b = bf16_b
            self.stage1_valid = True
            
            # 检查特殊情况
            is_special, result = self.check_special_cases(bf16_a, bf16_b)
            self.stage1_special_case = is_special
            self.stage1_result = result
        else:
            self.stage1_valid = False
            self.stage1_bf16_a = 0
            self.stage1_bf16_b = 0
            self.stage1_special_case = False
            self.stage1_result = 0
        
        # 返回当前周期的状态
        return {
            "cycle": self.cycle_count,
            "valid_output": self.stage5_valid,
            "pipeline_state": self.get_pipeline_state()
        }
    
    def get_pipeline_state(self):
        """返回流水线的当前状态，用于可视化"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "bf16_a": hex(self.stage1_bf16_a) if self.stage1_valid else "invalid",
                "bf16_b": hex(self.stage1_bf16_b) if self.stage1_valid else "invalid",
                "special": self.stage1_special_case
            },
            "stage2": {
                "valid": self.stage2_valid,
                "special": self.stage2_special_case,
                "sign_result": self.stage2_sign_result if self.stage2_valid else "invalid",
                "exp_a": self.stage2_exp_a if self.stage2_valid else "invalid",
                "exp_b": self.stage2_exp_b if self.stage2_valid else "invalid",
                "mant_a": hex(self.stage2_mant_a) if self.stage2_valid else "invalid",
                "mant_b": hex(self.stage2_mant_b) if self.stage2_valid else "invalid"
            },
            "stage3": {
                "valid": self.stage3_valid,
                "special": self.stage3_special_case,
                "sign_result": self.stage3_sign_result if self.stage3_valid else "invalid",
                "exp_result": self.stage3_exp_result if self.stage3_valid else "invalid",
                "mant_result": hex(self.stage3_mant_result) if self.stage3_valid else "invalid"
            },
            "stage4": {
                "valid": self.stage4_valid,
                "special": self.stage4_special_case,
                "sign_result": self.stage4_sign_result if self.stage4_valid else "invalid",
                "exp_result": self.stage4_exp_result if self.stage4_valid else "invalid",
                "mant_result": hex(self.stage4_mant_result) if self.stage4_valid else "invalid",
                "msb_pos": self.stage4_msb_pos if self.stage4_valid else "invalid"
            }
        }
    
    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - "
              f"BF16 A: {state['stage1']['bf16_a']}, BF16 B: {state['stage1']['bf16_b']} "
              f"{'(Special case)' if state['stage1']['special'] else ''}")
        print(f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - "
              f"{'Special case' if state['stage2']['special'] else 'Prepared operands'}")
        print(f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - "
              f"{'Special case' if state['stage3']['special'] else 'Multiplication result, Exp: ' + str(state['stage3']['exp_result'])}")
        print(f"  Stage 4: {'Valid' if state['stage4']['valid'] else 'Invalid'} - "
              f"{'Special case' if state['stage4']['special'] else 'Normalized result'}")
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

def convert_through_pipeline(value):
        """通过完整流水线模拟转换FP32到BF16"""
        temp_pipeline = FP32toBF16Pipeline()
        temp_pipeline.run_simulation([(value, True)], print_states=False)
        return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0

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
    for _ in range(10):
        a = random.uniform(-10, 10)
        test_data.append((a, True))
    
    
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
            
def bf16_add(bf16_a, bf16_b):
    sim = BF16AddPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], False)
    return sim.outputs[0]

def bf16_mul(bf16_a, bf16_b):
    sim = BF16MultiplyPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], False)
    return sim.outputs[0]

def test_bf16add():
    # 创建流水线实例
    pipeline = BF16AddPipeline()
    
    import torch
    import numpy as np
    
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
    
    def test_cases_bf16_to_float_add_to_bf16_to_float(test_cases):
        results = []
        for case in test_cases:
            results.append(bf16_to_float(case[0]) + bf16_to_float(case[1]))
        for i in range(len(results)):
            results[i] = convert_through_pipeline(results[i])
            results[i] = bf16_to_float(results[i])
        return results
    
    # 准备常规测试用例
    regular_cases = [
        (1.5, 2.25, True),        # 简单加法: 1.5 + 2.25 = 3.75
        (3.14159, -1.5, True),    # 异号加法: 3.14159 + (-1.5) ≈ 1.64159
        (-3.0, -2.0, True),       # 负数加法: (-3.0) + (-2.0) = -5.0
        (100.0, 0.001, True),     # 量级差异大: 100.0 + 0.001 ≈ 100.001
        (0.0, 0.0, True),         # 零加零: 0.0 + 0.0 = 0.0
        (1e4, 1e4, True),         # 大数: 10000 + 10000 = 20000
        (0.1, 0.2, True),         # 小数: 0.1 + 0.2 = 0.3 (注意精度问题)
        (1.0, -1.0, True)         # 正好抵消: 1.0 + (-1.0) = 0.0
    ]

    for _ in range(10):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        regular_cases.append((a, b, True))

    subnormal_cases = [
        (1e-38, 1e-38, True),      # 极小数 + 极小数
        (1e-38, -1e-38, True),     # 极小数 + 负极小数
        (1e-38, 0.0, True),        # 极小数 + 0
        (1e-38, 1e-20, True),      # 极小数 + 更大极小数
        (1e-38, -1e-20, True),     # 极小数 + 负极小数
        (1e-38, 1.0, True),        # 极小数 + 1
        (1e-38, -1.0, True),       # 极小数 + -1
        (1e-45, 1e-45, True),      # 更极小数 + 更极小数
        (1e-45, 0.0, True),        # 更极小数 + 0
        (1e-45, 1e-38, True),      # 更极小数 + 极小数
    ]
    
    # 准备特殊情况测试用例
    # 定义特殊值
    inf_pos = float('inf')  # 正无穷大
    inf_neg = float('-inf') # 负无穷大
    nan = float('nan')      # NaN
    
    special_cases = [
        (0.0, -0.0, True),         # 正零 + 负零 = 正零
        (-0.0, -0.0, True),        # 负零 + 负零 = 负零
        (inf_pos, 1.0, True),      # 正无穷大 + 任意数 = 正无穷大
        (inf_neg, 1.0, True),      # 负无穷大 + 任意数 = 负无穷大
        (inf_pos, inf_pos, True),  # 正无穷大 + 正无穷大 = 正无穷大
        (inf_pos, inf_neg, True),  # 正无穷大 + 负无穷大 = NaN
        (nan, 1.0, True),          # NaN + 任意数 = NaN
        (nan, nan, True)           # NaN + NaN = NaN
    ]
    
    # 合并所有测试用例
    test_cases = regular_cases + subnormal_cases + special_cases + [(0.0, 1.0, False)]
    
    

    # 转换测试用例为BF16格式
    bf16_test_cases = [(convert_through_pipeline(a), 
                        convert_through_pipeline(b), 
                        valid) for a, b, valid in test_cases]
    
    sim_cases = test_cases_bf16_to_float_add_to_bf16_to_float(bf16_test_cases)
    
    print("Starting BF16 Addition Pipeline Simulation")
    print("=" * 80)
    
    # 运行模拟
    results = pipeline.run_simulation(bf16_test_cases, print_states=True)
    
    print("=" * 80)
    print("Final Results:")
    print("{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15} {:<15}".format(
        "Test", "Input A", "Input B", "Expected Sum", "Custom BF16 Result", "PyTorch Result", "Error vs PyTorch", "Sim pytorch add"
    ))
    print("-" * 80)
    
    for i, output in enumerate(pipeline.outputs):
        if i < len(test_cases) and test_cases[i][2]:  # 只检查有效输入
            a, b, _ = test_cases[i]
            c = sim_cases[i]
            # 自定义实现的结果
            custom_result = bf16_to_float(output)
            
            # 获取PyTorch的BF16计算结果
            try:
                torch_a = torch.tensor(a, dtype=torch.float32).bfloat16()
                torch_b = torch.tensor(b, dtype=torch.float32).bfloat16()
                torch_sum = (torch_a + torch_b).float().item()
                
                # 计算与PyTorch结果的差异
                if np.isnan(custom_result) and np.isnan(torch_sum):
                    error = "N/A (Both NaN)"
                elif np.isinf(custom_result) and np.isinf(torch_sum) and np.sign(custom_result) == np.sign(torch_sum):
                    error = "N/A (Both Inf)"
                else:
                    error = abs(custom_result - torch_sum)
            except:
                torch_sum = "N/A"
                error = "N/A"
            
            # 预期结果
            if a == 0.0 and b == -0.0 or a == -0.0 and b == 0.0:
                expected = 0.0  # 预期输出正零
            else:
                expected = a + b
            
            print("{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15} {:<15}".format(
                i,
                f"{a:.6g}",
                f"{b:.6g}",
                f"{expected:.6g}" if not np.isnan(expected) else "NaN",
                f"{custom_result:.6g} ({hex(output)})" if not np.isnan(custom_result) else f"NaN ({hex(output)})",
                f"{torch_sum:.6g}" if torch_sum != "N/A" else torch_sum,
                f"{error:.6g}" if error != "N/A (Both NaN)" and error != "N/A (Both Inf)" and error != "N/A" else error,
                f"{c:.6g}"
            ))
    
    print("\nDetailed Analysis of Special Cases:")
    special_start = len(regular_cases)
    special_end = len(test_cases) - 1  # 除去无效输入
    
    for i in range(special_start, special_end):
        if i < len(pipeline.outputs):
            a, b, _ = test_cases[i]
            output = pipeline.outputs[i]
            custom_result = bf16_to_float(output)
            
            print(f"\nTest {i}: {a} + {b}")
            print(f"  BF16 Result: {hex(output)} -> {custom_result}")
            
            # 特殊情况的分析
            if np.isnan(a) or np.isnan(b):
                print("  Analysis: Input contains NaN, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) and np.isinf(b) and np.sign(a) != np.sign(b):
                print("  Analysis: Infinity - Infinity, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) or np.isinf(b):
                expected_sign = np.sign(a) if np.isinf(a) else np.sign(b)
                print(f"  Analysis: One input is Infinity, result should be {expected_sign}Infinity")
                print(f"  Correct: {np.isinf(custom_result) and np.sign(custom_result) == expected_sign}")
            elif a == 0.0 and b == 0.0:
                if (a == 0.0 and b == -0.0) or (a == -0.0 and b == 0.0):
                    print("  Analysis: Positive zero + Negative zero, result should be +0.0")
                    print(f"  Correct: {custom_result == 0.0 and not np.signbit(custom_result)}")
                else:
                    expected_sign = np.signbit(a)
                    print(f"  Analysis: Zero + Zero with same sign, result should maintain sign")
                    print(f"  Correct: {custom_result == 0.0 and np.signbit(custom_result) == expected_sign}")

def test_bf16multiply():
    # 创建流水线实例
    pipeline = BF16MultiplyPipeline()
    
    import torch
    import numpy as np
    
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
        
            
    
    # 准备常规测试用例
    regular_cases = [
        (2.0, 3.0, True),        # 简单乘法: 2.0 * 3.0 = 6.0
        (3.124, 2.249, True),
        (1.5, 2.25, True),       # 小数乘法: 1.5 * 2.25 = 3.375
        (3.14159, -1.5, True),   # 异号乘法: 3.14159 * (-1.5) ≈ -4.71
        (-3.0, -2.0, True),      # 负数乘法: (-3.0) * (-2.0) = 6.0
        (100.0, 0.01, True),     # 量级差异大: 100.0 * 0.01 = 1.0
        (0.0, 5.0, True),        # 零乘任何数: 0.0 * 5.0 = 0.0
        (1e-2, 1e2, True),       # 指数抵消: 0.01 * 100 = 1.0
        (0.1, 0.1, True)         # 小数平方: 0.1 * 0.1 = 0.01
    ]

    for _ in range(10):
        a = random.uniform(-10, 10)
        b = random.uniform(-10, 10)
        regular_cases.append((a, b, True))

    # 新增：下溢/非规格化测试用例
    subnormal_cases = [
        (1e-38, 1.0, True),      # 极小数 * 1
        (1e-38, 2.0, True),      # 极小数 * 2
        (1e-38, 0.5, True),      # 极小数 * 0.5
        (1e-38, 1e-2, True),     # 极小数 * 极小数
        (1e-38, -1.0, True),     # 极小数 * -1
        (1e-38, 1e38, True),     # 极小数 * 极大数
        (1e-38, 1e-38, True),    # 极小数 * 极小数
        (1e-20, 1e-20, True),    # 仍然很小
    ]
    
    # 准备特殊情况测试用例
    # 定义特殊值
    inf_pos = float('inf')  # 正无穷大
    inf_neg = float('-inf') # 负无穷大
    nan = float('nan')      # NaN
    
    special_cases = [
        (0.0, 0.0, True),         # 零乘零: 0.0 * 0.0 = 0.0
        (inf_pos, 0.0, True),     # 无穷大 * 零 = NaN
        (inf_pos, 2.0, True),     # 正无穷大 * 正数 = 正无穷大
        (inf_pos, -2.0, True),    # 正无穷大 * 负数 = 负无穷大
        (inf_neg, inf_pos, True), # 负无穷大 * 正无穷大 = 负无穷大
        (inf_pos, inf_pos, True), # 正无穷大 * 正无穷大 = 正无穷大
        (nan, 1.0, True),         # NaN * 任意数 = NaN
        (nan, nan, True)          # NaN * NaN = NaN
    ]
    
    # 合并所有测试用例
    test_cases = regular_cases + subnormal_cases + special_cases + [(0.0, 1.0, False)]
    
    

    # 转换测试用例为BF16格式
    bf16_test_cases = [(convert_through_pipeline(a), 
                        convert_through_pipeline(b), 
                        valid) for a, b, valid in test_cases]

    
    
    print("Starting BF16 Multiplication Pipeline Simulation")
    print("=" * 80)
    
    # 运行模拟
    results = pipeline.run_simulation(bf16_test_cases, print_states=True)
    
    print("=" * 80)
    print("Final Results:")
    print("{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
        "Test", "Input A", "Input B", "Expected Product", "Custom BF16 Result", "PyTorch Result", "Error vs PyTorch"
    ))
    print("-" * 80)
    
    for i, output in enumerate(pipeline.outputs):
        if i < len(test_cases) and test_cases[i][2]:  # 只检查有效输入
            a, b, _ = test_cases[i]
            
            # 自定义实现的结果
            custom_result = bf16_to_float(output)
            
            # 获取PyTorch的BF16计算结果
            try:
                torch_a = torch.tensor(a, dtype=torch.float32).bfloat16()
                torch_b = torch.tensor(b, dtype=torch.float32).bfloat16()
                torch_product = (torch_a * torch_b).float().item()
                
                # 计算与PyTorch结果的差异
                if np.isnan(custom_result) and np.isnan(torch_product):
                    error = "N/A (Both NaN)"
                elif np.isinf(custom_result) and np.isinf(torch_product) and np.sign(custom_result) == np.sign(torch_product):
                    error = "N/A (Both Inf)"
                else:
                    # 对于非常小的结果，使用相对误差
                    if abs(torch_product) > 1e-10:
                        error = abs((custom_result - torch_product) / torch_product)
                    else:
                        error = abs(custom_result - torch_product)
            except:
                torch_product = "N/A"
                error = "N/A"
            
            # 预期结果
            expected = a * b
            
            print("{:<5} {:<15} {:<15} {:<15} {:<20} {:<15} {:<15}".format(
                i,
                f"{a:.6g}",
                f"{b:.6g}",
                f"{expected:.6g}" if not np.isnan(expected) else "NaN",
                f"{custom_result:.6g} ({hex(output)})" if not np.isnan(custom_result) else f"NaN ({hex(output)})",
                f"{torch_product:.6g}" if torch_product != "N/A" else torch_product,
                f"{error:.6g}" if error != "N/A (Both NaN)" and error != "N/A (Both Inf)" and error != "N/A" else error
            ))
    
    print("\nDetailed Analysis of Special Cases:")
    special_start = len(regular_cases)
    special_end = len(test_cases) - 1  # 除去无效输入
    
    for i in range(special_start, special_end):
        if i < len(pipeline.outputs):
            a, b, _ = test_cases[i]
            output = pipeline.outputs[i]
            custom_result = bf16_to_float(output)
            
            print(f"\nTest {i}: {a} * {b}")
            print(f"  BF16 Result: {hex(output)} -> {custom_result}")
            
            # 特殊情况的分析
            if np.isnan(a) or np.isnan(b):
                print("  Analysis: Input contains NaN, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif (np.isinf(a) and b == 0.0) or (np.isinf(b) and a == 0.0):
                print("  Analysis: Infinity * Zero, result should be NaN")
                print(f"  Correct: {np.isnan(custom_result)}")
            elif np.isinf(a) or np.isinf(b):
                # 计算符号
                expected_sign = np.sign(a) * np.sign(b)
                print(f"  Analysis: One input is Infinity, result should be {expected_sign}Infinity")
                print(f"  Correct: {np.isinf(custom_result) and np.sign(custom_result) == expected_sign}")
            elif a == 0.0 or b == 0.0:
                print("  Analysis: Multiplication by zero, result should be zero")
                print(f"  Correct: {custom_result == 0.0}")
                
                # 检查零的符号（虽然通常乘法中零的符号并不那么重要）
                expected_sign = np.sign(a) * np.sign(b)
                if expected_sign == -0.0:
                    print(f"  Sign check: Result should be -0.0")
                    print(f"  Correct sign: {np.signbit(custom_result)}")
                else:
                    print(f"  Sign check: Result should be +0.0")
                    print(f"  Correct sign: {not np.signbit(custom_result)}")



# 测试模拟器
if __name__ == "__main__":
    #test_fp32_to_bf16()
    test_bf16add()
    #test_bf16multiply()
    #t = 0.0
    #return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
    #t_bits = struct.unpack('>I', struct.pack('>f', t))[0]
    #t_bits_float = struct.unpack('>f', struct.pack('>I', t_bits | 0X1))
    #print(t_bits_float)