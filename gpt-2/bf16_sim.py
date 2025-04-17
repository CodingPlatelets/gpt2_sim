import struct

class FP32toBF16Pipeline:
    def __init__(self):
        """初始化流水线寄存器和状态"""
        # 流水线寄存器
        self.stage1_fp32 = 0
        self.stage1_valid = False
        
        self.stage2_fp32 = 0
        self.stage2_high_bits = 0
        self.stage2_low_bits = 0
        self.stage2_valid = False
        
        self.stage3_high_bits = 0
        self.stage3_low_bits = 0
        self.stage3_valid = False
        
        self.stage4_bf16 = 0
        self.stage4_valid = False
        
        # 输出缓冲区
        self.outputs = []
        self.cycle_count = 0
    
    def reset(self):
        """重置流水线状态"""
        self.__init__()
    
    def clock_cycle(self, new_fp32=None, new_valid=False):
        """
        模拟一个时钟周期，推进流水线
        
        Args:
            new_fp32: 新的FP32输入值（可选）
            new_valid: 输入是否有效
        """
        # 更新周期计数
        self.cycle_count += 1
        
        # 阶段4（舍入阶段）- 从阶段3获取数据
        if self.stage3_valid:
            # 临近偶数位舍入
            if (self.stage3_low_bits > 0x8000 or 
                (self.stage3_low_bits == 0x8000 and self.stage3_high_bits & 1)):
                self.stage4_bf16 = (self.stage3_high_bits + 1) & 0xFFFF
            else:
                self.stage4_bf16 = self.stage3_high_bits
                
            # 如果是有效输出，添加到输出缓冲区
            if self.stage3_valid:
                self.outputs.append(self.stage4_bf16)
        
        self.stage4_valid = self.stage3_valid
        
        # 阶段3（舍入准备阶段）- 从阶段2获取数据
        self.stage3_high_bits = self.stage2_high_bits
        self.stage3_low_bits = self.stage2_low_bits
        self.stage3_valid = self.stage2_valid
        
        # 阶段2（提取阶段）- 从阶段1获取数据
        self.stage2_fp32 = self.stage1_fp32
        self.stage2_high_bits = (self.stage1_fp32 >> 16) & 0xFFFF
        self.stage2_low_bits = self.stage1_fp32 & 0xFFFF
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
            "valid_out": self.stage4_valid,
            "bf16_out": self.stage4_bf16 if self.stage4_valid else None,
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
                "high_bits": hex(self.stage3_high_bits) if self.stage3_valid else "invalid",
                "low_bits": hex(self.stage3_low_bits) if self.stage3_valid else "invalid",
                "valid": self.stage3_valid
            },
            "stage4": {
                "bf16": hex(self.stage4_bf16) if self.stage4_valid else "invalid",
                "valid": self.stage4_valid
            }
        }

    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - FP32: {state['stage1']['fp32']}")
        print(f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - High: {state['stage2']['high_bits']}, Low: {state['stage2']['low_bits']}")
        print(f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - High: {state['stage3']['high_bits']}, Low: {state['stage3']['low_bits']}")
        print(f"  Stage 4: {'Valid' if state['stage4']['valid'] else 'Invalid'} - BF16: {state['stage4']['bf16']}")
        if state['stage4']['valid']:
            print(f"  Output: BF16 = {state['stage4']['bf16']}")
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
        extended_inputs = list(inputs) + [(0, False)] * 4  # 加4个周期确保流水线清空
        
        # 运行流水线
        for i in range(len(extended_inputs)):
            fp32_value, valid = extended_inputs[i]
            result = self.clock_cycle(fp32_value, valid)
            results.append(result)
            
            if print_states:
                self.print_pipeline_state()
        
        return results


# 测试模拟器
if __name__ == "__main__":
    # 创建流水线实例
    pipeline = FP32toBF16Pipeline()
    
    # 准备测试数据：[(FP32值, 是否有效), ...]
    test_data = [
        (3.14159, True),  # Pi，十六进制表示为 0x40490FDB
        (1.5, True),      # 1.5，十六进制表示为 0x3FC00000
        (-2.25, True),    # -2.25，十六进制表示为 0xC0100000
        (0.0, False),     # 无效输入
        (65504.0, True),  # BF16能表示的最大正数
        (1e-20, True),    # 非常小的数
    ]
    
    print("Starting FP32 to BF16 Pipeline Simulation")
    print("=" * 50)
    
    # 运行模拟
    results = pipeline.run_simulation(test_data)
    
    print("=" * 50)
    print("Final Outputs:")
    for i, value in enumerate(pipeline.outputs):
        original_fp32 = test_data[i][0] if i < len(test_data) and test_data[i][1] else "N/A"
        # 将BF16转回FP32表示进行验证
        bf16_as_fp32 = struct.unpack('>f', struct.pack('>I', value << 16))[0]
        print(f"Input FP32: {original_fp32} -> Output BF16: {hex(value)} -> As FP32: {bf16_as_fp32}")
    
    print("\nVerification:")
    for i, (fp32, valid) in enumerate(test_data):
        if valid and i < len(pipeline.outputs):
            # 验证模拟结果与直接计算结果是否一致
            direct_result = pipeline.fp32_to_bf16(fp32)
            pipeline_result = pipeline.outputs[i]
            print(f"Input: {fp32}")
            print(f"  Direct conversion: {hex(direct_result)}")
            print(f"  Pipeline output:   {hex(pipeline_result)}")
            print(f"  Match: {direct_result == pipeline_result}")