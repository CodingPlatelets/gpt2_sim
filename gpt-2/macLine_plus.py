import numpy as np
import struct
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

def bf16_to_float_block(bf16_block):
    # 左移16位填充为32位表示
    float_block = []
    for bf16 in bf16_block:
        fp32_bits = bf16 << 16
        float_block.append(struct.unpack('>f', struct.pack('>I', fp32_bits))[0])
    # 转换为浮点数
    return float_block

class MACUnit:
    def __init__(self):
        self.multiply_pipline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.output = 0
        #self.valid = False
        
    def clock_cycle(self):
        result = self.multiply_pipline.clock_cycle(self.input1, self.input2, True)
        if result["valid_output"]:
            self.output = self.multiply_pipline.outputs.pop(0)
            


class MatmulPipeline:
    def __init__(self, mac_width=4):

        self.stage1_valid = False
        self.stage1_a_block_in = []
        self.stage1_b_block_in = []
        self.stage1_a_block_out = []
        self.stage1_b_block_out = []
        
        self.stage2_valid = False
        self.stage2_a_block_in = []
        self.stage2_b_block_in = []
        self.macline = [MACUnit() for _ in range(mac_width)]
                
        