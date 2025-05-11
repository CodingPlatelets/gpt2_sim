import struct
from collections import defaultdict


class Float32:
    @staticmethod
    def to_bits(f):
        return struct.unpack('!I', struct.pack('!f', f))[0]

    @staticmethod
    def from_bits(i):
        return struct.unpack('!f', struct.pack('!I', i))[0]


class PipelineRegister:
    def __init__(self):
        self.valid = False
        self.data = {}
        self.tag = None  # 用于跟踪不同向量元素的流水线


class MultiStageMAC:
    def __init__(self, vec_len=4):
        # 8级流水线寄存器 (3乘法 + 2对齐 + 3加法)
        self.registers = [PipelineRegister() for _ in range(8)]
        self.accumulator = 0.0
        self.clock = 0
        self.stall = False
        self.vec_len = vec_len  # 向量长度
        self.result_buffer = defaultdict(float)  # 用于乱序完成的累加

        # 乘法器流水线阶段
        self.mult_stages = [
            self._mult_stage1,  # 符号位处理和指数相加
            self._mult_stage2,  # 尾数乘法(高位部分)
            self._mult_stage3   # 尾数乘法(低位部分)和规格化
        ]

        # 加法器流水线阶段
        self.add_stages = [
            self._align_stage1,  # 指数比较
            self._align_stage2,  # 尾数对齐
            self._add_stage1,    # 尾数相加
            self._add_stage2,    # 规格化
            self._add_stage3     # 舍入和最终结果
        ]

    def feed_vector(self, a_vec, b_vec):
        """输入向量对进行MAC运算"""
        assert len(a_vec) == len(b_vec) == self.vec_len

        for i in range(self.vec_len):
            if not self.stall:
                self.registers[0].valid = True
                self.registers[0].data = {'a': a_vec[i], 'b': b_vec[i]}
                self.registers[0].tag = i  # 标记向量元素位置
                self._advance_pipeline()

    def run_cycles(self, n_cycles):
        """运行指定数量的时钟周期"""
        for _ in range(n_cycles):
            self._advance_pipeline()
            self.clock += 1

    def _advance_pipeline(self):
        """推进流水线"""
        # 向后推进流水线寄存器
        for i in range(len(self.registers)-1, 0, -1):
            self.registers[i] = self.registers[i-1]

        # 清空第一级寄存器
        self.registers[0] = PipelineRegister()

        # 执行乘法阶段
        for i in range(3):
            if self.registers[i].valid:
                self.mult_stages[i](i)

        # 执行加法阶段
        for i in range(3, 8):
            if self.registers[i].valid:
                stage_idx = i - 3  # 对齐加法阶段索引
                if stage_idx < len(self.add_stages):
                    self.add_stages[stage_idx](i)

        # 处理写回
        if self.registers[7].valid:
            result = self.registers[7].data['result']
            tag = self.registers[7].tag
            self.result_buffer[tag] = result

    # 乘法器三级流水实现
    def _mult_stage1(self, stage_idx):
        """乘法阶段1: 符号位处理和指数相加"""
        a = self.registers[stage_idx].data['a']
        b = self.registers[stage_idx].data['b']

        a_bits = Float32.to_bits(a)
        b_bits = Float32.to_bits(b)

        # 符号位
        sign_a = (a_bits >> 31) & 0x1
        sign_b = (b_bits >> 31) & 0x1
        sign = sign_a ^ sign_b

        # 指数部分 (减去127的偏置)
        exp_a = ((a_bits >> 23) & 0xFF) - 127
        exp_b = ((b_bits >> 23) & 0xFF) - 127
        exp_sum = exp_a + exp_b

        # 尾数部分 (加上隐含的1)
        mantissa_a = (a_bits & 0x7FFFFF) | 0x800000
        mantissa_b = (b_bits & 0x7FFFFF) | 0x800000

        self.registers[stage_idx].data.update({
            'sign': sign,
            'exp_sum': exp_sum,
            'mantissa_a': mantissa_a,
            'mantissa_b': mantissa_b
        })

    def _mult_stage2(self, stage_idx):
        """乘法阶段2: 尾数乘法(高位部分)"""
        mantissa_a = self.registers[stage_idx].data['mantissa_a']
        mantissa_b = self.registers[stage_idx].data['mantissa_b']

        # 24x24位乘法分解为高位和低位部分
        # 这里简化为直接乘法，实际硬件会使用Wallace树等结构
        product = mantissa_a * mantissa_b

        self.registers[stage_idx].data['product'] = product

    def _mult_stage3(self, stage_idx):
        """乘法阶段3: 规格化乘法结果"""
        sign = self.registers[stage_idx].data['sign']
        exp_sum = self.registers[stage_idx].data['exp_sum']
        product = self.registers[stage_idx].data['product']

        # 规格化处理
        if product == 0:
            normalized = 0.0
        else:
            # 找到最高有效位
            leading_zeros = 47 - product.bit_length()
            shift = 23 - leading_zeros

            # 调整指数
            exp = exp_sum + shift - 23

            # 处理溢出
            if exp > 127:
                # 上溢
                mantissa = 0x7FFFFF
                exp = 127
            elif exp < -126:
                # 下溢
                mantissa = 0
                exp = -126
            else:
                # 正常情况
                mantissa = (product >> (shift - 23)) & 0x7FFFFF

            # 组合浮点数
            result_bits = (sign << 31) | ((exp + 127) << 23) | mantissa
            normalized = Float32.from_bits(result_bits)

        self.registers[stage_idx].data['product_result'] = normalized

    # 加法器五级流水实现
    def _align_stage1(self, stage_idx):
        """对齐阶段1: 指数比较"""
        product = self.registers[stage_idx].data['product_result']
        acc_bits = Float32.to_bits(self.accumulator)
        prod_bits = Float32.to_bits(product)

        # 提取指数
        acc_exp = (acc_bits >> 23) & 0xFF
        prod_exp = (prod_bits >> 23) & 0xFF

        # 确定哪个操作数需要移位
        if acc_exp > prod_exp:
            larger = 'acc'
            shift = acc_exp - prod_exp
        else:
            larger = 'prod'
            shift = prod_exp - acc_exp

        self.registers[stage_idx].data.update({
            'product': product,
            'shift': shift,
            'larger_exp': larger,
            'accumulator': self.accumulator
        })

    def _align_stage2(self, stage_idx):
        """对齐阶段2: 尾数对齐"""
        product = self.registers[stage_idx].data['product']
        accumulator = self.registers[stage_idx].data['accumulator']
        shift = self.registers[stage_idx].data['shift']
        larger = self.registers[stage_idx].data['larger_exp']

        # 对齐尾数
        if larger == 'acc':
            # 乘积需要右移
            aligned_product = product / (2.0 ** shift)
            aligned_acc = accumulator
        else:
            # 累加器需要右移
            aligned_acc = accumulator / (2.0 ** shift)
            aligned_product = product

        self.registers[stage_idx].data.update({
            'aligned_acc': aligned_acc,
            'aligned_product': aligned_product
        })

    def _add_stage1(self, stage_idx):
        """加法阶段1: 尾数相加"""
        acc = self.registers[stage_idx].data['aligned_acc']
        product = self.registers[stage_idx].data['aligned_product']

        sum_result = acc + product

        self.registers[stage_idx].data['sum'] = sum_result

    def _add_stage2(self, stage_idx):
        """加法阶段2: 规格化"""
        sum_result = self.registers[stage_idx].data['sum']

        # 简化的规格化处理
        if sum_result == 0:
            normalized = 0.0
        else:
            sum_bits = Float32.to_bits(sum_result)
            sign = (sum_bits >> 31) & 0x1
            exp = (sum_bits >> 23) & 0xFF
            mantissa = sum_bits & 0x7FFFFF

            # 检查是否需要规格化
            if (mantissa >> 23) != 1:
                # 需要规格化
                leading_zeros = 23 - mantissa.bit_length()
                mantissa <<= leading_zeros
                exp -= leading_zeros

            # 处理溢出
            if exp > 254:  # 254 = 127*2
                exp = 254
                mantissa = 0x7FFFFF

            normalized = Float32.from_bits(
                (sign << 31) | (exp << 23) | (mantissa & 0x7FFFFF))

        self.registers[stage_idx].data['normalized'] = normalized

    def _add_stage3(self, stage_idx):
        """加法阶段3: 舍入和最终结果"""
        normalized = self.registers[stage_idx].data['normalized']

        # 简单舍入模式(向最近偶数舍入)
        result = normalized
        self.registers[stage_idx].data['result'] = result

    def get_result(self):
        """获取最终内积结果"""
        # 等待所有流水线操作完成
        while any(r.valid for r in self.registers):
            self._advance_pipeline()
            self.clock += 1

        # 累加所有部分结果
        final_result = sum(self.result_buffer.values())
        self.accumulator = final_result  # 更新累加器
        self.result_buffer.clear()

        return final_result


# 测试用例
if __name__ == "__main__":
    # 创建4通道的MAC Lane
    mac = MultiStageMAC(vec_len=4)

    # 测试向量
    a = [1.5, 2.0, 3.5, 4.0]
    b = [2.0, 3.0, 1.0, 0.5]

    print("计算内积:", a, "·", b)

    # 输入向量
    mac.feed_vector(a, b)

    # 运行足够周期完成计算
    mac.run_cycles(15)

    # 获取结果
    result = mac.get_result()
    expected = sum(x*y for x, y in zip(a, b))

    print("\nMAC Lane结果:", result)
    print("精确结果:   ", expected)
    print("相对误差:    {:.2e}".format(abs(result - expected)/expected))
    print("总时钟周期:  ", mac.clock)
