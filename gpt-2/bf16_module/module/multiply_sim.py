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
        self.NAN = 0x7FC0  # NaN：0 11111111 1000000

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
        if (exp_a == 0xFF and mant_a != 0) or (exp_b == 0xFF and mant_b != 0):
            return True, self.NAN

        # 2. 处理 无穷 × 0 = NaN 的情况
        if (exp_a == 0xFF and exp_b == 0 and mant_b == 0) or (
            exp_b == 0xFF and exp_a == 0 and mant_a == 0
        ):
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
                        exp_result += 1  # 右移小数点
                        round_bits = mant_result & 0xFF
                        half_point = 0x80
                        mant_result >>= 8
                    else:
                        round_bits = mant_result & 0x7F
                        half_point = 0x70
                        mant_result >>= 7

                    if round_bits > half_point or (
                        round_bits == half_point and (mant_result & 0x1)
                    ):
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
                            sticky_bits = (
                                denorm_mant & ((1 << (shift_amount - 1)) - 1)
                            ) != 0

                            denorm_mant >>= shift_amount

                            # 应用舍入
                            if round_bit and (sticky_bits or (denorm_mant & 1)):
                                denorm_mant += 1

                            result_bf16 = self.compose_bf16(
                                sign_result, 0, denorm_mant & 0x7F
                            )

                    elif exp_result >= 0xFF:  # 上溢，返回无穷大
                        result_bf16 = self.compose_bf16(sign_result, 0xFF, 0)

                    else:  # 正常情况
                        result_bf16 = self.compose_bf16(
                            sign_result, exp_result, mant_result
                        )

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
                    while temp_mant and not (
                        temp_mant & 0x80
                    ):  # 0x40 = 0100 0000, 第6位
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
                    while temp_mant and not (
                        temp_mant & 0x80
                    ):  # 0x40 = 0100 0000, 第6位
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
            #"pipeline_state": self.get_pipeline_state(),
        }

    def get_pipeline_state(self):
        """返回流水线的当前状态，用于可视化"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "bf16_a": hex(self.stage1_bf16_a) if self.stage1_valid else "invalid",
                "bf16_b": hex(self.stage1_bf16_b) if self.stage1_valid else "invalid",
                "special": self.stage1_special_case,
            },
            "stage2": {
                "valid": self.stage2_valid,
                "special": self.stage2_special_case,
                "sign_result": (
                    self.stage2_sign_result if self.stage2_valid else "invalid"
                ),
                "exp_a": self.stage2_exp_a if self.stage2_valid else "invalid",
                "exp_b": self.stage2_exp_b if self.stage2_valid else "invalid",
                "mant_a": hex(self.stage2_mant_a) if self.stage2_valid else "invalid",
                "mant_b": hex(self.stage2_mant_b) if self.stage2_valid else "invalid",
            },
            "stage3": {
                "valid": self.stage3_valid,
                "special": self.stage3_special_case,
                "sign_result": (
                    self.stage3_sign_result if self.stage3_valid else "invalid"
                ),
                "exp_result": (
                    self.stage3_exp_result if self.stage3_valid else "invalid"
                ),
                "mant_result": (
                    hex(self.stage3_mant_result) if self.stage3_valid else "invalid"
                ),
            },
            "stage4": {
                "valid": self.stage4_valid,
                "special": self.stage4_special_case,
                "sign_result": (
                    self.stage4_sign_result if self.stage4_valid else "invalid"
                ),
                "exp_result": (
                    self.stage4_exp_result if self.stage4_valid else "invalid"
                ),
                "mant_result": (
                    hex(self.stage4_mant_result) if self.stage4_valid else "invalid"
                ),
                "msb_pos": self.stage4_msb_pos if self.stage4_valid else "invalid",
            },
        }

    def print_pipeline_state(self):
        """打印当前流水线状态"""
        state = self.get_pipeline_state()
        print(f"Cycle {self.cycle_count}:")
        print(
            f"  Stage 1: {'Valid' if state['stage1']['valid'] else 'Invalid'} - "
            f"BF16 A: {state['stage1']['bf16_a']}, BF16 B: {state['stage1']['bf16_b']} "
            f"{'(Special case)' if state['stage1']['special'] else ''}"
        )
        print(
            f"  Stage 2: {'Valid' if state['stage2']['valid'] else 'Invalid'} - "
            f"{'Special case' if state['stage2']['special'] else 'Prepared operands'}"
        )
        print(
            f"  Stage 3: {'Valid' if state['stage3']['valid'] else 'Invalid'} - "
            f"{'Special case' if state['stage3']['special'] else 'Multiplication result, Exp: ' + str(state['stage3']['exp_result'])}"
        )
        print(
            f"  Stage 4: {'Valid' if state['stage4']['valid'] else 'Invalid'} - "
            f"{'Special case' if state['stage4']['special'] else 'Normalized result'}"
        )
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

    def is_active(self):
        return (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.stage4_valid
            or self.stage5_valid
        )
