from ..utils import min_bits_needed

class ShiftUnitPipeline:
    def __init__(self, bit_width=4):
        """初始化ShiftUnit流水线"""
        # 基本配置参数
        self.bit_width = bit_width
        self.min_bits_num = min_bits_needed(bit_width)

        # stage1 获取输入
        self.stage1_valid = False
        self.stage1_bit_mask = 0
        self.stage1_ec_idx = []
        self.stage1_values_len = 0
        self.stage1_offset = 0

        # stage2 统计0元
        self.stage2_zero_count = []
        self.stage2_valid = False
        self.stage2_offset = 0
        self.stage2_ec_idx = []
        self.stage2_values_len = 0
        self.stage2_bit_mask = 0

        # stage3 得到zero count bit
        self.stage3_valid = False
        self.stage3_zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.stage3_offset = 0
        self.stage3_ec_idx = []
        self.stage3_values_len = 0
        self.stage3_bit_mask = 0

        # stage4 shift
        self.stage4_valid = False

        # 输出结果
        self.output = []
        self.cycle_count = 0

    def reset(self):
        """重置流水线状态"""
        # 重置所有阶段状态
        # stage1 获取输入
        self.stage1_valid = False
        self.stage1_bit_mask = 0
        self.stage1_ec_idx = []
        self.stage1_values_len = 0
        self.stage1_offset = 0

        # stage2 统计0元
        self.stage2_zero_count = []
        self.stage2_valid = False
        self.stage2_offset = 0
        self.stage2_ec_idx = []
        self.stage2_values_len = 0
        self.stage2_bit_mask = 0

        # stage3 得到zero count bit
        self.stage3_valid = False
        self.stage3_zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.stage3_offset = 0
        self.stage3_ec_idx = []
        self.stage3_values_len = 0
        self.stage3_bit_mask = 0

        # stage4 shift
        self.stage4_valid = False

        # 重置输出和计数器
        self.output = []
        self.cycle_count = 0

    def clock_cycle(self, valid, bit_mask, ec_idx, values_len, offset):
        """
        执行一个时钟周期的流水线操作

        Args:
            start_new: 是否开始新的处理（将输入送入流水线）
        """
        self.cycle_count += 1

        # 阶段4 处理右移
        self.stage4_valid = self.stage3_valid
        if self.stage3_valid:
            #hifted_ec_idx = self.stage3_ec_idx.copy()

            #or bit_level in range(self.min_bits_num):
            #   temp_result = [0] * len(shifted_ec_idx)
            #   for i in range(len(self.stage3_zero_count_bit_vec[bit_level])):
            #       if self.stage3_zero_count_bit_vec[bit_level][i] == 1:
            #           target_idx = i - (1 << bit_level)
            #           if target_idx >= 0 and target_idx < len(temp_result):
            #               temp_result[target_idx] = shifted_ec_idx[i]
            #       else:
            #           temp_result[i] = shifted_ec_idx[i]

            #   shifted_ec_idx = temp_result.copy()

            #elf.output = [0] * self.stage3_values_len
            #or i in range(len(shifted_ec_idx)):
            #   target_idx = i + self.stage3_offset
            #   if target_idx < self.stage3_values_len:
            #       self.output[target_idx] = shifted_ec_idx[i]

            bit_width = len(self.stage3_ec_idx)
            
            # 直接基于mask提取有效索引
            valid_indices = []
            valid_positions = []
            
            for i in range(bit_width):
                bit = (self.stage3_bit_mask >> (bit_width - 1 - i)) & 1
                if bit == 1:
                    valid_indices.append(self.stage3_ec_idx[i])
                    valid_positions.append(i)
            
            #print(f"  优化版 - 有效位置: {valid_positions}")
            #print(f"  优化版 - 有效索引: {valid_indices}")
            
            # 计算移位后的位置
            result = [0] * self.stage3_values_len
            
            # ✅ 修复：使用self.stage3_offset而不是函数参数offset
            result_pos = self.stage3_offset
            for idx in valid_indices:
                if result_pos < self.stage3_values_len:
                    result[result_pos] = idx
                    result_pos += 1
            
            self.output = result

        # 阶段3: get 0 bit

        self.stage3_valid = self.stage2_valid
        self.stage3_offset = self.stage2_offset
        self.stage3_ec_idx = self.stage2_ec_idx.copy()
        self.stage3_values_len = self.stage2_values_len
        self.stage3_bit_mask = self.stage2_bit_mask
        if self.stage2_valid:
            #zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]

            #bit_vec = []
            #for count in self.stage2_zero_count:
            #    bit_vec.append(bin(count)[2:].zfill(self.min_bits_num))

            #for bit in bit_vec:
            #    for i in range(self.min_bits_num):
            #        zero_count_bit_vec[i].append(int(bit[self.min_bits_num - i - 1]))

            #self.stage3_zero_count_bit_vec = zero_count_bit_vec
            pass

        # 阶段2: 统计0元
        self.stage2_valid = self.stage1_valid
        self.stage2_offset = self.stage1_offset
        self.stage2_values_len = self.stage1_values_len
        self.stage2_ec_idx = self.stage1_ec_idx.copy()
        self.stage2_bit_mask = self.stage1_bit_mask

        if self.stage1_valid:
            #zero_count = []
            #count_zeros = 0
            #for i in range(self.bit_width):
            #    zero_count.append(count_zeros)
            #    bit = (self.stage1_bit_mask >> (self.bit_width - 1 - i)) & 1
            #    if bit == 0:
            #        count_zeros += 1
            #self.stage2_zero_count = zero_count
            pass

        # 阶段1: 接收新数据
        self.stage1_valid = valid
        if (
            valid
            and bit_mask is not None
            and ec_idx is not None
            and values_len is not None
            and offset is not None
        ):
            self.stage1_bit_mask = bit_mask
            self.stage1_ec_idx = ec_idx.copy()
            self.stage1_values_len = values_len
            self.stage1_offset = offset

        else:
            self.stage1_bit_mask = 0
            self.stage1_ec_idx = []
            self.stage1_values_len = 0
            self.stage1_offset = 0

        return {
            "cycle": self.cycle_count,
            "valid": self.stage4_valid,
            "output": self.output if self.stage4_valid else None,
            "pipeline_state": self.get_pipeline_state(),
        }

    def is_active(self):
        return (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.stage4_valid
        )

    def get_pipeline_state(self):
        """返回流水线当前状态"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "bit_mask": (
                    bin(self.stage1_bit_mask)[2:].zfill(self.bit_width)
                    if self.stage1_valid
                    else "invalid"
                ),
                "ec_idx_len": len(self.stage1_ec_idx) if self.stage1_valid else 0,
                "values_len": self.stage1_values_len if self.stage1_valid else 0,
                "offset": self.stage1_offset if self.stage1_valid else "invalid",
            },
            "stage2": {
                "valid": self.stage2_valid,
                "zero_count": (
                    self.stage2_zero_count if self.stage2_valid else "invalid"
                ),
                "ec_idx_len": len(self.stage2_ec_idx) if self.stage2_valid else 0,
                "values_len": self.stage2_values_len if self.stage2_valid else 0,
                "offset": self.stage2_offset if self.stage2_valid else "invalid",
            },
            "stage3": {
                "valid": self.stage3_valid,
                "bit_vectors": (
                    [v for v in self.stage3_zero_count_bit_vec]
                    if self.stage3_valid
                    else "invalid"
                ),
                "ec_idx_len": len(self.stage3_ec_idx) if self.stage3_valid else 0,
                "values_len": self.stage3_values_len if self.stage3_valid else 0,
                "offset": self.stage3_offset if self.stage3_valid else "invalid",
            },
            "stage4": {
                "valid": self.stage4_valid,
                "output_len": len(self.output) if self.output else 0,
            },
        }

    def print_state(self):
        """打印流水线当前状态"""
        state = self.get_pipeline_state()
        print(f"\n==== ShiftUnit Pipeline State (Cycle {self.cycle_count}) ====")

        print(f"Stage 1 (Input): {'Valid' if state['stage1']['valid'] else 'Invalid'}")
        if state["stage1"]["valid"]:
            print(f"  Bit mask: {state['stage1']['bit_mask']}")
            print(f"  EC idx length: {state['stage1']['ec_idx_len']}")
            print(f"  Values length: {state['stage1']['values_len']}")
            print(f"  Offset: {state['stage1']['offset']}")

        print(
            f"Stage 2 (Zero Count): {'Valid' if state['stage2']['valid'] else 'Invalid'}"
        )
        if state["stage2"]["valid"]:
            print(f"  Zero count: {state['stage2']['zero_count']}")
            print(f"  EC idx length: {state['stage2']['ec_idx_len']}")
            print(f"  Offset: {state['stage2']['offset']}")

        print(
            f"Stage 3 (Bit Vector): {'Valid' if state['stage3']['valid'] else 'Invalid'}"
        )
        if state["stage3"]["valid"] and state["stage3"]["bit_vectors"] != "invalid":
            for i, bit_vec in enumerate(state["stage3"]["bit_vectors"]):
                if i < 3 or i == len(state["stage3"]["bit_vectors"]) - 1:
                    print(
                        f"  Bit Level {i}: {bit_vec[:20]}..."
                        if len(bit_vec) > 20
                        else f"  Bit Level {i}: {bit_vec}"
                    )
                elif i == 3 and len(state["stage3"]["bit_vectors"]) > 4:
                    print(
                        f"  ... ({len(state['stage3']['bit_vectors']) - 4} more levels)"
                    )
            print(f"  EC idx length: {state['stage3']['ec_idx_len']}")
            print(f"  Offset: {state['stage3']['offset']}")

        print(f"Stage 4 (Shift): {'Valid' if state['stage4']['valid'] else 'Invalid'}")
        if state["stage4"]["valid"]:
            print(f"  Output length: {state['stage4']['output_len']}")
            if self.output and len(self.output) <= 20:
                print(f"  Output: {self.output}")
            elif self.output:
                print(f"  Output: {self.output[:10]}...{self.output[-10:]}")

        print("==============================")

    def run_pipeline(self, print_states=False):
        """完整运行流水线，生成最终结果"""
        # 重置状态
        self.reset()

        # 启动流水线并运行所需的周期
        results = []

        # 开始新的处理
        results.append(self.clock_cycle(start_new=True))
        if print_states:
            self.print_state()

        # 运行直至所有阶段处理完毕（需要额外3个周期）
        for _ in range(3):
            results.append(self.clock_cycle(start_new=False))
            if print_states:
                self.print_state()

        return self.output
