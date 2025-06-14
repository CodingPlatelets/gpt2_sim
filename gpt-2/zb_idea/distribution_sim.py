import numpy as np
import math
from scipy.sparse import csr_matrix
from distribution import get_values_offset_mask
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline


def min_bits_needed(bit_width):
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))


class MulUnit:
    """乘法单元，执行值的乘法操作"""

    def __init__(self):
        self.sft_index = None
        self.val_a = None
        self.val_b = None

    def get_val_a(self, val_a):
        """设置输入值A"""
        self.val_a = val_a

    def get_val_b(self, val_b):
        """设置输入值B"""
        self.val_b = val_b

    def get_sft_index(self, sft_index):
        """设置移位索引"""
        self.sft_index = sft_index

    def run(self):
        """执行乘法运算"""
        return self.val_a * self.val_b

    def print_status(self):
        """打印乘法单元当前状态"""
        print("\n==== MulUnit 状态 ====")
        print(f"Shift Index: {self.sft_index}")
        print(f"Value A: {self.val_a}")
        print(f"Value B: {self.val_b}")
        print("======================")


class AddUnit:
    """加法单元，执行值的加法操作"""

    def __init__(self):
        self.sft_index_1 = None
        self.sft_index_2 = None
        self.val_1 = None
        self.val_2 = None

    def get(self, val_1, val_2, sft_index_1, sft_index_2):
        """设置输入值和索引"""
        self.sft_index_1 = sft_index_1
        self.sft_index_2 = sft_index_2
        self.val_1 = val_1
        self.val_2 = val_2

    def run(self):
        """执行加法运算，如果索引相同则返回和值和第一个索引，否则返回None和两个索引"""
        if self.sft_index_1 == self.sft_index_2 and self.sft_index_1 != -1:
            return self.val_1 + self.val_2, self.sft_index_1
        return None, self.sft_index_1, self.sft_index_2


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

        # stage3 得到zero count bit
        self.stage3_valid = False
        self.stage3_zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.stage3_offset = 0
        self.stage3_ec_idx = []
        self.stage3_values_len = 0

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

        # stage3 得到zero count bit
        self.stage3_valid = False
        self.stage3_zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.stage3_offset = 0
        self.stage3_ec_idx = []
        self.stage3_values_len = 0

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
            shifted_ec_idx = self.stage3_ec_idx.copy()

            for bit_level in range(self.min_bits_num):
                temp_result = [0] * len(shifted_ec_idx)
                for i in range(len(self.stage3_zero_count_bit_vec[bit_level])):
                    if self.stage3_zero_count_bit_vec[bit_level][i] == 1:
                        target_idx = i - (1 << bit_level)
                        if target_idx >= 0 and target_idx < len(temp_result):
                            temp_result[target_idx] = shifted_ec_idx[i]
                    else:
                        temp_result[i] = shifted_ec_idx[i]

                shifted_ec_idx = temp_result.copy()

            self.output = [0] * self.stage3_values_len
            for i in range(len(shifted_ec_idx)):
                target_idx = i + self.stage3_offset
                if target_idx < self.stage3_values_len:
                    self.output[target_idx] = shifted_ec_idx[i]

        # 阶段3: get 0 bit

        self.stage3_valid = self.stage2_valid
        self.stage3_offset = self.stage2_offset
        self.stage3_ec_idx = self.stage2_ec_idx.copy()
        self.stage3_values_len = self.stage2_values_len

        if self.stage2_valid:
            zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]

            bit_vec = []
            for count in self.stage2_zero_count:
                bit_vec.append(bin(count)[2:].zfill(self.min_bits_num))

            for bit in bit_vec:
                for i in range(self.min_bits_num):
                    zero_count_bit_vec[i].append(int(bit[self.min_bits_num - i - 1]))

            self.stage3_zero_count_bit_vec = zero_count_bit_vec

        # 阶段2: 统计0元
        self.stage2_valid = self.stage1_valid
        self.stage2_offset = self.stage1_offset
        self.stage2_values_len = self.stage1_values_len
        self.stage2_ec_idx = self.stage1_ec_idx.copy()

        if self.stage1_valid:
            zero_count = []
            count_zeros = 0
            for i in range(self.bit_width):
                zero_count.append(count_zeros)
                bit = (self.stage1_bit_mask >> (self.bit_width - 1 - i)) & 1
                if bit == 0:
                    count_zeros += 1

            self.stage2_zero_count = zero_count

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


class MFIUPipeline:
    def __init__(self, width=None, bit_width=None):
        # 基础配置
        # TODO 处理 bit_width , width的输入，应该是一开始就确定的常量
        self.width = width
        self.bit_width = bit_width
        self.shift_unit_pipeline_vec_a = [
            ShiftUnitPipeline(bit_width) for _ in range(self.width)
        ]
        self.shift_unit_pipeline_vec_b = [
            ShiftUnitPipeline(bit_width) for _ in range(self.width)
        ]

        # stage1 处理输入，并进行一些预处理

        # 预处理结果
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_A_bit_mask_vec = [0] * self.width
        self.stage1_B_bit_mask_vec = [0] * self.width
        self.stage1_A_row_offset_vec = [0] * self.width
        self.stage1_B_col_offset_vec = [0] * self.width

        # stage2 B bitmask & A bitmask
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)

        self.stage2_len_values_A = 0
        self.stage2_len_values_B = 0
        self.stage2_A_bit_mask_vec = [0] * self.width
        self.stage2_B_bit_mask_vec = [0] * self.width
        self.stage2_A_row_offset_vec = [0] * self.width
        self.stage2_B_col_offset_vec = [0] * self.width

        # stage3 prefix sum
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        self.stage3_bit_seq = np.array([], dtype=int)

        self.stage3_len_values_A = 0
        self.stage3_len_values_B = 0
        self.stage3_A_bit_mask_vec = [0] * self.width
        self.stage3_B_bit_mask_vec = [0] * self.width
        self.stage3_A_row_offset_vec = [0] * self.width
        self.stage3_B_col_offset_vec = [0] * self.width

        # stage4 get ec_idx
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.width)]

        self.stage4_len_values_A = 0
        self.stage4_len_values_B = 0
        self.stage4_A_bit_mask_vec = [0] * self.width
        self.stage4_B_bit_mask_vec = [0] * self.width
        self.stage4_A_row_offset_vec = [0] * self.width
        self.stage4_B_col_offset_vec = [0] * self.width

        # stage5 shift
        self.stage5_valid = False

        self.cycle_count = 0
        self.output = ([], [])

    def clock_cycle(
        self,
        valid,
        mask_A_row,
        mask_B_col,
        offset_A_row,
        offset_B_col,
        len_values_A,
        len_values_B,
    ):

        self.cycle_count += 1

        # stage5 shift

        for i, (shift_unit_a, shift_unit_b) in enumerate(
            zip(self.shift_unit_pipeline_vec_a, self.shift_unit_pipeline_vec_b)
        ):
            results_a = shift_unit_a.clock_cycle(
                self.stage4_valid,
                self.stage4_A_bit_mask_vec[i],
                self.stage4_ec_idx_vec[i],
                self.stage4_len_values_A,
                self.stage4_A_row_offset_vec[i],
            )
            results_b = shift_unit_b.clock_cycle(
                self.stage4_valid,
                self.stage4_B_bit_mask_vec[i],
                self.stage4_ec_idx_vec[i],
                self.stage4_len_values_B,
                self.stage4_B_col_offset_vec[i],
            )
            self.stage5_valid = results_a["valid"]
            self.output[0].append(results_a["output"])
            self.output[1].append(results_b["output"])

        # stage4 get ec_idx
        self.stage4_valid = self.stage3_valid
        self.stage4_len_values_A = self.stage3_len_values_A
        self.stage4_len_values_B = self.stage3_len_values_B
        self.stage4_A_bit_mask_vec = self.stage3_A_bit_mask_vec.copy()
        self.stage4_B_bit_mask_vec = self.stage3_B_bit_mask_vec.copy()
        self.stage4_A_row_offset_vec = self.stage3_A_row_offset_vec.copy()
        self.stage4_B_col_offset_vec = self.stage3_B_col_offset_vec.copy()
        if self.stage3_valid:
            ec_idx_seq = np.where(
                self.stage3_bit_seq, self.stage3_AB_prefix_sum, 0
            ).tolist()
            for i in range(self.width):
                temp = []
                for j in range(self.bit_width):
                    temp.append(ec_idx_seq[i * self.bit_width + j])
                self.stage4_ec_idx_vec[i] = temp

        # stage3 prefix sum
        self.stage3_valid = self.stage2_valid
        self.stage3_len_values_A = self.stage2_len_values_A
        self.stage3_len_values_B = self.stage2_len_values_B
        self.stage3_A_bit_mask_vec = self.stage2_A_bit_mask_vec.copy()
        self.stage3_B_bit_mask_vec = self.stage2_B_bit_mask_vec.copy()
        self.stage3_A_row_offset_vec = self.stage2_A_row_offset_vec.copy()
        self.stage3_B_col_offset_vec = self.stage2_B_col_offset_vec.copy()
        self.stage3_bit_seq = self.stage2_bit_seq.copy()
        if self.stage2_valid:
            self.stage3_AB_prefix_sum = np.cumsum(self.stage2_bit_seq).tolist()

        # stage2 B bitmask & A bitmask
        self.stage2_valid = self.stage1_valid
        self.stage2_len_values_A = self.stage1_len_values_A
        self.stage2_len_values_B = self.stage1_len_values_B
        self.stage2_A_bit_mask_vec = self.stage1_A_bit_mask_vec.copy()
        self.stage2_B_bit_mask_vec = self.stage1_B_bit_mask_vec.copy()
        self.stage2_A_row_offset_vec = self.stage1_A_row_offset_vec.copy()
        self.stage2_B_col_offset_vec = self.stage1_B_col_offset_vec.copy()
        if self.stage1_valid:
            AB_bit_vec = [
                a & b
                for a, b in zip(self.stage1_A_bit_mask_vec, self.stage1_B_bit_mask_vec)
            ]
            all_bits = []
            for val in AB_bit_vec:
                for i in range(self.bit_width):
                    bit = (val >> (self.bit_width - 1 - i)) & 1
                    all_bits.append(bit)
            self.stage2_bit_seq = np.array(all_bits)

        # stage1 处理输入，并进行一些预处理
        self.stage1_valid = valid
        if valid:
            self.stage1_len_values_A = len_values_A
            self.stage1_len_values_B = len_values_B
            idx = 0
            for i in range(len(mask_B_col)):
                for j in range(len(mask_A_row)):
                    self.stage1_B_bit_mask_vec[idx] = mask_B_col[i]
                    self.stage1_B_col_offset_vec[idx] = offset_B_col[i]
                    self.stage1_A_bit_mask_vec[idx] = mask_A_row[j]
                    self.stage1_A_row_offset_vec[idx] = offset_A_row[j]
                    idx += 1
        else:
            self.stage1_len_values_A = 0
            self.stage1_len_values_B = 0
            self.stage1_A_bit_mask_vec = [0] * self.width
            self.stage1_B_bit_mask_vec = [0] * self.width
            self.stage1_A_row_offset_vec = [0] * self.width
            self.stage1_B_col_offset_vec = [0] * self.width

        return {
            "cycle": self.cycle_count,
            "valid": self.stage5_valid,
            "output": self.output if self.stage5_valid else None,
            "pipeline_state": self.get_pipeline_state(),
        }

    def reset(self):
        """重置流水线状态"""
        # 重置时钟计数器
        self.cycle_count = 0

        # 重置输出
        self.output = ([], [])

        # 重置stage1状态
        self.stage1_valid = False
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_A_bit_mask_vec = [0] * self.width
        self.stage1_B_bit_mask_vec = [0] * self.width
        self.stage1_A_row_offset_vec = [0] * self.width
        self.stage1_B_col_offset_vec = [0] * self.width

        # 重置stage2状态
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)
        self.stage2_len_values_A = 0
        self.stage2_len_values_B = 0
        self.stage2_A_bit_mask_vec = [0] * self.width
        self.stage2_B_bit_mask_vec = [0] * self.width
        self.stage2_A_row_offset_vec = [0] * self.width
        self.stage2_B_col_offset_vec = [0] * self.width

        # 重置stage3状态
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        self.stage3_bit_seq = np.array([], dtype=int)
        self.stage3_len_values_A = 0
        self.stage3_len_values_B = 0
        self.stage3_A_bit_mask_vec = [0] * self.width
        self.stage3_B_bit_mask_vec = [0] * self.width
        self.stage3_A_row_offset_vec = [0] * self.width
        self.stage3_B_col_offset_vec = [0] * self.width

        # 重置stage4状态
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.width)]
        self.stage4_len_values_A = 0
        self.stage4_len_values_B = 0
        self.stage4_A_bit_mask_vec = [0] * self.width
        self.stage4_B_bit_mask_vec = [0] * self.width
        self.stage4_A_row_offset_vec = [0] * self.width
        self.stage4_B_col_offset_vec = [0] * self.width

        # 重置stage5状态
        self.stage5_valid = False

        # 重置所有ShiftUnitPipeline
        for shift_unit in self.shift_unit_pipeline_vec_a:
            shift_unit.reset()

        for shift_unit in self.shift_unit_pipeline_vec_b:
            shift_unit.reset()

    def get_pipeline_state(self):
        """返回流水线当前状态"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "len_values_A": self.stage1_len_values_A,
                "len_values_B": self.stage1_len_values_B,
                "A_bit_mask_sample": self.stage1_A_bit_mask_vec[
                    : min(3, len(self.stage1_A_bit_mask_vec))
                ],
                "B_bit_mask_sample": self.stage1_B_bit_mask_vec[
                    : min(3, len(self.stage1_B_bit_mask_vec))
                ],
            },
            "stage2": {
                "valid": self.stage2_valid,
                "bit_seq_len": (
                    len(self.stage2_bit_seq) if self.stage2_bit_seq is not None else 0
                ),
                "bit_seq_sample": (
                    self.stage2_bit_seq[: min(10, len(self.stage2_bit_seq))]
                    if self.stage2_bit_seq is not None
                    else None
                ),
            },
            "stage3": {
                "valid": self.stage3_valid,
                "prefix_sum_len": (
                    len(self.stage3_AB_prefix_sum)
                    if self.stage3_AB_prefix_sum is not None
                    else 0
                ),
                "prefix_sum_sample": (
                    self.stage3_AB_prefix_sum[: min(10, len(self.stage3_AB_prefix_sum))]
                    if self.stage3_AB_prefix_sum is not None
                    else None
                ),
            },
            "stage4": {
                "valid": self.stage4_valid,
                "ec_idx_vec_len": len(self.stage4_ec_idx_vec),
                "ec_idx_vec_sample": (
                    self.stage4_ec_idx_vec[: min(3, len(self.stage4_ec_idx_vec))]
                    if self.stage4_ec_idx_vec
                    else []
                ),
            },
            "stage5": {
                "valid": self.stage5_valid,
                "output_a_len": (
                    len(self.output[0]) if self.output and len(self.output) > 0 else 0
                ),
                "output_b_len": (
                    len(self.output[1]) if self.output and len(self.output) > 1 else 0
                ),
            },
            "shift_units": {
                "a_count": len(self.shift_unit_pipeline_vec_a),
                "b_count": len(self.shift_unit_pipeline_vec_b),
            },
        }

    def print_state(self):
        """打印流水线当前状态"""
        state = self.get_pipeline_state()
        print(f"\n==== MFIU Pipeline State (Cycle {self.cycle_count}) ====")
        print(f"Configuration: width={self.width}, bit_width={self.bit_width}")

        print(f"Stage 1 (Input): {'Valid' if state['stage1']['valid'] else 'Invalid'}")
        if state["stage1"]["valid"]:
            print(
                f"  Values length: A={state['stage1']['len_values_A']}, B={state['stage1']['len_values_B']}"
            )
            print(f"  A bit mask sample: {state['stage1']['A_bit_mask_sample']}...")
            print(f"  B bit mask sample: {state['stage1']['B_bit_mask_sample']}...")

        print(f"Stage 2 (AND): {'Valid' if state['stage2']['valid'] else 'Invalid'}")
        if state["stage2"]["valid"] and state["stage2"]["bit_seq_len"] > 0:
            print(f"  Bit sequence length: {state['stage2']['bit_seq_len']}")
            print(f"  Bit sequence sample: {state['stage2']['bit_seq_sample']}...")

        print(
            f"Stage 3 (PrefixSum): {'Valid' if state['stage3']['valid'] else 'Invalid'}"
        )
        if state["stage3"]["valid"] and state["stage3"]["prefix_sum_len"] > 0:
            print(f"  Prefix sum length: {state['stage3']['prefix_sum_len']}")
            print(f"  Prefix sum sample: {state['stage3']['prefix_sum_sample']}...")

        print(f"Stage 4 (EcIdx): {'Valid' if state['stage4']['valid'] else 'Invalid'}")
        if state["stage4"]["valid"] and state["stage4"]["ec_idx_vec_len"] > 0:
            print(f"  EC index vector length: {state['stage4']['ec_idx_vec_len']}")
            print(
                f"  EC index vector sample: {state['stage4']['ec_idx_vec_sample']}..."
            )

        print(f"Stage 5 (Shift): {'Valid' if state['stage5']['valid'] else 'Invalid'}")
        if state["stage5"]["valid"]:
            print(
                f"  Output length: A={state['stage5']['output_a_len']}, B={state['stage5']['output_b_len']}"
            )

        print(
            f"Shift Units: {state['shift_units']['a_count']} A units, {state['shift_units']['b_count']} B units"
        )
        print("==============================")

    def run_pipeline(
        self,
        mask_A_row,
        mask_B_col,
        offset_A_row,
        offset_B_col,
        len_values_A,
        len_values_B,
        print_states=False,
    ):
        """
        完整运行流水线，生成最终结果

        Args:
            mask_A_row: A矩阵行的掩码
            mask_B_col: B矩阵列的掩码
            offset_A_row: A矩阵行的偏移量
            offset_B_col: B矩阵列的偏移量
            len_values_A: A矩阵值的长度
            len_values_B: B矩阵值的长度
            print_states: 是否打印状态
        """
        # 重置状态
        self.reset()

        # 启动流水线并运行所需的周期
        results = []

        # 开始新的处理
        results.append(
            self.clock_cycle(
                valid=True,
                mask_A_row=mask_A_row,
                mask_B_col=mask_B_col,
                offset_A_row=offset_A_row,
                offset_B_col=offset_B_col,
                len_values_A=len_values_A,
                len_values_B=len_values_B,
            )
        )

        if print_states:
            self.print_state()

        # 运行直至所有阶段处理完毕（通常需要额外5个周期，因为有5个阶段）
        max_cycles = 20  # 设置一个最大循环次数以防无限循环
        for cycle in range(max_cycles):
            result = self.clock_cycle(
                valid=False,  # 后续周期不再输入新数据
                mask_A_row=None,
                mask_B_col=None,
                offset_A_row=None,
                offset_B_col=None,
                len_values_A=0,
                len_values_B=0,
            )
            results.append(result)

            if print_states:
                self.print_state()

            # 检查是否已完成（当stage5有效且至少已运行5个周期）
            if result["valid"] and cycle >= 4:
                if print_states:
                    print(f"流水线在第{cycle + 2}个周期完成处理")
                break

        # 如果没有中途break，说明可能有问题
        else:
            if print_states:
                print(f"警告：流水线在{max_cycles}个周期内未产生有效输出")

        # 返回最终结果
        return {
            "output_a": self.output[0] if self.stage5_valid else None,
            "output_b": self.output[1] if self.stage5_valid else None,
            "results": results,
        }


class MACUnit:
    def __init__(self):
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.output = 0
        self.valid = False
        self.input_valid = False

    def get_input(self, input1, input2, input_valid):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid

    def clock_cycle(self):
        result = self.multiply_pipeline.clock_cycle(
            self.input1, self.input2, self.input_valid
        )
        self.valid = result["valid_output"]
        if self.valid:
            self.output = self.multiply_pipeline.outputs.pop(0)


class AddUnit:
    def __init__(self):
        self.add_pipeline = BF16AddPipeline()
        self.input1 = 0
        self.input2 = 0
        self.sft_index_1 = -1
        self.sft_index_2 = -1
        self.valid = False
        self.input_valid = False

    def get_input(self, input_valid, input1, input2, sft_index_1, sft_index_2):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        self.sft_index_1 = sft_index_1
        self.sft_index_2 = sft_index_2

    def clock_cycle(self):
        result = self.add_pipeline.clock_cycle(self.input1, self.input2, self.input_valid) 
        self.valid = result["valid_output"]
        if self.valid:
            if self.sft_index_1 == self.sft_index_2:
                return self.add_pipeline.outputs.pop(0), self.sft_index_1
            return None, self.sft_index_1, self.sft_index_2
        return None


class DoubleLayerAddUnit:
    def __init__(self, width, M, N, c_values):
        self.add_lower = [AddUnit(), AddUnit()]
        self.add_upper = AddUnit()
        self.width = width
        self.M = M
        self.N = N

        self.stage1_valid = False
        self.stage2_valid = False

        # TODO: remember to empty
        self.index_value_map = [[] for _ in range(width)]
        self.add_lower_value_queue = []
        self.add_lower_index_queue = []
        
        self.output_value_queue = []
        self.output_index_queue = []
        
        self.c_values = c_values # TODO: 后面需要更细致地处理这个C块加法逻辑

    def add_lower_clock_cycle(self, input_valid, value_input, sft_index):
        valid = False
        for i, add in enumerate(self.add_lower):
            idx = i * 2
            add.get_input(input_valid, value_input[idx], value_input[idx + 1], sft_index[idx], sft_index[idx + 1])
            result = add.clock_cycle()
            if result is None:
                self.add_lower_value_queue = []
                self.add_lower_index_queue = []
                self.index_value_map = [[] for _ in range(self.width)]
            else:
                if result[0] is None:
                    self.add_lower_index_queue.append(result[1])
                    self.add_lower_index_queue.append(result[2])
                    self.add_lower_value_queue.append(add.input1)
                    self.add_lower_value_queue.append(add.input2)
                else:
                    self.add_lower_index_queue.append(result[1])
                    self.add_lower_value_queue.append(result[0])
                valid = True    
        for i, index in enumerate(self.add_lower_index_queue):
            if index != -1:
                self.index_value_map[index].append(self.add_lower_value_queue[i])
        return valid
        
    def add_upper_clock_cycle(self, input_valid):
        
        if input_valid == False:
            return False
        
        valid = False
        flag = False
        for index, values in enumerate(self.index_value_map):
            m = index % self.M 
            n = int(index / self.M)
            if len(values) == 2:
                self.add_upper.get_input(input_valid ,values[0], values[1], index, index)
                result = self.add_upper.clock_cycle()
                flag = True
                if result is None:
                    self.output_value_queue = []
                    self.output_index_queue = []
                else:
                    self.output_value_queue.append(result[0])
                    self.output_index_queue.append(result[1])
                    valid = True
            elif len(values) == 1:            
                self.c_values[m * self.N + n] += values[0] #TODO: 目前简化了这个逻辑，未来需要处理下 
        if flag == False: #如果没有用到这个加法器，我们需要等待几个周期，以同步
            self.add_upper.get_input(True, 0, 0, 0 ,0)
            result = self.add_upper.clock_cycle()
            if result is not None:
                valid = True
        return valid
    
    def clock_cycle(self, valid, value_input, sft_index):
        
        self.stage2_valid = self.add_upper_clock_cycle(self.stage1_valid)
        self.stage1_valid = self.add_lower_clock_cycle(valid, value_input, sft_index)
        
        return {
            "valid": self.stage2_valid,
            "output_value_queue": self.output_value_queue if self.stage2_valid else [],
            "output_index_queue":
        }
        
        


class TrapezoidPipeline:
    def __init__(self, M, K, N, PE_num=4):
        # 默认配置
        self.width = M * N
        self.bit_width = K
        self.PE_num = PE_num
        self.mfiu = MFIUPipeline()

        # stage1: 预输入处理数据
        self.stage1_valid = False
        self.stage1_values_A = np.array([])
        self.stage1_values_B = np.array([])
        self.stage1_offset_A = np.array([])
        self.stage1_offset_B = np.array([])
        self.stage1_masks_A = []
        self.stage1_masks_B = []

        # stage2: mfiu模块
        self.stage2_valid = False
        self.stage2_index = ([], [])
        self.stage2_values_A = np.array([])
        self.stage2_values_B = np.array([])

        # stage3: 获得输入队列
        self.stage3_valid = False
        self.stage3_mul_queue_a = [[] for _ in range(self.PE_num)]
        self.stage3_mul_queue_b = [[] for _ in range(self.PE_num)]
        self.stage3_sft_index_queue = [[] for _ in range(self.PE_num)]

        self.cycle_count = 0

    def pad_queues(self):
        max_len_a = (
            max([len(queue) for queue in self.stage3_mul_queue_a])
            if self.stage3_mul_queue_a
            else 0
        )
        max_len_b = (
            max([len(queue) for queue in self.stage3_mul_queue_b])
            if self.stage3_mul_queue_b
            else 0
        )
        assert max_len_a == max_len_b

        for queue in self.stage3_mul_queue_a:
            while len(queue) < max_len_a:
                queue.append(0)

        for queue in self.stage3_mul_queue_b:
            while len(queue) < max_len_b:
                queue.append(0)

        for queue in self.stage3_sft_index_queue:
            while len(queue) < max_len_a:
                queue.append(-1)
        # self.round_num = max_len_a

    def clock_cycle(self, valid, A: np.array, B: np.array):

        self.cycle_count += 1

        # stage3 获得输入队列
        self.stage3_valid = self.stage2_valid
        if self.stage2_valid:
            sft_index_a = self.stage2_index[0]
            sft_index_b = self.stage2_index[1]
            for sft_index in range(len(sft_index_a)):
                sft_row_a = sft_index_a[sft_index]
                sft_row_b = sft_index_b[sft_index]
                for i in range(len(sft_row_a)):
                    if sft_row_a[i] != 0:
                        self.stage3_mul_queue_a[
                            (sft_row_a[i] - 1) % self.PE_num
                        ].append(self.stage2_values_A[i])
                        self.stage3_sft_index_queue[
                            (sft_row_a[i] - 1) % self.PE_num
                        ].append(
                            sft_index
                        )  # A and B are same

                for i in range(len(sft_row_b)):
                    if sft_row_b[i] != 0:
                        self.stage3_mul_queue_b[
                            (sft_row_b[i] - 1) % self.PE_num
                        ].append(self.stage1_values_B[i])
            self.pad_queues()

        # stage2: mfiu模块
        results = self.mfiu.clock_cycle(
            self.stage1_valid,
            self.stage1_masks_A,
            self.stage1_masks_B,
            self.stage1_offset_A,
            self.stage1_offset_B,
            len(self.stage1_values_A),
            len(self.stage1_values_B),
        )
        self.stage2_valid = results["valid"]
        if self.stage2_valid:
            self.stage2_index = results["output"]
            self.stage2_values_A = self.stage1_values_A.copy()
            self.stage2_values_B = self.stage1_values_B.copy()
        else:
            self.stage2_index = ([], [])

        # stage1: 预输入处理数据
        self.stage1_valid = valid
        if valid:
            csr_A = csr_matrix(A)
            csr_B = csr_matrix(B.T)
            self.stage1_values_A, self.stage1_offset_A, self.stage1_masks_A = (
                get_values_offset_mask(csr_A)
            )
            self.stage1_values_B, self.stage1_offset_B, self.stage1_masks_B = (
                get_values_offset_mask(csr_B)
            )
        else:
            self.stage1_values_A = np.array([])
            self.stage1_values_B = np.array([])
            self.stage1_offset_A = np.array([])
            self.stage1_offset_B = np.array([])
            self.stage1_masks_A = []
            self.stage1_masks_B = []


# 测试代码
def run_tests(A: np.array, B: np.array):
    print("\n===== 测试 MFIU Pipeline =====")
    csr_A = csr_matrix(A)
    csr_B = csr_matrix(B.T)

    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]

    width = M * N
    bit_width = K

    mfiu_pipeline = MFIUPipeline(width, bit_width)

    values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    values_B, offset_B, masks_B = get_values_offset_mask(csr_B)

    results = mfiu_pipeline.run_pipeline(
        masks_A, masks_B, offset_A, offset_B, len(values_A), len(values_B), True
    )
    print(results["output_a"])
    print(results["output_b"])


if __name__ == "__main__":
    # 运行测试
    print("\n===== 测试用例1：简单矩阵 =====")
    A1 = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])

    B1 = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    run_tests(A1, B1)
