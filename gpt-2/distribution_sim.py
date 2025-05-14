import numpy as np
import math
import struct
from scipy.sparse import csr_matrix
from distribution import get_values_offset_mask
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
from collections import Counter

def convert_through_pipeline(value):
    """通过完整流水线模拟转换FP32到BF16"""
    temp_pipeline = FP32toBF16Pipeline()
    temp_pipeline.run_simulation([(value, True)], print_states=False)
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0   

def min_bits_needed(bit_width):
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))


def bf16_to_float(bf16):
    # 左移16位填充为32位表示
    fp32_bits = bf16 << 16
    # 转换为浮点数
    return struct.unpack(">f", struct.pack(">I", fp32_bits))[0]

def find_singles(lst):
    # 统计每个元素出现的次数
    counts = Counter(lst)
    # 返回只出现一次的元素
    return [item for item, count in counts.items() if count == 1]



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
        self.index_queue = []  # 只有相同的index才能被送入加法单元
        self.valid = False 
        self.input_valid = False

    def get_input(self, input_valid, input1, input2, sft_index):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        if input_valid: #防止污染数据
            self.index_queue.append(sft_index) # 存放至sft_index队列中 
    
    def clock_cycle(self):
        result = self.add_pipeline.clock_cycle(self.input1, self.input2, self.input_valid) 
        self.valid = result["valid_output"]
        if self.valid:
            return self.add_pipeline.outputs.pop(0)
        return None
    
    def is_active(self):
        return self.add_pipeline.is_active()


class AdvanceAddUnit:
    def __init__(self, c_values, M, N):

        self.cycle_count = 0

        self.c_values = c_values
        self.M = M
        self.N = N
        self.add = AddUnit()

        self.direct_value_index_map_queue = []

        # stage1 get input and merge
        self.stage1_valid = False
        self.stage1_merge_index_value_map = {}
        self.stage1_evict_index_queue = []

        # stage2 evict values
        self.stage2_valid = False
        self.stage2_merge_index_value_map = {}

        # stage3 add
        self.stage3_valid = False
        self.output = {}

    def clock_cycle(
        self, valid, index_value_map_input1, index_value_map_input2, evict_index_queue):
        
        self.cycle_count += 1
        # stage3

        ## 对于一个add单元，它的输入index队列可能会有以下5种情况
        ## 1. None  2. AAB  3. AB  4. A  5. AA
        ## 无论是否用到加法器，我们都需要等待加法器的周期以保证同步
        ## 同时我们的output需要收集经过加法器和未经过加法器的index value对
        ## 我们在add.valid = True 时需要同时弹出之前保存的index 和 index value对
        ## 有两个queue 一个是 index queue 一个是 index value map queue
        ## 其中AAB情况下 这两个queue可以同步弹出
        ## 但在None AA A AB情况下，如果同步弹出出必定会导致有一个queue是没有的
        ## 所以需要在None AB A AA的情况下padding 用index=-2进行padding

         
        self.output = {}
        temp = {}
        add_used = False
        no_add_used = False

        case_AA = False
        case_AB_A = False
        case_AAB = False
        case_None = False

        # case None
        case_None = len(self.stage1_merge_index_value_map) == 0 
        if case_None:
            self.add.get_input(self.stage2_valid, 0, 0, -1) #也会包含stage2_valid=false的情况

        for _, values in self.stage2_merge_index_value_map.items():
            if len(values) == 2:
                add_used = True
            if len(values) == 1:
                no_add_used = True

        case_AA = (add_used) and (no_add_used == False)
        case_AB_A = (add_used == False) and (no_add_used)
        case_AAB = add_used and no_add_used
        for index, values in self.stage2_merge_index_value_map.items():
            assert len(values) <= 2
            if len(values) == 2:
                self.add.get_input(self.stage2_valid, values[0], values[1], index)
            elif len(values) == 1:
                if index not in temp:
                    temp[index] = []
                temp[index].append(values[0])#
                self.direct_value_index_map_queue.append(temp)
        # padding
        if self.stage2_valid:
            if case_AA or case_None:
                padding = {-2 : []}
                self.direct_value_index_map_queue.append(padding)
            elif case_AB_A:
                self.add.get_input(self.stage2_valid, 0, 0, -2)
            elif case_AAB:
                pass # do nothing

        result = self.add.clock_cycle()
        self.stage3_valid = self.add.valid
        if self.add.valid:
            index = self.add.index_queue.pop(0)
            index_value_map = self.direct_value_index_map_queue.pop(0)
            if index != -2:
                if index not in self.output:
                    self.output[index] = []
                self.output[index].append(result)
            if -2 not in index_value_map.keys():
                for index_s, values in index_value_map.items():
                    assert len(values) == 1
                    if index_s not in self.output:
                        self.output[index_s] = []
                    self.output[index_s].append(values[0])
        else:
            self.output = {}

        # stage2 evict values
        self.stage2_valid = self.stage1_valid
        self.stage2_merge_index_value_map = self.stage1_merge_index_value_map
        if self.stage1_valid:
            for evict_index in self.stage1_evict_index_queue:
                if evict_index in self.stage1_merge_index_value_map:
                    m = evict_index % self.M
                    n = int(evict_index / self.M)
                    assert len(self.stage1_merge_index_value_map[evict_index]) == 1
                    evict_val = self.stage1_merge_index_value_map.pop(evict_index)  # 获取列表中的值
                    if evict_index != -1:
                        self.c_values[m * self.N + n] += evict_val[0]

        # stage1
        self.stage1_valid = valid
        if valid:
            self.stage1_merge_index_value_map = index_value_map_input1.copy()

            for index, values in index_value_map_input2.items():
                if index in self.stage1_merge_index_value_map:
                    self.stage1_merge_index_value_map[index].extend(values)
                else:
                    self.stage1_merge_index_value_map[index] = values.copy()

            self.stage1_evict_index_queue = evict_index_queue
        else:
            self.stage1_evict_index_queue = []
            self.stage1_merge_index_value_map = {}

        return {
            "cycle": self.cycle_count,
            "valid": self.stage3_valid,
            "output": self.output if self.stage3_valid else None,
            "pipeline_state": self.get_pipeline_state(),
        }
        
    def reset(self):
        """重置流水线状态"""
        self.cycle_count = 0
        
        # 重置队列
        self.direct_value_index_map_queue = []
        
        # 重置第一阶段
        self.stage1_valid = False
        self.stage1_merge_index_value_map = {}
        self.stage1_evict_index_queue = []
        
        # 重置第二阶段
        self.stage2_valid = False
        self.stage2_merge_index_value_map = {}
        
        # 重置第三阶段
        self.stage3_valid = False
        self.output = {}
        
        # 重置加法单元
        self.add = AddUnit()

    def get_pipeline_state(self):
        """获取流水线的当前状态"""
        return {
            "cycle_count": self.cycle_count,
            "stage1": {
                "valid": self.stage1_valid,
                "merge_index_value_map": self.stage1_merge_index_value_map,
                "evict_index_queue": self.stage1_evict_index_queue
            },
            "stage2": {
                "valid": self.stage2_valid,
                "merge_index_value_map": self.stage2_merge_index_value_map
            },
            "stage3": {
                "valid": self.stage3_valid,
                "output": self.output
            },
            "add_unit": {
                "valid": self.add.valid if hasattr(self.add, 'valid') else False,
                "index_queue": self.add.index_queue if hasattr(self.add, 'index_queue') else []
            },
            "direct_value_index_map_queue": self.direct_value_index_map_queue
        }

    def print_state(self):
        """打印流水线的当前状态"""
        state = self.get_pipeline_state()
        
        print("\n==== AdvanceAddUnit 状态 (周期 {}) ====".format(state["cycle_count"]))
        
        # 打印阶段1状态
        print("\n[阶段1] 合并输入:")
        print("  有效: {}".format(state["stage1"]["valid"]))
        if state["stage1"]["valid"]:
            print("  合并索引-值映射:")
            for index, values in state["stage1"]["merge_index_value_map"].items():
                print(f"    索引 {index}: 值 {values}")
            print("  驱逐索引队列: {}".format(state["stage1"]["evict_index_queue"]))
        
        # 打印阶段2状态
        print("\n[阶段2] 驱逐值:")
        print("  有效: {}".format(state["stage2"]["valid"]))
        if state["stage2"]["valid"]:
            print("  合并索引-值映射:")
            for index, values in state["stage2"]["merge_index_value_map"].items():
                print(f"    索引 {index}: 值 {values}")
        
        # 打印阶段3状态
        print("\n[阶段3] 加法运算:")
        print("  有效: {}".format(state["stage3"]["valid"]))
        if state["stage3"]["valid"]:
            print("  输出:")
            for index, values in state["stage3"]["output"].items():
                print(f"    索引 {index}: 值 {values}")
        
        # 打印加法单元状态
        print("\n[加法单元]:")
        print("  有效: {}".format(state["add_unit"]["valid"]))
        print("  索引队列: {}".format(state["add_unit"]["index_queue"]))
        
        # 打印直接值索引映射队列
        print("\n[直接值索引映射队列]:")
        for idx, item in enumerate(state["direct_value_index_map_queue"]):
            print(f"  项目 {idx}: {item}")
        
        print("\n====================================")
    
    def run_pipeline(self, input_maps_pairs, evict_indices=None, max_cycles=20, print_states=False):
        """
        运行整个AdvanceAddUnit流水线，处理一系列输入并返回结果
        
        Args:
            input_maps_pairs: 列表，每个元素是(map1, map2)元组，表示每个时钟周期的两个输入映射
            evict_indices: 列表，每个元素是要驱逐的索引列表，对应每个输入对
            max_cycles: 最大运行周期数，防止无限循环
            print_states: 是否打印每个周期的状态
            
        Returns:
            results: 包含每个时钟周期输出的列表
        """
        # 重置流水线状态
        self.reset()
        
        # 初始化结果列表
        results = []
        
        # 如果没有提供驱逐索引，则使用空列表
        if evict_indices is None:
            evict_indices = [[] for _ in range(len(input_maps_pairs))]
        
        # 确保evict_indices长度匹配input_maps_pairs
        assert len(evict_indices) >= len(input_maps_pairs), "驱逐索引列表长度应不小于输入对列表长度"
        
        # 创建输入队列
        input_queue = list(zip(input_maps_pairs, evict_indices))
        input_idx = 0
        
        # 运行流水线，直到处理完所有输入并且没有更多有效数据
        cycle = 0
        while (input_idx < len(input_queue) or 
            self.is_active()) and cycle < max_cycles:
            
            # 获取当前周期的输入，如果有的话
            if input_idx < len(input_queue):
                (map1, map2), evict_list = input_queue[input_idx]
                valid = True
                input_idx += 1
            else:
                map1, map2, evict_list = {}, {}, []
                valid = False
            
            # 运行一个时钟周期
            result = self.clock_cycle(valid, map1, map2, evict_list)
            results.append(result)
            
            # 如果需要，打印当前状态
            if print_states:
                print(f"\n--- 周期 {cycle + 1} ---")
                self.print_state()
            
            cycle += 1
        
        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (input_idx < len(input_queue) or 
                                self.stage1_valid or self.stage2_valid or self.stage3_valid):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")
        
        return results
    
    def run_pipeline_with_bf16(self, input_maps_pairs, evict_indices=None, max_cycles=20, print_states=False):
        """
        运行流水线，自动将输入值转换为BF16格式
        
        Args:
            input_maps_pairs: 列表，每个元素是(map1, map2)元组，表示每个时钟周期的两个输入映射
            evict_indices: 列表，每个元素是要驱逐的索引列表
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态
            
        Returns:
            results: 包含每个时钟周期输出的列表
        """
        # 转换所有输入值为BF16格式
        converted_input_pairs = []
        
        for map1, map2 in input_maps_pairs:
            # 转换第一个映射
            converted_map1 = {}
            for index, values in map1.items():
                converted_values = [convert_through_pipeline(float(val)) for val in values]
                converted_map1[index] = converted_values
            
            # 转换第二个映射
            converted_map2 = {}
            for index, values in map2.items():
                converted_values = [convert_through_pipeline(float(val)) for val in values]
                converted_map2[index] = converted_values
            
            # 添加到转换后的列表
            converted_input_pairs.append((converted_map1, converted_map2))
        
        # 使用转换后的输入运行流水线
        return self.run_pipeline(converted_input_pairs, evict_indices, max_cycles, print_states)

    def is_active(self):
        """
        检查流水线是否仍在处理数据
        
        Returns:
            bool: 如果流水线中还有活跃的数据则返回True
        """
        return (self.stage1_valid or 
                self.stage2_valid or 
                self.stage3_valid or
                self.add.is_active())

class AddTree:
    def __init__(self, PE_num, c_values, M, N):
        self.cycle_count = 0
        
        self.PE_num = PE_num
        self.c_values = c_values
        self.M = M
        self.N = N
        self.tree = self.create_tree() 
        
        self.stage_valid_vec = [False] * (self.tree_levels + 1)
        self.stage_output_vec = [[] for _ in range(self.tree_levels + 1)] 
        self.stage_evict_vec = [[] for _ in range(self.tree_levels)]
        
    def create_tree(self):
        self.tree_levels = int(math.log2(self.PE_num))
        tree = []
        level_size = self.PE_num // 2
        for _ in range(0, self.tree_levels):
            tree.append(
                [AdvanceAddUnit(self.c_values, self.M, self.N) for _ in range(level_size)]
            )
            level_size = level_size // 2
        #tree.reverse()
        return tree
    
    def get_level_evict_index(self, map_queue):
        merge_index = []
        for m in map_queue:
            for index, _ in m.items():
                merge_index.append(index)
        evict_index = find_singles(merge_index)
        return evict_index
    
    def clock_cycle(self, valid, map_queue):
        
        # add.clock_cycle(self, valid, index_value_map_input1, index_value_map_input2, evict_index_queue):
        # return {
        #    "cycle": self.cycle_count,
        #    "valid": self.stage3_valid,
        #    "output": self.output if self.stage3_valid else None,
        #    "pipeline_state": self.get_pipeline_state(),
        #}
        self.cycle_count += 1
        for i in range(self.tree_levels - 1):
            pass
        
        for i in range(self.tree_levels - 1 , 0, -1):
            add_layer_valid = False
            for add in self.tree[i]:
                result = add.clock_cycle(
                self.stage_valid_vec[i - 1], 
                self.stage_output_vec[i - 1].pop(0) if self.stage_valid_vec[i - 1] else [],
                self.stage_output_vec[i - 1].pop(0) if self.stage_valid_vec[i - 1] else [],
                self.stage_evict_vec[i - 1]
                )
                if result["valid"]:
                    self.stage_output_vec[i].append(result["output"])
                    add_layer_valid = result["valid"]
            if add_layer_valid:
                self.stage_evict_vec[i] = self.get_level_evict_index(self.stage_output_vec[i])
            else:
                self.stage_evict_vec[i] = []
                self.stage_output_vec[i] = []
        
        self.stage_valid_vec[0] = valid
        if valid:
            self.stage_output_vec[0] = map_queue.copy()
            self.stage_evict_vec[0] = self.get_level_evict_index(self.stage_output_vec[0])
        else:
            self.stage_evict_vec[0] = []
            self.stage_output_vec[0] = []
                        
        

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
def test_MFIU_unit(A: np.array, B: np.array):
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

def test_MFIU():
    # 运行测试
    print("\n===== 测试用例1：简单矩阵 =====")
    A1 = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])

    B1 = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
    test_MFIU_unit(A1, B1)
    
    np.random.seed(42)
    A3 = np.random.choice([0, 1], size=(1, 64), p=[0, 1])
    B3 = np.random.choice([0, 1], size=(64, 10), p=[0.9, 0.1])
    test_MFIU_unit(A3, B3)

def AdvanceAdd_output_to_float(output:dict):
    new_output = {}
    for key, values in output.items():
        temp = [bf16_to_float(v) for v in values]
        new_output[key] = temp
    return new_output


def test_AdvanceAdd():
    # 示例用法
    adder = AdvanceAddUnit(c_values=[0]*4, M=2, N=2)

    # 准备输入数据
    input_pairs = [
        ({0: [5], -1: [5]}, {-1: [3]}),           # 周期1: 合并得到 {1: [5, 3]}
        ({2: [4]}, {3: [7]}),           # 周期2: {2: [4]}, {3: [7]}
        ({}, {})                        # 周期3: 空输入
    ]

    evict_indices = [
        [],                           # 周期1: 驱逐索引1
        [3],                           # 周期2: 驱逐索引2
        []                             # 周期3: 无驱逐
    ]

    # 运行流水线
    results = adder.run_pipeline_with_bf16(input_pairs, evict_indices, print_states=True)

    # 检查结果
    for i, res in enumerate(results):
        if res["valid"]:
            print(f"周期 {i+1} 输出: {AdvanceAdd_output_to_float(res['output'])}") 
    c_values_float = [bf16_to_float(c) for c in adder.c_values]
    print(c_values_float)

if __name__ == "__main__":
    #test_MFIU()
    test_AdvanceAdd()
