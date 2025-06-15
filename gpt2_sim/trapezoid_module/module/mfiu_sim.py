import numpy as np
from .shift_sim import ShiftUnitPipeline

class MFIUPipeline:
    def __init__(self, width=None, bit_width=None):
        # 基础配置
        # TODO width 其实可以是动态的
        self.width = width
        self.bit_width = bit_width
        #self.shift_unit_pipeline_vec_a = [
        #    ShiftUnitPipeline(bit_width) for _ in range(self.width)
        #]
        #self.shift_unit_pipeline_vec_b = [
        #    ShiftUnitPipeline(bit_width) for _ in range(self.width)
        #]

        # 批量处理
        self.shift_unit_pipeline_batch_process_a = ShiftUnitPipeline(bit_width)
        self.shift_unit_pipeline_batch_process_b = ShiftUnitPipeline(bit_width)

        # stage1 处理输入，并进行一些预处理

        # 预处理结果
        self.stage1_valid = False
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_A_bit_mask_vec = []
        self.stage1_B_bit_mask_vec = []
        self.stage1_A_row_offset_vec = []
        self.stage1_B_col_offset_vec = []
        self.stage1_AB_width = 0

        # stage2 B bitmask & A bitmask
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)

        self.stage2_len_values_A = 0
        self.stage2_len_values_B = 0
        self.stage2_A_bit_mask_vec = []
        self.stage2_B_bit_mask_vec = []
        self.stage2_A_row_offset_vec = []
        self.stage2_B_col_offset_vec = []
        self.stage2_AB_width = 0

        # stage3 prefix sum
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        self.stage3_bit_seq = np.array([], dtype=int)

        self.stage3_len_values_A = 0
        self.stage3_len_values_B = 0
        self.stage3_A_bit_mask_vec = []
        self.stage3_B_bit_mask_vec = []
        self.stage3_A_row_offset_vec = []
        self.stage3_B_col_offset_vec = []
        self.stage3_AB_width = 0

        # stage4 get ec_idx
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.stage3_AB_width)]

        self.stage4_len_values_A = 0
        self.stage4_len_values_B = 0
        self.stage4_A_bit_mask_vec = []
        self.stage4_B_bit_mask_vec = []
        self.stage4_A_row_offset_vec = []
        self.stage4_B_col_offset_vec = []

        # stage5 shift
        self.stage5_valid = False

        self.cycle_count = 0
        self.output = [[], []]

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
        
        self.output = [[], []]
        """
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
            if self.stage5_valid:
                self.output[0].append(results_a["output"])
                self.output[1].append(results_b["output"])
            else:
                self.output = ([], [])
        """
        result_a = self.shift_unit_pipeline_batch_process_a.clock_cycle(
            self.stage4_valid,
            self.stage4_A_bit_mask_vec,
            self.stage4_ec_idx_vec,
            self.stage4_len_values_A,
            self.stage4_A_row_offset_vec,
        )
        result_b = self.shift_unit_pipeline_batch_process_b.clock_cycle(
            self.stage4_valid,
            self.stage4_B_bit_mask_vec,
            self.stage4_ec_idx_vec,
            self.stage4_len_values_B,
            self.stage4_B_col_offset_vec,
        )
        self.stage5_valid = result_a["valid"]
        if self.stage5_valid:
            self.output[0] = result_a["output"]
            self.output[1] = result_b["output"]
        else:
            self.output = [[], []]

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
            )
            #for i in range(self.width):
            #    start_idx = i * self.bit_width
            #    end_idx = start_idx + self.bit_width
            #    self.stage4_ec_idx_vec[i] = ec_idx_seq[start_idx : end_idx].tolist()
            self.stage4_ec_idx_vec = [[] for _ in range(self.stage3_AB_width)]
            for i in range(self.stage3_AB_width):
                start_idx = i * self.bit_width
                end_idx = start_idx + self.bit_width
                self.stage4_ec_idx_vec[i] = ec_idx_seq[start_idx : end_idx].tolist()
        # stage3 prefix sum
        self.stage3_valid = self.stage2_valid
        self.stage3_len_values_A = self.stage2_len_values_A
        self.stage3_len_values_B = self.stage2_len_values_B
        self.stage3_A_bit_mask_vec = self.stage2_A_bit_mask_vec.copy()
        self.stage3_B_bit_mask_vec = self.stage2_B_bit_mask_vec.copy()
        self.stage3_A_row_offset_vec = self.stage2_A_row_offset_vec.copy()
        self.stage3_B_col_offset_vec = self.stage2_B_col_offset_vec.copy()
        self.stage3_bit_seq = self.stage2_bit_seq.copy()
        self.stage3_AB_width = self.stage2_AB_width
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
        self.stage2_AB_width = self.stage1_AB_width
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
            self.stage1_B_bit_mask_vec = []
            self.stage1_B_col_offset_vec = []
            self.stage1_A_bit_mask_vec = []
            self.stage1_A_row_offset_vec = []
            self.stage1_len_values_A = len_values_A
            self.stage1_len_values_B = len_values_B
            self.stage1_AB_width = len(mask_B_col) * len(mask_A_row)
            for i in range(len(mask_B_col)):
                for j in range(len(mask_A_row)):
                    self.stage1_B_bit_mask_vec.append(mask_B_col[i])
                    self.stage1_B_col_offset_vec.append(offset_B_col[i])
                    self.stage1_A_bit_mask_vec.append(mask_A_row[j])
                    self.stage1_A_row_offset_vec.append(offset_A_row[j])

        else:
            self.stage1_len_values_A = 0
            self.stage1_len_values_B = 0
            self.stage1_A_bit_mask_vec = []
            self.stage1_B_bit_mask_vec = []
            self.stage1_A_row_offset_vec = []
            self.stage1_B_col_offset_vec = []

        return {
            "cycle": self.cycle_count,
            "valid": self.stage5_valid,
            "output": self.output if self.stage5_valid else None,
            #"pipeline_state": self.get_pipeline_state(),
        }

    def is_active(self):
        if (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.stage4_valid
            or self.stage5_valid
        ):
            return True
        if self.shift_unit_pipeline_batch_process_a.is_active():
            return True
        if self.shift_unit_pipeline_batch_process_b.is_active():
            return True
        return False

    def reset(self, width=-1, bit_width=-1):
        """重置流水线状态"""
        # 重置时钟计数器
        self.cycle_count = 0

        if width ==-1 and bit_width==-1:
            pass
        else:
            self.width = width
            self.bit_width = width

        # 重置输出
        self.output = [[], []]

        # 重置stage1状态
        self.stage1_valid = False
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_A_bit_mask_vec = []
        self.stage1_B_bit_mask_vec = []
        self.stage1_A_row_offset_vec = []
        self.stage1_B_col_offset_vec = []
        self.stage1_AB_width = 0
        # 重置stage2状态
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)
        self.stage2_len_values_A = 0
        self.stage2_len_values_B = 0
        self.stage2_A_bit_mask_vec = []
        self.stage2_B_bit_mask_vec = []
        self.stage2_A_row_offset_vec = []
        self.stage2_B_col_offset_vec = []
        self.stage2_AB_width = 0
        # 重置stage3状态
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        self.stage3_bit_seq = np.array([], dtype=int)
        self.stage3_len_values_A = 0
        self.stage3_len_values_B = 0
        self.stage3_A_bit_mask_vec = []
        self.stage3_B_bit_mask_vec = []
        self.stage3_A_row_offset_vec = []
        self.stage3_B_col_offset_vec = []
        self.stage3_AB_width = 0
        # 重置stage4状态
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.stage3_AB_width)]
        self.stage4_len_values_A = 0
        self.stage4_len_values_B = 0
        self.stage4_A_bit_mask_vec = []
        self.stage4_B_bit_mask_vec = []
        self.stage4_A_row_offset_vec = []
        self.stage4_B_col_offset_vec = []
        self.stage4_AB_width = 0
        # 重置stage5状态
        self.stage5_valid = False

        # 重置所有ShiftUnitPipeline
        self.shift_unit_pipeline_batch_process_a.reset(self.bit_width)
        self.shift_unit_pipeline_batch_process_b.reset(self.bit_width)

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
                "a_count": 1,
                "b_count": 1,
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
        mask_A_rows,
        mask_B_cols,
        offset_A_rows,
        offset_B_cols,
        len_values_A,
        len_values_B,
        max_cycles=20,
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

        input_idx = 0
        cycle = 0

        while (input_idx < len(mask_A_rows) or self.is_active()) and cycle < max_cycles:

            if input_idx < len(mask_A_rows):
                mask_A_row = mask_A_rows[input_idx]
                mask_B_col = mask_B_cols[input_idx]
                offset_A_row = offset_A_rows[input_idx]
                offset_B_col = offset_B_cols[input_idx]
                valid = True
                input_idx += 1
            else:
                mask_A_row = []
                mask_B_col = []
                offset_A_row = []
                offset_B_col = []
                valid = False

            result = self.clock_cycle(valid, mask_A_row, mask_B_col, offset_A_row, offset_B_col, len_values_A, len_values_B)
            results.append(result)
            if print_states:
                print(f"\n--- 周期 {cycle + 1} ---")
                self.print_state()
            cycle += 1

        if cycle >= max_cycles and (
            self.is_active()
        ):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        return results
