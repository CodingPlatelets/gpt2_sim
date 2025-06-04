import numpy as np
from .shift_sim import ShiftUnitPipeline

class MFIUPipelineDenseA:
    """优化版MFIU流水线，专门针对稠密A矩阵（mask全为1）"""
    
    def __init__(self, width=None, bit_width=None):
        self.width = width
        self.bit_width = bit_width
        
        # 稠密A矩阵的预计算缓存
        self.dense_A_mask = (1 << bit_width) - 1  # 全1的mask
        self.dense_A_precomputed = True
        
        # 其他初始化保持不变
        self.shift_unit_pipeline_vec_a = [ShiftUnitPipeline() for _ in range(width)]
        self.shift_unit_pipeline_vec_b = [ShiftUnitPipeline() for _ in range(width)]
        
        # 各个阶段的状态
        self.stage1_valid = False
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_B_bit_mask_vec = [0] * self.width
        self.stage1_B_col_offset_vec = [0] * self.width
        # A相关的向量可以预计算，因为都是全1
        
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)
        
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.width)]
        
        self.stage5_valid = False
        self.cycle_count = 0
        self.output = ([], [])

    def clock_cycle(
        self,
        valid,
        mask_B_col,
        offset_B_col,
        len_values_A,
        len_values_B,
    ):
        """优化版clock_cycle，假设A矩阵是稠密的"""
        
        self.cycle_count += 1

        # stage5 shift - 保持不变
        self.output = ([], [])
        for i, (shift_unit_a, shift_unit_b) in enumerate(
            zip(self.shift_unit_pipeline_vec_a, self.shift_unit_pipeline_vec_b)
        ):
            # A矩阵相关的处理简化
            results_a = shift_unit_a.clock_cycle(
                self.stage4_valid,
                self.dense_A_mask,  # 使用预计算的全1 mask
                self.stage4_ec_idx_vec[i],
                self.stage4_len_values_A,
                0,  # A矩阵offset始终为0（稠密）
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

        # stage4 get ec_idx - 保持不变
        self.stage4_valid = self.stage3_valid
        self.stage4_len_values_A = self.stage3_len_values_A
        self.stage4_len_values_B = self.stage3_len_values_B
        self.stage4_B_bit_mask_vec = self.stage3_B_bit_mask_vec.copy()
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

        # stage3 prefix sum - 保持不变
        self.stage3_valid = self.stage2_valid
        self.stage3_len_values_A = self.stage2_len_values_A
        self.stage3_len_values_B = self.stage2_len_values_B
        self.stage3_B_bit_mask_vec = self.stage2_B_bit_mask_vec.copy()
        self.stage3_B_col_offset_vec = self.stage2_B_col_offset_vec.copy()
        self.stage3_bit_seq = self.stage2_bit_seq.copy()
        if self.stage2_valid:
            self.stage3_AB_prefix_sum = np.cumsum(self.stage2_bit_seq).tolist()

        # stage2 优化：A为全1，直接使用B的mask
        self.stage2_valid = self.stage1_valid
        self.stage2_len_values_A = self.stage1_len_values_A
        self.stage2_len_values_B = self.stage1_len_values_B
        self.stage2_B_bit_mask_vec = self.stage1_B_bit_mask_vec.copy()
        self.stage2_B_col_offset_vec = self.stage1_B_col_offset_vec.copy()
        
        if self.stage1_valid:
            # 优化：A为全1，所以 A & B = B
            all_bits = []
            for b_mask in self.stage1_B_bit_mask_vec:
                for i in range(self.bit_width):
                    bit = (b_mask >> (self.bit_width - 1 - i)) & 1
                    all_bits.append(bit)
            self.stage2_bit_seq = np.array(all_bits)

        # stage1 优化：只处理B相关数据
        self.stage1_valid = valid
        if valid:
            self.stage1_len_values_A = len_values_A
            self.stage1_len_values_B = len_values_B
            
            # 优化：只填充B相关的向量，A相关的都是预知的
            for i in range(len(mask_B_col)):
                self.stage1_B_bit_mask_vec[i] = mask_B_col[i]
                self.stage1_B_col_offset_vec[i] = offset_B_col[i]
                
           
        else:
            self.stage1_len_values_A = 0
            self.stage1_len_values_B = 0
            self.stage1_B_bit_mask_vec = [0] * self.width
            self.stage1_B_col_offset_vec = [0] * self.width

        return {
            "cycle": self.cycle_count,
            "valid": self.stage5_valid,
            "output": self.output if self.stage5_valid else None,
            "pipeline_state": self.get_pipeline_state(),
        }

    def is_active(self):
        """检查流水线是否活跃"""
        if (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.stage4_valid
            or self.stage5_valid
        ):
            return True
        for shift in self.shift_unit_pipeline_vec_a:
            if shift.is_active():
                return True
        for shift in self.shift_unit_pipeline_vec_b:
            if shift.is_active():
                return True
        return False

    def reset(self):
        """重置流水线状态"""
        self.cycle_count = 0
        self.output = ([], [])
        
        # 重置各阶段状态
        self.stage1_valid = False
        self.stage1_len_values_A = 0
        self.stage1_len_values_B = 0
        self.stage1_B_bit_mask_vec = [0] * self.width
        self.stage1_B_col_offset_vec = [0] * self.width
        
        self.stage2_valid = False
        self.stage2_bit_seq = np.array([], dtype=int)
        self.stage2_len_values_A = 0
        self.stage2_len_values_B = 0
        self.stage2_B_bit_mask_vec = [0] * self.width
        self.stage2_B_col_offset_vec = [0] * self.width
        
        self.stage3_valid = False
        self.stage3_AB_prefix_sum = np.array([], dtype=int)
        self.stage3_bit_seq = np.array([], dtype=int)
        self.stage3_len_values_A = 0
        self.stage3_len_values_B = 0
        self.stage3_B_bit_mask_vec = [0] * self.width
        self.stage3_B_col_offset_vec = [0] * self.width
        
        self.stage4_valid = False
        self.stage4_ec_idx_vec = [[] for _ in range(self.width)]
        self.stage4_len_values_A = 0
        self.stage4_len_values_B = 0
        self.stage4_B_bit_mask_vec = [0] * self.width
        self.stage4_B_col_offset_vec = [0] * self.width
        
        self.stage5_valid = False
        
        # 重置所有ShiftUnitPipeline
        for shift_unit in self.shift_unit_pipeline_vec_a:
            shift_unit.reset()
        for shift_unit in self.shift_unit_pipeline_vec_b:
            shift_unit.reset()

    def get_pipeline_state(self):
        """返回当前流水线状态"""
        return {
            "stage1": {"valid": self.stage1_valid, "dense_A_optimized": True},
            "stage2": {"valid": self.stage2_valid},
            "stage3": {"valid": self.stage3_valid},
            "stage4": {"valid": self.stage4_valid},
            "stage5": {"valid": self.stage5_valid},
        } 

    def run_pipeline(
        self,
        mask_B_cols,
        offset_B_cols,
        len_values_A,
        len_values_B,
        max_cycles=20,
        print_states=False,
    ):
        """
        运行优化的MFIU流水线，专门处理稠密A矩阵
        
        Args:
            mask_B_cols: B矩阵列的掩码列表
            offset_B_cols: B矩阵列的偏移量列表
            len_values_A: A矩阵值的长度
            len_values_B: B矩阵值的长度
            max_cycles: 最大周期数
            print_states: 是否打印状态
        
        Returns:
            results: 包含每个周期输出的列表
        """
        # 重置状态
        self.reset()
        
        results = []
        input_idx = 0
        cycle = 0
        
        while (input_idx < len(mask_B_cols) or self.is_active()) and cycle < max_cycles:
            
            if input_idx < len(mask_B_cols):
                mask_B_col = mask_B_cols[input_idx]
                offset_B_col = offset_B_cols[input_idx]
                valid = True
                input_idx += 1
            else:
                mask_B_col = []
                offset_B_col = []
                valid = False

            result = self.clock_cycle(
                valid, mask_B_col, offset_B_col, len_values_A, len_values_B
            )
            results.append(result)
            
            if print_states:
                print(f"\n--- 周期 {cycle + 1} ---")
                self.print_state()
            
            cycle += 1

        if cycle >= max_cycles and (input_idx < len(mask_B_cols) or self.is_active()):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        return results

    def print_state(self):
        """打印流水线当前状态"""
        state = self.get_pipeline_state()
        print(f"\n==== MFIU Dense A Pipeline State (Cycle {self.cycle_count}) ====")
        print(f"Configuration: width={self.width}, bit_width={self.bit_width}")
        print(f"Dense A Optimization: ENABLED")

        print(f"Stage 1 (Input): {'Valid' if state['stage1']['valid'] else 'Invalid'}")
        if state["stage1"]["valid"]:
            print(f"  Values length: A={self.stage1_len_values_A}, B={self.stage1_len_values_B}")
            print(f"  B bit mask sample: {self.stage1_B_bit_mask_vec[:min(3, len(self.stage1_B_bit_mask_vec))]}...")

        print(f"Stage 2 (AND-Optimized): {'Valid' if state['stage2']['valid'] else 'Invalid'}")
        if state["stage2"]["valid"] and len(self.stage2_bit_seq) > 0:
            print(f"  Bit sequence length: {len(self.stage2_bit_seq)}")
            print(f"  Bit sequence sample: {self.stage2_bit_seq[:min(10, len(self.stage2_bit_seq))]}...")

        print(f"Stage 3 (PrefixSum): {'Valid' if state['stage3']['valid'] else 'Invalid'}")
        if state["stage3"]["valid"] and len(self.stage3_AB_prefix_sum) > 0:
            print(f"  Prefix sum length: {len(self.stage3_AB_prefix_sum)}")
            print(f"  Prefix sum sample: {self.stage3_AB_prefix_sum[:min(10, len(self.stage3_AB_prefix_sum))]}...")

        print(f"Stage 4 (EcIdx): {'Valid' if state['stage4']['valid'] else 'Invalid'}")
        if state["stage4"]["valid"] and len(self.stage4_ec_idx_vec) > 0:
            print(f"  EC index vector length: {len(self.stage4_ec_idx_vec)}")
            print(f"  EC index vector sample: {self.stage4_ec_idx_vec[:min(3, len(self.stage4_ec_idx_vec))]}...")

        print(f"Stage 5 (Shift): {'Valid' if state['stage5']['valid'] else 'Invalid'}")
        if state["stage5"]["valid"]:
            print(f"  Output length: A={len(self.output[0])}, B={len(self.output[1])}")

        print(f"Shift Units: {len(self.shift_unit_pipeline_vec_a)} A units, {len(self.shift_unit_pipeline_vec_b)} B units")
        print("=====================================")

    def get_pipeline_state(self):
        """返回当前流水线状态的详细信息"""
        return {
            "stage1": {
                "valid": self.stage1_valid,
                "dense_A_optimized": True,
                "len_values_A": self.stage1_len_values_A,
                "len_values_B": self.stage1_len_values_B,
                "B_bit_mask_sample": self.stage1_B_bit_mask_vec[:min(3, len(self.stage1_B_bit_mask_vec))],
            },
            "stage2": {
                "valid": self.stage2_valid,
                "bit_seq_len": len(self.stage2_bit_seq) if hasattr(self, 'stage2_bit_seq') else 0,
                "bit_seq_sample": self.stage2_bit_seq[:min(10, len(self.stage2_bit_seq))] if hasattr(self, 'stage2_bit_seq') and len(self.stage2_bit_seq) > 0 else None,
            },
            "stage3": {
                "valid": self.stage3_valid,
                "prefix_sum_len": len(self.stage3_AB_prefix_sum) if hasattr(self, 'stage3_AB_prefix_sum') else 0,
                "prefix_sum_sample": self.stage3_AB_prefix_sum[:min(10, len(self.stage3_AB_prefix_sum))] if hasattr(self, 'stage3_AB_prefix_sum') and len(self.stage3_AB_prefix_sum) > 0 else None,
            },
            "stage4": {
                "valid": self.stage4_valid,
                "ec_idx_vec_len": len(self.stage4_ec_idx_vec),
                "ec_idx_vec_sample": self.stage4_ec_idx_vec[:min(3, len(self.stage4_ec_idx_vec))] if self.stage4_ec_idx_vec else [],
            },
            "stage5": {
                "valid": self.stage5_valid,
                "output_a_len": len(self.output[0]) if self.output and len(self.output) > 0 else 0,
                "output_b_len": len(self.output[1]) if self.output and len(self.output) > 1 else 0,
            },
            "shift_units": {
                "a_count": len(self.shift_unit_pipeline_vec_a),
                "b_count": len(self.shift_unit_pipeline_vec_b),
            },
        } 