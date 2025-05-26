import numpy as np
from scipy.sparse import csr_matrix

from .utils import FP32toBF16Pipeline, convert_through_pipeline, bf16_add, get_values_offset_mask, bf16_to_float, get_values_offset_mask_direct
from .module.mfiu_sim import MFIUPipeline 
from .module.add_tree_sim import AddTree
from .module.compute_sim import MultiplyUnit

class TrapezoidPipeline:
    def __init__(self, M, K, N, PE_num=4):
        # 默认配置
        self.width = M * N
        self.bit_width = K
        self.PE_num = PE_num
        self.M = M
        self.N = N
        self.K = K

        self.c_values = [0] * self.M * self.N

        self.mfiu = MFIUPipeline(self.width, self.bit_width)
        self.mul_vec = [MultiplyUnit() for _ in range(self.PE_num)]
        self.add_tree = AddTree(self.PE_num, self.c_values, self.M, self.N)

        self.values_A_queue = []
        self.values_B_queue = []

        self.start_index_queue = []
        self.nums_rows_queue = []

        # stage1: 预输入处理数据
        self.stage1_valid = False
        self.stage1_offset_A = np.array([])
        self.stage1_offset_B = np.array([])
        self.stage1_masks_A = []
        self.stage1_masks_B = []
        self.stage1_values_A = np.array([])
        self.stage1_values_B = np.array([])

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

        # stage4: 乘法单元
        self.stage4_valid = False
        self.stage4_input_map_queue = []

        # stage5: add tree
        self.stage5_valid = False

        self.cycle_count = 0

    def init_c_values(self):
        c_values = [0] * self.M * self.N
        for i in range(len(c_values)):
            c_values[i] = FP32toBF16Pipeline(c_values[i])
        self.c_values_bf16 = c_values

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

    # TODO: now only A is a dense vector
    def clock_cycle(self, valid, A: np.array, B: np.array, 
                    is_hbm=False, start_index=0,
                    values_B_input=[], col_indices_input=[], row_ptr_input=[] 
                    ):
        if is_hbm and A.shape[0] != 1:
            raise ValueError("如果是hbm传输模式, A需要是稠密的向量")

        self.cycle_count += 1

        # stage5 add tree
        result = self.add_tree.clock_cycle(
            self.stage4_valid, self.stage4_input_map_queue
        )
        self.stage5_valid = result["valid"]
        if result["valid"]:
            assert len(result["output"]) == 1
            if result["output"][0]:
                index = next(iter(result["output"][0]))
                value = result["output"][0][index][0]
                m = index % self.M
                n = int(index / self.M)
                if index != -1:
                    # self.c_values[m * self.N + n] += value
                    self.c_values[m * self.N + n] = bf16_add(
                        self.c_values[m * self.N + n], value
                    )

        # stage4 乘法单元
        self.stage4_input_map_queue = []
        for i, mul in enumerate(self.mul_vec):
            assert len(self.stage3_mul_queue_a[i]) == len(self.stage3_mul_queue_b[i])
            assert len(self.stage3_mul_queue_b[i]) == len(
                self.stage3_sft_index_queue[i]
            )

            mul.get_input(
                self.stage3_valid,
                self.stage3_mul_queue_a[i].pop(0) if self.stage3_valid else 0,
                self.stage3_mul_queue_b[i].pop(0) if self.stage3_valid else 0,
                self.stage3_sft_index_queue[i].pop(0) if self.stage3_valid else 0,
            )
            result = mul.clock_cycle()
            self.stage4_valid = mul.valid
            if mul.valid:
                index = mul.index_queue.pop(0)
                temp_map = {index: [result]}
                self.stage4_input_map_queue.append(temp_map)
            else:
                self.stage4_input_map_queue = []

        # stage3 获得输入队列
        # self.stage3_valid = self.stage2_valid
        # 事实上stage3_valid通过self.mul_queue的长度判断
        if self.stage2_valid:
            if len(self.start_index_queue):
                start_index_val = self.start_index_queue.pop(0)
            else:
                start_index_val = 0
                
            if len(self.nums_rows_queue):
                index_len = self.nums_rows_queue.pop(0)
            else:
                index_len = len(sft_index_b)
            
            sft_index_a = self.stage2_index[0]
            sft_index_b = self.stage2_index[1]
            for sft_index in range(index_len):
                sft_row_a = sft_index_a[sft_index]
                sft_row_b = sft_index_b[sft_index]
                for i in range(len(sft_row_a)):
                    if sft_row_a[i] != 0:
                        self.stage3_mul_queue_a[
                            (sft_row_a[i] - 1) % self.PE_num
                        ].append(self.stage2_values_A[i])

                for i in range(len(sft_row_b)):
                    if sft_row_b[i] != 0:
                        self.stage3_mul_queue_b[
                            (sft_row_b[i] - 1) % self.PE_num
                        ].append(self.stage2_values_B[i])
                       
                        self.stage3_sft_index_queue[
                            (sft_row_b[i] - 1) % self.PE_num
                        ].append(
                            sft_index + start_index_val
                        )  # A and B are same
            self.pad_queues()
        for i, _ in enumerate(self.mul_vec):
            self.stage3_valid = len(self.stage3_mul_queue_a[i]) > 0
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
            self.stage2_values_A = self.values_A_queue.pop(0)
            self.stage2_values_B = self.values_B_queue.pop(0)
        else:
            self.stage2_index = ([], [])
            self.stage2_values_A = np.array([])
            self.stage2_values_B = np.array([])

        # stage1: 预输入处理数据
        self.stage1_valid = valid
        if valid and not is_hbm:
            csr_A = csr_matrix(A)
            csr_B = csr_matrix(B.T)
            values_A, self.stage1_offset_A, self.stage1_masks_A = (
                get_values_offset_mask(csr_A)
            )
            self.stage1_values_A = values_A
            self.values_A_queue.append(values_A)
            values_B, self.stage1_offset_B, self.stage1_masks_B = (
                get_values_offset_mask(csr_B)
            )
            self.values_B_queue.append(values_B)
            self.stage1_values_B = values_B
        elif valid and is_hbm:
            csr_A = csr_matrix(A)
            values_A, self.stage1_offset_A, self.stage1_masks_A = (
                get_values_offset_mask(csr_A)
            )
            self.stage1_values_A = values_A
            self.values_A_queue.append(values_A)

            values_B, self.stage1_offset_B, self.stage1_masks_B, nums_rows = (
                get_values_offset_mask_direct(values_B_input, col_indices_input, row_ptr_input, self.K)
            )
            self.start_index_queue.append(start_index)
            self.values_B_queue.append(values_B)
            self.nums_rows_queue.append(nums_rows)
            self.stage1_values_B = values_B

        else:
            self.stage1_values_A = np.array([])
            self.stage1_values_B = np.array([])
            self.stage1_offset_A = np.array([])
            self.stage1_offset_B = np.array([])
            self.stage1_masks_A = []
            self.stage1_masks_B = []

        return {
            "cycle": self.cycle_count,
            "valid": self.stage5_valid,
            "output": self.c_values if self.stage5_valid else None,
        }

    def is_active(self):
        """检查流水线是否仍在活跃处理数据"""
        # 检查各阶段是否有效
        if (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.stage4_valid
            or self.stage5_valid
        ):
            return True
        if self.add_tree.is_active():
            return True
        if self.mfiu.is_active():
            return True
        for mul in self.mul_vec:
            if mul.is_active():
                return True
        return False

    def reset(self):
        """重置TrapezoidPipeline的所有状态"""
        # 重置周期计数
        self.cycle_count = 0

        # 重置C矩阵值
        self.c_values = [0] * self.M * self.N

        # 重置各子模块
        self.mfiu.reset()
        for mac in self.mul_vec:
            mac.multiply_pipeline.reset()
            mac.input1 = 0
            mac.input2 = 0
            mac.index_queue = []
            mac.valid = False
            mac.input_valid = False

        self.add_tree.reset()

        self.values_A_queue = []
        self.values_B_queue = []
        self.start_index_queue = []

        # 重置stage1状态
        self.stage1_valid = False
        self.stage1_values_A = np.array([])
        self.stage1_values_B = np.array([])
        self.stage1_offset_A = np.array([])
        self.stage1_offset_B = np.array([])
        self.stage1_masks_A = []
        self.stage1_masks_B = []

        # 重置stage2状态
        self.stage2_valid = False
        self.stage2_index = ([], [])
        self.stage2_values_A = np.array([])
        self.stage2_values_B = np.array([])

        # 重置stage3状态
        self.stage3_valid = False
        self.stage3_mul_queue_a = [[] for _ in range(self.PE_num)]
        self.stage3_mul_queue_b = [[] for _ in range(self.PE_num)]
        self.stage3_sft_index_queue = [[] for _ in range(self.PE_num)]

        # 重置stage4状态
        self.stage4_valid = False
        self.stage4_input_map_queue = []

        # 重置stage5状态
        self.stage5_valid = False

    def get_pipeline_state(self):
        """获取TrapezoidPipeline的当前状态"""
        # 收集各个乘法单元的状态
        mac_states = []
        for i, mac in enumerate(self.mul_vec):
            mac_states.append(
                {
                    "index": i,
                    "valid": mac.valid,
                    "input_valid": mac.input_valid,
                    "index_queue_length": len(mac.index_queue),
                    "pipeline_active": mac.is_active(),
                }
            )

        # 提取当前各阶段队列的大小和状态
        stage3_queue_sizes = []
        for i in range(self.PE_num):
            stage3_queue_sizes.append(
                {
                    "PE": i,
                    "mul_queue_a_size": len(self.stage3_mul_queue_a[i]),
                    "mul_queue_b_size": len(self.stage3_mul_queue_b[i]),
                    "sft_index_queue_size": len(self.stage3_sft_index_queue[i]),
                }
            )

        # 汇总整个流水线的状态
        return {
            "cycle_count": self.cycle_count,
            "configuration": {
                "M": self.M,
                "K": self.K,
                "N": self.N,
                "PE_num": self.PE_num,
                "width": self.width,
                "bit_width": self.bit_width,
            },
            "stages": {
                "stage1": {
                    "valid": self.stage1_valid,
                    "values_A_size": len(self.stage1_values_A),
                    "values_B_size": len(self.stage1_values_B),
                    "masks_A_size": len(self.stage1_masks_A),
                    "masks_B_size": len(self.stage1_masks_B),
                },
                "stage2": {
                    "valid": self.stage2_valid,
                    "index_size": (
                        (len(self.stage2_index[0]), len(self.stage2_index[1]))
                        if self.stage2_index
                        else (0, 0)
                    ),
                    "values_A_size": len(self.stage2_values_A),
                    "values_B_size": len(self.stage2_values_B),
                },
                "stage3": {
                    "valid": self.stage3_valid,
                    "queue_sizes": stage3_queue_sizes,
                },
                "stage4": {
                    "valid": self.stage4_valid,
                    "input_map_queue_size": len(self.stage4_input_map_queue),
                },
                "stage5": {"valid": self.stage5_valid},
            },
            "components": {
                "mac_units": mac_states,
                "mfiu_state": self.mfiu.get_pipeline_state(),
                "add_tree_state": self.add_tree.get_pipeline_state(),
            },
            "result": {"c_values_non_zero": sum(1 for v in self.c_values if v != 0)},
        }

    def print_state(self):
        """打印TrapezoidPipeline的当前状态"""
        state = self.get_pipeline_state()

        print(f"\n====== TrapezoidPipeline 状态 (周期 {state['cycle_count']}) ======")
        print(
            f"配置: M={state['configuration']['M']}, K={state['configuration']['K']}, "
            f"N={state['configuration']['N']}, PE数量={state['configuration']['PE_num']}"
        )

        # 打印各阶段状态
        print("\n--- 流水线阶段状态 ---")

        print(
            f"Stage 1 (输入处理): {'有效' if state['stages']['stage1']['valid'] else '无效'}"
        )
        if state["stages"]["stage1"]["valid"]:
            print(f"  A矩阵值数量: {state['stages']['stage1']['values_A_size']}")
            print(f"  B矩阵值数量: {state['stages']['stage1']['values_B_size']}")
            print(f"  A矩阵掩码数量: {state['stages']['stage1']['masks_A_size']}")
            print(f"  B矩阵掩码数量: {state['stages']['stage1']['masks_B_size']}")

        print(
            f"Stage 2 (MFIU): {'有效' if state['stages']['stage2']['valid'] else '无效'}"
        )
        if state["stages"]["stage2"]["valid"]:
            print(
                f"  索引大小: A={state['stages']['stage2']['index_size'][0]}, B={state['stages']['stage2']['index_size'][1]}"
            )
            print(f"  A矩阵值数量: {state['stages']['stage2']['values_A_size']}")
            print(f"  B矩阵值数量: {state['stages']['stage2']['values_B_size']}")

        print(
            f"Stage 3 (输入队列): {'有效' if state['stages']['stage3']['valid'] else '无效'}"
        )
        if state["stages"]["stage3"]["valid"]:
            print("  PE输入队列大小:")
            for pe in state["stages"]["stage3"]["queue_sizes"]:
                print(
                    f"    PE{pe['PE']}: A队列={pe['mul_queue_a_size']}, B队列={pe['mul_queue_b_size']}, 索引队列={pe['sft_index_queue_size']}"
                )

        print(
            f"Stage 4 (乘法单元): {'有效' if state['stages']['stage4']['valid'] else '无效'}"
        )
        if state["stages"]["stage4"]["valid"]:
            print(
                f"  输入映射队列大小: {state['stages']['stage4']['input_map_queue_size']}"
            )

        print(
            f"Stage 5 (加法树): {'有效' if state['stages']['stage5']['valid'] else '无效'}"
        )

        # 打印组件状态摘要
        print("\n--- 组件状态摘要 ---")

        # MAC单元状态
        active_macs = sum(
            1
            for mac in state["components"]["mac_units"]
            if mac["valid"] or mac["pipeline_active"]
        )
        print(f"乘法单元: {active_macs}/{len(state['components']['mac_units'])} 活跃")

        # MFIU状态
        mfiu_active = any(
            state["components"]["mfiu_state"][f"stage{i}"]["valid"] for i in range(1, 6)
        )
        print(f"MFIU: {'活跃' if mfiu_active else '空闲'}")

        # 加法树状态
        tree_active = any(state["components"]["add_tree_state"]["stage_valid"])
        print(f"加法树: {'活跃' if tree_active else '空闲'}")

        # 结果矩阵状态
        print(
            f"\n结果矩阵: {state['result']['c_values_non_zero']}/{self.M * self.N} 非零元素"
        )

        # 如果需要查看C矩阵的非零元素，取决于大小是否合理显示
        if self.M <= 8 and self.N <= 8:
            c_matrix = np.array(self.c_values).reshape(self.M, self.N)
            print("\nC矩阵值 (BF16格式):")
            print(c_matrix)

            # 转换为浮点数显示
            c_matrix_float = np.array(
                [bf16_to_float(v) for v in self.c_values]
            ).reshape(self.M, self.N)
            print("\nC矩阵值 (浮点数):")
            print(c_matrix_float)

        print("=====================================")

    def run_pipeline(self, input_matrices, max_cycles=100, print_states=False):
        """
        运行整个Trapezoid流水线处理一组输入矩阵

        Args:
            input_matrices: 列表，每个元素是(A矩阵, B矩阵)元组，表示每个时钟周期的输入
            max_cycles: 最大运行周期数，防止无限循环
            print_states: 是否打印每个周期的状态

        Returns:
            results: 包含每个时钟周期输出的列表
            final_c_values: 最终的C矩阵值
        """
        # 重置流水线状态
        # self.reset()

        # 初始化结果列表
        results = []

        # 创建输入队列
        input_idx = 0

        # 运行流水线，直到处理完所有输入并且没有更多有效数据
        cycle = 0
        while (
            input_idx < len(input_matrices) or self.is_active()
        ) and cycle < max_cycles:
            # 获取当前周期的输入，如果有的话
            if input_idx < len(input_matrices):
                A, B = input_matrices[input_idx]
                valid = True
                input_idx += 1
            else:
                A, B = np.array([]), np.array([])
                valid = False

            # 运行一个时钟周期
            result = self.clock_cycle(valid, A, B)
            results.append(result)

            # 如果需要，打印当前状态
            if print_states and (
                cycle % 10 == 0 or cycle < 5 or cycle >= len(input_matrices) - 3
            ):
                print(f"\n--- 周期 {cycle + 1} ---")
                self.print_state()

            cycle += 1

        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (
            input_idx < len(input_matrices) or self.is_active()
        ):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        # 转换结果矩阵为浮点数
        final_c_matrix = np.array([bf16_to_float(v) for v in self.c_values]).reshape(
            self.M, self.N
        )

        return {
            "results": results,
            "cycles": cycle,
            "c_matrix": final_c_matrix,
            "c_values_bf16": self.c_values.copy(),
        }

    def run_pipeline_with_bf16(
        self, input_matrices, max_cycles=1000, print_states=False
    ):
        """
        运行Trapezoid流水线，将输入矩阵转换为BF16格式

        Args:
            input_matrices: 列表，每个元素是(A矩阵, B矩阵)元组
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态

        Returns:
            结果字典，包含运行结果和最终矩阵
        """
        # 将所有输入转换为BF16格式
        bf16_input_matrices = []

        for A, B in input_matrices:
            # 转换A矩阵
            A_bf16 = np.zeros_like(A)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] != 0:
                        A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))

            # 转换B矩阵
            B_bf16 = np.zeros_like(B)
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    if B[i, j] != 0:
                        B_bf16[i, j] = convert_through_pipeline(float(B[i, j]))

            bf16_input_matrices.append((A_bf16, B_bf16))

        # 运行流水线
        return self.run_pipeline(bf16_input_matrices, max_cycles, print_states)

    def run_pipeline_hbm(
        self, A_matrices, B_data_list, max_cycles=100, print_states=False
    ):
        """
        运行整个Trapezoid流水线处理HBM格式的输入矩阵
        
        Args:
            A_matrices: 稠密向量A的列表，每个元素是一个形状为(1, K)的numpy数组
            B_data_list: 列表，每个元素是包含B矩阵CSR格式的字典，格式为
                        {"values": [...], "col_indices": [...], "row_ptr": [...], "row_start_index": int}
            max_cycles: 最大运行周期数，防止无限循环
            print_states: 是否打印每个周期的状态
            
        Returns:
            结果字典，包含每个时钟周期输出和最终矩阵
        """
        # 重置流水线状态
        # self.reset()

        # 初始化结果列表
        results = []

        # 验证输入
        if not all(A.shape[0] == 1 for A in A_matrices):
            raise ValueError("使用HBM模式时，所有A矩阵必须是稠密向量(形状为(1, K))")

        if len(A_matrices) != len(B_data_list):
            raise ValueError(f"A矩阵数量({len(A_matrices)})与B数据数量({len(B_data_list)})不匹配")

        # 创建输入队列
        input_idx = 0

        # 运行流水线，直到处理完所有输入并且没有更多有效数据
        cycle = 0
        while (input_idx < len(B_data_list) or self.is_active()) and cycle < max_cycles:
            # 获取当前周期的输入，如果有的话
            if input_idx < len(B_data_list):
                # 总是获取A矩阵
                A = A_matrices[0] 

                # 从B数据列表获取当前B的CSR格式数据
                B_data = B_data_list[input_idx]
                values_B = B_data.get("values", [])
                col_indices = B_data.get("col_indices", [])
                row_ptr = B_data.get("row_ptr", [])
                start_index = B_data.get("row_start_index", 0)

                valid = True
                input_idx += 1
            else:
                # 没有更多输入
                A = np.array([[]])
                values_B = []
                col_indices = []
                row_ptr = []
                start_index = 0
                valid = False

            # 运行一个时钟周期，使用HBM模式
            result = self.clock_cycle(
                valid=valid, 
                A=A, 
                B=np.array([]),  # 在HBM模式下B矩阵为空
                is_hbm=True, 
                start_index=start_index,
                values_B_input=values_B, 
                col_indices_input=col_indices, 
                row_ptr_input=row_ptr
            )
            results.append(result)

            # 如果需要，打印当前状态
            if print_states and (cycle % 10 == 0 or cycle < 5 or cycle >= len(A_matrices) - 3):
                print(f"\n--- 周期 {cycle + 1} (HBM模式) ---")
                self.print_state()

            cycle += 1

        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (input_idx < len(A_matrices) or self.is_active()):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        # 转换结果矩阵为浮点数
        final_c_matrix = np.array([bf16_to_float(v) for v in self.c_values]).reshape(
            self.M, self.N
        )

        return {
            "results": results,
            "cycles": cycle,
            "c_matrix": final_c_matrix,
            "c_values_bf16": self.c_values.copy(),
        }

    def run_pipeline_hbm_with_bf16(
        self, A_matrices, B_data_list, max_cycles=1000, print_states=False
    ):
        """
        运行Trapezoid流水线，将HBM格式输入转换为BF16格式
        
        Args:
            A_matrices: 稠密向量A的列表，每个元素是一个形状为(1, K)的numpy数组
            B_data_list: 列表，每个元素是包含B矩阵CSR格式的字典
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态
            
        Returns:
            结果字典，包含运行结果和最终矩阵
        """
        # 将所有A矩阵转换为BF16格式
        bf16_A_matrices = []
        bf16_B_data_list = []

        for B_data in B_data_list:
            # 转换A矩阵
            A = A_matrices[0]
            A_bf16 = np.zeros_like(A)
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    if A[i, j] != 0:
                        A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
            bf16_A_matrices.append(A_bf16)

            # 转换B矩阵的values
            values_B = B_data.get("values", [])
            bf16_values_B = []
            for val in values_B:
                if val != 0:
                    bf16_values_B.append(convert_through_pipeline(float(val)))
                else:
                    bf16_values_B.append(0)

            # 创建新的B数据字典，保持col_indices和row_ptr不变
            bf16_B_data = {
                "values": bf16_values_B,
                "col_indices": B_data.get("col_indices", []),
                "row_ptr": B_data.get("row_ptr", []),
                "row_start_index": B_data.get("row_start_index", 0)
            }
            bf16_B_data_list.append(bf16_B_data)

        # 运行流水线
        return self.run_pipeline_hbm(bf16_A_matrices, bf16_B_data_list, max_cycles, print_states)
