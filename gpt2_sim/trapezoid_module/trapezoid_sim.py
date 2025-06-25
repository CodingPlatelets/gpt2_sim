import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from collections import deque

from .utils import FP32toBF16Pipeline, convert_through_pipeline, bf16_add, bf16_add_list, get_values_offset_mask, bf16_to_float, get_values_offset_mask_direct
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

        self.values_A_queue = deque()
        self.values_B_queue = deque()

        self.start_index_queue = deque()
        self.nums_rows_queue = deque()

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
        self.stage3_mul_queue_a = [deque() for _ in range(self.PE_num)]
        self.stage3_mul_queue_b = [deque() for _ in range(self.PE_num)]
        self.stage3_sft_index_queue = [deque() for _ in range(self.PE_num)]

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
                self.stage3_mul_queue_a[i].popleft() if self.stage3_valid else 0,
                self.stage3_mul_queue_b[i].popleft() if self.stage3_valid else 0,
                self.stage3_sft_index_queue[i].popleft() if self.stage3_valid else 0,
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
                start_index_val = self.start_index_queue.popleft()
            else:
                start_index_val = 0
            
            sft_index_a = self.stage2_index[0]
            sft_index_b = self.stage2_index[1]

            #assert len(sft_index_a) == len(sft_index_b)

            #if len(self.nums_rows_queue):
            #    index_len = self.nums_rows_queue.popleft()
            #else:
            #    index_len = len(sft_index_b)
            index_len = len(sft_index_b)
            for sft_index in range(index_len):
                sft_row_a = sft_index_a[sft_index]
                sft_row_b = sft_index_b[sft_index]
                #assert len(sft_row_a) == len(self.stage2_values_A) 
                #assert len(sft_row_b) == len(self.stage2_values_B)

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
            self.stage2_values_A = self.values_A_queue.popleft()
            self.stage2_values_B = self.values_B_queue.popleft()
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

    def reset(self, M=-1, K=-1, N=-1):
        """重置TrapezoidPipeline的所有状态并根据需要更新"""
        # 重置周期计数
        self.cycle_count = 0

        # 设置M, K, N
        if M == -1 and K == -1 and N == -1:
            pass
        else:
            self.M = M
            self.N = N
            self.K = K

        self.width = self.M * self.N
        self.bit_width = self.K

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

        self.add_tree.reset(self.M, self.N, self.c_values)
        self.mfiu.reset(self.width, self.bit_width)

        # 重置为deque
        self.values_A_queue = deque()
        self.values_B_queue = deque()
        self.start_index_queue = deque()
        self.nums_rows_queue = deque()

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
        self.stage3_mul_queue_a = [deque() for _ in range(self.PE_num)]
        self.stage3_mul_queue_b = [deque() for _ in range(self.PE_num)]
        self.stage3_sft_index_queue = [deque() for _ in range(self.PE_num)]

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

        # 估算总周期数：输入数据 + 流水线深度的缓冲
        estimated_cycles = len(input_matrices) + 20  # 20是估算的流水线深度
        
        # 创建进度条
        with tqdm(total=estimated_cycles, desc="流水线处理", unit="cycle") as pbar:
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

                # 更新进度条
                cycle += 1
                
                # 动态更新进度条描述
                if input_idx < len(input_matrices):
                    pbar.set_description(f"流水线处理 (输入 {input_idx}/{len(input_matrices)})")
                else:
                    pbar.set_description(f"流水线处理 (排空中)")
                
                # 如果超出估算周期数，扩展进度条
                if cycle >= pbar.total:
                    pbar.total = cycle + 10
                    pbar.refresh()
                
                pbar.update(1)

                # 如果需要，打印当前状态
                if print_states and (
                    cycle % 10 == 0 or cycle < 5 or cycle >= len(input_matrices) - 3
                ):
                    # 暂时禁用进度条输出，打印状态，然后重新启用
                    pbar.write(f"\n--- 周期 {cycle} ---")
                    # 将状态信息写入到tqdm的输出中，避免与进度条冲突
                    state_info = self.get_pipeline_state()
                    pbar.write(f"流水线状态: 活跃阶段数 {sum(1 for stage in ['stage1', 'stage2', 'stage3', 'stage4', 'stage5'] if state_info['stages'][stage]['valid'])}")

        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (
            input_idx < len(input_matrices) or self.is_active()
        ):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        # 转换结果矩阵为浮点数
        final_c_matrix = np.array([bf16_to_float(v) for v in self.c_values]).reshape(
            self.M, self.N
        )

        print(f"✅ 流水线处理完成，总共 {cycle} 个周期")

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
        print(f"🔄 转换输入数据为BF16格式...")
        
        # 将所有输入转换为BF16格式
        bf16_input_matrices = []

        # 为转换过程添加进度条
        with tqdm(total=len(input_matrices), desc="BF16转换", unit="matrix") as conv_pbar:
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
                conv_pbar.update(1)

        print(f"✅ BF16转换完成")

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

        # 估算总周期数：输入数据 + 流水线深度的缓冲
        estimated_cycles = len(B_data_list) + 20  # 20是估算的流水线深度
        
        # 创建进度条
        with tqdm(total=estimated_cycles, desc="HBM流水线处理", unit="cycle") as pbar:
            # 运行流水线，直到处理完所有输入并且没有更多有效数据
            cycle = 0
            while (input_idx < len(B_data_list) or self.is_active()) and (cycle < max_cycles or max_cycles == -1):
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

                # 更新进度条
                cycle += 1
                
                # 动态更新进度条描述
                if input_idx < len(B_data_list):
                    pbar.set_description(f"HBM流水线处理 (输入 {input_idx}/{len(B_data_list)})")
                else:
                    pbar.set_description(f"HBM流水线处理 (排空中)")
                
                # 如果超出估算周期数，扩展进度条
                if cycle >= pbar.total:
                    pbar.total = cycle + 10
                    pbar.refresh()
                
                pbar.update(1)

                # 如果需要，打印当前状态
                if print_states and (cycle % 10 == 0 or cycle < 5 or cycle >= len(A_matrices) - 3):
                    # 暂时禁用进度条输出，打印状态，然后重新启用
                    pbar.write(f"\n--- 周期 {cycle} (HBM模式) ---")
                    # 将状态信息写入到tqdm的输出中
                    state_info = self.get_pipeline_state()
                    pbar.write(f"流水线状态: {state_info}")

        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (input_idx < len(A_matrices) or self.is_active()):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        # 转换结果矩阵为浮点数
        final_c_matrix = np.array([bf16_to_float(v) for v in self.c_values]).reshape(
            self.M, self.N
        )

        print(f"✅ HBM流水线处理完成，总共 {cycle} 个周期")

        return {
            "results": results,
            "cycles": cycle,
            "c_matrix": final_c_matrix,
            "c_values_bf16": self.c_values.copy(),
        }
    
    def run_pipeline_hbm_multi_batch_for_weight(
        self, A_batch_matrices, B_data_list, trapezoid_list, max_cycles=100, print_states=False
    ):
        
        # 1. batch <= pe_row and pe_row % batch == 0 
        # 此时 pe_row_per_batch = pe_row / batch 
        # 例如 pe_row = 32, batch = 4, pe_row_per_batch = 8
        # 即每个batch分得8个pe_row，也就相当于划分了4组pe_row
        # 这四组pe_row之间共享同一份权重矩阵，即同一份B_data_list
        # 同时pe_row之内共享一个batch的A_matrice
        # 每组perow内，又采用轮循的方式去给b_data_list
        # 例如 
        #     group0              group1                  group2                  group3       
        # 0 1 2 3 4 5 6 7  8 9 10 11 12 13 14 15  16 17 18 19 20 21 22 23  24 25 26 27 28 29 30 31
        # 0 1 2 3 4 5 6 7  0 1 2  3  4  5  6  7   0  1  2  3  4  5  6  7   0  1  2  3  4  5  6  7
        #      A_batch0           A_batch1                A_batch2               A_batch3
        # 
        # 2.  batch <= pe_row and pe_row % batch == 1
        # 此时 pe_row_per_batch = pe_row // batch
        # 此时最后一组肯定会多几个pe_row
        # 例如 pe_row = 32, batch = 5, pe_row_per_batch = 6, pe_row_per_batch_last = 8
        # 例如
        #     group0         group1          group2             group3              group4
        # 0 1 2 3 4 5  6 7 8 9 10 11  12 13 14 15 16 17  18 19 20 21 22 23  24 25 26 27 28 29 30 31
        # 0 1 2 3 4 5  0 1 2 3 4  5    0  1  2  3  4  5   0  1  2  3  4  5   0  1  2  3  4  5
        #     A_batch0    A_batch1           A_batch2           A_batch3            A_batch4
        # 这种情况下会有30 31 这两个pe row闲置
        #
        # 3. batch > pe_row
        # 此时的策略为，一轮一轮的算
        # 例如 batch = 64 pe_row = 32
        # 此时我们需要算 两轮，第一轮先算前32个batch，第二轮算后32个batch
        # 同理如果最后一轮batch 出现了不能整除的情况，即情况2里的例子，就用情况2的方式来解决
        

        final_results_all_rounds = {}
        final_batch_results = {}  # 存储每个batch的最终结果 {batch_idx: result}
        total_cycles = 0

        all_results = []
        num_trapezoids = len(trapezoid_list)  # pe_row
        batch = len(A_batch_matrices)

        if not all(A.shape[0] == 1 for A in A_batch_matrices):
            raise ValueError("使用HBM模式时，所有A矩阵必须是稠密向量(形状为(1, K))")

        print(f"🚀 开始批处理权重共享HBM处理: {num_trapezoids}个流水线, {batch}个batch")
        
        # 确定处理策略
        if batch > num_trapezoids:
            # 情况3: batch > pe_row，分轮处理
            num_rounds = (batch + num_trapezoids - 1) // num_trapezoids
            print(f"📋 策略: 分{num_rounds}轮处理，每轮最多{num_trapezoids}个batch")
        else:
            # 情况1和2: batch <= pe_row
            num_rounds = 1
            pe_row_per_batch = num_trapezoids // batch
            if num_trapezoids % batch == 0:
                print(f"📋 策略: 完全均匀分组，每个batch分配{pe_row_per_batch}个trapezoid")
            else:
                print(f"📋 策略: 不均匀分组，每个batch至少分配{pe_row_per_batch}个trapezoid，最后一组会分配更多")

        final_results_all_rounds = {}
        total_cycles = 0
        
        # 按轮次处理
        for round_idx in range(num_rounds):
            print(f"\n--- 第 {round_idx + 1}/{num_rounds} 轮处理 ---")
            
            # 确定当前轮次的batch范围
            start_batch = round_idx * num_trapezoids
            end_batch = min(start_batch + num_trapezoids, batch)
            current_round_batches = A_batch_matrices[start_batch:end_batch]
            current_batch_count = len(current_round_batches)
            
            if current_batch_count == 0:
                break
                
            # 为当前轮次分配trapezoid
            if current_batch_count <= num_trapezoids:
                # 情况1和2: batch <= pe_row (在当前轮次中)
                pe_row_per_batch = num_trapezoids // current_batch_count
                
                # 创建trapezoid到batch的映射
                trapezoid_to_batch = {}
                trapezoid_to_group_idx = {}
                
                for batch_idx in range(current_batch_count):
                    start_trap = batch_idx * pe_row_per_batch
                    if batch_idx == current_batch_count - 1:
                        # 最后一个batch分配剩余的所有trapezoid
                        end_trap = num_trapezoids
                    else:
                        end_trap = start_trap + pe_row_per_batch
                    
                    for trap_idx in range(start_trap, end_trap):
                        if trap_idx < num_trapezoids:
                            trapezoid_to_batch[trap_idx] = batch_idx
                            trapezoid_to_group_idx[trap_idx] = trap_idx - start_trap
                            
                print(f"当前轮次trapezoid分配: {trapezoid_to_batch}")
            
            # 重置trapezoid状态
            if round_idx == 0:
                for trapezoid in trapezoid_list:
                    trapezoid.reset()
            else:
                for trapezoid in trapezoid_list:
                    saved_c_values = trapezoid.c_values.copy()
                    trapezoid.reset()
                    trapezoid.c_values = saved_c_values
            
            input_idx = 0
            estimated_cycles = len(B_data_list) + 20
            
            with tqdm(total=estimated_cycles, desc=f"第{round_idx+1}轮HBM处理", unit="cycle") as pbar:
                cycle = 0
                while (input_idx < len(B_data_list) or any(trap.is_active() for trap in trapezoid_list)) and (cycle < max_cycles or max_cycles == -1):
                    
                    cycle_results = []
                    
                    for trap_idx, trapezoid in enumerate(trapezoid_list):
                        # 确定当前trapezoid对应的batch和组内索引
                        if trap_idx in trapezoid_to_batch:
                            batch_idx = trapezoid_to_batch[trap_idx]
                            group_idx = trapezoid_to_group_idx[trap_idx]
                            
                            # 计算当前trapezoid应该处理的B_data索引（组内轮询）
                            current_b_data_idx = input_idx + group_idx
                            
                            if current_b_data_idx < len(B_data_list):
                                # 获取对应batch的A矩阵
                                A = current_round_batches[batch_idx]
                                
                                # 从B数据列表获取当前B的CSR格式数据
                                B_data = B_data_list[current_b_data_idx]
                                values_B = B_data.get("values", [])
                                col_indices = B_data.get("col_indices", [])
                                row_ptr = B_data.get("row_ptr", [])
                                start_index = B_data.get("row_start_index", 0)
                                
                                valid = True
                            else:
                                # 没有更多输入给这个trapezoid
                                A = np.array([[]])
                                values_B = []
                                col_indices = []
                                row_ptr = []
                                start_index = 0
                                valid = False
                        else:
                            # 当前trapezoid在此轮次中闲置
                            A = np.array([[]])
                            values_B = []
                            col_indices = []
                            row_ptr = []
                            start_index = 0
                            valid = False
                        
                        # 运行当前trapezoid的一个时钟周期
                        result = trapezoid.clock_cycle(
                            valid=valid,
                            A=A,
                            B=np.array([]),  # 在HBM模式下B矩阵为空
                            is_hbm=True,
                            start_index=start_index,
                            values_B_input=values_B,
                            col_indices_input=col_indices,
                            row_ptr_input=row_ptr
                        )
                        
                        # 添加trapezoid和轮次标识
                        result["trapezoid_id"] = trap_idx
                        result["round_id"] = round_idx
                        cycle_results.append(result)
                    
                    # 更新输入索引（每个周期前进当前轮次active trapezoid组数个步长）
                    if input_idx < len(B_data_list):
                        # 计算当前轮次有多少个active的组
                        if current_batch_count <= num_trapezoids:
                            active_groups = pe_row_per_batch
                        else:
                            active_groups = num_trapezoids
                        input_idx += active_groups
                    
                    all_results.append(cycle_results)
                    cycle += 1
                    
                    # 更新进度条
                    active_traps = sum(1 for trap in trapezoid_list if trap.is_active())
                    if input_idx < len(B_data_list):
                        pbar.set_description(f"第{round_idx+1}轮处理 (进度 {input_idx}/{len(B_data_list)}, 活跃:{active_traps})")
                    else:
                        pbar.set_description(f"第{round_idx+1}轮处理 (排空中, 活跃:{active_traps})")
                    
                    if cycle >= pbar.total:
                        pbar.total = cycle + 10
                        pbar.refresh()
                    
                    pbar.update(1)
                    
                    # 打印状态
                    if print_states and (cycle % 10 == 0 or cycle < 5):
                        pbar.write(f"\n--- 轮次{round_idx+1} 周期 {cycle} ---")
                        for i, trap in enumerate(trapezoid_list):
                            if trap.is_active():
                                batch_info = f"batch{trapezoid_to_batch.get(i, 'idle')}" if i in trapezoid_to_batch else "idle"
                                pbar.write(f"  Trapezoid {i}({batch_info}): 周期{trap.cycle_count}")
            
            total_cycles += cycle
            
            round_results = {}
            batch_results_current_round = {}  # 当前轮次每个batch的累积结果
            
            for i, trapezoid in enumerate(trapezoid_list):
                if i in trapezoid_to_batch:
                    batch_idx = trapezoid_to_batch[i]
                    actual_batch_idx = start_batch + batch_idx  # 全局batch索引
                    
                    c_matrix = np.array([bf16_to_float(v) for v in trapezoid.c_values]).reshape(
                        trapezoid.M, trapezoid.N
                    )
                    
                    # 保存单个trapezoid的结果（用于调试）
                    trap_key = f"round_{round_idx}_batch_{batch_idx}_trap_{i}"
                    round_results[trap_key] = {
                        "c_matrix": c_matrix,
                        "c_values_bf16": trapezoid.c_values.copy(),
                        "cycles": trapezoid.cycle_count,
                        "batch_idx": actual_batch_idx,
                        "trap_idx": i
                    }
                    
                    # 累加到对应batch的结果中
                    if actual_batch_idx not in batch_results_current_round:
                        batch_results_current_round[actual_batch_idx] = {
                            "c_values_bf16": trapezoid.c_values.copy(),
                            "trap_count": 1
                        }
                    else:
                        # 累加当前batch内多个trapezoid的结果
                        batch_results_current_round[actual_batch_idx]["c_values_bf16"] = bf16_add_list(
                            batch_results_current_round[actual_batch_idx]["c_values_bf16"],
                            trapezoid.c_values
                        )
                        batch_results_current_round[actual_batch_idx]["trap_count"] += 1
            
            final_results_all_rounds.update(round_results)
            
            # 将当前轮次的batch结果存储到全局batch结果中
            for batch_idx, batch_data in batch_results_current_round.items():
                if batch_idx not in final_batch_results:
                    final_batch_results[batch_idx] = {
                        "c_values_bf16": batch_data["c_values_bf16"].copy(),
                        "trap_count": batch_data["trap_count"]
                    }
                else:
                    # 如果是多轮处理，需要累加不同轮次中同一batch的结果
                    final_batch_results[batch_idx]["c_values_bf16"] = bf16_add_list(
                        final_batch_results[batch_idx]["c_values_bf16"],
                        batch_data["c_values_bf16"]
                    )
                    final_batch_results[batch_idx]["trap_count"] += batch_data["trap_count"]
            
            print(f"✅ 第{round_idx+1}轮处理完成，用时 {cycle} 个周期")
            print(f"   当前轮次处理了 {len(batch_results_current_round)} 个batch")
        
        batch_c_matrices = {}  # {batch_idx: c_matrix}
        
        for batch_idx, batch_data in final_batch_results.items():
            c_matrix = np.array([bf16_to_float(v) for v in batch_data["c_values_bf16"]]).reshape(
                trapezoid_list[0].M, trapezoid_list[0].N
            )
            c_matrix_bf16 = np.array(batch_data["c_values_bf16"]).reshape(
                trapezoid_list[0].M, trapezoid_list[0].N
            )
            
            batch_c_matrices[batch_idx] = {
                "c_matrix": c_matrix,
                "c_matrix_bf16": c_matrix_bf16,
                "trap_count": batch_data["trap_count"]
            }
        
        print(f"🎉 批处理权重共享HBM处理完成，总共 {total_cycles} 个周期，处理了 {batch} 个batch")
        print(f"📊 生成了 {len(batch_c_matrices)} 个独立的batch结果")
        
        return {
            "all_cycle_results": all_results,
            "cycles": total_cycles,
            "individual_results": final_results_all_rounds,  # 保留单个trapezoid结果用于调试
            "batch_results": batch_c_matrices,  # 新增：每个batch的独立结果
            "num_trapezoids": num_trapezoids,
            "num_batches": batch,
            "num_rounds": num_rounds
        }
    
    def run_pipeline_hbm_multi_batch_for_weight_fast(
        self, A_batch_matrices, B_data_list, trapezoid_list, max_cycles=100, print_states=False
    ):
        
        num_trapezoids = len(trapezoid_list)
        batch = A_batch_matrices.shape[0]

        if batch & (batch - 1) != 0 or batch == 0:
            raise ValueError("batch 必须是2的幂次方！")

        round_num = 1
      
        pe_row = num_trapezoids
        if batch > pe_row:
            round_num = batch // pe_row

        # 计算每个batch分配的PE行数
        pe_row_per_batch = pe_row // batch
        if batch == 64 or batch == 128:
            pe_row_per_batch = 1
        

        
        # 为每个batch创建独立的结果累加器
        batch_c_values = {}
        for batch_idx in range(batch):
            # 初始化每个batch的C矩阵累加器
            batch_c_values[batch_idx] = [0] * trapezoid_list[0].M * trapezoid_list[0].N

        # 创建输入队列索引，采用轮询方式
        

        # 估算总周期数：输入数据 + 流水线深度的缓冲
        estimated_cycles = len(B_data_list) + 20  # 20是估算的流水线深度
        
        print(f"🚀 开始多Trapezoid批量权重共享HBM处理:")
        print(f"   {num_trapezoids}个流水线, {batch}个batch, 每batch分配{pe_row_per_batch}个PE")
        
        # 创建进度条
        cycle = 0
        for round_idx in range(round_num):
            input_idx = 0
            with tqdm(total=estimated_cycles, desc="批量权重共享HBM处理", unit="cycle") as pbar:
                # 运行流水线，直到处理完所有输入并且没有更多有效数据
                
                while (input_idx < len(B_data_list) or any(trap.is_active() for trap in trapezoid_list)) and (cycle < max_cycles or max_cycles == -1):
                    
                    for trap_idx, trapezoid in enumerate(trapezoid_list):
                        # 确定当前trapezoid属于哪个batch组
                        batch_group = trap_idx // pe_row_per_batch + round_idx * pe_row
                        # 当前trapezoid在其batch组内的局部索引
                        local_trap_idx = trap_idx % pe_row_per_batch
                        
                        # 计算当前trapezoid应该处理的B_data索引
                        # 在每个batch组内采用轮询分配策略
                        current_b_data_idx = input_idx + local_trap_idx
                        
                        if current_b_data_idx < len(B_data_list):
                            # 获取对应batch的A矩阵
                            A = A_batch_matrices[batch_group]

                            # 从B数据列表获取当前B的CSR格式数据（所有batch共享）
                            B_data = B_data_list[current_b_data_idx]
                            values_B = B_data.get("values", [])
                            col_indices = B_data.get("col_indices", [])
                            row_ptr = B_data.get("row_ptr", [])
                            start_index = B_data.get("row_start_index", 0)

                            valid = True
                        else:
                            # 没有更多输入给这个trapezoid
                            A = np.array([[]])
                            values_B = []
                            col_indices = []
                            row_ptr = []
                            start_index = 0
                            valid = False

                        # 运行当前trapezoid的一个时钟周期
                        result = trapezoid.clock_cycle(
                            valid=valid, 
                            A=A, 
                            B=np.array([]),  # 在HBM模式下B矩阵为空
                            is_hbm=True, 
                            start_index=start_index,
                            values_B_input=values_B, 
                            col_indices_input=col_indices, 
                            row_ptr_input=row_ptr
                        )
                        
                    
                    # 更新输入索引（每个周期前进pe_row_per_batch个步长）
                    if input_idx < len(B_data_list):
                        input_idx += pe_row_per_batch
                    

                    # 更新进度条
                    cycle += 1
                    
                    # 动态更新进度条描述
                    active_traps = sum(1 for trap in trapezoid_list if trap.is_active())
                    if input_idx < len(B_data_list):
                        pbar.set_description(f"批量处理 (数据 {input_idx//pe_row_per_batch}/{len(B_data_list)//pe_row_per_batch}, 活跃:{active_traps})")
                    else:
                        pbar.set_description(f"批量处理 (排空中, 活跃:{active_traps})")
                    
                    # 如果超出估算周期数，扩展进度条
                    if cycle >= pbar.total:
                        pbar.total = cycle + 10
                        pbar.refresh()
                    
                    pbar.update(1)


            # 检查是否因为达到最大周期数而退出
            if cycle >= max_cycles and (input_idx < len(B_data_list) or any(trap.is_active() for trap in trapezoid_list)):
                print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

            # 收集每个batch的最终结果
            # 按batch组收集trapezoid的结果
            for trap_idx, trapezoid in enumerate(trapezoid_list):
                batch_group = trap_idx // pe_row_per_batch + round_idx * pe_row
                
                # 将当前trapezoid的结果累加到对应batch的结果中
                if batch_group < batch:  # 确保batch_group有效
                    batch_c_values[batch_group] = bf16_add_list(
                        batch_c_values[batch_group], 
                        trapezoid.c_values
                )
            for trapezoid in trapezoid_list:
                trapezoid.reset()
            

        # 构建最终的输出矩阵，shape为(batch, M, N)
        M, N = trapezoid_list[0].M, trapezoid_list[0].N
        combined_c_matrix = np.zeros((batch, M, N))
        combined_c_matrix_bf16 = np.zeros((batch, M, N))

        for batch_idx in range(batch):
            # 转换BF16结果为浮点数
            c_matrix = np.array([bf16_to_float(v) for v in batch_c_values[batch_idx]]).reshape(M, N)
            c_matrix_bf16 = np.array(batch_c_values[batch_idx]).reshape(M, N)
            
            # 填入组合矩阵
            combined_c_matrix[batch_idx] = c_matrix
            combined_c_matrix_bf16[batch_idx] = c_matrix_bf16
            

        print(f"✅ 批量权重共享HBM处理完成，总共 {cycle} 个周期")
        print(f"📊 生成了 {batch} 个batch的结果，每个结果形状为 ({M}, {N})")

        return {
            "cycles": cycle,
            "combined_c_matrix": combined_c_matrix,  # shape: (batch, M, N)
            "combined_c_matrix_bf16": combined_c_matrix_bf16,  # shape: (batch, M, N)
            "num_trapezoids": num_trapezoids,
            "num_batches": batch,
            "pe_row_per_batch": pe_row_per_batch
        }
    
    def run_pipeline_hbm_multi_batch(
        self, A_matrices_batch, B_data_list_batch, trapezoid_list, max_cycles=100, print_states=False      
    ):
        M, N = trapezoid_list[0].M, trapezoid_list[0].N

        
        num_trapezoids = len(trapezoid_list)
        assert A_matrices_batch.shape[0] == len(B_data_list_batch)
        batch = A_matrices_batch.shape[0]

        combined_c_matrix = np.zeros((batch, M, N))
        combined_c_matrix_bf16 = np.zeros((batch, M, N))

        estimated_cycles = len(B_data_list_batch[0]) + 20
        print(f"🚀 开始多Trapezoid HBM处理: {num_trapezoids}个流水线")

        with tqdm(total=estimated_cycles, desc="批处理HBM非共享", unit="cycle") as pbar:
            cycle = 0
            for bx in range(0, batch):
                for trapezoid in trapezoid_list:
                    trapezoid.reset()
                input_idx = 0
                while (input_idx < len(B_data_list_batch[bx]) or any(trap.is_active() for trap in trapezoid_list)) and (cycle < max_cycles or max_cycles == -1):
                    for trap_idx, trapezoid in enumerate(trapezoid_list):
                        current_b_data_idx = input_idx + trap_idx

                        if current_b_data_idx < len(B_data_list_batch[bx]):
                            A = A_matrices_batch[bx]
                            B_data = B_data_list_batch[bx][current_b_data_idx]
                            values_B = B_data.get("values", [])
                            col_indices = B_data.get("col_indices", [])
                            row_ptr = B_data.get("row_ptr", [])
                            start_index = B_data.get("row_start_index", 0)

                            valid = True
                        else:
                            A = np.array([[]])
                            values_B = []
                            col_indices = []
                            row_ptr = []
                            start_index = 0
                            valid = False
                        
                        result = trapezoid.clock_cycle(
                            valid=valid, 
                            A=A, 
                            B=np.array([]),  # 在HBM模式下B矩阵为空
                            is_hbm=True, 
                            start_index=start_index,
                            values_B_input=values_B, 
                            col_indices_input=col_indices, 
                            row_ptr_input=row_ptr
                        )
                    
                    if input_idx < len(B_data_list_batch[bx]):
                        input_idx += num_trapezoids
                    
                    cycle += 1

                    # 动态更新进度条描述
                    active_traps = sum(1 for trap in trapezoid_list if trap.is_active())
                    if input_idx < len(B_data_list_batch[bx]):
                        pbar.set_description(f"多Trapezoid处理 (批次 {input_idx//num_trapezoids}/{len(B_data_list_batch[bx])//num_trapezoids}, 活跃:{active_traps})")
                    else:
                        pbar.set_description(f"多Trapezoid处理 (排空中, 活跃:{active_traps})")
                    
                    # 如果超出估算周期数，扩展进度条
                    if cycle >= pbar.total:
                        pbar.total = cycle + 10
                        pbar.refresh()
                    
                    pbar.update(1)
                if cycle >= max_cycles and (input_idx < len(B_data_list_batch[bx]) or any(trap.is_active() for trap in trapezoid_list)):
                    print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")
                
                combined_c_matrix_one_batch = None
                combined_c_matrix_one_batch_bf16 = None

                for i, trapezoid in enumerate(trapezoid_list):
                    c_matrix = np.array([bf16_to_float(v) for v in trapezoid.c_values]).reshape(
                    trapezoid.M, trapezoid.N
                )
                
                    if combined_c_matrix_one_batch is None:
                        combined_c_matrix_one_batch = c_matrix.copy()
                    else:
                        combined_c_matrix_one_batch += c_matrix

                    if combined_c_matrix_one_batch_bf16 is None:
                        combined_c_matrix_one_batch_bf16 = trapezoid.c_values.copy()
                    else:
                        combined_c_matrix_one_batch_bf16 = bf16_add_list(combined_c_matrix_one_batch_bf16, trapezoid.c_values)
                combined_c_matrix_one_batch_bf16 = np.array(combined_c_matrix_one_batch_bf16).reshape(trapezoid_list[0].M, trapezoid_list[0].N)
            
                combined_c_matrix[bx] = combined_c_matrix_one_batch
                combined_c_matrix_bf16[bx] = combined_c_matrix_one_batch_bf16

        print(f"✅ 批量非共享HBM处理完成，总共 {cycle} 个周期")

        return{
            "cycles": cycle,
            "combined_c_matrix": combined_c_matrix,
            "combined_c_matrix_bf16": combined_c_matrix_bf16
        }
            
    
    def run_pipeline_hbm_multi(
        self, A_matrices, B_data_list, trapezoid_list, max_cycles=100, print_states=False
    ):
        """
        运行多个Trapezoid流水线处理HBM格式的输入矩阵
        
        Args:
            A_matrices: 稠密向量A的列表，每个元素是一个形状为(1, K)的numpy数组
            B_data_list: 列表，每个元素是包含B矩阵CSR格式的字典，格式为
                        {"values": [...], "col_indices": [...], "row_ptr": [...], "row_start_index": int}
            trapezoid_list: TrapezoidPipeline实例的列表
            max_cycles: 最大运行周期数，防止无限循环
            print_states: 是否打印每个周期的状态
            
        Returns:
            结果字典，包含每个时钟周期输出和最终矩阵
        """
        # 初始化结果列表
        all_results = []
        num_trapezoids = len(trapezoid_list)
        
        # 验证输入
        if not all(A.shape[0] == 1 for A in A_matrices):
            raise ValueError("使用HBM模式时，所有A矩阵必须是稠密向量(形状为(1, K))")

        # 创建输入队列
        input_idx = 0

        # 估算总周期数：输入数据 + 流水线深度的缓冲
        estimated_cycles = len(B_data_list) + 20  # 20是估算的流水线深度
        
        print(f"🚀 开始多Trapezoid HBM处理: {num_trapezoids}个流水线")
        
        # 创建进度条
        with tqdm(total=estimated_cycles, desc="多Trapezoid HBM处理", unit="cycle") as pbar:
            # 运行流水线，直到处理完所有输入并且没有更多有效数据
            cycle = 0
            while (input_idx < len(B_data_list) or any(trap.is_active() for trap in trapezoid_list)) and (cycle < max_cycles or max_cycles == -1):
                
                # 为每个trapezoid准备当前周期的输入
                cycle_results = []
                
                for trap_idx, trapezoid in enumerate(trapezoid_list):
                    # 计算当前trapezoid应该处理的B_data索引
                    # 采用轮询分配策略
                    current_b_data_idx = input_idx + trap_idx
                    
                    if current_b_data_idx < len(B_data_list):
                        # 总是获取A矩阵（所有trapezoid共享同一个A）
                        A = A_matrices[0] 

                        # 从B数据列表获取当前B的CSR格式数据
                        B_data = B_data_list[current_b_data_idx]
                        values_B = B_data.get("values", [])
                        col_indices = B_data.get("col_indices", [])
                        row_ptr = B_data.get("row_ptr", [])
                        start_index = B_data.get("row_start_index", 0)

                        valid = True
                    else:
                        # 没有更多输入给这个trapezoid
                        A = np.array([[]])
                        values_B = []
                        col_indices = []
                        row_ptr = []
                        start_index = 0
                        valid = False

                    # 运行当前trapezoid的一个时钟周期
                    result = trapezoid.clock_cycle(
                        valid=valid, 
                        A=A, 
                        B=np.array([]),  # 在HBM模式下B矩阵为空
                        is_hbm=True, 
                        start_index=start_index,
                        values_B_input=values_B, 
                        col_indices_input=col_indices, 
                        row_ptr_input=row_ptr
                    )
                    
                    # 添加trapezoid标识
                    result["trapezoid_id"] = trap_idx
                    cycle_results.append(result)
                
                # 更新输入索引（每个周期前进trapezoid数量个步长）
                if input_idx < len(B_data_list):
                    input_idx += num_trapezoids
                
                all_results.append(cycle_results)

                # 更新进度条
                cycle += 1
                
                # 动态更新进度条描述
                active_traps = sum(1 for trap in trapezoid_list if trap.is_active())
                if input_idx < len(B_data_list):
                    pbar.set_description(f"多Trapezoid处理 (批次 {input_idx//num_trapezoids}/{len(B_data_list)//num_trapezoids}, 活跃:{active_traps})")
                else:
                    pbar.set_description(f"多Trapezoid处理 (排空中, 活跃:{active_traps})")
                
                # 如果超出估算周期数，扩展进度条
                if cycle >= pbar.total:
                    pbar.total = cycle + 10
                    pbar.refresh()
                
                pbar.update(1)

                # 如果需要，打印当前状态
                if print_states and (cycle % 10 == 0 or cycle < 5):
                    pbar.write(f"\n--- 周期 {cycle} (多Trapezoid HBM模式) ---")
                    for i, trap in enumerate(trapezoid_list):
                        state = trap.get_pipeline_state()
                        pbar.write(f"  Trapezoid {i}: 周期{state['cycle_count']}, "
                                f"活跃{'是' if trap.is_active() else '否'}, "
                                f"非零结果{state['result']['c_values_non_zero']}")

        # 检查是否因为达到最大周期数而退出
        if cycle >= max_cycles and (input_idx < len(B_data_list) or any(trap.is_active() for trap in trapezoid_list)):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        # 收集所有trapezoid的最终结果
        final_results = {}
        combined_c_matrix = None
        combined_c_matrix_bf16 = None

        for i, trapezoid in enumerate(trapezoid_list):
            # 转换结果矩阵为浮点数
            c_matrix = np.array([bf16_to_float(v) for v in trapezoid.c_values]).reshape(
                trapezoid.M, trapezoid.N
            )


            final_results[f"trapezoid_{i}"] = {
                "c_matrix": c_matrix,
                "c_values_bf16": trapezoid.c_values.copy(),
                "cycles": trapezoid.cycle_count
            }
            
            # 如果所有trapezoid的矩阵大小相同，可以合并结果
            if combined_c_matrix is None:
                combined_c_matrix = c_matrix.copy()
            else:
                combined_c_matrix += c_matrix

            if combined_c_matrix_bf16 is None:
                combined_c_matrix_bf16 = trapezoid.c_values.copy()
            else:
                combined_c_matrix_bf16 = bf16_add_list(combined_c_matrix_bf16, trapezoid.c_values)
        combined_c_matrix_bf16 = np.array(combined_c_matrix_bf16).reshape(trapezoid_list[0].M, trapezoid_list[0].N)

        #combined_c_matrix_bf16_test = np.array([bf16_to_float(v) for v in combined_c_matrix_bf16]).reshape(trapezoid_list[0].M, trapezoid_list[0].N)
        #print(combined_c_matrix_bf16_test)

        print(f"✅ 多Trapezoid HBM处理完成，总共 {cycle} 个周期")

        return {
            "all_cycle_results": all_results,
            "cycles": cycle,
            "individual_results": final_results,
            "combined_c_matrix": combined_c_matrix,
            "combined_c_matrix_bf16": combined_c_matrix_bf16,
            "num_trapezoids": num_trapezoids
        }


    def run_pipeline_hbm_multi_with_bf16(
        self, A_matrices, B_data_list, trapezoid_list, max_cycles=1000, print_states=False
    ):
        """
        运行多个Trapezoid流水线处理HBM格式的输入矩阵，将输入转换为BF16格式
        
        Args:
            A_matrices: 稠密向量A的列表，每个元素是一个形状为(1, K)的numpy数组
            B_data_list: 列表，每个元素是包含B矩阵CSR格式的字典
            trapezoid_list: TrapezoidPipeline实例的列表
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态
            
        Returns:
            结果字典，包含运行结果和最终矩阵
        """
        print(f"🔄 转换输入数据为BF16格式...")
        
        # 优化：A矩阵只转换一次
        A = A_matrices[0]
        A_bf16 = np.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    A_bf16[i, j] = convert_through_pipeline(float(A[i, j]))
        
        # 将所有A矩阵转换为BF16格式
        bf16_A_matrices = []
        bf16_B_data_list = []

        for B_data in B_data_list:
            # 优化：直接使用已转换的A矩阵
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

        print(f"✅ BF16转换完成，开始多Trapezoid处理...")
        
        # 运行多Trapezoid流水线
        return self.run_pipeline_hbm_multi(bf16_A_matrices, bf16_B_data_list, trapezoid_list, max_cycles, print_states)
