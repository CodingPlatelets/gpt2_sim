import math
from .advance_add_sim import AdvanceAddUnit
from ..utils import find_singles, convert_through_pipeline
from collections import deque

class AddTree:
    def __init__(self, PE_num, c_values, M, N):
        self.cycle_count = 0

        self.PE_num = PE_num
        self.c_values = c_values
        self.M = M
        self.N = N
        self.tree = self.create_tree()

        self.stage_valid_vec = [False] * (self.tree_levels + 1)
        self.stage_output_vec = [deque() for _ in range(self.tree_levels + 1)]
        self.stage_evict_vec = [[] for _ in range(self.tree_levels + 1)]

    def create_tree(self):
        self.tree_levels = int(math.log2(self.PE_num))
        tree = []
        level_size = self.PE_num // 2
        for _ in range(0, self.tree_levels):
            tree.append(
                [
                    AdvanceAddUnit(self.c_values, self.M, self.N)
                    for _ in range(level_size)
                ]
            )
            level_size = level_size // 2
        # tree.reverse()
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
        # }
        self.cycle_count += 1

        for i in range(self.tree_levels - 1, -1, -1):
            add_layer_valid = False
            self.stage_output_vec[i + 1] = deque()
            for add in self.tree[i]:
                result = add.clock_cycle(
                    self.stage_valid_vec[i],
                    self.stage_output_vec[i].popleft() if self.stage_valid_vec[i] and self.stage_output_vec[i] else [],
                    self.stage_output_vec[i].popleft() if self.stage_valid_vec[i] and self.stage_output_vec[i] else [],
                    self.stage_evict_vec[i],
                )
                if result["valid"]:
                    self.stage_output_vec[i + 1].append(result["output"])
                    add_layer_valid = result["valid"]
            if add_layer_valid:
                self.stage_evict_vec[i + 1] = self.get_level_evict_index(
                    list(self.stage_output_vec[i + 1])
                )
                self.stage_valid_vec[i + 1] = True
            else:
                self.stage_evict_vec[i + 1] = []
                self.stage_output_vec[i + 1] = deque()
                self.stage_valid_vec[i + 1] = False

        self.stage_valid_vec[0] = valid
        if valid:
            self.stage_output_vec[0] = deque(map_queue)
            self.stage_evict_vec[0] = self.get_level_evict_index(
                list(self.stage_output_vec[0])
            )
        else:
            self.stage_evict_vec[0] = []
            self.stage_output_vec[0] = deque()

        return {
            "cycle": self.cycle_count,
            "valid": self.stage_valid_vec[-1],  # 最后阶段的有效标志
            "output": list(self.stage_output_vec[-1]) if self.stage_valid_vec[-1] else None,
        }

    def reset(self):
        """重置加法树的所有状态"""
        self.cycle_count = 0

        # 重置所有阶段的有效标志
        self.stage_valid_vec = [False] * (self.tree_levels + 1)

        # 重置输出和驱逐索引向量
        self.stage_output_vec = [deque() for _ in range(self.tree_levels + 1)]
        self.stage_evict_vec = [[] for _ in range(self.tree_levels + 1)]

        # 重置树中每个AdvanceAddUnit的状态
        for level in self.tree:
            for adder in level:
                adder.reset()

    def get_pipeline_state(self):
        """获取加法树的当前状态"""
        # 收集树中每个加法单元的状态
        tree_state = []
        for level_idx, level in enumerate(self.tree):
            level_state = []
            for unit_idx, adder in enumerate(level):
                level_state.append(
                    {"unit_idx": unit_idx, "state": adder.get_pipeline_state()}
                )
            tree_state.append({"level": level_idx, "units": level_state})

        return {
            "cycle_count": self.cycle_count,
            "tree_levels": self.tree_levels,
            "PE_num": self.PE_num,
            "stage_valid": self.stage_valid_vec,
            "stage_output_sizes": [len(outputs) for outputs in self.stage_output_vec],
            "stage_evict_sizes": [len(evicts) for evicts in self.stage_evict_vec],
            "tree_state": tree_state,
        }

    def print_state(self):
        """打印当前加法树的状态"""
        state = self.get_pipeline_state()

        print(f"\n==== AddTree 状态 (周期 {state['cycle_count']}) ====")
        print(f"树级数: {state['tree_levels']}, PE数量: {state['PE_num']}")

        # 打印各阶段状态
        print("\n阶段有效状态:")
        for i, valid in enumerate(state["stage_valid"]):
            print(f"  阶段 {i}: {'有效' if valid else '无效'}")

        print("\n阶段输出大小:")
        for i, size in enumerate(state["stage_output_sizes"]):
            print(f"  阶段 {i} 输出: {size} 项")

        # 添加输出向量内容的打印
        print("\n阶段输出向量内容:")
        for i, outputs in enumerate(self.stage_output_vec):
            if outputs:
                # 格式化输出向量内容
                formatted_outputs = []
                for out in outputs:
                    # 只取前几个键值对，避免过长
                    sample = {k: v for idx, (k, v) in enumerate(out.items())}
                    # if len(out) > 3:
                    #    sample_str = str(sample)[:-1] + ", ...}"
                    # else:
                    sample_str = str(sample)
                    formatted_outputs.append(sample_str)

                # if len(formatted_outputs) > 2:
                #    print(f"  阶段 {i}: [{formatted_outputs[0]}, {formatted_outputs[1]}, ...]")
                # else:
                print(f"  阶段 {i}: {formatted_outputs}")
            else:
                print(f"  阶段 {i}: []")

        print("\n阶段驱逐索引大小:")
        for i, size in enumerate(state["stage_evict_sizes"]):
            print(f"  阶段 {i} 驱逐索引: {size} 项")

        # 添加驱逐向量内容的打印
        print("\n阶段驱逐向量内容:")
        for i, evict_indices in enumerate(self.stage_evict_vec):
            if len(evict_indices) > 5:
                print(f"  阶段 {i}: {evict_indices[:5]}...")
            else:
                print(f"  阶段 {i}: {evict_indices}")

        # 打印树的简要状态
        print("\n树结构状态摘要:")
        for level in state["tree_state"]:
            level_idx = level["level"]
            units_count = len(level["units"])
            active_units = sum(
                1
                for unit in level["units"]
                if unit["state"]["stage1"]["valid"]
                or unit["state"]["stage2"]["valid"]
                or unit["state"]["stage3"]["valid"]
            )

            print(f"  级别 {level_idx}: {units_count} 个单元, {active_units} 个活跃")

        print("=====================================")

    def run_pipeline_with_bf16(
        self, input_queues_batches, max_cycles=100, print_states=False
    ):
        """
        运行加法树流水线，处理输入队列并将输入转换为BF16格式

        Args:
            input_queues_batches: 输入队列批次列表，每个元素是一个包含多个处理单元输入映射的列表
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态

        Returns:
            results: 包含每个周期输出的列表
        """
        # 重置加法树状态
        self.reset()

        # 将输入值转换为BF16格式
        bf16_input_queues_batches = []

        for batch in input_queues_batches:
            bf16_batch = []

            for queue in batch:
                converted_queue = {}
                for index, values in queue.items():
                    # 将每个值转换为BF16格式
                    converted_values = [
                        convert_through_pipeline(float(val)) for val in values
                    ]
                    converted_queue[index] = converted_values
                bf16_batch.append(converted_queue)

            bf16_input_queues_batches.append(bf16_batch)

        # 运行流水线
        return self.run_pipeline(bf16_input_queues_batches, max_cycles, print_states)

    def run_pipeline(self, input_queues, max_cycles=100, print_states=False):
        """
        运行加法树流水线，处理输入队列

        Args:
            input_queues: 输入队列列表，每个元素对应一个处理单元的输入映射
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态

        Returns:
            results: 包含每个周期输出的列表
        """
        # 重置加法树状态
        self.reset()

        # 初始化结果列表
        results = []
        input_idx = 0
        cycle = 0
        # 运行流水线，直到完成处理或达到最大周期
        while (
            input_idx < len(input_queues) or self.is_active()
        ) and cycle < max_cycles:
            # while cycle < 20:
            if input_idx < len(input_queues):
                input_queue = input_queues[input_idx]
                valid = True
                input_idx += 1
            else:
                input_queue = []
                valid = False

            result = self.clock_cycle(valid, input_queue)

            results.append(result)

            # 如果需要，打印当前状态
            if print_states:
                print(f"\n--- 周期 {cycle + 1} ---")
                self.print_state()

            # 检查流水线是否还在处理数据
            # if not self.is_active() and cycle > self.tree_levels:
            #    break
            cycle += 1

        # 如果达到最大周期数而流水线仍在处理，发出警告
        if cycle >= max_cycles - 1 and self.is_active():
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        print(cycle)

        return results

    def is_active(self):
        """
        检查加法树是否仍在处理数据

        Returns:
            bool: 如果任何阶段有效或任何加法单元活跃则返回True
        """
        # 检查所有阶段的有效标志
        if any(self.stage_valid_vec):
            return True

        # 检查树中的每个加法单元
        for level in self.tree:
            for adder in level:
                if adder.is_active():
                    return True

        return False