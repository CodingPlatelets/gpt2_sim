from .compute_sim import AddUnit
from ..utils import bf16_add, convert_through_pipeline
from collections import deque


class AdvanceAddUnit:
    def __init__(self, c_values, M, N):

        self.cycle_count = 0

        self.c_values = c_values
        self.M = M
        self.N = N
        self.add = AddUnit()

        # 优化1: 使用deque替代list，提高pop(0)性能
        self.direct_value_index_map_queue = deque()

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
        self, valid, index_value_map_input1, index_value_map_input2, evict_index_queue
    ):

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
        case_None = len(self.stage2_merge_index_value_map) == 0
        if case_None:
            self.add.get_input(
                self.stage2_valid, 0, 0, -2
            )  # 也会包含stage2_valid=false的情况

        for _, values in self.stage2_merge_index_value_map.items():
            values_len = len(values)
            if values_len == 2:
                add_used = True
            elif values_len == 1:
                no_add_used = True

        case_AA = add_used and (not no_add_used)
        case_AB_A = (not add_used) and no_add_used
        case_AAB = add_used and no_add_used
        
        for index, values in self.stage2_merge_index_value_map.items():
            values_len = len(values)
            assert values_len <= 2
            if values_len == 2:
                self.add.get_input(self.stage2_valid, values[0], values[1], index)
            elif values_len == 1:
                if index not in temp:
                    temp[index] = []
                temp[index].append(values[0])
                
        if temp:  # 只有在temp非空时才添加
            self.direct_value_index_map_queue.append(temp.copy())
            
        # padding
        if self.stage2_valid:
            if case_AA or case_None:
                padding = {-2: []}
                self.direct_value_index_map_queue.append(padding)
            elif case_AB_A:
                self.add.get_input(self.stage2_valid, 0, 0, -2)
            elif case_AAB:
                pass  # do nothing

        result = self.add.clock_cycle()
        self.stage3_valid = self.add.valid
        if self.add.valid:
            index = self.add.index_queue.pop(0)
            # 优化4: 使用popleft()替代pop(0)
            index_value_map = self.direct_value_index_map_queue.popleft()
            if index != -2:
                if index not in self.output:
                    self.output[index] = []
                self.output[index].append(result)
            if -2 not in index_value_map:
                for index_s, values in index_value_map.items():
                    assert len(values) == 1
                    if index_s not in self.output:
                        self.output[index_s] = []
                    self.output[index_s].append(values[0])
        else:
            self.output = {}

        # stage2 evict values
        self.stage2_valid = self.stage1_valid

        if self.stage1_valid:
            for evict_index in self.stage1_evict_index_queue:
                if evict_index in self.stage1_merge_index_value_map:
                    m = evict_index % self.M
                    n = evict_index // self.M  # 使用整数除法替代int(/)
                    assert len(self.stage1_merge_index_value_map[evict_index]) == 1
                    evict_val = self.stage1_merge_index_value_map.pop(evict_index)
                    if evict_index != -1:
                        # self.c_values[m * self.N + n] += evict_val[0]
                        self.c_values[m * self.N + n] = bf16_add(
                            self.c_values[m * self.N + n], evict_val[0]
                        )
        self.stage2_merge_index_value_map = self.stage1_merge_index_value_map.copy()

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
            #"pipeline_state": self.get_pipeline_state(),
        }

    def reset(self):
        """重置流水线状态"""
        self.cycle_count = 0

        # 重置队列
        self.direct_value_index_map_queue.clear()  # 优化9: 使用clear()替代重新赋值

        # 重置第一阶段
        self.stage1_valid = False
        self.stage1_merge_index_value_map.clear()  # 优化10: 使用clear()
        self.stage1_evict_index_queue.clear()

        # 重置第二阶段
        self.stage2_valid = False
        self.stage2_merge_index_value_map.clear()

        # 重置第三阶段
        self.stage3_valid = False
        self.output.clear()

        # 重置加法单元
        self.add = AddUnit()

    def get_pipeline_state(self):
        """获取流水线的当前状态"""
        return {
            "cycle_count": self.cycle_count,
            "stage1": {
                "valid": self.stage1_valid,
                "merge_index_value_map": self.stage1_merge_index_value_map,
                "evict_index_queue": self.stage1_evict_index_queue,
            },
            "stage2": {
                "valid": self.stage2_valid,
                "merge_index_value_map": self.stage2_merge_index_value_map,
            },
            "stage3": {"valid": self.stage3_valid, "output": self.output},
            "add_unit": {
                "valid": self.add.valid if hasattr(self.add, "valid") else False,
                "index_queue": (
                    self.add.index_queue if hasattr(self.add, "index_queue") else []
                ),
            },
            "direct_value_index_map_queue": list(self.direct_value_index_map_queue),  # 转换为list用于显示
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

    def run_pipeline(
        self, input_maps_pairs, evict_indices=None, max_cycles=20, print_states=False
    ):
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
        assert len(evict_indices) >= len(
            input_maps_pairs
        ), "驱逐索引列表长度应不小于输入对列表长度"

        # 创建输入队列
        input_queue = list(zip(input_maps_pairs, evict_indices))
        input_idx = 0

        # 运行流水线，直到处理完所有输入并且没有更多有效数据
        cycle = 0
        while (input_idx < len(input_queue) or self.is_active()) and cycle < max_cycles:

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
        if cycle >= max_cycles and (
            input_idx < len(input_queue)
            or self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
        ):
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")

        return results

    def run_pipeline_with_bf16(
        self, input_maps_pairs, evict_indices=None, max_cycles=20, print_states=False
    ):
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
                converted_values = [
                    convert_through_pipeline(float(val)) for val in values
                ]
                converted_map1[index] = converted_values

            # 转换第二个映射
            converted_map2 = {}
            for index, values in map2.items():
                converted_values = [
                    convert_through_pipeline(float(val)) for val in values
                ]
                converted_map2[index] = converted_values

            # 添加到转换后的列表
            converted_input_pairs.append((converted_map1, converted_map2))

        # 使用转换后的输入运行流水线
        return self.run_pipeline(
            converted_input_pairs, evict_indices, max_cycles, print_states
        )

    def is_active(self):
        """
        检查流水线是否仍在处理数据

        Returns:
            bool: 如果流水线中还有活跃的数据则返回True
        """
        return (
            self.stage1_valid
            or self.stage2_valid
            or self.stage3_valid
            or self.add.is_active()
        )
