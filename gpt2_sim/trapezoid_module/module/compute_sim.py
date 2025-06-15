from ...bf16_module import BF16AddPipeline, BF16MultiplyPipeline
from ...bf16_module.utils import convert_through_pipeline
from collections import deque

class MultiplyUnit:
    def __init__(self):
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.index_queue = []
        self.valid = False
        self.input_valid = False

    def get_input(self, input_valid, input1, input2, sft_index):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        if input_valid:
            self.index_queue.append(sft_index)

    def clock_cycle(self):
        result = self.multiply_pipeline.clock_cycle(
            self.input1, self.input2, self.input_valid
        )
        self.valid = result["valid_output"]
        if self.valid:
            return self.multiply_pipeline.outputs.pop(0)
        return None

    def is_active(self):
        return self.multiply_pipeline.is_active()

    def reset(self):
        """重置乘法单元状态"""
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.index_queue = []
        self.valid = False
        self.input_valid = False

    def get_pipeline_state(self):
        """获取乘法单元的当前状态"""
        return {
            "input1": self.input1,
            "input2": self.input2,
            "input_valid": self.input_valid,
            "valid": self.valid,
            "index_queue_size": len(self.index_queue),
            "pipeline_active": self.multiply_pipeline.is_active()
        }


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
        if input_valid:  # 防止污染数据
            self.index_queue.append(sft_index)  # 存放至sft_index队列中

    def clock_cycle(self):
        result = self.add_pipeline.clock_cycle(
            self.input1, self.input2, self.input_valid
        )
        self.valid = result["valid_output"]
        if self.valid:
            return self.add_pipeline.outputs.pop(0)
        return None

    def is_active(self):
        return self.add_pipeline.is_active()

    def reset(self):
        """重置加法单元状态"""
        self.add_pipeline = BF16AddPipeline()
        self.input1 = 0
        self.input2 = 0
        self.index_queue = []
        self.valid = False
        self.input_valid = False

    def get_pipeline_state(self):
        """获取加法单元的当前状态"""
        return {
            "input1": self.input1,
            "input2": self.input2,
            "input_valid": self.input_valid,
            "valid": self.valid,
            "index_queue_size": len(self.index_queue),
            "pipeline_active": self.add_pipeline.is_active()
        }


class MacUnit:
    def __init__(self):
        self.multiply_unit = MultiplyUnit()
        self.add_unit = AddUnit()
        self.input1 = 0
        self.input2 = 0

        self.multiply_result = None
        self.multiply_valid = False

        self.acc_valid = False
        
        self.input_valid = False
        self.output_valid = False

        self.acc_queue = []
        self.multiply_queue = []

        self.cycle_count = 0
        self.outputs = []  # 存储输出结果

    def set_initial_acc(self, initial_acc):
        self.acc_queue.append(initial_acc)

    def get_input(self, input_valid, input1, input2):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid

    def clock_cycle(self):
        self.cycle_count += 1
        

        # 处理加法单元
        add_input_valid = len(self.acc_queue) > 0 and len(self.multiply_queue) > 0
        add_input1 = self.multiply_queue.pop(0) if add_input_valid else 0
        add_input2 = self.acc_queue.pop(0) if add_input_valid else 0
        
        self.add_unit.get_input(add_input_valid, add_input1, add_input2, 0)
        add_result = self.add_unit.clock_cycle()
        
        if self.add_unit.valid:
            self.acc_queue.append(add_result)
            self.outputs.append(add_result)
        
        # 处理乘法单元
        self.multiply_unit.get_input(self.input_valid, self.input1, self.input2, 0)
        multiply_result = self.multiply_unit.clock_cycle()
        
        if self.multiply_unit.valid:
            self.multiply_queue.append(multiply_result)
            self.multiply_valid = True
        else:
            self.multiply_valid = False
        
        return self.outputs.pop(0) if len(self.outputs) else None

    def is_active(self):
        return (self.add_unit.is_active() or 
                self.multiply_unit.is_active()
        )

    def reset(self):
        """重置MAC单元的所有状态"""
        self.multiply_unit.reset()
        self.add_unit.reset()
        self.input1 = 0
        self.input2 = 0
        self.multiply_result = None
        self.multiply_valid = False
        self.input_valid = False
        self.output_valid = False
        self.acc_queue = []
        self.cycle_count = 0
        self.outputs = []

    def get_pipeline_state(self):
        """获取MAC单元的当前状态"""
        return {
            "cycle_count": self.cycle_count,
            "input1": self.input1,
            "input2": self.input2,
            "input_valid": self.input_valid,
            "multiply_result": self.multiply_result,
            "multiply_valid": self.multiply_valid,
            "output_valid": self.output_valid,
            "acc_queue_size": len(self.acc_queue),
            "acc_queue": self.acc_queue.copy(),
            "outputs_count": len(self.outputs),
            "multiply_unit_state": self.multiply_unit.get_pipeline_state(),
            "add_unit_state": self.add_unit.get_pipeline_state(),
            "is_active": self.is_active()
        }

    def print_state(self):
        """打印当前MAC单元的状态"""
        state = self.get_pipeline_state()
        
        print(f"\n==== MacUnit 状态 (周期 {state['cycle_count']}) ====")
        print(f"输入: input1={state['input1']}, input2={state['input2']}")
        print(f"输入有效: {state['input_valid']}")
        print(f"乘法结果: {state['multiply_result']}, 有效: {state['multiply_valid']}")
        print(f"输出有效: {state['output_valid']}")
        print(f"累加器队列大小: {state['acc_queue_size']}")
        if state['acc_queue']:
            print(f"累加器队列内容: {state['acc_queue']}")
        print(f"输出结果数量: {state['outputs_count']}")
        print(f"单元活跃状态: {state['is_active']}")
        
        print("\n乘法单元状态:")
        mul_state = state['multiply_unit_state']
        print(f"  输入: ({mul_state['input1']}, {mul_state['input2']})")
        print(f"  输入有效: {mul_state['input_valid']}, 输出有效: {mul_state['valid']}")
        print(f"  流水线活跃: {mul_state['pipeline_active']}")
        print(f"  索引队列大小: {mul_state['index_queue_size']}")
        
        print("\n加法单元状态:")
        add_state = state['add_unit_state']
        print(f"  输入: ({add_state['input1']}, {add_state['input2']})")
        print(f"  输入有效: {add_state['input_valid']}, 输出有效: {add_state['valid']}")
        print(f"  流水线活跃: {add_state['pipeline_active']}")
        print(f"  索引队列大小: {add_state['index_queue_size']}")
        
        print("=====================================")

    def run_pipeline_with_bf16(self, input_data, initial_acc, max_cycles=100, print_states=False):
        """
        运行MAC单元流水线，处理输入数据并将输入转换为BF16格式
        
        Args:
            input_data: 输入数据列表，每个元素是(input1, input2, acc_initial)的元组
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态
            
        Returns:
            results: 包含每个周期输出的列表
        """
        # 将输入数据转换为BF16格式
        bf16_input_data = []
        for input1, input2 in input_data:
            bf16_input1 = convert_through_pipeline(float(input1))
            bf16_input2 = convert_through_pipeline(float(input2))
            bf16_input_data.append((bf16_input1, bf16_input2))
        
        # 运行流水线
        return self.run_pipeline(bf16_input_data, initial_acc, max_cycles, print_states)

    def run_pipeline(self, input_data, initial_acc, max_cycles=100, print_states=False):
        """
        运行MAC单元流水线，处理输入数据
        
        Args:
            input_data: 输入数据列表，每个元素是(input1, input2, acc_initial)的元组
            max_cycles: 最大运行周期数
            print_states: 是否打印每个周期的状态
            
        Returns:
            results: 包含每个周期输出的列表
        """
        # 重置MAC单元状态
        self.reset()
        
        # 初始化结果列表
        results = []
        input_idx = 0
        cycle = 0
        
        self.set_initial_acc(initial_acc)

        # 运行流水线，直到完成处理或达到最大周期
        while (input_idx < len(input_data) or self.is_active()) and cycle < max_cycles:
            # 获取当前周期的输入
            if input_idx < len(input_data):
                input1, input2 = input_data[input_idx]
                self.get_input(True, input1, input2)
                input_idx += 1
            else:
                # 没有更多输入，发送无效输入
                self.get_input(False, 0, 0)
            
            # 执行一个时钟周期
            result = self.clock_cycle()
            results.append(result)
            
            # 如果需要，打印当前状态
            if print_states:
                print(f"\n--- 周期 {cycle + 1} ---")
                if result is not None:
                    print(f"输出结果: {result}")
                self.print_state()
            
            cycle += 1
        
        # 如果达到最大周期数而流水线仍在处理，发出警告
        if cycle >= max_cycles and self.is_active():
            print(f"警告: 达到最大周期数 {max_cycles}，流水线可能未完全排空")
        
        print(f"总共运行了 {cycle} 个周期")
        
        return results
        
                
