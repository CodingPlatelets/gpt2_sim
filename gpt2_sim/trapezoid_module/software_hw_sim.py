import numpy as np
from collections import deque
from tqdm import tqdm
import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


try:
    # 尝试相对导入（当作为模块使用时）
    from .utils import FP32toBF16Pipeline, convert_through_pipeline, bf16_add, bf16_to_float
    from ..temp.bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline as BF16Convert
except ImportError:
    # 当直接运行此文件时，使用绝对导入
    from gpt2_sim.trapezoid_module.utils import FP32toBF16Pipeline, convert_through_pipeline, bf16_add, bf16_to_float
    from gpt2_sim.temp.bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline as BF16Convert

# BF16 常量定义
BF16_MAX_BF16 = convert_through_pipeline(65504.0)  # bf16 的最大正数
BF16_MIN_BF16 = convert_through_pipeline(6.103515625e-05)  # bf16 的最小正数
BF16_EXP_MAX_INPUT_BF16 = convert_through_pipeline(11.0)  # exp(11) ≈ 59874, 接近bf16上限
BF16_EXP_MIN_INPUT_BF16 = convert_through_pipeline(-17.0)  # exp(-17) ≈ 4e-8, 接近bf16下限
BF16_THRESHOLD_BF16 = convert_through_pipeline(52403.2)  # BF16_MAX * 0.8

class ExpUnit:
    """简单的指数运算单元，使用软件exp函数但所有数值都是BF16格式"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.valid = False
        self.input_val = 0
        self.output_val = 0
    
    def compute(self, input_val_bf16, valid=True):
        """计算exp函数，输入输出都是BF16格式"""
        self.valid = valid
        if not valid:
            self.output_val = 0
            return
            
        self.input_val = input_val_bf16
        
        # 转换为浮点数进行exp计算
        input_float = bf16_to_float(input_val_bf16)
        
        # 检查输入范围
        if input_float > bf16_to_float(BF16_EXP_MAX_INPUT_BF16):
            # 溢出情况，返回NaN的BF16表示
            self.output_val = 0x7FC0  # NaN in BF16
        elif input_float < bf16_to_float(BF16_EXP_MIN_INPUT_BF16):
            # 下溢情况，返回0
            self.output_val = 0
        else:
            try:
                # todo: using hw exp to compute
                result_float = math.exp(input_float)
                self.output_val = convert_through_pipeline(result_float)
            except OverflowError:
                self.output_val = 0x7FC0  # NaN in BF16

class CompareUnit:
    """比较运算单元，用于寻找最大值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.valid = False
        self.result = 0
        self.is_greater = False
    
    def compute(self, val_a_bf16, val_b_bf16, valid=True):
        """比较两个BF16值，返回较大值"""
        self.valid = valid
        if not valid:
            self.result = 0
            self.is_greater = False
            return
            
        val_a_float = bf16_to_float(val_a_bf16)
        val_b_float = bf16_to_float(val_b_bf16)
        
        # todo: using hw float to compare
        if val_a_float > val_b_float:
            self.result = val_a_bf16
            self.is_greater = True
        else:
            self.result = val_b_bf16
            self.is_greater = False

class DivideUnit:
    """除法运算单元，实现BF16除法"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.valid = False
        self.result = 0
    
    def compute(self, dividend_bf16, divisor_bf16, valid=True):
        """计算dividend / divisor"""
        self.valid = valid
        if not valid:
            self.result = 0
            return
            
        dividend_float = bf16_to_float(dividend_bf16)
        divisor_float = bf16_to_float(divisor_bf16)
        
        if divisor_float == 0.0:
            # 除零情况
            if dividend_float > 0:
                self.result = 0x7F80  # +Inf in BF16
            elif dividend_float < 0:
                self.result = 0xFF80  # -Inf in BF16
            else:
                self.result = 0x7FC0  # NaN in BF16
        else:
            try:
                # todo: using hw float to divide
                result_float = dividend_float / divisor_float
                self.result = convert_through_pipeline(result_float)
            except (OverflowError, ZeroDivisionError):
                self.result = 0x7FC0  # NaN in BF16

class RowState:
    """单行数据的处理状态"""
    def __init__(self, row_id, row_length, front_window, back_window):
        self.row_id = row_id
        self.row_length = row_length
        self.front_window = front_window
        self.back_window = back_window
        
        # 处理阶段
        self.phase = "collecting_window"  # collecting_window, processing_window_exp, processing_remaining_exp, division, completed
        
        # 数据收集
        self.window_data = {}  # {col_idx: val_bf16}
        self.remaining_data = {}  # {col_idx: val_bf16}
        self.all_data = {}     # {col_idx: val_bf16}
        self.received_count = 0
        
        # 窗口收集状态
        self.front_window_received = set()
        self.back_window_received = set()
        self.window_complete = False
        
        # 计算状态
        self.window_max = None  # 窗口内的最大值
        self.global_max = None  # 全局最大值
        self.current_max = None  # 当前使用的最大值（初始为窗口最大值，可能会更新）
        self.exp_values = {}  # {col_idx: exp_val_bf16}
        self.exp_sum = convert_through_pipeline(0.0)
        self.final_results = {}  # {col_idx: softmax_val_bf16}
        
        # 处理进度
        self.window_exp_idx = 0  # 窗口exp处理索引
        self.remaining_exp_idx = 0  # 非窗口exp处理索引
        self.division_processing_idx = 0
        
        # 溢出检测
        self.overflow_detected = False
        self.max_updated = False
        self.window_exp_completed = False
        self.remaining_exp_completed = False
        
        # 窗口和非窗口数据的有序列表
        self.window_indices = []
        self.remaining_indices = []

    def is_window_data(self, col_idx):
        """判断是否为窗口数据"""
        return (col_idx < self.front_window or 
                col_idx >= (self.row_length - self.back_window))
    
    def add_data(self, col_idx, val_bf16):
        """添加数据点"""
        self.all_data[col_idx] = val_bf16
        self.received_count += 1
        
        if self.is_window_data(col_idx):
            self.window_data[col_idx] = val_bf16
            if col_idx < self.front_window:
                self.front_window_received.add(col_idx)
            if col_idx >= (self.row_length - self.back_window):
                self.back_window_received.add(col_idx)
            
            # 更新窗口最大值
            if self.window_max is None:
                self.window_max = val_bf16
                self.current_max = val_bf16
                self.global_max = val_bf16
            else:
                # 比较并更新窗口最大值
                val_float = bf16_to_float(val_bf16)
                max_float = bf16_to_float(self.window_max)
                if val_float > max_float:
                    self.window_max = val_bf16
                    self.current_max = val_bf16
                    self.global_max = val_bf16
            
            # 添加到窗口索引列表
            if col_idx not in self.window_indices:
                self.window_indices.append(col_idx)
            
            # 检查窗口是否收集完成
            front_needed = min(self.front_window, self.row_length)
            back_needed = min(self.back_window, self.row_length)
            
            if (len(self.front_window_received) == front_needed and 
                len(self.back_window_received) == back_needed and 
                not self.window_complete):
                self.window_complete = True
                self.phase = "processing_window_exp"
                print(f"   行{self.row_id}: 窗口收集完成，开始处理窗口exp，最大值 {hex(self.window_max)} ({bf16_to_float(self.window_max):.3f})")
                # 对窗口索引排序以便有序处理
                self.window_indices.sort()
        else:
            # 非窗口数据
            self.remaining_data[col_idx] = val_bf16
            
            # 添加到非窗口索引列表
            if col_idx not in self.remaining_indices:
                self.remaining_indices.append(col_idx)
            
            # 如果已经在处理非窗口exp，则需要检查是否需要更新最大值
            if self.phase == "processing_remaining_exp":
                val_float = bf16_to_float(val_bf16)
                max_float = bf16_to_float(self.current_max)
                
                if val_float > max_float:
                    # 新的最大值，需要重新计算
                    self.current_max = val_bf16
                    self.global_max = val_bf16
                    self.overflow_detected = True
                    self.max_updated = True
                    print(f"   行{self.row_id}: 非窗口数据更新最大值 {hex(self.current_max)} ({bf16_to_float(self.current_max):.3f})")
    
    def can_process_window_exp(self):
        """检查是否可以处理窗口exp值"""
        return (self.phase == "processing_window_exp" and 
                self.window_exp_idx < len(self.window_indices) and
                not self.window_exp_completed)
    
    def get_next_window_exp_data(self):
        """获取下一个需要处理exp的窗口数据"""
        if not self.can_process_window_exp():
            return None, None
            
        if self.window_exp_idx < len(self.window_indices):
            col_idx = self.window_indices[self.window_exp_idx]
            val_bf16 = self.window_data[col_idx]
            return col_idx, val_bf16
        return None, None
    
    def process_window_exp_result(self, col_idx, exp_result, is_overflow=False):
        """处理窗口exp计算结果"""
        if is_overflow:
            print(f"   行{self.row_id}: 窗口exp溢出检测 列{col_idx}")
            # 窗口内不应该有溢出，如果有，说明有问题
            self.overflow_detected = True
        else:
            # 存储exp结果
            self.exp_values[col_idx] = exp_result
            
            # 累加到exp_sum
            current_sum = bf16_to_float(self.exp_sum)
            exp_val = bf16_to_float(exp_result)
            new_sum = convert_through_pipeline(current_sum + exp_val)
            self.exp_sum = new_sum
        
        self.window_exp_idx += 1
        
        # 检查是否完成窗口exp处理
        if self.window_exp_idx >= len(self.window_indices):
            self.window_exp_completed = True
            self.phase = "processing_remaining_exp"
            print(f"   行{self.row_id}: 窗口exp处理完成，开始处理非窗口exp")
            # 对非窗口索引排序以便有序处理
            self.remaining_indices.sort()
    
    def can_process_remaining_exp(self):
        """检查是否可以处理非窗口exp值"""
        return (self.phase == "processing_remaining_exp" and 
                self.remaining_exp_idx < len(self.remaining_indices) and
                not self.remaining_exp_completed)
    
    def get_next_remaining_exp_data(self):
        """获取下一个需要处理exp的非窗口数据"""
        if not self.can_process_remaining_exp():
            return None, None
            
        if self.remaining_exp_idx < len(self.remaining_indices):
            col_idx = self.remaining_indices[self.remaining_exp_idx]
            val_bf16 = self.remaining_data[col_idx]
            return col_idx, val_bf16
        return None, None
    
    def process_remaining_exp_result(self, col_idx, exp_result, is_overflow=False):
        """处理非窗口exp计算结果"""
        if is_overflow:
            print(f"   行{self.row_id}: 非窗口exp溢出检测 列{col_idx}")
            self.overflow_detected = True
            # 更新最大值
            val_bf16 = self.remaining_data[col_idx]
            if bf16_to_float(val_bf16) > bf16_to_float(self.current_max):
                self.current_max = val_bf16
                self.global_max = val_bf16
                self.max_updated = True
                print(f"   行{self.row_id}: 更新最大值 {hex(self.current_max)} ({bf16_to_float(self.current_max):.3f})")
        else:
            # 存储exp结果
            self.exp_values[col_idx] = exp_result
            
            # 累加到exp_sum
            current_sum = bf16_to_float(self.exp_sum)
            exp_val = bf16_to_float(exp_result)
            new_sum = convert_through_pipeline(current_sum + exp_val)
            self.exp_sum = new_sum
        
        self.remaining_exp_idx += 1
        
        # 检查是否完成非窗口exp处理
        if self.remaining_exp_idx >= len(self.remaining_indices):
            self.remaining_exp_completed = True
            
            if self.overflow_detected and self.max_updated:
                # 需要重新计算所有exp值
                self._restart_exp_processing()
            else:
                # 开始除法阶段
                self.phase = "division"
                print(f"   行{self.row_id}: 开始除法，exp_sum = {bf16_to_float(self.exp_sum):.6f}")
    
    def _restart_exp_processing(self):
        """重新开始exp处理（溢出后）"""
        print(f"   行{self.row_id}: 回滚重新计算exp，使用新的最大值 {bf16_to_float(self.current_max):.3f}")
        self.exp_values.clear()
        self.exp_sum = convert_through_pipeline(0.0)
        self.window_exp_idx = 0
        self.remaining_exp_idx = 0
        self.window_exp_completed = False
        self.remaining_exp_completed = False
        self.overflow_detected = False
        self.max_updated = False
        self.phase = "processing_window_exp"
    
    def can_process_division(self):
        """检查是否可以处理除法"""
        return (self.phase == "division" and 
                self.division_processing_idx < len(self.exp_values))
    
    def get_next_division_data(self):
        """获取下一个需要除法的数据"""
        if not self.can_process_division():
            return None, None
            
        sorted_cols = sorted(self.exp_values.keys())
        if self.division_processing_idx < len(sorted_cols):
            col_idx = sorted_cols[self.division_processing_idx]
            exp_val = self.exp_values[col_idx]
            return col_idx, exp_val
        return None, None
    
    def process_division_result(self, col_idx, division_result):
        """处理除法结果"""
        self.final_results[col_idx] = division_result
        self.division_processing_idx += 1
        
        # 检查是否完成
        if self.division_processing_idx >= len(self.exp_values):
            self.phase = "completed"
            print(f"   行{self.row_id}: 处理完成！")
    
    def is_completed(self):
        """检查是否完成"""
        return self.phase == "completed"
    
    def get_results(self):
        """获取结果"""
        if self.is_completed():
            return {col_idx: self.final_results[col_idx] 
                   for col_idx in sorted(self.final_results.keys())}
        return None

class SoftmaxPipeline:
    def __init__(self, front_window=8, back_window=8, max_rows=4):
        """
        初始化Softmax流水线 - 支持多行并行处理
        
        Args:
            front_window: 前窗口大小，用于估算最大值
            back_window: 后窗口大小，用于估算最大值
            max_rows: 最大同时处理的行数
        """
        self.front_window = front_window
        self.back_window = back_window
        self.max_rows = max_rows
        
        # 硬件单元 - 每行一套
        self.exp_units = [ExpUnit() for _ in range(max_rows)]
        self.compare_units = [CompareUnit() for _ in range(max_rows)]
        self.divide_units = [DivideUnit() for _ in range(max_rows)]
        
        # BF16计算单元 - 共享
        self.bf16_add = BF16AddPipeline()
        self.bf16_multiply = BF16MultiplyPipeline()
        
        # 每行的处理状态
        self.row_states = {}  # {row_id: RowState}
        
        # 全局状态
        self.cycle_count = 0
        
        print(f"   SoftmaxPipeline 初始化完成")
        print(f"   前窗口大小: {self.front_window}")
        print(f"   后窗口大小: {self.back_window}")
        print(f"   最大并行行数: {self.max_rows}")

    def get_or_create_row_state(self, row_id, row_length):
        """获取或创建行状态"""
        if row_id not in self.row_states:
            self.row_states[row_id] = RowState(row_id, row_length, 
                                             self.front_window, self.back_window)
            print(f"   创建行{row_id}状态，长度{row_length}")
        return self.row_states[row_id]

    def clock_cycle(self, valid=False, val_bf16=None, row_id=None, col_idx=None, row_length=None):
        """
        时钟周期推进 - 实现真正的流水线计算
        
        Args:
            valid: 输入是否有效
            val_bf16: BF16格式的输入值
            row_id: 行ID
            col_idx: 列索引
            row_length: 行长度（新行时需要）
        
        Returns:
            dict: 当前周期的状态信息
        """
        self.cycle_count += 1
        
        result = {
            "cycle": self.cycle_count,
            "completed_rows": {},
            "active_rows": list(self.row_states.keys())
        }
        
        # 处理新输入
        if valid and val_bf16 is not None and row_id is not None and col_idx is not None:
            if row_length is None:
                row_length = 0  # 需要从外部提供
            
            row_state = self.get_or_create_row_state(row_id, row_length)
            row_state.add_data(col_idx, val_bf16)
        
        # 处理各行的窗口exp计算
        for row_id, row_state in list(self.row_states.items()):
            if row_state.can_process_window_exp():
                col_idx, val_bf16 = row_state.get_next_window_exp_data()
                if col_idx is not None:
                    # 计算 val - max
                    val_float = bf16_to_float(val_bf16)
                    max_float = bf16_to_float(row_state.current_max)
                    diff_float = val_float - max_float
                    diff_bf16 = convert_through_pipeline(diff_float)
                    
                    # 计算exp
                    exp_unit = self.exp_units[row_id % self.max_rows]
                    exp_unit.compute(diff_bf16)
                    exp_result = exp_unit.output_val
                    
                    # 检查溢出
                    is_overflow = (exp_result == 0x7FC0 or exp_result == 0x7F80 or exp_result == 0xFF80)
                    
                    row_state.process_window_exp_result(col_idx, exp_result, is_overflow)
        
        # 处理各行的非窗口exp计算
        for row_id, row_state in list(self.row_states.items()):
            if row_state.can_process_remaining_exp():
                col_idx, val_bf16 = row_state.get_next_remaining_exp_data()
                if col_idx is not None:
                    # 计算 val - max
                    val_float = bf16_to_float(val_bf16)
                    max_float = bf16_to_float(row_state.current_max)
                    diff_float = val_float - max_float
                    diff_bf16 = convert_through_pipeline(diff_float)
                    
                    # 计算exp
                    exp_unit = self.exp_units[row_id % self.max_rows]
                    exp_unit.compute(diff_bf16)
                    exp_result = exp_unit.output_val
                    
                    # 检查溢出
                    is_overflow = (exp_result == 0x7FC0 or exp_result == 0x7F80 or exp_result == 0xFF80)
                    
                    row_state.process_remaining_exp_result(col_idx, exp_result, is_overflow)
        
        # 处理各行的除法计算
        for row_id, row_state in list(self.row_states.items()):
            if row_state.can_process_division():
                col_idx, exp_val = row_state.get_next_division_data()
                if col_idx is not None:
                    # 计算除法
                    divide_unit = self.divide_units[row_id % self.max_rows]
                    divide_unit.compute(exp_val, row_state.exp_sum)
                    division_result = divide_unit.result
                    
                    row_state.process_division_result(col_idx, division_result)
        
        # 推进BF16流水线
        self.bf16_add.clock_cycle()
        self.bf16_multiply.clock_cycle()
        
        # 收集完成的行
        completed_rows = {}
        for row_id, row_state in list(self.row_states.items()):
            if row_state.is_completed():
                completed_rows[row_id] = row_state.get_results()
                # 清理已完成的行状态，避免内存累积
                del self.row_states[row_id]
        
        result["completed_rows"] = completed_rows
        
        return result

    def is_active(self):
        """检查流水线是否活跃"""
        # 检查是否有未完成的行
        has_active_rows = any(not row_state.is_completed() for row_state in self.row_states.values())
        
        return (has_active_rows or 
                self.bf16_add.is_active() or 
                self.bf16_multiply.is_active())

    def get_completed_results(self):
        """获取所有已完成的结果"""
        results = {}
        for row_id, row_state in self.row_states.items():
            if row_state.is_completed():
                results[row_id] = row_state.get_results()
        return results

    def run_pipeline(self, input_data, max_cycles=1000, print_progress=True):
        """
        运行流水线处理数据
        
        Args:
            input_data: 输入数据列表，每个元素是(val, row_id, col_idx, row_length)
            max_cycles: 最大周期数
            print_progress: 是否打印进度
            
        Returns:
            dict: 处理结果
        """
        if print_progress:
            print(f"🚀 开始流水线处理，共 {len(input_data)} 个数据点")
        
        results = {}
        input_idx = 0
        
        with tqdm(total=len(input_data) + 100, desc="流水线处理", disable=not print_progress) as pbar:
            cycle = 0
            while (input_idx < len(input_data) or self.is_active()) and cycle < max_cycles:
                # 获取当前输入
                if input_idx < len(input_data):
                    val, row_id, col_idx, row_length = input_data[input_idx]
                    val_bf16 = convert_through_pipeline(val) if isinstance(val, float) else val
                    result = self.clock_cycle(True, val_bf16, row_id, col_idx, row_length)
                    input_idx += 1
                else:
                    result = self.clock_cycle(False)
                
                # 收集完成的结果
                for row_id, row_result in result["completed_rows"].items():
                    if row_id not in results:
                        results[row_id] = row_result
                        if print_progress:
                            print(f"✅ 行{row_id}处理完成")
                
                cycle += 1
                pbar.update(1)
                
                if cycle >= pbar.total:
                    pbar.total = cycle + 50
                    pbar.refresh()
        
        if print_progress:
            print(f"🎉 流水线处理完成，总共 {cycle} 个周期")
        
        return results

def example_test():
    """测试新实现的流水线架构 - 逐个数据点输入"""
    print("=" * 60)
    print("🌟 测试新实现的SoftmaxPipeline流水线架构")
    print("=" * 60)
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=3, back_window=3, max_rows=2)
    
    # 测试数据集：包含窗口内和窗口外的最大值
    print("\n📋 测试：窗口外有更大值的情况")
    input_data = [1.0, 2.0, 3.0, 50.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    row_length = len(input_data)
    
    # 准备输入数据：(val, row_id, col_idx, row_length)
    # 窗口数据优先：前3个(0,1,2)和后3个(7,8,9)先发送
    pipeline_input = []
    
    # 前窗口数据
    for i in [0, 1, 2]:  # front_window = 3
        pipeline_input.append((input_data[i], 0, i, row_length))
    
    # 后窗口数据
    for i in [7, 8, 9]:  # back_window = 3
        pipeline_input.append((input_data[i], 0, i, row_length))
    
    # 剩余数据，包含真正的最大值50.0
    for i in [3, 4, 5, 6]:
        pipeline_input.append((input_data[i], 0, i, row_length))
    
    print("🔄 开始逐个时钟周期处理数据...")
    print("   总输入数据点: {}".format(len(pipeline_input)))
    
    # 手动推进时钟周期
    results = {}
    cycle = 0
    input_idx = 0
    
    print("\n🕒 开始时钟周期推进:")
    while (input_idx < len(pipeline_input) or pipeline.is_active()) and cycle < 500:
        cycle += 1
        
        # 当前周期的输入
        if input_idx < len(pipeline_input):
            val, row_id, col_idx, row_length = pipeline_input[input_idx]
            val_bf16 = convert_through_pipeline(val) if isinstance(val, float) else val
            
            # 打印当前输入
            if cycle <= 20 or cycle % 50 == 0 or input_idx == len(pipeline_input) - 1:
                print(f"   周期 {cycle}: 输入 位置{col_idx}={val}")
            
            # 推进时钟周期
            result = pipeline.clock_cycle(True, val_bf16, row_id, col_idx, row_length)
            input_idx += 1
        else:
            # 无输入数据，继续推进时钟
            if cycle <= 20 or cycle % 50 == 0:
                print(f"   周期 {cycle}: 无输入，继续处理")
            result = pipeline.clock_cycle(False)
        
        # 收集完成的结果
        for row_id, row_result in result["completed_rows"].items():
            if row_id not in results:
                results[row_id] = row_result
                print(f"✅ 周期 {cycle}: 行{row_id}处理完成")
    
    print(f"\n🎉 时钟周期推进完成，总共 {cycle} 个周期")
    
    # 验证结果
    if 0 in results:
        result_row0 = results[0]
        print(f"\n📊 处理结果：")
        
        # 转换为浮点数并排序
        softmax_results = []
        for col_idx in sorted(result_row0.keys()):
            softmax_val = bf16_to_float(result_row0[col_idx])
            softmax_results.append(softmax_val)
            print(f"  位置{col_idx}: {softmax_val:.6f}")
        
        # 验证和为1
        total_sum = sum(softmax_results)
        print(f"\n✅ 概率和: {total_sum:.6f} (应该接近1.0)")
        
        # 验证最大值位置的softmax应该最大
        max_idx = input_data.index(max(input_data))
        max_softmax = softmax_results[max_idx]
        print(f"✅ 最大值位置{max_idx}的softmax: {max_softmax:.6f}")
        print(f"   应该是所有值中最大的: {max_softmax == max(softmax_results)}")
        
        # 与PyTorch BF16结果对比
        import torch
        torch_input = torch.tensor(input_data, dtype=torch.float32).bfloat16()
        torch_softmax = torch.softmax(torch_input, dim=0).float()
        
        print(f"\n🔍 与PyTorch BF16结果对比：")
        max_error = 0.0
        for i, (custom, torch_val) in enumerate(zip(softmax_results, torch_softmax.tolist())):
            error = abs(custom - torch_val)
            max_error = max(max_error, error)
            print(f"  位置{i}: 自定义={custom:.6f}, PyTorch={torch_val:.6f}, 误差={error:.6f}")
        print(f"📈 最大误差: {max_error:.6f}")
    else:
        print("❌ 测试失败：未获得行0的结果")
    
    # 测试2：多行并行处理
    print(f"\n📋 测试2：多行并行处理 - 逐个时钟周期")
    
    # 重新创建流水线
    pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=2)
    
    # 准备两行数据
    row0_data = [1.0, 2.0, 3.0, 4.0, 5.0]  # 行0
    row1_data = [10.0, 20.0, 30.0, 5.0, 15.0]  # 行1，注意第3个位置有最大值
    
    # 准备交错的输入数据
    multi_input = []
    
    # 交错发送窗口数据
    multi_input.append((row0_data[0], 0, 0, len(row0_data)))  # 行0前窗口
    multi_input.append((row1_data[0], 1, 0, len(row1_data)))  # 行1前窗口
    multi_input.append((row0_data[1], 0, 1, len(row0_data)))  # 行0前窗口
    multi_input.append((row1_data[1], 1, 1, len(row1_data)))  # 行1前窗口
    multi_input.append((row0_data[3], 0, 3, len(row0_data)))  # 行0后窗口
    multi_input.append((row1_data[3], 1, 3, len(row1_data)))  # 行1后窗口
    multi_input.append((row0_data[4], 0, 4, len(row0_data)))  # 行0后窗口
    multi_input.append((row1_data[4], 1, 4, len(row1_data)))  # 行1后窗口
    
    # 发送剩余数据
    multi_input.append((row0_data[2], 0, 2, len(row0_data)))  # 行0剩余
    multi_input.append((row1_data[2], 1, 2, len(row1_data)))  # 行1剩余，包含最大值
    
    print("🔄 开始多行逐个时钟周期处理...")
    print("   总输入数据点: {}".format(len(multi_input)))
    
    # 手动推进时钟周期
    multi_results = {}
    cycle = 0
    input_idx = 0
    
    print("\n🕒 开始时钟周期推进:")
    while (input_idx < len(multi_input) or pipeline.is_active()) and cycle < 500:
        cycle += 1
        
        # 当前周期的输入
        if input_idx < len(multi_input):
            val, row_id, col_idx, row_length = multi_input[input_idx]
            val_bf16 = convert_through_pipeline(val) if isinstance(val, float) else val
            
            # 打印当前输入
            if cycle <= 20 or cycle % 50 == 0 or input_idx == len(multi_input) - 1:
                print(f"   周期 {cycle}: 输入 行{row_id}位置{col_idx}={val}")
            
            # 推进时钟周期
            result = pipeline.clock_cycle(True, val_bf16, row_id, col_idx, row_length)
            input_idx += 1
        else:
            # 无输入数据，继续推进时钟
            if cycle <= 20 or cycle % 50 == 0:
                print(f"   周期 {cycle}: 无输入，继续处理")
            result = pipeline.clock_cycle(False)
        
        # 收集完成的结果
        for row_id, row_result in result["completed_rows"].items():
            if row_id not in multi_results:
                multi_results[row_id] = row_result
                print(f"✅ 周期 {cycle}: 行{row_id}处理完成")
    
    print(f"\n🎉 时钟周期推进完成，总共 {cycle} 个周期")
    
    # 验证多行结果
    for row_id in [0, 1]:
        if row_id in multi_results:
            result_row = multi_results[row_id]
            print(f"\n📊 行{row_id}结果：")
            
            softmax_results = []
            for col_idx in sorted(result_row.keys()):
                softmax_val = bf16_to_float(result_row[col_idx])
                softmax_results.append(softmax_val)
                print(f"  位置{col_idx}: {softmax_val:.6f}")
            
            total_sum = sum(softmax_results)
            print(f"✅ 行{row_id}概率和: {total_sum:.6f}")
            
            # 验证最大值
            if row_id == 1:  # 行1有明确的最大值
                max_idx = row1_data.index(max(row1_data))
                max_softmax = softmax_results[max_idx]
                print(f"✅ 行1最大值位置{max_idx}的softmax: {max_softmax:.6f}")
                print(f"   应该是所有值中最大的: {max_softmax == max(softmax_results)}")
        else:
            print(f"❌ 测试2失败：未获得行{row_id}的结果")
    
    print("\n🎯 流水线行为验证:")
    print("   1. 窗口数据收集过程中就开始计算最大值")
    print("   2. 窗口数据收集完成后立即开始处理窗口exp")
    print("   3. 处理非窗口数据时，如果发现更大的值，会触发重新计算")
    print("   4. 多行数据可以并行处理")
    print("   5. 每行有独立的最大值计算和exp处理")
    print("   6. 流水线实现了真正的逐个时钟周期处理")
    
    print(f"\n🎉 新实现的SoftmaxPipeline测试完成！")

if __name__ == "__main__":
    example_test()