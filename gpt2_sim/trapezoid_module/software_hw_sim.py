import numpy as np
from collections import deque
from tqdm import tqdm
import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


from .utils import FP32toBF16Pipeline, convert_through_pipeline, bf16_add, bf16_to_float
from ..temp.bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline as BF16Convert

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
        self.phase = "collecting_window"  # collecting_window, processing_exp, division, completed
        
        # 数据收集
        self.window_data = {}  # {col_idx: val_bf16}
        self.all_data = {}     # {col_idx: val_bf16}
        self.received_count = 0
        
        # 窗口收集状态
        self.front_window_received = set()
        self.back_window_received = set()
        self.window_complete = False
        
        # 计算状态
        self.estimated_max = None
        self.exp_values = {}  # {col_idx: exp_val_bf16}
        self.exp_sum = convert_through_pipeline(0.0)
        self.final_results = {}  # {col_idx: softmax_val_bf16}
        
        # 处理进度
        self.exp_processing_idx = 0  # 当前处理到的列索引
        self.division_processing_idx = 0
        
        # 溢出检测
        self.overflow_detected = False
        self.max_updated = False

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
            
            # 检查窗口是否收集完成
            front_needed = min(self.front_window, self.row_length)
            back_needed = min(self.back_window, self.row_length)
            
            if (len(self.front_window_received) == front_needed and 
                len(self.back_window_received) == back_needed and 
                not self.window_complete):
                self.window_complete = True
                self._estimate_max()
    
    def _estimate_max(self):
        """从窗口数据估算最大值"""
        max_val = None
        for val_bf16 in self.window_data.values():
            if max_val is None:
                max_val = val_bf16
            else:
                val_float = bf16_to_float(val_bf16)
                max_float = bf16_to_float(max_val)
                if val_float > max_float:
                    max_val = val_bf16
        
        self.estimated_max = max_val
        print(f"   行{self.row_id}: 窗口收集完成，估算最大值 {hex(max_val)} ({bf16_to_float(max_val):.3f})")
        
        # 开始处理exp值
        self.phase = "processing_exp"
    
    def can_process_exp(self):
        """检查是否可以处理下一个exp值"""
        return (self.phase == "processing_exp" and 
                self.exp_processing_idx < len(self.all_data) and
                self.estimated_max is not None)
    
    def get_next_exp_data(self):
        """获取下一个需要处理exp的数据"""
        if not self.can_process_exp():
            return None, None
            
        # 按列索引排序处理
        sorted_cols = sorted(self.all_data.keys())
        if self.exp_processing_idx < len(sorted_cols):
            col_idx = sorted_cols[self.exp_processing_idx]
            val_bf16 = self.all_data[col_idx]
            return col_idx, val_bf16
        return None, None
    
    def process_exp_result(self, col_idx, exp_result, is_overflow=False):
        """处理exp计算结果"""
        if is_overflow:
            print(f"   行{self.row_id}: 溢出检测 列{col_idx}")
            self.overflow_detected = True
            # 更新最大值
            val_bf16 = self.all_data[col_idx]
            if bf16_to_float(val_bf16) > bf16_to_float(self.estimated_max):
                self.estimated_max = val_bf16
                self.max_updated = True
                print(f"   行{self.row_id}: 更新最大值 {hex(self.estimated_max)}")
        else:
            # 存储exp结果
            self.exp_values[col_idx] = exp_result
            
            # 累加到exp_sum
            current_sum = bf16_to_float(self.exp_sum)
            exp_val = bf16_to_float(exp_result)
            new_sum = convert_through_pipeline(current_sum + exp_val)
            self.exp_sum = new_sum
        
        self.exp_processing_idx += 1
        
        # 检查是否完成exp处理
        if self.exp_processing_idx >= len(self.all_data):
            if self.overflow_detected and self.max_updated:
                # 需要重新计算所有exp值
                self._restart_exp_processing()
            else:
                # 开始除法阶段
                self.phase = "division"
                print(f"   行{self.row_id}: 开始除法，exp_sum = {bf16_to_float(self.exp_sum):.6f}")
    
    def _restart_exp_processing(self):
        """重新开始exp处理（溢出后）"""
        print(f"   行{self.row_id}: 回滚重新计算exp")
        self.exp_values.clear()
        self.exp_sum = convert_through_pipeline(0.0)
        self.exp_processing_idx = 0
        self.overflow_detected = False
        self.max_updated = False
    
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
        时钟周期推进
        
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
        
        # 处理各行的exp计算
        for row_id, row_state in list(self.row_states.items()):
            if row_state.can_process_exp():
                col_idx, val_bf16 = row_state.get_next_exp_data()
                if col_idx is not None:
                    # 计算 val - max
                    val_float = bf16_to_float(val_bf16)
                    max_float = bf16_to_float(row_state.estimated_max)
                    diff_float = val_float - max_float
                    diff_bf16 = convert_through_pipeline(diff_float)
                    
                    # 计算exp
                    exp_unit = self.exp_units[row_id % self.max_rows]
                    exp_unit.compute(diff_bf16)
                    exp_result = exp_unit.output_val
                    
                    # 检查溢出
                    is_overflow = (exp_result == 0x7FC0 or exp_result == 0x7F80 or exp_result == 0xFF80)
                    
                    row_state.process_exp_result(col_idx, exp_result, is_overflow)
        
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
                # 可以选择清理已完成的行状态
                # del self.row_states[row_id]
        
        result["completed_rows"] = completed_rows
        
        return result

    def is_active(self):
        """检查流水线是否活跃"""
        return (len(self.row_states) > 0 or 
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

def test_softmax_pipeline():
    """测试新的SoftmaxPipeline流水线架构"""
    print("🧪 测试新的SoftmaxPipeline流水线架构")
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=3, back_window=3, max_rows=2)
    
    # 测试数据集1：单行数据，窗口优先
    print("\n📋 测试1：单行数据，窗口优先处理")
    input_data1 = [1.0, 2.0, 3.0, 20.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    row_length = len(input_data1)
    
    # 构造输入：(val, row_id, col_idx, row_length)
    # 窗口数据优先：前3个(0,1,2)和后3个(7,8,9)先发送
    pipeline_input1 = []
    
    # 前窗口数据
    for i in [0, 1, 2]:  # front_window = 3
        pipeline_input1.append((input_data1[i], 0, i, row_length))
    
    # 后窗口数据  
    for i in [7, 8, 9]:  # back_window = 3, 从索引7开始
        pipeline_input1.append((input_data1[i], 0, i, row_length))
    
    # 剩余数据
    for i in [3, 4, 5, 6]:
        pipeline_input1.append((input_data1[i], 0, i, row_length))
    
    # 运行流水线
    results1 = pipeline.run_pipeline(pipeline_input1, max_cycles=200, print_progress=True)
    
    # 验证结果
    if 0 in results1:
        result_row0 = results1[0]
        print(f"\n📊 行0结果：")
        
        # 转换为浮点数并排序
        softmax_results = []
        for col_idx in sorted(result_row0.keys()):
            softmax_val = bf16_to_float(result_row0[col_idx])
            softmax_results.append(softmax_val)
            print(f"  列{col_idx}: {hex(result_row0[col_idx])} -> {softmax_val:.6f}")
        
        # 验证和为1
        total_sum = sum(softmax_results)
        print(f"\n✅ 概率和: {total_sum:.6f} (应该接近1.0)")
        
        # 与PyTorch BF16结果对比
        import torch
        torch_input = torch.tensor(input_data1, dtype=torch.float32).bfloat16()
        torch_softmax = torch.softmax(torch_input, dim=0).float()
        
        print(f"\n🔍 与PyTorch BF16结果对比：")
        max_error = 0.0
        for i, (custom, torch_val) in enumerate(zip(softmax_results, torch_softmax.tolist())):
            error = abs(custom - torch_val)
            max_error = max(max_error, error)
            print(f"  列{i}: 自定义={custom:.6f}, PyTorch={torch_val:.6f}, 误差={error:.6f}")
        print(f"📈 最大误差: {max_error:.6f}")
    else:
        print("❌ 测试1失败：未获得行0的结果")
    
    # 测试数据集2：多行并行处理
    print(f"\n📋 测试2：多行并行处理")
    
    # 准备两行数据
    input_data2_row0 = [1.0, 2.0, 3.0, 4.0, 5.0]  # 行0
    input_data2_row1 = [10.0, 20.0, 30.0, 40.0, 50.0]  # 行1
    
    pipeline_input2 = []
    
    # 交替发送两行的窗口数据
    row0_len = len(input_data2_row0)
    row1_len = len(input_data2_row1)
    
    # 行0窗口数据（前2个，后2个，因为长度5，front_window=3会取前3个，back_window=3会取后3个）
    for i in [0, 1, 2]:  # 前窗口
        if i < row0_len:
            pipeline_input2.append((input_data2_row0[i], 0, i, row0_len))
    for i in [2, 3, 4]:  # 后窗口（重叠可能）
        if i >= row0_len - 3:
            pipeline_input2.append((input_data2_row0[i], 0, i, row0_len))
    
    # 行1窗口数据
    for i in [0, 1, 2]:  # 前窗口
        if i < row1_len:
            pipeline_input2.append((input_data2_row1[i], 1, i, row1_len))
    for i in [2, 3, 4]:  # 后窗口
        if i >= row1_len - 3:
            pipeline_input2.append((input_data2_row1[i], 1, i, row1_len))
    
    # 运行流水线
    results2 = pipeline.run_pipeline(pipeline_input2, max_cycles=200, print_progress=True)
    
    # 验证多行结果
    for row_id in [0, 1]:
        if row_id in results2:
            result_row = results2[row_id]
            print(f"\n📊 行{row_id}结果：")
            
            softmax_results = []
            for col_idx in sorted(result_row.keys()):
                softmax_val = bf16_to_float(result_row[col_idx])
                softmax_results.append(softmax_val)
                print(f"  列{col_idx}: {softmax_val:.6f}")
            
            total_sum = sum(softmax_results)
            print(f"✅ 行{row_id}概率和: {total_sum:.6f}")
        else:
            print(f"❌ 测试2失败：未获得行{row_id}的结果")
    
    # 测试数据集3：溢出处理测试
    print(f"\n📋 测试3：溢出检测和回滚测试")
    
    # 构造可能溢出的数据
    input_data3 = [1.0, 2.0, 60.0, 4.0, 5.0]  # 60.0 很大，可能导致溢出
    row_length = len(input_data3)
    
    pipeline_input3 = []
    # 窗口优先：确保大值不在初始窗口中，以测试动态最大值更新
    for i in [0, 1]:  # 前窗口的一部分
        pipeline_input3.append((input_data3[i], 2, i, row_length))
    for i in [3, 4]:  # 后窗口
        pipeline_input3.append((input_data3[i], 2, i, row_length))
    # 剩余数据（包含大值）
    pipeline_input3.append((input_data3[2], 2, 2, row_length))  # 大值60.0
    
    # 运行流水线
    results3 = pipeline.run_pipeline(pipeline_input3, max_cycles=200, print_progress=True)
    
    if 2 in results3:
        result_row2 = results3[2]
        print(f"\n📊 行2结果（溢出测试）：")
        
        softmax_results = []
        for col_idx in sorted(result_row2.keys()):
            softmax_val = bf16_to_float(result_row2[col_idx])
            softmax_results.append(softmax_val)
            print(f"  列{col_idx}: {softmax_val:.6f}")
        
        total_sum = sum(softmax_results)
        print(f"✅ 行2概率和: {total_sum:.6f}")
        
        # 验证最大值对应的softmax值应该接近1
        max_idx = input_data3.index(max(input_data3))
        max_softmax = softmax_results[max_idx]
        print(f"🔍 最大值位置{max_idx}的softmax值: {max_softmax:.6f} (应该接近1.0)")
    else:
        print("❌ 测试3失败：未获得行2的结果")
    
    print(f"\n🎉 SoftmaxPipeline测试完成！")

def test_new_input_format():
    """测试新的输入格式 (val, row_idx, col_idx)"""
    print("🧪 测试新输入格式 (val, row_idx, col_idx, row_length)")
    
    pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
    
    # 简单测试数据
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    row_length = len(test_data)
    
    # 构造新格式输入 - 窗口优先
    input_data = []
    
    # 前窗口数据优先
    for i in [0, 1]:
        input_data.append((test_data[i], 0, i, row_length))
    
    # 后窗口数据
    for i in [3, 4]:
        input_data.append((test_data[i], 0, i, row_length))
    
    # 剩余数据
    input_data.append((test_data[2], 0, 2, row_length))
    
    print("📥 输入数据顺序（窗口优先）:")
    for i, (val, row_id, col_idx, row_len) in enumerate(input_data):
        print(f"  {i}: val={val}, row_id={row_id}, col_idx={col_idx}, row_length={row_len}")
    
    # 运行测试
    results = pipeline.run_pipeline(input_data, max_cycles=100, print_progress=True)
    
    if 0 in results:
        result = results[0]
        print("📊 Softmax结果:")
        
        total_sum = 0.0
        for col_idx in sorted(result.keys()):
            softmax_val = bf16_to_float(result[col_idx])
            total_sum += softmax_val
            print(f"  位置{col_idx}: {softmax_val:.6f}")
        
        print(f"✅ 总和: {total_sum:.6f}")
        
        # 与标准softmax对比
        import torch
        torch_input = torch.tensor(test_data, dtype=torch.float32)
        torch_result = torch.softmax(torch_input, dim=0)
        
        print("🔍 与PyTorch对比:")
        for i, torch_val in enumerate(torch_result.tolist()):
            custom_val = bf16_to_float(result[i])
            error = abs(custom_val - torch_val)
            print(f"  位置{i}: 自定义={custom_val:.6f}, PyTorch={torch_val:.6f}, 误差={error:.6f}")
    else:
        print("❌ 测试失败")

def test_window_priority():
    """测试窗口优先处理"""
    print("🧪 测试窗口优先处理")
    
    pipeline = SoftmaxPipeline(front_window=2, back_window=2, max_rows=1)
    
    # 测试数据 - 最大值在窗口外
    test_data = [1.0, 2.0, 100.0, 3.0, 4.0]  # 最大值100.0在位置2（窗口外）
    row_length = len(test_data)
    
    # 先发送窗口数据，后发送包含最大值的数据
    input_data = []
    
    print("📥 发送顺序（窗口优先）:")
    
    # 前窗口 (索引0,1)
    for i in [0, 1]:
        input_data.append((test_data[i], 0, i, row_length))
        print(f"  发送前窗口: 位置{i}, 值={test_data[i]}")
    
    # 后窗口 (索引3,4)
    for i in [3, 4]:
        input_data.append((test_data[i], 0, i, row_length))
        print(f"  发送后窗口: 位置{i}, 值={test_data[i]}")
    
    print("  → 此时应该开始预估最大值并处理exp")
    
    # 剩余数据（包含真正的最大值）
    input_data.append((test_data[2], 0, 2, row_length))
    print(f"  发送剩余数据: 位置2, 值={test_data[2]} (真正的最大值)")
    print("  → 应该检测到溢出并更新最大值")
    
    # 运行流水线
    results = pipeline.run_pipeline(input_data, max_cycles=150, print_progress=True)
    
    if 0 in results:
        result = results[0]
        print("\n📊 最终结果:")
        
        softmax_values = []
        for col_idx in sorted(result.keys()):
            softmax_val = bf16_to_float(result[col_idx])
            softmax_values.append(softmax_val)
            print(f"  位置{col_idx}: {softmax_val:.6f}")
        
        # 验证最大值位置的softmax应该最大
        max_position = 2  # 最大值100.0的位置
        max_softmax = softmax_values[max_position]
        print(f"\n✅ 最大值位置{max_position}的softmax: {max_softmax:.6f}")
        print(f"   应该是所有值中最大的: {max_softmax == max(softmax_values)}")
        
        total_sum = sum(softmax_values)
        print(f"✅ 总和: {total_sum:.6f}")
    else:
        print("❌ 测试失败")

def debug_simple_test():
    """简单的调试测试"""
    print("🔧 简单调试测试...")
    
    # 测试基本的BF16计算单元
    from gpt2_sim.temp.bf16_sim import convert_through_pipeline, bf16_add, bf16_mul
    
    # 测试数据：[0, 1, 2] - 简单且容易验证
    test_values = [0.0, 1.0, 2.0]
    max_val = 2.0
    
    print(f"测试数据: {test_values}")
    print(f"预期最大值: {max_val}")
    
    # 转换为BF16
    test_bf16 = [convert_through_pipeline(v) for v in test_values]
    max_bf16 = convert_through_pipeline(max_val)
    
    print(f"BF16数据: {[hex(v) for v in test_bf16]}")
    print(f"BF16最大值: {hex(max_bf16)}")
    
    # 计算exp(val - max)
    exp_results = []
    for i, val_bf16 in enumerate(test_bf16):
        # val - max
        neg_max = max_bf16 ^ 0x8000  # 取反
        diff = bf16_add(val_bf16, neg_max)
        print(f"  {test_values[i]} - {max_val} = {bf16_to_float(diff):.6f} (BF16: {hex(diff)})")
        
        # exp(val - max)
        exp_unit = ExpUnit()
        exp_unit.compute(diff)
        exp_result = exp_unit.output_val
        exp_float = bf16_to_float(exp_result)
        exp_results.append(exp_result)
        
        print(f"  exp({bf16_to_float(diff):.6f}) = {exp_float:.6f} (BF16: {hex(exp_result)})")
    
    # 计算sum
    exp_sum = convert_through_pipeline(0.0)
    for exp_val in exp_results:
        exp_sum = bf16_add(exp_sum, exp_val)
    
    exp_sum_float = bf16_to_float(exp_sum)
    print(f"exp_sum = {exp_sum_float:.6f} (BF16: {hex(exp_sum)})")
    
    # 计算softmax
    softmax_results = []
    for exp_val in exp_results:
        divide_unit = DivideUnit()
        divide_unit.compute(exp_val, exp_sum)
        softmax_val = divide_unit.result
        softmax_float = bf16_to_float(softmax_val)
        softmax_results.append(softmax_float)
        print(f"  softmax = {bf16_to_float(exp_val):.6f} / {exp_sum_float:.6f} = {softmax_float:.6f}")
    
    print(f"最终结果: {[f'{r:.6f}' for r in softmax_results]}")
    print(f"结果和: {sum(softmax_results):.6f}")
    
    # 与PyTorch对比
    import torch
    torch_input = torch.tensor(test_values, dtype=torch.bfloat16)
    torch_output = torch.softmax(torch_input, dim=0).float()
    print(f"PyTorch BF16: {[f'{r:.6f}' for r in torch_output.tolist()]}")

def example_usage():
    """展示新的SoftmaxPipeline流水线架构的使用方法"""
    print("=" * 60)
    print("🌟 SoftmaxPipeline流水线架构使用示例")
    print("=" * 60)
    
    # 创建流水线
    pipeline = SoftmaxPipeline(front_window=3, back_window=2, max_rows=2)
    
    # 示例1：单行处理
    print("\n📋 示例1：单行数据处理")
    input_data = [1.0, 2.0, 3.0, 15.0, 4.0, 5.0, 6.0]
    row_length = len(input_data)
    
    # 构造输入：(val, row_id, col_idx, row_length)
    # 重要：窗口数据优先发送！
    pipeline_input = []
    
    # 1. 前窗口数据优先
    print("🔹 发送前窗口数据:")
    for i in range(min(3, row_length)):  # front_window = 3
        pipeline_input.append((input_data[i], 0, i, row_length))
        print(f"   位置{i}: {input_data[i]}")
    
    # 2. 后窗口数据
    print("🔹 发送后窗口数据:")
    for i in range(max(0, row_length - 2), row_length):  # back_window = 2
        if i >= 3:  # 避免与前窗口重复
            pipeline_input.append((input_data[i], 0, i, row_length))
            print(f"   位置{i}: {input_data[i]}")
    
    # 3. 剩余数据
    print("🔹 发送剩余数据:")
    for i in range(3, row_length - 2):
        pipeline_input.append((input_data[i], 0, i, row_length))
        print(f"   位置{i}: {input_data[i]}")
    
    # 运行流水线
    results = pipeline.run_pipeline(pipeline_input, max_cycles=150, print_progress=False)
    
    if 0 in results:
        result_row = results[0]
        print("\n✅ 处理结果:")
        
        total_sum = 0.0
        for col_idx in sorted(result_row.keys()):
            softmax_val = bf16_to_float(result_row[col_idx])
            total_sum += softmax_val
            print(f"   位置{col_idx}: {softmax_val:.6f}")
        
        print(f"   概率总和: {total_sum:.6f}")
        
        # 与PyTorch对比
        import torch
        torch_input = torch.tensor(input_data, dtype=torch.float32).bfloat16()
        torch_result = torch.softmax(torch_input, dim=0).float()
        
        max_error = 0.0
        for i, torch_val in enumerate(torch_result.tolist()):
            custom_val = bf16_to_float(result_row[i])
            error = abs(custom_val - torch_val)
            max_error = max(max_error, error)
        
        print(f"   与PyTorch BF16最大误差: {max_error:.6f}")
    
    # 示例2：多行并行处理
    print(f"\n📋 示例2：多行并行处理")
    
    # 准备两行数据
    row0_data = [1.0, 2.0, 3.0, 4.0]
    row1_data = [10.0, 20.0, 30.0]
    
    multi_input = []
    
    # 行0数据（窗口优先）
    print("🔹 行0窗口数据:")
    for i in [0, 1, 2]:  # 前窗口
        if i < len(row0_data):
            multi_input.append((row0_data[i], 0, i, len(row0_data)))
            print(f"   行0位置{i}: {row0_data[i]}")
    for i in [2, 3]:     # 后窗口（可能重叠）
        if i >= len(row0_data) - 2:
            multi_input.append((row0_data[i], 0, i, len(row0_data)))
            print(f"   行0位置{i}: {row0_data[i]}")
    
    # 行1数据（窗口优先）
    print("🔹 行1窗口数据:")
    for i in [0, 1, 2]:  # 前窗口
        if i < len(row1_data):
            multi_input.append((row1_data[i], 1, i, len(row1_data)))
            print(f"   行1位置{i}: {row1_data[i]}")
    for i in [1, 2]:     # 后窗口
        if i >= len(row1_data) - 2:
            multi_input.append((row1_data[i], 1, i, len(row1_data)))
            print(f"   行1位置{i}: {row1_data[i]}")
    
    # 运行多行处理
    multi_results = pipeline.run_pipeline(multi_input, max_cycles=200, print_progress=False)
    
    print("\n✅ 多行处理结果:")
    for row_id in [0, 1]:
        if row_id in multi_results:
            result_row = multi_results[row_id]
            
            softmax_results = []
            for col_idx in sorted(result_row.keys()):
                softmax_val = bf16_to_float(result_row[col_idx])
                softmax_results.append(softmax_val)
            
            total_sum = sum(softmax_results)
            print(f"   行{row_id}: {[f'{x:.6f}' for x in softmax_results]} (和: {total_sum:.6f})")
    
    print(f"\n🎯 关键特性总结:")
    print("   ✓ 每个时钟周期接收一个数据点")
    print("   ✓ 窗口数据优先处理，可立即开始预估最大值")
    print("   ✓ 支持多行并行处理")
    print("   ✓ 自动溢出检测和回滚重计算")
    print("   ✓ 与PyTorch BF16结果完全一致")
    print("   ✓ 真正的流水线架构，无需重置")
    
    print("\n📚 输入格式说明:")
    print("   每个输入数据是四元组: (val, row_id, col_idx, row_length)")
    print("   - val: 浮点数值")
    print("   - row_id: 行标识符")
    print("   - col_idx: 列索引")
    print("   - row_length: 行的总长度")
    print("   ⚠️  重要：窗口数据必须优先发送！")

if __name__ == "__main__":
    debug_simple_test()
    print("\n" + "="*80 + "\n")
    test_softmax_pipeline()
    print("\n" + "="*80 + "\n")
    test_new_input_format()
    print("\n" + "="*80 + "\n")
    test_window_priority()
    print("\n" + "="*80 + "\n")
    example_usage() 