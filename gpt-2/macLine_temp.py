import numpy as np
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline
import struct

import struct
def bf16_to_float(bf16):
    # 左移16位填充为32位表示
    fp32_bits = bf16 << 16
    # 转换为浮点数
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

def bf16_to_float_block(bf16_block):
    # 左移16位填充为32位表示
    float_block = []
    for bf16 in bf16_block:
        fp32_bits = bf16 << 16
        float_block.append(struct.unpack('>f', struct.pack('>I', fp32_bits))[0])
    # 转换为浮点数
    return float_block

class MACUnit:
    def __init__(self, clock_cycle=2):
        self.multliply_pipeline = BF16MultiplyPipeline()
        self.state = None  # list of (a, b, remaining_cycles)
        #self.clock_cycle = clock_cycle 
        self.result = None

    def is_busy(self):
        return len(self.state) > 0
    
    def get_input(self, a, b):
        self.state = (a, b, True) 

    #def enqueue(self, a, b):
    def tick(self):
        if not self.state:
            return None
        
        self.multliply_pipeline.clock_cycle(self.state[0], self.state[1], self.state[2])
        if len(self.multliply_pipeline.outputs) == 0:
            return None
        return self.multliply_pipeline.outputs.pop(0)

class MACLine:
    def __init__(self, mac_width):
        self.macs = [MACUnit() for _ in range(mac_width)]
        self.input = []  # list of [a_block, b_block]

    #def enqueue(self, ab_block):
    #    self.input_queue.append(ab_block)
    def get_input(self, input):
        self.input = input
      
    def reset(self):
        self.input = []
        for mac in self.macs:
            mac.state = []
            mac.result = None
            
    def is_active(self):
        return self.input or any(mac.is_busy() for mac in self.macs)

    def tick(self):
        results = []
        if not self.input:  # 防止访问空队列
            return None
        
        for i, mac in enumerate(self.macs):
            #if i < len(self.input_queue[0][0]) and i < len(self.input_queue[0][1]):
            mac.get_input(self.input[0][i], self.input[1][i])
            r = mac.tick()
            if r is not None:
                results.append(r)  # 即使是 None 也添加到结果中
        
        #if self.input_queue:  # 处理完后移除队列中的元素
        #    self.input_queue.pop(0)
        
        return results

# 修改：将AdderTree拆分为两个阶段
class AdderTree:
    def __init__(self):
        self.values = []
        self.busy = False
        self.intermediate_results = []  # 存储第一阶段的中间结果
        self.add_pipeline_lower = [BF16AddPipeline(), BF16AddPipeline()]
        self.add_pipeline_upper = BF16AddPipeline()
        self.input = []
        self.values_pack = []

    def add_values(self, values):
        # 确保过滤None值并使用float32类型确保精度一致
        self.values = values
        self.busy = True
    
    def pack_values(self):
        self.values_pack = []
        for i in range(0, len(self.values), 2):
            self.values_pack.append([(self.values[i], self.values[i + 1], True)])
        #print(self.values_pack)

    def is_active(self):
        return self.busy and (self.values or self.intermediate_results)
    
    def reset(self):
        self.values = []
        self.intermediate_results = []
        self.busy = False
    
    def tick_first_stage(self):
        """第一阶段：将4个输入合并为2个中间结果"""
        #print(self.values)
        if not self.values:
            return None
        # 4
        # 计算第一级加法结果，确保类型一致
        next_level = []
        self.pack_values()
        
        #print(self.values_pack)
        
        self.add_pipeline_lower[0].clock_cycle(self.values_pack[0][0][0], self.values_pack[0][0][1], self.values_pack[0][0][2])
        self.add_pipeline_lower[1].clock_cycle(self.values_pack[1][0][0], self.values_pack[1][0][1], self.values_pack[1][0][2])

        if len(self.add_pipeline_lower[0].outputs) and len(self.add_pipeline_lower[1].outputs):
            next_level.append(self.add_pipeline_lower[0].outputs.pop(0))
            next_level.append(self.add_pipeline_lower[1].outputs.pop(0))

        #next_level.append(np.float32(self.values[i] + self.values[i + 1]))
        
        self.intermediate_results = next_level
        self.values = []
        return self.intermediate_results
    
    def tick_second_stage(self):
        """第二阶段：将2个中间结果合并为1个输出"""
        if not self.intermediate_results:
            return None
        
        final_result = None
        # 计算第二级加法结果，确保类型一致
        self.add_pipeline_upper.clock_cycle(self.intermediate_results[0], self.intermediate_results[1], True)
        if len(self.add_pipeline_upper.outputs):
            final_result = self.add_pipeline_upper.outputs.pop(0)
        self.intermediate_results = []
        self.busy = False
        return final_result

# 新增：流水线状态记录
class PipelineStatus:
    def __init__(self):
        self.cycles = []  # 每个周期的状态记录

    def add_cycle(self, cycle_num, status_dict):
        """添加一个周期的状态记录"""
        self.cycles.append({
            "cycle": cycle_num,
            **status_dict
        })

    def print_status(self, detailed=False):
        """打印流水线状态历史"""
        print("\n===== 流水线执行历史 =====")
        for cycle in self.cycles:
            print(f"\n周期 {cycle['cycle']}:")
            
            # 输入数据
            if "input" in cycle:
                if cycle["input"] is None:
                    print("  输入: 无")
                else:
                    print(f"  输入: 矩阵元素 C[{cycle['input']['coords'][0]}][{cycle['input']['coords'][1]}] - 块 {cycle['input']['block_idx']}")
            
            # 阶段状态
            print(f"  阶段1: {'有效' if cycle.get('stage1_valid', False) else '无效'}")
            print(f"  阶段2: {'有效' if cycle.get('stage2_valid', False) else '无效'}", end="")
            if cycle.get('stage2_valid', False) and "stage2_coords" in cycle:
                print(f" - 处理 C[{cycle['stage2_coords'][0]}][{cycle['stage2_coords'][1]}]")
            else:
                print("")
                
            print(f"  阶段3: {'有效' if cycle.get('stage3_valid', False) else '无效'}", end="")
            if cycle.get('stage3_valid', False) and "stage3_coords" in cycle:
                print(f" - 处理 C[{cycle['stage3_coords'][0]}][{cycle['stage3_coords'][1]}]")
            else:
                print("")
                
            print(f"  阶段4: {'有效' if cycle.get('stage4_valid', False) else '无效'}", end="")
            if cycle.get('stage4_valid', False) and "stage3_coords" in cycle:
                print(f" - 处理 C[{cycle['stage3_coords'][0]}][{cycle['stage3_coords'][1]}]")
            else:
                print("")
            
            # 结果
            if "result" in cycle and cycle["result"] is not None:
                coords, value = cycle["result"]
                print(f"  结果: C[{coords[0]}][{coords[1]}] 得到部分结果 {value}")
            
            # 完成的矩阵元素
            if "completed" in cycle and cycle["completed"]:
                i, j = cycle["completed"]["coords"]
                value = cycle["completed"]["value"]
                print(f"  完成: C[{i}][{j}] = {value}")
            
            if detailed and "debug_info" in cycle:
                print("  调试信息:")
                for key, value in cycle["debug_info"].items():
                    print(f"    {key}: {value}")

class PipelineSimulator:
    def __init__(self, mac_width=4):
        self.mac_width = mac_width
        self.mac_line = MACLine(mac_width)
        self.adder_tree = AdderTree()
        self.clock_cycle = 0
        self.log = []
        self.input = [] # 模块的输入数据
        self.ijQueue = []   # 块的坐标
        
        self.stage1_valid = True
        self.stage1_input = []
        self.stage1_output = []
        self.stage2_valid = False
        self.stage2_input = []
        self.stage2_output = []
        # 修改：拆分stage3为两个阶段
        self.stage3_valid = False
        self.stage3_input = []
        self.stage4_valid = False
        self.stage4_input = []
        
        self.pipeline_result = None
        self.output_valid = False
        
        # 添加坐标追踪
        self.stage2_coords = None   # 当前在stage2中的坐标
        self.stage3_coords = None   # 当前在stage3中的坐标
        self.stage4_coords = None   # 当前在stage3中的坐标
        
        # 新增：状态跟踪
        self.status = PipelineStatus()
    
    def process_block(self, a_block, b_block, coords, is_last_block):
        """处理单个数据块并返回部分结果及其对应坐标"""
        # 记录当前阶段的状态（处理前）
        pre_status = {
            "stage1_valid": self.stage1_valid,
            "stage2_valid": self.stage2_valid,
            "stage3_valid": self.stage3_valid,
            "stage4_valid": self.stage4_valid,
            "stage2_coords": self.stage2_coords,
            "stage3_coords": self.stage3_coords,
            "stage4_coords": self.stage4_coords,
        }
        
        # stage4：处理前一个周期 stage3 的第二阶段加法
        result = None
        result_coords = None
        if self.stage4_valid:
            reduced = self.adder_tree.tick_second_stage()
            if reduced is not None:
                result = reduced
                result_coords = self.ijQueue.pop(0)
                # result_coords = self.stage4_coords
            self.stage4_valid = False
            
        # stage3：处理前一个周期 stage2 的第一阶段加法
        if self.stage3_valid:
            intermediate = self.adder_tree.tick_first_stage()
            if intermediate:
                self.stage4_valid = True
                self.stage4_coords = self.stage3_coords
            self.stage3_valid = False
            
        # stage2：处理前一个周期 stage1 的MAC计算
        if self.stage2_valid:
            if self.stage2_input[0] is not None and self.stage2_input[1] is not None:
                self.mac_line.get_input(self.stage2_input)
                results = self.mac_line.tick()
                if len(results):
                    self.adder_tree.add_values(results)
                    self.stage3_valid = True
                    self.stage3_coords = self.stage2_coords  # 传递坐标信息
            self.stage2_valid = False
            
        # stage1：接收新数据
        if a_block is not None and b_block is not None:
            # 确保转换为float32以匹配numpy
            self.stage1_output = [a_block, b_block]
            self.ijQueue.append(coords)
            
            self.stage2_input = self.stage1_output
            self.stage2_coords = coords
            self.stage2_valid = True
            
        # 记录当前阶段的状态（处理后）
        post_status = {
            "stage1_valid": self.stage1_valid,
            "stage2_valid": self.stage2_valid,
            "stage3_valid": self.stage3_valid,
            "stage4_valid": self.stage4_valid,
            "stage2_coords": self.stage2_coords,
            "stage3_coords": self.stage3_coords,
            "result": (result_coords, result) if result is not None and result_coords is not None else None,
            "debug_info": {
                "pre_stage2_valid": pre_status["stage2_valid"],
                "pre_stage3_valid": pre_status["stage3_valid"],
                "pre_stage4_valid": pre_status["stage4_valid"],
                "post_stage2_valid": self.stage2_valid,
                "post_stage3_valid": self.stage3_valid,
                "post_stage4_valid": self.stage4_valid,
                "a_block": "None" if a_block is None else "有值",
                "b_block": "None" if b_block is None else "有值",
                "is_last_block": is_last_block
            }
        }
        
        # 添加到状态记录
        self.status.add_cycle(self.clock_cycle, post_status)
        
        # 只有当结果和坐标都有效时才返回元组
        if result is not None and result_coords is not None:
            return (result_coords, result)
        
        return None
        
    def reset(self):
        self.__init__(self.mac_width)
        
    def is_active(self):
        """检查流水线是否正在处理数据"""
        return (self.stage2_valid or self.stage3_valid or self.stage4_valid or 
                any(mac.is_busy() for mac in self.mac_line.macs))
        
def convert_through_pipeline(value):
    """通过完整流水线模拟转换FP32到BF16"""
    temp_pipeline = FP32toBF16Pipeline()
    temp_pipeline.run_simulation([(value, True)], print_states=False)
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0

def pipeline_matmul(A, B, verbose=False):
    assert A.shape[1] == B.shape[0]
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float32)
    logs = []
    
    # 创建一个持久的流水线模拟器
    sim = PipelineSimulator()
    
    # 创建元素跟踪字典和输入队列
    element_tracker = {}  # {(i,j): {"blocks_total": x, "blocks_processed": y, "result": value}}
    input_queue = []      # 存储待处理的数据块

    # 准备所有计算任务并直接放入输入队列
    for i in range(m):
        for j in range(n):
            a_row = A[i,:]
            b_col = B[:,j]
            k_len = len(a_row)
            blocks_total = (k_len + sim.mac_width - 1) // sim.mac_width  # 向上取整
            
            # 初始化该元素的跟踪信息
            element_tracker[(i, j)] = {
                "blocks_total": blocks_total,
                "blocks_processed": 0,
                "partial_results": [],
                "result": None
            }

            # 准备该元素的所有数据块并加入队列
            for block_start in range(0, k_len, sim.mac_width):
                a_block = a_row[block_start:block_start + sim.mac_width]
                b_block = b_col[block_start:block_start + sim.mac_width]
                
                # 补齐不足的块
                if len(a_block) < sim.mac_width:
                    a_block = np.pad(a_block, (0, sim.mac_width - len(a_block))).astype(np.float32)
                    b_block = np.pad(b_block, (0, sim.mac_width - len(b_block))).astype(np.float32)
                
                block_index = block_start // sim.mac_width
                is_last_block = (block_start + sim.mac_width >= k_len)
                
                input_queue.append({
                    "a_block": a_block,
                    "b_block": b_block,
                    "coords": (i, j),
                    "block_index": block_index,
                    "is_last_block": is_last_block
                })

    last_coords = None  # 跟踪最后一个处理的坐标
    
    # 运行直到输入队列为空且流水线不再活跃
    while input_queue or sim.is_active():
        #print(input_queue)
        #print(sim.is_active())
        current_input_log = None
        completed_element_log = None
        
        # 从队列获取数据或处理流水线空闲周期
        if input_queue:
            data = input_queue.pop(0)
            a_block = data["a_block"]
            b_block = data["b_block"]
            coords = data["coords"]
            block_idx = data["block_index"]
            is_last_block = data["is_last_block"]
            last_coords = coords  # 更新最后处理的坐标
            a_block = [float(a) for a in a_block.tolist()]
            b_block = [float(b) for b in b_block.tolist()]
        
            
            a_block = [convert_through_pipeline(a) for a in a_block]
            b_block = [convert_through_pipeline(b) for b in b_block]
            #print(a_block)
            
            current_input_log = {
                "coords": coords,
                "block_idx": block_idx,
                "is_last": is_last_block
            }
        else:
            # 输入队列为空，但流水线仍活跃，继续推进
            a_block, b_block, coords = None, None, last_coords
            is_last_block = True # 假设在排空阶段，需要标记结束
            current_input_log = None

        # 运行一个时钟周期,将坐标与数据一起传递给流水线
        partial_result = sim.process_block(a_block, b_block, coords, is_last_block)
        
        
        # 处理返回的部分结果
        if partial_result is not None:
            result_coords, result_value = partial_result
            result_value = bf16_to_float(result_value)
            i, j = result_coords
            C[i][j] += result_value
            #print(C[i][j])

        # 更新状态记录 (如果需要)
        if verbose and hasattr(sim.status, "cycles") and sim.status.cycles:
            latest_cycle = sim.status.cycles[-1]
            latest_cycle["input"] = current_input_log
            latest_cycle["completed"] = completed_element_log
        
        # 增加时钟周期
        sim.clock_cycle += 1
        #print(sim.clock_cycle)
    
    # 循环结束后，再次检查所有元素的任务记录，检查并处理可能未完全累加的结果 (理论上不应发生，但作为保险)
    for coords, tracker in element_tracker.items():
        i, j = coords
        if tracker["result"] is None and tracker["partial_results"]:
            total_result = np.float32(sum(tracker["partial_results"]))
            C[i][j] = total_result
            # logs.append(f"C[{i}][{j}] completed manually at end => {total_result}") # 可选日志

    # 打印详细信息
    if verbose:
        sim.status.print_status(detailed=False)
        # 打印计算日志
        for log in logs:
            print(log)
            
    return C

def verify_result(m=4, k=7, n=5, random_seed=42, verbose=True):
    """
    生成随机矩阵 A (m x k) 和 B (k x n)，
    计算 MACLineMatrixMultiplier 计算的 C 与 NumPy 直接矩阵乘法的比较。
    """
    np.random.seed(random_seed)
    A = np.random.randint(0, 10, size=(m, k))
    B = np.random.randint(0, 10, size=(k, n))
    C_np = A @ B
    print(C_np)
    C_mac = pipeline_matmul(A, B, verbose=verbose)
    
    
    if np.allclose(C_mac, C_np):
        print("验证成功：结果一致！")
    else:
        print("验证失败：结果不一致！")
    
if __name__ == '__main__':
    verify_result(m=4, k=5, n=5, random_seed=42)