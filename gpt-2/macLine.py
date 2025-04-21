import numpy as np

class MACUnit:
    def __init__(self, clock_cycle=2):
        self.state = []  # list of (a, b, remaining_cycles)
        self.clock_cycle = clock_cycle 
        self.result = None

    def is_busy(self):
        return len(self.state) > 0

    def enqueue(self, a, b):
        self.state.append([a, b, self.clock_cycle])  # 每个MAC运算耗2周期

    def tick(self):
        if not self.state:
            return None
        a, b, _ = self.state.pop(0)
        return a * b

class MACLine:
    def __init__(self, mac_width):
        self.macs = [MACUnit() for _ in range(mac_width)]
        self.input_queue = []  # list of (a_block, b_block)

    def enqueue(self, a_block, b_block):
        self.input_queue.append((a_block, b_block))
    
    def reset(self):
        self.input_queue = []
        for mac in self.macs:
            mac.state = []
            mac.result = None
            
    def is_active(self):
        return self.input_queue or any(mac.is_busy() for mac in self.macs)

    def tick(self):
        results = []
        if not self.input_queue:  # 防止访问空队列
            return None
        
        for i, mac in enumerate(self.macs):
            if i < len(self.input_queue[0][0]) and i < len(self.input_queue[0][1]):
                mac.enqueue(self.input_queue[0][0][i], self.input_queue[0][1][i])
                r = mac.tick()
                results.append(r)  # 即使是 None 也添加到结果中
        
        if self.input_queue:  # 处理完后移除队列中的元素
            self.input_queue.pop(0)
        
        return results

# 修改：将AdderTree拆分为两个阶段
# 修改：将AdderTree拆分为两个阶段，修复精度问题
class AdderTree:
    def __init__(self):
        self.values = []
        self.busy = False
        self.intermediate_results = []  # 存储第一阶段的中间结果

    def add_values(self, values):
        # 确保过滤None值并使用float32类型确保精度一致
        self.values.extend([np.float32(v) for v in values if v is not None])
        self.busy = True

    def is_active(self):
        return self.busy and (self.values or self.intermediate_results)
    
    def reset(self):
        self.values = []
        self.intermediate_results = []
        self.busy = False
    
    def tick_first_stage(self):
        """第一阶段：将4个输入合并为2个中间结果"""
        if not self.values:
            return None
        
        # 计算第一级加法结果，确保类型一致
        next_level = []
        for i in range(0, len(self.values), 2):
            if i + 1 < len(self.values):
                next_level.append(np.float32(self.values[i] + self.values[i + 1]))
            else:
                next_level.append(np.float32(self.values[i]))
        
        self.intermediate_results = next_level
        self.values = []
        return self.intermediate_results
    
    def tick_second_stage(self):
        """第二阶段：将2个中间结果合并为1个输出"""
        if not self.intermediate_results:
            return None
        
        # 计算第二级加法结果，确保类型一致
        final_result = np.float32(sum(self.intermediate_results))
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
                result_coords = self.stage4_coords
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
                self.mac_line.enqueue(self.stage2_input[0], self.stage2_input[1])
                results = self.mac_line.tick()
                if results:
                    self.adder_tree.add_values(results)
                    self.stage3_valid = True
                    self.stage3_coords = self.stage2_coords  # 传递坐标信息
            self.stage2_valid = False
            
        # stage1：接收新数据
        if a_block is not None and b_block is not None:
            # 确保转换为float32以匹配numpy
            self.stage1_output = [np.array(a_block, dtype=np.float32), 
                                np.array(b_block, dtype=np.float32)]
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


def pipeline_matmul(A, B, verbose=True):
    assert A.shape[1] == B.shape[0]
    m, k = A.shape
    n = B.shape[1]
    C = np.zeros((m, n), dtype=np.float32)
    logs = []
    
    # 创建一个持久的流水线模拟器
    sim = PipelineSimulator()
    
    # 创建元素跟踪字典，记录每个矩阵元素的状态
    element_tracker = {}  # {(i,j): {"blocks_total": x, "blocks_processed": y, "result": value}}
    
    # 准备所有矩阵元素的计算任务
    all_tasks = []
    for i in range(m):
        for j in range(n):
            all_tasks.append((i, j, A[i,:], B[:, j]))
    
    # 连续向流水线中添加任务
    task_index = 0
    current_element = None  # 当前正在准备数据的元素坐标
    last_coords = None  # 跟踪最后一个处理的坐标
    
    # 运行直到所有任务都被处理完
    while task_index < len(all_tasks) or sim.input or sim.is_active():
        # 记录当前周期的输入数据
        current_input = None
        completed_element = None
        
        # 如果流水线的stage1空闲并且还有任务，就添加新任务
        if task_index < len(all_tasks) and not sim.input:
            i, j, a_row, b_col = all_tasks[task_index]
            current_element = (i, j)
            last_coords = current_element  # 更新最后处理的坐标
            
            # 初始化该元素的跟踪信息
            k_len = len(a_row)
            blocks_total = (k_len + sim.mac_width - 1) // sim.mac_width  # 向上取整
            element_tracker[(i, j)] = {
                "blocks_total": blocks_total,
                "blocks_processed": 0,
                "partial_results": [],
                "result": None
            }
            
            # 准备该元素的所有数据块
            for block_start in range(0, k_len, sim.mac_width):
                a_block = a_row[block_start:block_start + sim.mac_width]
                b_block = b_col[block_start:block_start + sim.mac_width]
                
                if len(a_block) < sim.mac_width:
                    a_block = np.pad(a_block, (0, sim.mac_width - len(a_block)))
                    b_block = np.pad(b_block, (0, sim.mac_width - len(b_block)))
                
                block_index = block_start // sim.mac_width
                is_last_block = (block_start + sim.mac_width >= k_len)
                
                # 数据块附带元素坐标和块信息
                sim.input.append({
                    "a_block": a_block,
                    "b_block": b_block,
                    "coords": (i, j),
                    "block_index": block_index,
                    "is_last_block": is_last_block
                })
            
            task_index += 1
        
        # 运行一个时钟周期
        if sim.input:
            data = sim.input.pop(0)
            a_block = data["a_block"]
            b_block = data["b_block"]
            coords = data["coords"]
            block_idx = data["block_index"]
            last_coords = coords  # 更新最后处理的坐标
            is_last_block = data["is_last_block"]
            
            # 记录当前输入
            current_input = {
                "coords": coords,
                "block_idx": block_idx,
                "is_last": is_last_block
            }
            
            # 使用修改后的process_block方法
            partial_result = sim.process_block(a_block, b_block, coords, is_last_block)
        else:
            # 如果没有输入但流水线仍活跃，继续处理
            # 关键修改：使用最后一个有效坐标而不是(-1,-1)
            partial_result = sim.process_block(None, None, last_coords, True)
            current_input = None
        
        # 如果有结果返回，则更新元素的部分结果
        if partial_result is not None:
            result_coords, result_value = partial_result
            i, j = result_coords
            if (i, j) in element_tracker:  # 确保这个坐标在tracker中
                element_tracker[(i, j)]["blocks_processed"] += 1
                element_tracker[(i, j)]["partial_results"].append(result_value)
                
                # 如果所有块都处理完了，计算最终结果
                if element_tracker[(i, j)]["blocks_processed"] == element_tracker[(i, j)]["blocks_total"]:
                    total_result = sum(element_tracker[(i, j)]["partial_results"])
                    element_tracker[(i, j)]["result"] = total_result
                    C[i][j] = total_result
                    logs.append(f"C[{i}][{j}] computed in {sim.clock_cycle} cycles => {total_result}")
                    completed_element = {"coords": (i, j), "value": total_result}
        
        # 更新状态记录，添加输入信息和完成的元素信息
        if hasattr(sim.status, "cycles") and sim.status.cycles:
            latest_cycle = sim.status.cycles[-1]
            latest_cycle["input"] = current_input
            latest_cycle["completed"] = completed_element
        
        # 增加时钟周期
        sim.clock_cycle += 1
    
    # 检查是否有未完成的元素
    for coords, tracker in element_tracker.items():
        i, j = coords
        if tracker["result"] is None and tracker["partial_results"]:
            # 有部分结果但未完成的元素，手动完成
            total_result = sum(tracker["partial_results"])
            C[i][j] = total_result
    
    
    # 打印流水线状态历史
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
    C_mac = pipeline_matmul(A, B, verbose=verbose)
    C_np = A @ B
    if np.allclose(C_mac, C_np):
        print("\n验证成功：结果一致！")
    else:
        print("\n验证失败：结果不一致！")
    
if __name__ == '__main__':
    verify_result(m=4, k=4, n=5, random_seed=42)