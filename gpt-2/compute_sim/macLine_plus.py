import numpy as np
import struct
import torch
from bf16_sim import BF16AddPipeline, BF16MultiplyPipeline, FP32toBF16Pipeline

def bf16_to_float_block(bf16_block):
    # 左移16位填充为32位表示
    float_block = []
    for bf16 in bf16_block:
        fp32_bits = bf16 << 16
        float_block.append(struct.unpack('>f', struct.pack('>I', fp32_bits))[0])
    # 转换为浮点数
    return float_block

class MACUnit:
    def __init__(self):
        self.multiply_pipeline = BF16MultiplyPipeline()
        self.input1 = 0
        self.input2 = 0
        self.output = 0
        self.valid = False
        self.input_valid = False

    def get_input(self, input1, input2, input_valid):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        
    def clock_cycle(self):
        result = self.multiply_pipeline.clock_cycle(self.input1, self.input2, self.input_valid)
        self.valid = result["valid_output"]
        if self.valid:
            self.output = self.multiply_pipeline.outputs.pop(0)

class AddUnit:
    def __init__(self):
        self.add_pipeline = BF16AddPipeline()
        self.input1 = 0
        self.input2 = 0
        self.output = 0
        self.valid = False
        self.input_valid = False
    
    def get_input(self, input1, input2, input_valid):
        self.input1 = input1
        self.input2 = input2
        self.input_valid = input_valid
        
    def clock_cycle(self):
        result = self.add_pipeline.clock_cycle(self.input1, self.input2, self.input_valid) 
        self.valid = result["valid_output"]
        if self.valid:
            self.output = self.add_pipeline.outputs.pop(0) 


class MatmulPipeline:
    def __init__(self, mac_width=4):
        self.mac_width = mac_width
        self.clock = 0
        
        self.stage1_valid = False
        self.stage1_a_block = [0] * 4
        self.stage1_b_block = [0] * 4
        
        self.macline = [MACUnit() for _ in range(mac_width)]
        self.macline_valid = False
        self.stage2_valid = False
        self.stage2_result = [0] * 4
        
        self.addvec = [AddUnit(), AddUnit()]
        self.addvec_valid = False
        self.stage3_valid = False
        self.stage3_result = [0] * 2
        
        self.add = AddUnit()
        self.add_valid = False
        self.stage4_valid = False
        self.coords_queue = []
        
    def is_active(self):
        # 只要有任何一个阶段还在处理，或队列未空，就认为流水线活跃
        return (
            self.stage1_valid or
            self.macline_valid or
            self.stage2_valid or
            self.addvec_valid or
            self.stage3_valid or
            self.add_valid or
            self.stage4_valid or
            len(self.coords_queue) > 0
        )
    
    def print_stage_valid(self):
        print(
            f"stage1_valid: {self.stage1_valid}, "
            f"macline_valid: {self.macline_valid}, "
            f"stage2_valid: {self.stage2_valid}, "
            f"addvec_valid: {self.addvec_valid}, "
            f"stage3_valid: {self.stage3_valid}, "
            f"add_valid: {self.add_valid}, "
            f"stage4_valid: {self.stage4_valid}, "
            f"coords_queue: {len(self.coords_queue)}"
        )
        


    def clock_cycle(self, a_block, b_block, coords, valid):
        
        result = None
        result_coords = None
        
        #stage3 -> stage4: 处理第二阶段加法
        
        #if self.stage3_valid:
        self.add.get_input(self.stage3_result[0], self.stage3_result[1], self.stage3_valid)
        self.add.clock_cycle()
        self.add_valid = self.add.valid
        if self.add.valid:
            result = self.add.output
            if self.coords_queue:
                result_coords = self.coords_queue.pop(0)
            else:
                result_coords = None
        self.stage4_valid = self.add_valid
        #stage2 -> stage3 : 处理第一阶段加法
        
        #if self.stage2_valid:
        temp_result = [0] * 2
        self.addvec[0].get_input(self.stage2_result[0], self.stage2_result[1], self.stage2_valid)
        self.addvec[1].get_input(self.stage2_result[2], self.stage2_result[3], self.stage2_valid)
        self.addvec[0].clock_cycle()
        self.addvec[1].clock_cycle()
        self.addvec_valid = self.addvec[0].valid
        if self.addvec[0].valid and self.addvec[1].valid:
            temp_result[0] = self.addvec[0].output
            temp_result[1] = self.addvec[1].output
        self.stage3_result = temp_result
        self.stage3_valid = self.addvec_valid

        #stage1 -> stage2 : 处理mac
        
        temp_result = [0] * 4
        for i, mac in enumerate(self.macline):
            mac.get_input(self.stage1_a_block[i], self.stage1_b_block[i], self.stage1_valid)
            mac.clock_cycle()
            self.macline_valid = mac.valid
            if mac.valid:
                temp_result[i] = mac.output
        self.stage2_result = temp_result
        self.stage2_valid = self.macline_valid


        #stage1：接收新数据
        if valid and a_block is not None and b_block is not None:
            self.stage1_a_block = a_block
            self.stage1_b_block = b_block
            self.stage1_valid = True
            self.coords_queue.append(coords)

        else:
            self.stage1_valid = False
            self.stage1_a_block = [0] * 4 
            self.stage1_b_block = [0] * 4
        
        if result is not None and result_coords is not None:
            return (result_coords, result)
        
        return None
        
def convert_through_pipeline(value):
    """通过完整流水线模拟转换FP32到BF16"""
    temp_pipeline = FP32toBF16Pipeline()
    temp_pipeline.run_simulation([(value, True)], print_states=False)
    return temp_pipeline.outputs[0]["bf16"] if temp_pipeline.outputs else 0   
  
def bf16_mul(bf16_a, bf16_b):
    sim = BF16MultiplyPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], False)
    return sim.outputs[0]

def bf16_add(bf16_a, bf16_b):
    sim = BF16AddPipeline()
    sim.run_simulation([(bf16_a, bf16_b, True)], print_states=False)
    return sim.outputs[0]

def pipeline_matmul(A, B, verbose=False):
    assert A.shape[1] == B.shape[0]
    m, k = A.shape
    n = B.shape[1]

    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = convert_through_pipeline(float(C[i][j]))

    # 创建一个持久的流水线模拟器
    sim = MatmulPipeline()
    
    # 创建元素跟踪字典和输入队列
    element_tracker = {}  # {(i,j): {"blocks_total": x, "blocks_processed": y, "result": value}}
    input_queue = []      # 存储待处理的数据块

    # 准备所有计算任务并直接放入输入队列
    for i in range(m):
        for j in range(n):
            a_row = A[i,:]
            b_col = B[:,j]
            k_len = len(a_row)

            # 准备该元素的所有数据块并加入队列
            for block_start in range(0, k_len, sim.mac_width):
                a_block = a_row[block_start:block_start + sim.mac_width]
                b_block = b_col[block_start:block_start + sim.mac_width]
                
                # 补齐不足的块
                if len(a_block) < sim.mac_width:
                    a_block = np.pad(a_block, (0, sim.mac_width - len(a_block))).astype(np.float32)
                    b_block = np.pad(b_block, (0, sim.mac_width - len(b_block))).astype(np.float32)
                
                input_queue.append({
                    "a_block": a_block,
                    "b_block": b_block,
                    "coords": (i, j),
                })

    last_coords = None  # 跟踪最后一个处理的坐标
    # 运行直到输入队列为空且流水线不再活跃
    while input_queue or sim.is_active():
        sim.print_stage_valid()
      
        # 从队列获取数据或处理流水线空闲周期
        if input_queue:
            data = input_queue.pop(0)
            a_block = data["a_block"]
            b_block = data["b_block"]
            coords = data["coords"]
            # is_last_block = data["is_last_block"]
            a_block = [float(a) for a in a_block.tolist()]
            b_block = [float(b) for b in b_block.tolist()]
            
            a_block = [convert_through_pipeline(a) for a in a_block]
            b_block = [convert_through_pipeline(b) for b in b_block]
            
            valid = True
        else:
            # 输入队列为空，但流水线仍活跃，继续推进
            a_block, b_block, coords = None, None, None
            valid = False

        # 运行一个时钟周期,将坐标与数据一起传递给流水线
        partial_result = sim.clock_cycle(a_block, b_block, coords, valid)
        import struct
        def bf16_to_float(bf16):
            # 左移16位填充为32位表示
            fp32_bits = bf16 << 16
            # 转换为浮点数
            return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]
        
        # 处理返回的部分结果
        if partial_result is not None:
            result_coords, result_value = partial_result
            #result_value = bf16_to_float(result_value)
            i, j = result_coords
            C[i][j] = bf16_add(C[i][j], result_value)


        
        # 增加时钟周期
        sim.clock += 1

    for i in range(m):
        for j in range(n):
            C[i][j] = bf16_to_float(C[i][j])

    return C

def verify_result(m=4, k=7, n=5, random_seed=42, verbose=True):
    """
    生成随机矩阵 A (m x k) 和 B (k x n)，
    计算 MACLineMatrixMultiplier 计算的 C 与 NumPy 直接矩阵乘法的比较。
    """
    np.random.seed(random_seed)
    A = np.random.randint(0, 10, size=(m, k))
    B = np.random.randint(0, 10, size=(k, n))
    print(A)
    print(B)
    C_np = A @ B
    #print(C_np)
    C_mac = pipeline_matmul(A, B, verbose=verbose)
    print(C_mac)
    print(C_np)
    
    if np.allclose(C_mac, C_np):
        print("验证成功：结果一致！")
    else:
        print("验证失败：结果不一致！")

def test_bf16_pipeline_matmul(m=4, k=7, n=5, random_seed=123):
    """
    使用 PyTorch 生成 bfloat16 小数矩阵，验证 pipeline_matmul 的正确性
    """
    torch.manual_seed(random_seed)
    # 生成小数矩阵，范围在 -10 到 10
    A = (torch.rand((m, k))).to(torch.bfloat16)
    B = (torch.rand((k, n))).to(torch.bfloat16)

    # PyTorch 计算 bfloat16 结果
    C_torch = torch.matmul(A, B).to(torch.float32).numpy()

    # 转为 numpy float32，传给 pipeline_matmul
    A_np = A.to(torch.float32).numpy()
    B_np = B.to(torch.float32).numpy()

    # 用你的流水线仿真
    C_sim = pipeline_matmul(A_np, B_np, verbose=False)

    print("PyTorch bfloat16 结果：")
    print(C_torch)
    print("pipeline_matmul 仿真结果：")
    print(C_sim)

    # 允许一定误差（bfloat16 精度较低）
    if np.allclose(C_sim, C_torch, atol=1e-1, rtol=1e-2):
        print("验证成功：pipeline_matmul 与 PyTorch bfloat16 结果一致！")
    else:
        print("验证失败：结果不一致！")

def test_bf16_pytorch_matmul_gpu(m=4, k=7, n=5, random_seed=123):
    """
    使用 PyTorch 生成 bfloat16 小数矩阵，验证 pipeline_matmul 的正确性
    """
    torch.manual_seed(random_seed)
    # 生成小数矩阵，范围在 -10 到 10
    A = (torch.rand((m, k))).to(torch.bfloat16)
    B = (torch.rand((k, n))).to(torch.bfloat16)

    # PyTorch 计算 bfloat16 结果
    C_torch = torch.matmul(A, B).to(torch.float32).numpy()
    print(C_torch)
    
if __name__ == '__main__':
    #verify_result(m=4, k=10, n=5, random_seed=42)
    test_bf16_pipeline_matmul(m=4, k=7, n=5)
    test_bf16_pytorch_matmul_gpu()
                
        