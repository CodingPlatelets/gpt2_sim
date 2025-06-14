import numpy as np
import torch
from scipy.sparse import csr_matrix
from distribution import MFIU, get_Sequence
from macLine import MatmulPipeline, convert_through_pipeline, bf16_to_float , bf16_add


def create_coords_mapping(mfiu, A, B):
    """
    创建从序列索引到C矩阵位置的映射，正确处理列优先顺序
    """
    coords_map = {}
    m, n = len(A), len(B[0])
    # 重要：MFIU使用列优先顺序！
    position_map = {}
    for i in range(len(mfiu.shift_unit_a)):
        col = i // m   # 列索引是单元号除以行数
        row = i % m    # 行索引是单元号对行数取模 
        if row < m and col < n:
            position_map[i] = (row, col)
    
    # 遍历所有单元，处理有效索引
    for unit_idx, (unit_a, unit_b) in enumerate(zip(mfiu.shift_unit_a, mfiu.shift_unit_b)):
        if unit_idx not in position_map:
            continue
            
        row, col = position_map[unit_idx]
        
        # 查找所有索引对
        for i in range(len(unit_a.index)):
            idx_a = unit_a.index[i]
            if idx_a > 0:
                for j in range(len(unit_b.index)):
                    idx_b = unit_b.index[j]
                    if idx_b > 0:
                        # 两个索引都存在，表示需要计算
                        coords_map[idx_a-1] = (row, col)
                        # 可能需要同时映射B索引
                        if idx_b-1 not in coords_map:
                            coords_map[idx_b-1] = (row, col)
    
    return coords_map

# 使用Trapezoid加速的矩阵乘法实现
def trapezoid_matmul(A, B):
    # 1. 经过distribution网络处理，得到变换后的矩阵
    A_seq, B_seq, mfiu = get_Sequence(A, B)
    A_seq = torch.tensor(A_seq, dtype=torch.float32)
    B_seq = torch.tensor(B_seq, dtype=torch.float32)
    
    print(A_seq)
    print(B_seq)
    
    # 2.  创建坐标映射
    coords_map = create_coords_mapping(mfiu, A, B)
    
     # 3.  初始化结果矩阵
    m, n = len(A), len(B[0])
    C = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = convert_through_pipeline(float(C[i][j]))
    
    # 4.  创建MatmulPipeline并准备输入队列
    mac_width = 4  # 使用4个PE
    pipeline = MatmulPipeline(mac_width=mac_width)
    # 创建与macLine.py一致的输入队列
    input_queue = []
    
    # 对数据分组处理并准备输入队列
    # for i, j in sorted(set(coords_map.values())):
    for i, j in sorted(set(coords_map.values())):
        # 找出所有映射到当前坐标(i,j)的索引
        indices = [idx for idx, coord in coords_map.items() if coord == (i, j)]
        
        # 提取对应的A,B元素并分块
        for block_start in range(0, len(indices), mac_width):
            block_indices = indices[block_start:block_start+mac_width]
            
            # 提取对应的A和B值，并打印详细信息
            a_values = []
            b_values = []
            for idx in block_indices:
                a_val = A_seq[idx].item() if idx < len(A_seq) else 0
                b_val = B_seq[idx].item() if idx < len(B_seq) else 0
                a_values.append(a_val)
                b_values.append(b_val)

            # 补足到mac_width个元素
            while len(a_values) < mac_width:
                a_values.append(0)
                b_values.append(0)

            
            # 使用与macLine.py相同的字典格式添加到输入队列
            input_queue.append({
                "a_block": [np.float32(a) for a in a_values],
                "b_block": [np.float32(b) for b in b_values],
                "coords": (i, j),
            })
   
    # print(input_queue)

    # 运行直到输入队列为空且流水线不再活跃
    while input_queue or pipeline.is_active():
      
        # 从队列获取数据或处理流水线空闲周期
        if input_queue:
            data = input_queue.pop(0)
            a_block = data["a_block"]
            b_block = data["b_block"]
            coords = data["coords"]
            
            a_block = [convert_through_pipeline(float(a)) for a in a_block]
            b_block = [convert_through_pipeline(float(b)) for b in b_block]
            
            valid = True
        else:
            # 输入队列为空，但流水线仍活跃，继续推进
            a_block, b_block, coords = None, None, None
            valid = False

        # 运行一个时钟周期,将坐标与数据一起传递给流水线
        partial_result = pipeline.clock_cycle(a_block, b_block, coords, valid)
        
        # 处理返回的部分结果
        if partial_result is not None:
            result_coords, result_value = partial_result
            i, j = result_coords
            C[i][j] = bf16_add(C[i][j], result_value)

        # 增加时钟周期
        pipeline.clock += 1

    for i in range(m):
        for j in range(n):
            C[i][j] = bf16_to_float(C[i][j])

    return C
def verify_result(m=2, k=7, n=3, random_seed=42, verbose=True):
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
    C_mac = trapezoid_matmul(A, B)
    print(C_mac)
    print(C_np)
    
    if np.allclose(C_mac, C_np):
        print("验证成功：结果一致！")
    else:
        print("验证失败：结果不一致！")
        
def test_bf16_pipeline_matmul(m=4, k=7, n=5, random_seed=12):
    """
    使用 PyTorch 生成 bfloat16 小数矩阵，验证 pipeline_matmul 的正确性
    """
    torch.manual_seed(random_seed)
    # 生成小数矩阵，范围在 -10 到 10
    # A = (torch.rand((m, k))).to(torch.bfloat16)
    # B = (torch.rand((k, n))).to(torch.bfloat16)
    A = (torch.rand((m, k))).to(torch.float32)
    B = (torch.rand((k, n))).to(torch.float32)
    print(A)
    print(B)

    # PyTorch 计算 bfloat16 结果
    C_torch = torch.matmul(A, B).to(torch.float32).numpy()
    print(A @ B)

    # 转为 numpy float32，传给 pipeline_matmul
    A_np = A.to(torch.float32).numpy()
    B_np = B.to(torch.float32).numpy()

    # 用你的流水线仿真
    C_sim = trapezoid_matmul(A_np, B_np)

    print("PyTorch bfloat16 结果：")
    print(C_torch)
    print("pipeline_matmul 仿真结果：")
    print(C_sim)

    # 允许一定误差（bfloat16 精度较低）
    if np.allclose(C_sim, C_torch, atol=1e-1, rtol=1e-2):
        print("验证成功：pipeline_matmul 与 PyTorch bfloat16 结果一致！")
    else:
        print("验证失败：结果不一致！")
        
def test():
    # A = np.array([[1, 0, 2, 0], [0, 3, 4, 0]])
    # B = np.array([[5, 7], [0, 0], [0, 8], [6, 0]])
    A = np.array([[1, 2, 0, 4 , 5], [6, 0, 8, 9, 0 ],[10, 11, 12, 0, 13]])
    B = np.array([[1, 4, 0 ], [5, 0 , 7], [0, 6 , 8], [4 , 0 , 2], [3, 0 , 9 ]])
    # A = np.array([[0.232, 0, 0.342, 0], [0, 0.573, 0.498, 0]])
    # B = np.array([[0.945, 0.957], [0.432, 0], [0, 0.978], [0.856, 0]])
    
    C = trapezoid_matmul(A, B)
    print(C)
    print(A @ B)
    
if __name__ == "__main__":
    test()
    # verify_result()
#    test_bf16_pipeline_matmul()