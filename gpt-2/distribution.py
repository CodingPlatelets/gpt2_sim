import numpy as np
import math
from scipy.sparse import csr_matrix

def min_bits_needed(bit_width):
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))

class Distribution:
    def __init__(self):
        self.data_buffer = []
        self.data_mask_1 = None
        self.data_mask_2 = None
    
    def get_data(self, data):
        self.data_buffer = data

class ShiftUnit:
    def __init__(self, bit_mask, ec_idx, values_len, bit_width=4, offset=0):
        self.bit_width = bit_width
        self.bit_mask = bit_mask
        self.ec_idx = ec_idx
        self.offset = offset
        self.min_bits_num = min_bits_needed(bit_width)
        self.zero_count = []
        self.zero_count_bit_vec = [[] for _ in range(self.min_bits_num)]
        self.index = [0] * len(self.ec_idx)
        self.values_len = values_len

    def get_zero_count(self):
        self.zero_count = []
        count_zeros = 0
        for i in range(self.bit_width):
            self.zero_count.append(count_zeros)
            bit = (self.bit_mask >> (self.bit_width - 1 - i)) & 1
            if bit == 0:
                count_zeros += 1
    

    def get_zero_count_bit_vec(self):
        bit_vec = []
        for count in self.zero_count:
            bit_vec.append(bin(count)[2:].zfill(self.min_bits_num))

        for bit in bit_vec:
            for i in range(self.min_bits_num):
                self.zero_count_bit_vec[i].append(int(bit[self.min_bits_num - i - 1]))

    def shift(self):

        shifted_ec_idx = self.ec_idx.copy()
        
        for bit_level in range(self.min_bits_num):
            temp_result = [0] * len(shifted_ec_idx)
            for i in range(len(self.zero_count_bit_vec[bit_level])):
                if self.zero_count_bit_vec[bit_level][i] == 1:
                    target_idx = i - (1 << bit_level)
                    if target_idx < len(temp_result):
                        temp_result[target_idx] = shifted_ec_idx[i]
                else:
                    temp_result[i] = shifted_ec_idx[i]
            
            shifted_ec_idx = temp_result.copy()
     
        self.index = [0] * self.values_len
        for i in range(len(shifted_ec_idx)):
            target_idx = i + self.offset
            if target_idx < self.values_len:  
                self.index[target_idx] = shifted_ec_idx[i]

    def print_state(self):
        """打印 ShiftUnit 对象的关键信息"""
        print("\n==== ShiftUnit 状态 ====")
        print(f"bit_width: {self.bit_width}")
        print(f"offset: {self.offset}")
        print(f"bit_mask: {bin(self.bit_mask)[2:].zfill(self.bit_width)}")
        print(f"min_bits_num: {self.min_bits_num}")
        
        print("\nzero_count: {self.zero_count}")
        
        print("\nzero_count_bit_vec:")
        for i, bit_vec in enumerate(self.zero_count_bit_vec):
            print(f"  Bit {i}: {bit_vec}")
        
        print("\nInput ec_idx: {self.ec_idx}")
        print(f"Output index: {self.index}")
        print("=======================")


    def __call__(self):           
        self.get_zero_count()
        self.get_zero_count_bit_vec()
        self.shift()


        

class MFIU:
    def __init__(self, width=None, bit_width=None):
        self.width = width
        self.bit_width = bit_width
        #self.A_bit_mask_vec = [0] * width
        #self.B_bit_mask_vec = [0] * width
        #self.A_row_offset_vec = [0] * width
        #self.B_col_offset_vec = [0] * width
        #self.AB_bit_vec = [0] * width
        #self.AB_prefix_sum = [] 
        #self.ec_idx_vec = [[] for _ in range(width)]
        

    def get_mask_offset_values(self, mask_A_row, mask_B_col, offset_A_row, offset_B_col, values_A, values_B):
        
        #assert(len(mask_B_col) * len(mask_A_row) == self.width)

        rows_A = len(mask_A_row)
        cols_B = len(mask_B_col)
        
        if self.width is None:
            self.width = rows_A * cols_B
        
        max_dim_A = len(bin(max(mask_A_row))) - 2 if mask_A_row else 0  
        max_dim_B = len(bin(max(mask_B_col))) - 2 if mask_B_col else 0
        
        if self.bit_width is None:
            self.bit_width = max(max_dim_A, max_dim_B)
        
       
        self.A_bit_mask_vec = [0] * self.width
        self.B_bit_mask_vec = [0] * self.width
        self.A_row_offset_vec = [0] * self.width
        self.B_col_offset_vec = [0] * self.width
        self.AB_bit_vec = [0] * self.width
        self.AB_prefix_sum = []
        self.ec_idx_vec = [[] for _ in range(self.width)]

        idx = 0
        for i in range(len(mask_B_col)):
            for j in range(len(mask_A_row)):
                self.B_bit_mask_vec[idx] = mask_B_col[i]
                self.B_col_offset_vec[idx] = offset_B_col[i]
                self.A_bit_mask_vec[idx] = mask_A_row[j]
                self.A_row_offset_vec[idx] = offset_A_row[j]
                idx += 1
        self.values_A = values_A
        self.values_B = values_B

    def bitvec_to_bitseq(self, bitvec):
        all_bits = []
        for val in bitvec:
            for i in range(self.bit_width):
                bit = (val >> (self.bit_width - 1 - i)) & 1
                all_bits.append(bit)
        return all_bits
    
    def ecseq_to_ecvec(self, ecseq):
        for i in range(self.width):
            temp = []
            for j in range(self.bit_width):
                temp.append(ecseq[i * self.bit_width + j])
            self.ec_idx_vec[i] = temp
    
    def init_shift_unit(self):
        self.shift_unit_a = [ShiftUnit(self.A_bit_mask_vec[i], self.ec_idx_vec[i], len(self.values_A), self.bit_width, self.A_row_offset_vec[i]) for i in range(self.width)]
        self.shift_unit_b = [ShiftUnit(self.B_bit_mask_vec[i], self.ec_idx_vec[i], len(self.values_B), self.bit_width, self.B_col_offset_vec[i]) for i in range(self.width)]

    def print_shift(self):
        print("shiftA index")
        for shift in self.shift_unit_a:
            print(shift.index)
        print("shiftB index")
        for shift in self.shift_unit_b:
            print(shift.index)
        

    def print_state(self):
        """打印MFIU对象的所有成员值"""
        print("==== MFIU状态 ====")
        print(f"width: {self.width}")
        print(f"bit_width: {self.bit_width}")
        print(f"A_bit_mask_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.A_bit_mask_vec]}")
        print(f"B_bit_mask_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.B_bit_mask_vec]}")
        print(f"AB_bit_vec: {[bin(x)[2:].zfill(self.bit_width) for x in self.AB_bit_vec]}")
        print(f"AB_prefix_sum: {self.AB_prefix_sum}")
        print(f"ec_idx_vec: {self.ec_idx_vec}")
        self.print_shift()
        print("==================")


    def __call__(self):
        self.AB_bit_vec = [a & b for a, b in zip(self.A_bit_mask_vec, self.B_bit_mask_vec)]
        
        bit_seq = np.array(self.bitvec_to_bitseq(self.AB_bit_vec))
        
        self.AB_prefix_sum = np.cumsum(bit_seq).tolist()
        
        ec_idx_seq = np.where(bit_seq, self.AB_prefix_sum, 0).tolist()
        
        self.ecseq_to_ecvec(ec_idx_seq)

        self.init_shift_unit()
        for shift in self.shift_unit_a:
            shift()
        for shift in self.shift_unit_b:
            shift()

class MulUnit:
    def __init__(self):
        self.sft_index = None
        self.val_a = None
        self.val_b = None

    def get_val_a(self, val_a):
        self.val_a = val_a

    def get_val_b(self, val_b):
        self.val_b  = val_b
    
    def get_sft_index(self, sft_index):
        self.sft_index = sft_index

    def run(self):
        return self.val_a * self.val_b
    
    def print_status(self):
        print("\n==== MulUnit 状态 ====")
        print(f"Shift Index: {self.sft_index}")
        print(f"Value A: {self.val_a}")
        print(f"Value B: {self.val_b}")
        print("======================")
        

class AddUnit:
    def __init__(self):
        self.sft_index_1 = None
        self.sft_index_2 = None
        self.val_1 = None
        self.val_2 = None

    def get(self, val_1, val_2, sft_index_1, sft_index_2):
        self.sft_index_1 = sft_index_1
        self.sft_index_2 = sft_index_2
        self.val_1 = val_1
        self.val_2 = val_2

    def run(self):
        if self.sft_index_1 == self.sft_index_2 and self.sft_index_1 != -1:
            return self.val_1 + self.val_2, self.sft_index_1
        return None, self.sft_index_1, self.sft_index_2
 

class Trapezoid:
    def __init__(self, PE_num=4):
        self.M = 0
        self.K = 0
        self.N = 0
        self.PE_num = PE_num
        self.mfiu = MFIU()
        self.mul_vec = [MulUnit() for _ in range(self.PE_num)]
        self.mul_queue_a = [[] for _ in range(self.PE_num)]
        self.mul_queue_b = [[] for _ in range(self.PE_num)]
        self.sft_index_queue = [[] for _ in range(self.PE_num)]
        self.c_index_set = set()
        self.c_values = None

        self.add_lower = [AddUnit() for _ in range(int(self.PE_num / 2))]
        

        self.add_upper = AddUnit()  #TODO: PE_num != 4 
        self.sft_index_a = []
        self.sft_index_b = []
        self.values_A = None
        self.values_B = None
        self.mul_result = [None for _ in range(self.PE_num)]

        self.round_num = 0
        self.C_matrix = None

    def pad_queues(self):
        max_len_a = max([len(queue) for queue in self.mul_queue_a]) if self.mul_queue_a else 0
        max_len_b = max([len(queue) for queue in self.mul_queue_b]) if self.mul_queue_b else 0
        
        assert(max_len_a == max_len_b)

        for queue in self.mul_queue_a:
            while len(queue) < max_len_a:
                queue.append(0)
                
        for queue in self.mul_queue_b:
            while len(queue) < max_len_b:
                queue.append(0)

        for queue in self.sft_index_queue:
            while len(queue) < max_len_a:
                queue.append(-1)
        self.round_num = max_len_a

    def run(self, A:np.array, B:np.array):

        self.M = A.shape[0]
        self.K = A.shape[1]
        self.N = B.shape[1]

        self.c_values = [0] * self.M * self.N

        csr_A = csr_matrix(A)
        csr_B = csr_matrix(B.T)


        values_A, offset_A, masks_A = get_values_offset_mask(csr_A)
        values_B, offset_B, masks_B = get_values_offset_mask(csr_B)
        self.values_A = values_A
        self.values_B = values_B
        self.mfiu.get_mask_offset_values(masks_A, masks_B, offset_A, offset_B, values_A, values_B)
        self.mfiu()
        
        for shift in self.mfiu.shift_unit_a:
            self.sft_index_a.append(shift.index)
        for shift in self.mfiu.shift_unit_b:
            self.sft_index_b.append(shift.index)

        for sft_index in range(len(self.sft_index_a)):
            sft_row_a = self.sft_index_a[sft_index]
            sft_row_b = self.sft_index_b[sft_index]
            for i in range(len(sft_row_a)):
                if sft_row_a[i] != 0:
                    self.mul_queue_a[(sft_row_a[i] - 1) % self.PE_num].append(self.values_A[i])
                    self.c_index_set.add(sft_index)
                    #self.mul_vec[sft_row_a[i] - 1].get_val_a(self.values_A[i])
                    self.sft_index_queue[(sft_row_a[i] - 1) % self.PE_num].append(sft_index) # A and B are same

            for i in range(len(sft_row_b)):
                if sft_row_b[i] != 0:
                    self.mul_queue_b[(sft_row_b[i] - 1) % self.PE_num].append(self.values_B[i])
                    self.c_index_set.add(sft_index)
                    #self.mul_vec[sft_row_b[i] - 1].get_val_b(self.values_B[i])
                    #self.mul_vec[sft_row_b[i] - 1].get_sft_index(sft_index)

        self.pad_queues()

        for _ in range(self.round_num):
            #TODO: remeber to pop queue
            for i, mul in enumerate(self.mul_vec):
                    mul.get_val_a(self.mul_queue_a[i][0])
                    mul.get_val_b(self.mul_queue_b[i][0])
                    mul.get_sft_index(self.sft_index_queue[i][0])
                    self.mul_result[i] = mul.run()
            add_lower_value_queue = []
            add_lower_index_queue = []

            index_value_map = [[] for _ in range(self.mfiu.width)]

            for i, add in enumerate(self.add_lower):
                idx = i * 2
                add.get(self.mul_result[idx], self.mul_result[idx + 1], self.mul_vec[idx].sft_index, self.mul_vec[idx + 1].sft_index)
                result = add.run()
                if result[0] is None:
                    add_lower_index_queue.append(result[1])
                    add_lower_index_queue.append(result[2])
                    add_lower_value_queue.append(add.val_1)
                    add_lower_value_queue.append(add.val_2)
                else:
                    add_lower_index_queue.append(result[1])
                    add_lower_value_queue.append(result[0])
                
            for i, index in enumerate(add_lower_index_queue):
                if index != -1:
                    index_value_map[index].append(add_lower_value_queue[i])
            #A {0 : 0, 1 : 1, 2 : 0, 3 : 1}
            #B {0 : 0, 1 : 0, 2 : 1, 3 : 1}

            for index, values in enumerate(index_value_map):
                m = index % self.M
                n = int(index / self.M)
                if len(values) == 2:
                    self.add_upper.get(values[0], values[1], index, index)
                    result = self.add_upper.run()
                    self.c_values[m * self.N + n] += result[0]
                elif len(values) == 1:
                    self.c_values[m * self.N + n] += values[0]

            for i in range(self.PE_num):
                if self.mul_queue_a[i]:
                    self.mul_queue_a[i].pop(0)
                if self.mul_queue_b[i]:
                    self.mul_queue_b[i].pop(0)
                if self.sft_index_queue[i]:
                    self.sft_index_queue[i].pop(0)
            
                    
        self.C_matrix = np.array(self.c_values).reshape(self.M, self.N)
        
    def print_status(self):
        """打印Trapezoid对象的完整状态信息"""
        print("\n==== Trapezoid状态 ====")
        print(f"矩阵维度: M={self.M}, K={self.K}, N={self.N}")
        print(f"PE数量: {self.PE_num}")
        print(f"执行轮数: {self.round_num}")
        
        print("\n== 移位索引 ==")
        print("A矩阵移位索引:")
        for i, index in enumerate(self.sft_index_a):
            print(f"  [{i}]: {index}")
        print("B矩阵移位索引:")
        for i, index in enumerate(self.sft_index_b):
            print(f"  [{i}]: {index}")
        
        print("\n== 队列状态 ==")
        for i in range(self.PE_num):
            print(f"PE {i}:")
            print(f"  A值队列: {self.mul_queue_a[i]}")
            print(f"  B值队列: {self.mul_queue_b[i]}")
            print(f"  移位索引队列: {self.sft_index_queue[i]}")
        
        print("\n== 乘法单元状态 ==")
        for i, mul in enumerate(self.mul_vec):
            print(f"乘法单元 {i}:")
            print(f"  Shift Index: {mul.sft_index}")
            print(f"  Value A: {mul.val_a}")
            print(f"  Value B: {mul.val_b}")
            print(f"  Result: {mul.run() if mul.val_a is not None and mul.val_b is not None else None}")
        
        print("\n== 加法单元状态 ==")
        print("下层加法单元:")
        for i, add in enumerate(self.add_lower):
            print(f"  加法单元 {i}:")
            print(f"    Index 1: {add.sft_index_1}")
            print(f"    Index 2: {add.sft_index_2}")
            print(f"    Value 1: {add.val_1}")
            print(f"    Value 2: {add.val_2}")
        
        print("上层加法单元:")
        print(f"  Index 1: {self.add_upper.sft_index_1}")
        print(f"  Index 2: {self.add_upper.sft_index_2}")
        print(f"  Value 1: {self.add_upper.val_1}")
        print(f"  Value 2: {self.add_upper.val_2}")
        
        print("\n== 矩阵C索引集合 ==")
        print(f"索引集合: {self.c_index_set}")
        
        print("\n== 计算结果 ==")
        # 格式化输出矩阵C
        
        
        print("矩阵C:")
        print(self.C_matrix)
        
        print("========================")






def get_values_offset_mask(matrix:csr_matrix):
    values = matrix.data
    col_indices = matrix.indices
    row_ptr = matrix.indptr
    
    # 获取矩阵大小
    num_rows = len(row_ptr) - 1
    num_cols = matrix.shape[1]  
    
    # 创建每行的二进制掩码
    masks = []
    
    for row in range(num_rows):
        start = row_ptr[row]
        end = row_ptr[row + 1]
        row_mask = 0
        
        for i in range(start, end):
            col = col_indices[i]
            row_mask |= (1 << (num_cols - 1 - col))
        
        masks.append(row_mask)
    
    return values, row_ptr, masks
    



#if __name__ == "__main__":
#
#    A = np.array([[1, 0, 1, 0], [0, 1, 1, 0]])
#
#    B = np.array([[1, 1], [0, 0], [0, 1], [1, 0]])
#
#
#    trapezoid = Trapezoid()
#    trapezoid.run(A, B)
#    trapezoid.print_status()

if __name__ == "__main__":
    
    # 测试用例1：简单的2x4和4x2矩阵
    print("\n===== 测试用例1：简单矩阵 =====")
    A1 = np.array([[1, 0, 1, 0], 
                   [0, 1, 1, 0]])
    
    B1 = np.array([[1, 1], 
                   [0, 0], 
                   [0, 1], 
                   [1, 0]])
    
    expected1 = np.matmul(A1, B1)
    print(f"预期结果:\n{expected1}")
    
    trapezoid1 = Trapezoid()
    trapezoid1.run(A1, B1)
    print(f"Trapezoid结果:\n{trapezoid1.C_matrix}")
    print(f"结果正确: {np.array_equal(expected1, trapezoid1.C_matrix)}")
    
    
    # 测试用例2：零矩阵
    print("\n===== 测试用例2：零矩阵 =====")
    A2 = np.zeros((3, 4), dtype=int)
    B2 = np.zeros((4, 2), dtype=int)
    
    expected2 = np.matmul(A2, B2)
    print(f"预期结果:\n{expected2}")
    
    trapezoid2 = Trapezoid()
    trapezoid2.run(A2, B2)
    print(f"Trapezoid结果:\n{trapezoid2.C_matrix}")
    print(f"结果正确: {np.array_equal(expected2, trapezoid2.C_matrix)}")
    
    
    # 测试用例3：随机稀疏矩阵
    print("\n===== 测试用例3：随机稀疏矩阵 =====")
    # 创建稀疏矩阵，只有20%的元素为1
    np.random.seed(42)  # 设定随机种子以便结果可复现
    A3 = np.random.choice([0, 1], size=(103, 23), p=[0.8, 0.2])
    B3 = np.random.choice([0, 1], size=(23, 32), p=[0.8, 0.2])
    
    expected3 = np.matmul(A3, B3)
    print(f"矩阵A:\n{A3}")
    print(f"矩阵B:\n{B3}")
    print(f"预期结果:\n{expected3}")
    
    trapezoid3 = Trapezoid()
    trapezoid3.run(A3, B3)
    print(f"Trapezoid结果:\n{trapezoid3.C_matrix}")
    print(f"结果正确: {np.array_equal(expected3, trapezoid3.C_matrix)}")
    #trapezoid3.print_status()
    
    
    # 测试用例4：单位矩阵
    print("\n===== 测试用例4：单位矩阵 =====")
    A4 = np.eye(4, dtype=int)
    B4 = np.eye(4, dtype=int)
    
    expected4 = np.matmul(A4, B4)  # 单位矩阵乘单位矩阵应该等于单位矩阵
    print(f"预期结果:\n{expected4}")
    
    trapezoid4 = Trapezoid()
    trapezoid4.run(A4, B4)
    print(f"Trapezoid结果:\n{trapezoid4.C_matrix}")
    print(f"结果正确: {np.array_equal(expected4, trapezoid4.C_matrix)}")

    # 测试用例5：稠密矩阵
    print("\n===== 测试用例5：稠密矩阵 =====")
    A2 = np.random.randint(0, 10, size=(12, 11))
    B2 = np.random.randint(0, 10, size=(11, 13))
    
    expected2 = np.matmul(A2, B2)
    print(f"预期结果:\n{expected2}")
    
    trapezoid2 = Trapezoid()
    trapezoid2.run(A2, B2)
    print(f"Trapezoid结果:\n{trapezoid2.C_matrix}")
    print(f"结果正确: {np.array_equal(expected2, trapezoid2.C_matrix)}")
    
    
   ## 测试用例5：不同PE数量
   #print("\n===== 测试用例5：不同PE数量 =====")
   #A5 = np.array([[1, 2, 0], 
   #               [0, 1, 1]])
   #
   #B5 = np.array([[1, 0], 
   #               [1, 1], 
   #               [0, 1]])
   #
   #expected5 = np.matmul(A5, B5)
   #print(f"预期结果:\n{expected5}")
    

    

   