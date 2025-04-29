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
    def __init__(self, width=4, bit_width=4):
        self.width = width
        self.bit_width = bit_width
        self.A_bit_mask_vec = [0] * width
        self.B_bit_mask_vec = [0] * width
        self.A_row_offset_vec = [0] * width
        self.B_col_offset_vec = [0] * width
        self.AB_bit_vec = [0] * width
        self.AB_prefix_sum = [] 
        self.ec_idx_vec = [[] for _ in range(width)]
        

    def get_mask_offset_values(self, mask_A_row, mask_B_col, offset_A_row, offset_B_col, values_A, values_B):
        
        assert(len(mask_B_col) * len(mask_A_row) == self.width)
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
    



if __name__ == "__main__":

    A = np.array([[1, 1, 1, 1],
                [1, 1, 1, 1]])
    csr_A = csr_matrix(A)
    print(csr_A)

    B = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
    csr_B = csr_matrix(B.T)

    value_A, offset_A, masks_A = get_values_offset_mask(csr_A)
    value_B, offset_B, masks_B = get_values_offset_mask(csr_B)

    print(offset_A)

    print("Masks as binary:", [bin(m) for m in masks_A])
    
    mfiu = MFIU(4, 4)
    mfiu.get_mask_offset_values(masks_A, masks_B, offset_A, offset_B, value_A, value_B)
    mfiu()
    mfiu.print_state()

   

   