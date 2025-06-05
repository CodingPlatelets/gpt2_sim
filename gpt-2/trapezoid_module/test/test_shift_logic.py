def test_shift_logic():
    """测试移位逻辑的简单实现"""
    
    # 测试数据
    mask_binary = "01100101"  # 0 1 1 0 0 1 0 1
    mask_int = int(mask_binary, 2)  # 转换为整数：101 (十进制)
    
    index = [0, 5, 0, 0, 0, 6, 0, 7]
    offset = 4
    values = "abcdefgh"
    
    print(f"输入:")
    print(f"  mask (binary): {mask_binary}")
    print(f"  mask (int): {mask_int} (0b{bin(mask_int)[2:].zfill(8)})")
    print(f"  index: {index}")
    print(f"  offset: {offset}")
    print(f"  values: {values}")
    print()
    
    # 方法1: 基础实现
    result1 = basic_shift_implementation(mask_int, index, offset, len(values))
    print(f"方法1 (基础实现) 结果: {result1}")
    
    # 方法2: 优化实现
    result2 = optimized_shift_implementation(mask_int, index, offset, len(values))
    print(f"方法2 (优化实现) 结果: {result2}")
    
    # 验证values中对应位置的字符
    print(f"\n验证 - 提取的字符:")
    for i, idx in enumerate(result1):
        if idx != 0:
            print(f"  位置 {i}: index={idx} -> values[{idx}] = '{values[idx]}'")
    
    # 验证结果一致性
    print(f"\n结果一致性: {result1 == result2}")
    
    return result1, result2

def basic_shift_implementation(mask_int, index, offset, values_len):
    """基础实现：模拟shift_sim.py的逻辑"""
    bit_width = len(index)
    
    # Step 1: 计算zero count (模拟stage2)
    zero_count = []
    count_zeros = 0
    
    for i in range(bit_width):
        zero_count.append(count_zeros)
        bit = (mask_int >> (bit_width - 1 - i)) & 1
        if bit == 0:
            count_zeros += 1
    
    print(f"  Stage2 - zero_count: {zero_count}")
    
    # Step 2: 转换为bit vectors (模拟stage3)
    min_bits_num = max(1, len(bin(max(zero_count) if zero_count else 0)[2:]))
    zero_count_bit_vec = [[] for _ in range(min_bits_num)]
    
    for count in zero_count:
        bit_repr = bin(count)[2:].zfill(min_bits_num)
        for bit_pos in range(min_bits_num):
            zero_count_bit_vec[bit_pos].append(int(bit_repr[min_bits_num - 1 - bit_pos]))
    
    print(f"  Stage3 - min_bits_num: {min_bits_num}")
    print(f"  Stage3 - zero_count_bit_vec: {zero_count_bit_vec}")
    
    # Step 3: 移位操作 (模拟stage4)
    shifted_ec_idx = index.copy()
    
    for bit_level in range(min_bits_num):
        temp_result = [0] * len(shifted_ec_idx)
        
        for i in range(len(zero_count_bit_vec[bit_level])):
            if zero_count_bit_vec[bit_level][i] == 1:
                target_idx = i - (1 << bit_level)
                if target_idx >= 0 and target_idx < len(temp_result):
                    temp_result[target_idx] = shifted_ec_idx[i]
            else:
                temp_result[i] = shifted_ec_idx[i]
        
        shifted_ec_idx = temp_result.copy()
        print(f"  Stage4 - bit_level {bit_level}: {shifted_ec_idx}")
    
    # Step 4: 应用offset
    output = [0] * values_len
    for i in range(len(shifted_ec_idx)):
        target_idx = i + offset
        if target_idx < values_len:
            output[target_idx] = shifted_ec_idx[i]
    
    return output

def optimized_shift_implementation(mask_int, index, offset, values_len):
    """优化实现：更直观的逻辑"""
    bit_width = len(index)
    
    # 直接基于mask提取有效索引
    valid_indices = []
    valid_positions = []
    
    for i in range(bit_width):
        bit = (mask_int >> (bit_width - 1 - i)) & 1
        if bit == 1:
            valid_indices.append(index[i])
            valid_positions.append(i)
    
    #print(f"  优化版 - 有效位置: {valid_positions}")
    #print(f"  优化版 - 有效索引: {valid_indices}")
    
    # 计算移位后的位置
    result = [0] * values_len
    
    # 对于每个有效索引，计算其在结果中的位置
    result_pos = offset
    for idx in valid_indices:
        if result_pos < values_len:
            result[result_pos] = idx
            result_pos += 1
    
    return result

def additional_test_cases():
    """额外的测试用例"""
    print("\n" + "="*50)
    print("额外测试用例")
    print("="*50)
    
    # 测试用例1: 全1 mask
    print("\n测试用例1: 全1 mask")
    mask1 = 0b11111111  # 255
    index1 = [1, 2, 3, 4, 5, 6, 7, 8]
    offset1 = 0
    values1 = "abcdefgh"
    
    result1_basic = basic_shift_implementation(mask1, index1, offset1, len(values1))
    result1_opt = optimized_shift_implementation(mask1, index1, offset1, len(values1))
    print(f"基础实现: {result1_basic}")
    print(f"优化实现: {result1_opt}")
    print(f"一致性: {result1_basic == result1_opt}")
    
    # 测试用例2: 稀疏mask
    print("\n测试用例2: 稀疏mask")
    mask2 = 0b10001000  # 136
    index2 = [10, 20, 30, 40, 50, 60, 70, 80]
    offset2 = 2
    values2 = "abcdefghij"
    
    result2_basic = basic_shift_implementation(mask2, index2, offset2, len(values2))
    result2_opt = optimized_shift_implementation(mask2, index2, offset2, len(values2))
    print(f"基础实现: {result2_basic}")
    print(f"优化实现: {result2_opt}")
    print(f"一致性: {result2_basic == result2_opt}")

def min_bits_needed(bit_width):
    """辅助函数：计算所需的最小位数"""
    import math
    if bit_width <= 0:
        return 0
    return math.ceil(math.log2(bit_width))

if __name__ == "__main__":
    print("Shift Logic 测试程序")
    print("="*50)
    
    # 运行主测试
    result1, result2 = test_shift_logic()
    
    # 运行额外测试
    additional_test_cases()
    
    print(f"\n期望结果: [0, 0, 0, 0, 5, 0, 6, 7]")
    print(f"实际结果: {result1}")
    print(f"测试通过: {result1 == [0, 0, 0, 0, 5, 0, 6, 7]}")