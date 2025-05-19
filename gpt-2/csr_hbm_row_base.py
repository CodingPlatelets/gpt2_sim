import numpy as np
from scipy.sparse import csr_matrix
import time


def store_csr_in_blocks_row_based(
    csr_matrix,
    target_block_size=2048,  # 目标块大小(字节)
    max_rows_per_block=None,  # 可选的每块最大行数限制
):
    """
    使用基于行的分块策略将CSR矩阵分块存储
    每个块包含完整的行数据(row_pointers、values和column_indices)

    Args:
        csr_matrix: 输入的CSR格式矩阵
        target_block_size: 目标块大小，单位为字节
        max_rows_per_block: 每块包含的最大行数，None表示不限制

    Returns:
        blocks: 分块后的矩阵数据列表
    """
    # 提取CSR数据
    values = csr_matrix.data  # 16位值
    col_indices = csr_matrix.indices  # 16位列索引
    row_pointers = csr_matrix.indptr  # 32位行指针
    num_rows = csr_matrix.shape[0]

    # 估计数据类型所需字节数
    value_size = 2  # 16位 = 2字节
    col_idx_size = 2  # 16位 = 2字节
    row_ptr_size = 4  # 32位 = 4字节

    blocks = []
    current_block = {
        "values": [],
        "col_indices": [],
        "row_pointers": [],
        "row_range": [0, 0],  # [起始行, 结束行]
        "row_offsets": [],  # 每行在块中的偏移
    }

    current_size = 0
    start_row = 0

    for row in range(num_rows):
        # 计算当前行的数据大小
        row_start = row_pointers[row]
        row_end = row_pointers[row + 1]
        nonzeros_in_row = row_end - row_start

        # 该行数据所需字节数
        row_data_size = (
            (nonzeros_in_row * value_size)
            + (nonzeros_in_row * col_idx_size)
            + row_ptr_size
        )

        # 检查是否应该创建新块
        if (
            (current_size + row_data_size > target_block_size)
            or (
                max_rows_per_block is not None and row - start_row >= max_rows_per_block
            )
        ) and current_size > 0:

            # 完成当前块
            current_block["row_range"] = [start_row, row - 1]
            blocks.append(current_block)

            # 创建新块
            current_block = {
                "values": [],
                "col_indices": [],
                "row_pointers": [],
                "row_range": [row, 0],  # 最终结束行将在后续设置
                "row_offsets": [],  # 记录每行在块内的起始位置
            }
            current_size = 0
            start_row = row

        # 添加行指针
        # 在块内部，我们使用相对偏移
        relative_offset = len(current_block["values"])
        current_block["row_offsets"].append(relative_offset)
        if row == start_row:
            current_block["row_pointers"].append(0)  # 第一行相对偏移为0
        else:
            current_block["row_pointers"].append(relative_offset)

        # 添加行数据
        for i in range(row_start, row_end):
            current_block["values"].append(values[i])
            current_block["col_indices"].append(col_indices[i])

        current_size += row_data_size

    # 确保最后一个块被添加
    if current_block["values"]:
        current_block["row_range"] = [start_row, num_rows - 1]
        blocks.append(current_block)

    # 每个块的最后添加行尾指针
    for block in blocks:
        start, end = block["row_range"]
        if end == num_rows - 1:  # 最后一行
            block["row_pointers"].append(len(block["values"]))
        else:
            next_row_offset = len(block["values"])
            block["row_pointers"].append(next_row_offset)

    print(f"总行数: {num_rows}")
    print(f"总块数: {len(blocks)}")
    print(f"平均每块行数: {num_rows/len(blocks):.1f}")

    # 计算块大小统计
    block_sizes = [
        len(b["values"]) * value_size
        + len(b["col_indices"]) * col_idx_size
        + len(b["row_pointers"]) * row_ptr_size
        for b in blocks
    ]
    avg_block_size = sum(block_sizes) / len(block_sizes)
    max_block_size = max(block_sizes)
    min_block_size = min(block_sizes)

    print(f"平均块大小: {avg_block_size:.1f} 字节 ({avg_block_size/1024:.2f} KB)")
    print(f"最大块大小: {max_block_size} 字节 ({max_block_size/1024:.2f} KB)")
    print(f"最小块大小: {min_block_size} 字节 ({min_block_size/1024:.2f} KB)")

    return blocks


def verify_csr_blocks(original_csr, reconstructed_csr):
    """验证重建的CSR矩阵是否与原始矩阵相同"""
    # 检查形状
    if original_csr.shape != reconstructed_csr.shape:
        print(
            f"形状不匹配! 原始: {original_csr.shape}, 重建: {reconstructed_csr.shape}"
        )
        return False

    # 检查非零元素数量
    if len(original_csr.data) != len(reconstructed_csr.data):
        print(
            f"非零元素数量不匹配! 原始: {len(original_csr.data)}, 重建: {len(reconstructed_csr.data)}"
        )
        return False

    # 检查数据值
    if not np.array_equal(original_csr.data, reconstructed_csr.data):
        print("Values不匹配!")
        print(f"原始: {original_csr.data}")
        print(f"重建: {reconstructed_csr.data}")
        return False

    # 检查列索引
    if not np.array_equal(original_csr.indices, reconstructed_csr.indices):
        print("Column indices不匹配!")
        print(f"原始: {original_csr.indices}")
        print(f"重建: {reconstructed_csr.indices}")
        return False

    # 检查行指针
    if not np.array_equal(original_csr.indptr, reconstructed_csr.indptr):
        print("Row pointers不匹配!")
        print(f"原始: {original_csr.indptr}")
        print(f"重建: {reconstructed_csr.indptr}")
        return False

    # 检查转换为密集矩阵后是否相同
    if not np.array_equal(original_csr.toarray(), reconstructed_csr.toarray()):
        print("转换为密集矩阵后不匹配!")
        return False

    return True


def reconstruct_csr_from_row_blocks(blocks, num_rows, num_cols):
    """从行分块重建CSR矩阵"""
    # 准备数组容器
    all_values = []
    all_col_indices = []
    all_row_pointers = [0]  # CSR格式的第一个行指针总是0

    current_nnz = 0

    # 按照行范围排序块
    sorted_blocks = sorted(blocks, key=lambda b: b["row_range"][0])

    # 处理每个块
    for block in sorted_blocks:
        start_row, end_row = block["row_range"]

        # 获取块内数据
        block_values = block["values"]
        block_col_indices = block["col_indices"]
        block_row_pointers = block["row_pointers"]

        # 收集values和column indices
        all_values.extend(block_values)
        all_col_indices.extend(block_col_indices)

        # 转换块内部相对行指针为全局行指针
        for row in range(start_row, end_row + 1):
            row_idx = row - start_row
            if row_idx < len(block_row_pointers) - 1:
                next_row_offset = block_row_pointers[row_idx + 1]
            else:
                next_row_offset = len(block_values)

            # 添加全局行指针 (上一个指针值 + 当前行中的元素数量)
            all_row_pointers.append(current_nnz + next_row_offset)

        # 更新全局非零元素计数
        current_nnz += len(block_values)

    # 确保行指针数量正确
    if len(all_row_pointers) != num_rows + 1:
        print(
            f"警告: 行指针数量 ({len(all_row_pointers)}) 与预期 ({num_rows + 1}) 不匹配!"
        )
        # 如果需要，补齐缺失的行指针
        while len(all_row_pointers) < num_rows + 1:
            all_row_pointers.append(current_nnz)

    # 重建CSR矩阵
    reconstructed_csr = csr_matrix(
        (all_values, all_col_indices, all_row_pointers), shape=(num_rows, num_cols)
    )

    return reconstructed_csr


def test_csr_row_blocking(matrix, target_block_size=2048, max_rows_per_block=None):
    """测试基于行的CSR矩阵分块存储和重建"""
    print(f"\n测试矩阵形状: {matrix.shape}")

    # 计算矩阵稀疏度
    total_elements = matrix.shape[0] * matrix.shape[1]
    nonzeros = np.count_nonzero(matrix)
    sparsity = nonzeros / total_elements
    print(f"矩阵稀疏度: {sparsity:.6f} ({nonzeros}/{total_elements} 非零元素)")

    # 转换为CSR格式
    csr = csr_matrix(matrix)

    # 分块存储
    start_time = time.time()
    blocks = store_csr_in_blocks_row_based(csr, target_block_size, max_rows_per_block)
    blocking_time = time.time() - start_time
    print(f"分块时间: {blocking_time:.6f} 秒")

    # 分析每个块的行指针、值和列索引的数据局部性
    print("\n=== 数据局部性分析 ===")
    full_coverage_blocks = 0
    partial_coverage_blocks = 0

    for i, block in enumerate(blocks[:5]):  # 只显示前5个块的详细信息
        start_row, end_row = block["row_range"]
        rows_in_block = end_row - start_row + 1
        values_count = len(block["values"])
        col_indices_count = len(block["col_indices"])
        row_pointers_count = len(block["row_pointers"])

        print(f"\n块 {i} (行 {start_row}-{end_row}):")
        print(f"  包含 {rows_in_block} 行")
        print(f"  Values: {values_count}")
        print(f"  Column Indices: {col_indices_count}")
        print(f"  Row Pointers: {row_pointers_count}")

        if row_pointers_count == rows_in_block + 1:
            full_coverage_blocks += 1
        else:
            partial_coverage_blocks += 1

    if len(blocks) > 5:
        print(f"\n... 以及其他 {len(blocks) - 5} 个块")

    print(f"\n完整行覆盖块: {full_coverage_blocks}/{len(blocks)}")
    print(f"部分行覆盖块: {partial_coverage_blocks}/{len(blocks)}")

    # 分析有效数据利用率
    total_bytes = 0
    effective_bytes = 0

    for block in blocks:
        values_bytes = len(block["values"]) * 2  # 16位 = 2字节
        col_indices_bytes = len(block["col_indices"]) * 2  # 16位 = 2字节
        row_pointers_bytes = len(block["row_pointers"]) * 4  # 32位 = 4字节

        block_bytes = values_bytes + col_indices_bytes + row_pointers_bytes
        # 假设块大小向上取整到目标大小
        padded_block_bytes = (
            (block_bytes + target_block_size - 1) // target_block_size
        ) * target_block_size

        total_bytes += padded_block_bytes
        effective_bytes += block_bytes

    utilization = effective_bytes / total_bytes
    print(f"\n带宽利用率: {utilization:.2%} ({effective_bytes}/{total_bytes} 字节)")

    # 重建CSR
    start_time = time.time()
    reconstructed_csr = reconstruct_csr_from_row_blocks(
        blocks, csr.shape[0], csr.shape[1]
    )
    reconstruction_time = time.time() - start_time
    print(f"重建时间: {reconstruction_time:.6f} 秒")

    # 验证
    is_valid = verify_csr_blocks(csr, reconstructed_csr)
    print(f"\n验证结果: {'成功' if is_valid else '失败'}")

    return is_valid, blocks


def run_tests():
    # ... 其他测试代码 ...

    # 测试行分块策略
    print("\n===== 测试基于行的CSR矩阵分块存储 =====")

    # 测试中等大小矩阵
    print("\n测试: 中等大小矩阵 (行分块)")
    medium_matrix = np.random.choice([0, 1], size=(1000, 1000), p=[0.95, 0.05])
    test_csr_row_blocking(medium_matrix, target_block_size=2048)

    # 测试不同稀疏度
    print("\n测试: 高稀疏度矩阵 (行分块)")
    sparse_matrix = np.random.choice([0, 1], size=(1000, 1000), p=[0.99, 0.01])
    test_csr_row_blocking(sparse_matrix, target_block_size=2048)

    # 测试4096x2048矩阵
    print("\n测试: 4096x4096矩阵 (行分块)")
    large_matrix = np.random.choice([0, 1], size=(4096, 4096), p=[0.90, 0.10])
    test_csr_row_blocking(large_matrix, target_block_size=2048)

if __name__ == "__main__":
    run_tests()