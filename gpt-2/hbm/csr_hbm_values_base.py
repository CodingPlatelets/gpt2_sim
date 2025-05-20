import numpy as np
from scipy.sparse import csr_matrix
import time


def store_csr_in_simple_blocks(csr_matrix, elements_per_block=2):
    """
    将CSR矩阵简单分块存储，每块包含固定数量的values和indices

    Args:
        csr_matrix: 输入的CSR格式矩阵
        elements_per_block: 每块包含的元素数量
    """
    values = csr_matrix.data
    col_indices = csr_matrix.indices
    row_pointers = csr_matrix.indptr
    num_rows = csr_matrix.shape[0]
    total_values = len(values)

    # 计算需要的块数量
    num_blocks = (total_values + elements_per_block - 1) // elements_per_block

    blocks = []

    # 对每个块进行处理
    for block_idx in range(num_blocks):
        start_idx = block_idx * elements_per_block
        end_idx = min(start_idx + elements_per_block, total_values)

        # 提取当前块的values和indices
        block_values = values[start_idx:end_idx].tolist()
        block_col_indices = col_indices[start_idx:end_idx].tolist()

        # 找出当前块涉及的行范围
        # 对每行检查是否有元素在当前块的范围内
        rows_in_block = []
        for row in range(num_rows):
            row_start_pos = row_pointers[row]
            row_end_pos = row_pointers[row + 1]

            # 如果行的元素范围与块范围有重叠
            if row_start_pos < end_idx and row_end_pos > start_idx:
                rows_in_block.append(row)

        if not rows_in_block:
            print(f"警告: 块 {block_idx} 没有找到相关行!")
            continue

        row_start = min(rows_in_block)

        # 创建块内行指针
        block_row_ptr = [0]
        current_pos = 0

        # 计算每行在块内的元素数量
        for row in range(row_start, max(rows_in_block) + 1):
            if row in rows_in_block:
                # 计算该行在当前块内的元素数量
                row_begin = max(row_pointers[row], start_idx)
                row_end_pos = min(row_pointers[row + 1], end_idx)
                row_elements = max(0, row_end_pos - row_begin)
            else:
                # 块内不存在的行
                row_elements = 0

            current_pos += row_elements
            block_row_ptr.append(int(current_pos))

        # 创建块
        block = {
            "values": block_values,
            "col_indices": block_col_indices,
            "row_ptr": block_row_ptr,
            "row_start_index": row_start,
        }

        blocks.append(block)

    return blocks


def test_csr_simple_blocks(
    matrix, elements_per_block=2, print_blocks=3, print_elements=10, debug=False
):
    """
    测试简单分块CSR存储和重建

    Args:
        matrix: 输入矩阵
        elements_per_block: 每块包含的元素数量
        print_blocks: 要打印详情的块数量
        print_elements: 每个块要打印的元素数量
        debug: 是否打印调试信息
    """
    print(f"\n测试矩阵形状: {matrix.shape}")

    # 计算矩阵稀疏度
    total_elements = matrix.shape[0] * matrix.shape[1]
    nonzeros = np.count_nonzero(matrix)
    sparsity = nonzeros / total_elements
    print(f"矩阵稀疏度: {sparsity:.6f} ({nonzeros}/{total_elements} 非零元素)")

    # 转换为CSR格式
    csr = csr_matrix(matrix)

    # 打印原始CSR数据
    if debug:
        print("\n原始CSR数据:")
        print(f"Values: {csr.data}")
        print(f"Column indices: {csr.indices}")
        print(f"Row pointers: {csr.indptr}")

    # 分块
    start_time = time.time()
    blocks = store_csr_in_simple_blocks(csr, elements_per_block)
    blocking_time = time.time() - start_time
    print(f"分块时间: {blocking_time:.6f} 秒")
    print(f"生成的块数: {len(blocks)}")

    # 分析分块效率
    print("\n=== 分块效率分析 ===")
    block_sizes = [len(b["values"]) for b in blocks]
    avg_block_size = sum(block_sizes) / len(blocks) if blocks else 0
    max_block_size = max(block_sizes) if blocks else 0
    min_block_size = min(block_sizes) if blocks else 0

    print(f"平均每块元素数: {avg_block_size:.2f}")
    print(f"元素数范围: {min_block_size} - {max_block_size}")

    # 分析行分布
    rows_per_block = {}
    for i, block in enumerate(blocks):
        row_start = block["row_start_index"]
        row_count = len(block["row_ptr"]) - 1
        rows_per_block[i] = row_count

    avg_rows = (
        sum(rows_per_block.values()) / len(rows_per_block) if rows_per_block else 0
    )
    max_rows = max(rows_per_block.values()) if rows_per_block else 0
    min_rows = min(rows_per_block.values()) if rows_per_block else 0

    print(f"平均每块行数: {avg_rows:.2f}")
    print(f"行数范围: {min_rows} - {max_rows}")

    # 打印块详情
    print("\n=== 块详情 ===")
    for i, block in enumerate(blocks):
        if i >= print_blocks:
            print(f"\n... 省略剩余 {len(blocks) - print_blocks} 个块的详情 ...")
            break

        row_start = block["row_start_index"]
        row_count = len(block["row_ptr"]) - 1

        print(f"\n块 #{i}:")
        print(f"  起始行索引: {row_start}")
        print(f"  行数: {row_count}")
        print(f"  元素数: {len(block['values'])}")

        # 显示行指针
        print(f"  行指针 ({len(block['row_ptr'])}): ", end="")
        if len(block["row_ptr"]) <= 10:
            print(f"{block['row_ptr']}")
        else:
            print(f"{block['row_ptr'][:5]}...{block['row_ptr'][-5:]}")

        # 显示元素值
        element_count = min(print_elements, len(block["values"]))
        print(f"  元素值 (前{element_count}个):")
        print("  索引\t值\t列索引")
        print("  -------------------")

        for j in range(element_count):
            value = block["values"][j]
            col_idx = block["col_indices"][j]
            print(f"  {j}\t{value}\t{col_idx}")

        if len(block["values"]) > element_count:
            print(f"  ... 以及其他 {len(block['values']) - element_count} 个元素")

    # 分析带宽利用率
    total_bytes = 0
    effective_bytes = 0
    metadata_bytes = 0

    for block in blocks:
        # 有效数据
        values_bytes = len(block["values"]) * 2  # 16位 = 2字节
        col_indices_bytes = len(block["col_indices"]) * 2  # 16位 = 2字节
        effective_data = values_bytes + col_indices_bytes

        # 元数据
        row_ptr_bytes = len(block["row_ptr"]) * 4  # 32位 = 4字节
        start_row_bytes = 4  # 起始行索引，32位 = 4字节
        metadata = row_ptr_bytes + start_row_bytes

        # 假设块大小为2KB
        block_size = 2048  # 固定2KB块大小

        total_bytes += block_size
        effective_bytes += effective_data
        metadata_bytes += metadata

    data_utilization = effective_bytes / total_bytes if total_bytes > 0 else 0
    metadata_ratio = metadata_bytes / total_bytes if total_bytes > 0 else 0
    total_utilization = (
        (effective_bytes + metadata_bytes) / total_bytes if total_bytes > 0 else 0
    )

    print(f"\n=== 带宽利用率分析 ===")
    print(f"有效数据 (values+indices): {effective_bytes} 字节 ({data_utilization:.2%})")
    print(f"元数据 (row_ptr+start_row): {metadata_bytes} 字节 ({metadata_ratio:.2%})")
    print(f"总体利用率: {total_utilization:.2%}")
    print(f"总传输字节: {total_bytes} 字节 ({total_bytes/1024:.2f} KB)")

    # 测试矩阵重建
    if blocks:
        print("\n=== 矩阵重建测试 ===")
        start_time = time.time()

        # 创建空矩阵进行重建
        reconstructed_full = np.zeros((csr.shape[0], csr.shape[1]))

        for block in blocks:
            # 提取数据
            values = block["values"]
            col_indices = block["col_indices"]
            row_ptr = block["row_ptr"]
            row_start = block["row_start_index"]

            # 计算行数
            num_rows = len(row_ptr) - 1

            # 创建子矩阵
            sub_matrix = csr_matrix(
                (values, col_indices, row_ptr), shape=(num_rows, csr.shape[1])
            )

            # 重建到完整矩阵对应位置
            reconstructed_full[row_start : row_start + num_rows] += sub_matrix.toarray()

        reconstruction_time = time.time() - start_time
        print(f"重建时间: {reconstruction_time:.6f} 秒")

        # 验证重建是否正确
        original_matrix = csr.toarray()
        are_equal = np.array_equal(original_matrix, reconstructed_full)
        print(f"重建验证结果: {'成功' if are_equal else '失败'}")

        # 如果是小矩阵，打印原始和重建的矩阵进行比较
        if matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
            print("\n原始矩阵:")
            print(original_matrix)
            print("\n重建矩阵:")
            print(reconstructed_full)

            if not are_equal:
                print("\n差异矩阵:")
                print(original_matrix - reconstructed_full)

    return blocks


def run_simple_blocks_tests():
    """运行多种矩阵大小和稀疏度的简单分块测试"""
    print("===== 测试简单分块CSR存储 =====")

    # 测试1: 简单示例 - 与原始例子相同
    print("\n测试1: 简单示例矩阵")
    simple_matrix = np.array([[1, 1, 1, 0], [0, 0, 0, 0], [0, 1, 1, 0]])

    blocks = store_csr_in_simple_blocks(csr_matrix(simple_matrix))
    
    print(blocks)

    # blocks = test_csr_simple_blocks(
    #    simple_matrix,
    #    elements_per_block=2,
    #    print_blocks=5,
    #    print_elements=5,
    #    debug=True,
    # )

    ## 测试2: 无空行矩阵
    # print("\n测试2: 无空行矩阵")
    # no_empty_rows = np.array([[1, 0, 2, 0], [0, 3, 0, 4], [5, 0, 6, 0]])
    # test_csr_simple_blocks(no_empty_rows, elements_per_block=3, debug=True)

    # 测试3: 中等大小矩阵
    # print("\n测试3: 中等大小矩阵 (5% 非零元素)")
    # np.random.seed(42)  # 设置随机种子以便结果可复现
    # medium_matrix = np.random.choice([0, 1], size=(4096, 4096), p=[0.9, 0.1])
    # test_csr_simple_blocks(medium_matrix, elements_per_block=256)

    ## 测试4: 不同分块大小对比
    # print("\n测试4: 不同分块大小对比")
    # matrix_for_sizes = np.random.choice([0, 1], size=(30, 30), p=[0.9, 0.1])
#
# print("\n分块大小: 5 元素/块")
# test_csr_simple_blocks(matrix_for_sizes, elements_per_block=5, print_blocks=2)
#
# print("\n分块大小: 10 元素/块")
# test_csr_simple_blocks(matrix_for_sizes, elements_per_block=10, print_blocks=2)
#
# print("\n分块大小: 20 元素/块")
# test_csr_simple_blocks(matrix_for_sizes, elements_per_block=20, print_blocks=2)
#
## 测试5: 一行密集，其余稀疏
# print("\n测试5: 行密度差异大的矩阵")
# uneven_matrix = np.zeros((10, 20))
## 一行非常密集
# uneven_matrix[5, :] = np.random.choice([0, 1, 2], size=20, p=[0.2, 0.4, 0.4])
## 其余行稀疏
# for i in range(10):
#    if i != 5:
#        uneven_matrix[i, :] = np.random.choice([0, 1], size=20, p=[0.9, 0.1])
#
# test_csr_simple_blocks(
#    uneven_matrix, elements_per_block=8, print_blocks=4, debug=True
# )


if __name__ == "__main__":
    run_simple_blocks_tests()
