import numpy as np
from scipy.sparse import csr_matrix
import time


def store_csr_in_blocks(
    csr_matrix,
    values_per_block=256,
    col_indices_per_block=256,
    row_pointers_per_block=256,
):
    """将CSR矩阵分块存储在2KB块中"""
    # 提取CSR数据
    values = csr_matrix.data  # 16位值
    col_indices = csr_matrix.indices  # 16位列索引
    row_pointers = csr_matrix.indptr  # 32位行指针

    # 计算需要的块数量
    total_values = len(values)
    total_col_indices = len(col_indices)
    total_row_pointers = len(row_pointers)

    blocks_for_values = (total_values + values_per_block - 1) // values_per_block
    blocks_for_indices = (
        total_col_indices + col_indices_per_block - 1
    ) // col_indices_per_block
    blocks_for_pointers = (
        total_row_pointers + row_pointers_per_block - 1
    ) // row_pointers_per_block

    total_blocks = max(blocks_for_values, blocks_for_indices, blocks_for_pointers)

    print(f"总块数: {total_blocks}")
    print(f"Values需要块数: {blocks_for_values}")
    print(f"Indices需要块数: {blocks_for_indices}")
    print(f"Row Pointers需要块数: {blocks_for_pointers}")

    # 创建块数组
    blocks = []

    # 填充每个块
    for block_idx in range(total_blocks):
        # 确定要复制的数据范围
        val_start = block_idx * values_per_block
        val_end = min(val_start + values_per_block, total_values)

        col_start = block_idx * col_indices_per_block
        col_end = min(col_start + col_indices_per_block, total_col_indices)

        row_start = block_idx * row_pointers_per_block
        row_end = min(row_start + row_pointers_per_block, total_row_pointers)

        # 创建块
        block = {
            "values": (
                values[val_start:val_end].tolist() if val_start < total_values else []
            ),
            "col_indices": (
                col_indices[col_start:col_end].tolist()
                if col_start < total_col_indices
                else []
            ),
            "row_pointers": (
                row_pointers[row_start:row_end].tolist()
                if row_start < total_row_pointers
                else []
            ),
        }

        # 填充到固定大小 (在真实场景中可能使用特殊标记)
        if len(block["values"]) < values_per_block:
            block["values"].extend([None] * (values_per_block - len(block["values"])))

        if len(block["col_indices"]) < col_indices_per_block:
            block["col_indices"].extend(
                [None] * (col_indices_per_block - len(block["col_indices"]))
            )

        if len(block["row_pointers"]) < row_pointers_per_block:
            block["row_pointers"].extend(
                [None] * (row_pointers_per_block - len(block["row_pointers"]))
            )

        blocks.append(block)

    return blocks


def reconstruct_csr_from_blocks(
    blocks, num_rows, num_cols, values_per_block=256, col_indices_per_block=256
):
    """从块重建CSR矩阵"""
    # 合并数据
    all_values = []
    all_col_indices = []
    all_row_pointers = []

    for block in blocks:
        # 提取非填充数据
        real_values = [v for v in block["values"] if v is not None]
        real_indices = [i for i in block["col_indices"] if i is not None]
        real_pointers = [p for p in block["row_pointers"] if p is not None]

        all_values.extend(real_values)
        all_col_indices.extend(real_indices)
        all_row_pointers.extend(real_pointers)

    # 确保行指针正确完整
    if len(all_row_pointers) < num_rows + 1:
        print("错误: 重建的行指针数量不足!")

    # 重建CSR矩阵
    reconstructed_csr = csr_matrix(
        (all_values, all_col_indices, all_row_pointers), shape=(num_rows, num_cols)
    )

    return reconstructed_csr


def analyze_block_sizes(blocks, display_all=False):
    """分析每个块中row_pointers、values和column indices的大小关系"""
    total_blocks = len(blocks)
    blocks_with_smaller_pointers = 0
    blocks_with_equal_pointers = 0
    blocks_with_larger_pointers = 0

    # 详细记录不同类型的块
    smaller_blocks = []
    equal_blocks = []
    larger_blocks = []

    for i, block in enumerate(blocks):
        # 计算每个数组的实际数据大小(不包括None填充)
        values_size = sum(1 for v in block["values"] if v is not None)
        col_indices_size = sum(1 for c in block["col_indices"] if c is not None)
        row_pointers_size = sum(1 for r in block["row_pointers"] if r is not None)

        # 判断大小关系
        if row_pointers_size < min(values_size, col_indices_size):
            blocks_with_smaller_pointers += 1
            smaller_blocks.append((i, row_pointers_size, values_size, col_indices_size))
        elif row_pointers_size == values_size and row_pointers_size == col_indices_size:
            blocks_with_equal_pointers += 1
            equal_blocks.append((i, row_pointers_size, values_size, col_indices_size))
        else:
            blocks_with_larger_pointers += 1
            larger_blocks.append((i, row_pointers_size, values_size, col_indices_size))

    # 打印分析结果
    print("\n=== 行指针大小分析 ===")
    print(f"总块数: {total_blocks}")
    print(
        f"行指针小于values和column indices的块数: {blocks_with_smaller_pointers} ({blocks_with_smaller_pointers/total_blocks*100:.1f}%)"
    )
    print(
        f"行指针等于values和column indices的块数: {blocks_with_equal_pointers} ({blocks_with_equal_pointers/total_blocks*100:.1f}%)"
    )
    print(
        f"行指针大于values或column indices的块数: {blocks_with_larger_pointers} ({blocks_with_larger_pointers/total_blocks*100:.1f}%)"
    )

    # 显示详细信息（可选）
    if display_all:
        if smaller_blocks:
            print(
                "\n行指针较小的块详细信息 (块索引, 行指针大小, values大小, column indices大小):"
            )
            for info in smaller_blocks:
                print(
                    f"  块 {info[0]}: 行指针={info[1]}, Values={info[2]}, Column indices={info[3]}"
                )

        if equal_blocks:
            print("\n行指针相等的块详细信息:")
            for info in equal_blocks:
                print(
                    f"  块 {info[0]}: 行指针={info[1]}, Values={info[2]}, Column indices={info[3]}"
                )

        if larger_blocks:
            print("\n行指针较大的块详细信息:")
            for info in larger_blocks:
                print(
                    f"  块 {info[0]}: 行指针={info[1]}, Values={info[2]}, Column indices={info[3]}"
                )
    else:
        # 只显示部分样例
        if smaller_blocks:
            print("\n行指针较小的块样例:")
            for info in smaller_blocks[:3]:
                print(
                    f"  块 {info[0]}: 行指针={info[1]}, Values={info[2]}, Column indices={info[3]}"
                )
            if len(smaller_blocks) > 3:
                print(f"  ... 以及其他 {len(smaller_blocks)-3} 个块")

        if larger_blocks:
            print("\n行指针较大的块样例:")
            for info in larger_blocks[:3]:
                print(
                    f"  块 {info[0]}: 行指针={info[1]}, Values={info[2]}, Column indices={info[3]}"
                )
            if len(larger_blocks) > 3:
                print(f"  ... 以及其他 {len(larger_blocks)-3} 个块")

    # 计算平均大小
    avg_values = (
        sum(info[2] for info in smaller_blocks + equal_blocks + larger_blocks)
        / total_blocks
    )
    avg_col_indices = (
        sum(info[3] for info in smaller_blocks + equal_blocks + larger_blocks)
        / total_blocks
    )
    avg_row_pointers = (
        sum(info[1] for info in smaller_blocks + equal_blocks + larger_blocks)
        / total_blocks
    )

    print(f"\n平均大小比较:")
    print(f"  平均values大小: {avg_values:.1f}")
    print(f"  平均column indices大小: {avg_col_indices:.1f}")
    print(f"  平均row pointers大小: {avg_row_pointers:.1f}")

    # 返回分析结果以供进一步使用
    return {
        "total_blocks": total_blocks,
        "blocks_with_smaller_pointers": blocks_with_smaller_pointers,
        "blocks_with_equal_pointers": blocks_with_equal_pointers,
        "blocks_with_larger_pointers": blocks_with_larger_pointers,
        "smaller_blocks": smaller_blocks,
        "equal_blocks": equal_blocks,
        "larger_blocks": larger_blocks,
    }


def calculate_block_size_bytes(
    values_per_block=256, col_indices_per_block=256, row_pointers_per_block=256
):
    """计算块的总大小(字节)"""
    values_size = values_per_block * 2  # 16位 = 2字节
    col_indices_size = col_indices_per_block * 2  # 16位 = 2字节
    row_pointers_size = row_pointers_per_block * 4  # 32位 = 4字节

    total_size = values_size + col_indices_size + row_pointers_size
    return total_size


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


def analyze_bandwidth_utilization(blocks, block_size_bytes=2048):
    """
    分析CSR分块存储的带宽利用率

    Args:
        blocks: CSR分块列表
        block_size_bytes: 每个数据块的大小，默认为2KB (2048字节)

    Returns:
        dict: 带宽利用率分析结果
    """
    total_blocks = len(blocks)
    total_effective_bytes = 0

    # 为每种数据类型单独计算有效字节数
    total_values_bytes = 0
    total_col_indices_bytes = 0
    total_row_pointers_bytes = 0

    # 为每种数据类型单独计算理论最大字节数
    max_values_bytes = 0
    max_col_indices_bytes = 0
    max_row_pointers_bytes = 0

    total_possible_bytes = total_blocks * block_size_bytes
    block_utilization = []

    for i, block in enumerate(blocks):
        # 计算每个数组的实际数据大小(不包括None填充)
        values_count = sum(1 for v in block["values"] if v is not None)
        col_indices_count = sum(1 for c in block["col_indices"] if c is not None)
        row_pointers_count = sum(1 for r in block["row_pointers"] if r is not None)

        # 计算各种数据类型的最大可能字节数
        max_values = len(block["values"]) * 2  # 16位 = 2字节
        max_col_indices = len(block["col_indices"]) * 2  # 16位 = 2字节
        max_row_pointers = len(block["row_pointers"]) * 4  # 32位 = 4字节

        # 累加最大可能字节数
        max_values_bytes += max_values
        max_col_indices_bytes += max_col_indices
        max_row_pointers_bytes += max_row_pointers

        # 计算有效字节数
        values_bytes = values_count * 2
        col_indices_bytes = col_indices_count * 2
        row_pointers_bytes = row_pointers_count * 4

        # 累加有效字节数
        total_values_bytes += values_bytes
        total_col_indices_bytes += col_indices_bytes
        total_row_pointers_bytes += row_pointers_bytes

        # 总有效字节数
        effective_bytes = values_bytes + col_indices_bytes + row_pointers_bytes
        total_effective_bytes += effective_bytes

        # 计算该块的利用率
        utilization = effective_bytes / block_size_bytes

        # 计算每种数据类型的块利用率
        values_utilization = values_bytes / max_values if max_values > 0 else 0
        col_indices_utilization = (
            col_indices_bytes / max_col_indices if max_col_indices > 0 else 0
        )
        row_pointers_utilization = (
            row_pointers_bytes / max_row_pointers if max_row_pointers > 0 else 0
        )

        block_utilization.append(
            (
                i,
                effective_bytes,
                block_size_bytes,
                utilization,
                values_bytes,
                col_indices_bytes,
                row_pointers_bytes,
                values_utilization,
                col_indices_utilization,
                row_pointers_utilization,
            )
        )

    # 计算总体利用率
    overall_utilization = total_effective_bytes / total_possible_bytes

    # 计算每种数据类型的总体利用率
    values_utilization = (
        total_values_bytes / max_values_bytes if max_values_bytes > 0 else 0
    )
    col_indices_utilization = (
        total_col_indices_bytes / max_col_indices_bytes
        if max_col_indices_bytes > 0
        else 0
    )
    row_pointers_utilization = (
        total_row_pointers_bytes / max_row_pointers_bytes
        if max_row_pointers_bytes > 0
        else 0
    )

    # 对块按总利用率排序
    sorted_blocks = sorted(block_utilization, key=lambda x: x[3])
    lowest_util_blocks = sorted_blocks[:3]
    highest_util_blocks = sorted_blocks[-3:]

    # 打印分析结果
    print("\n=== 带宽利用率分析 ===")
    print(f"总块数: {total_blocks}")
    print(f"每块大小: {block_size_bytes} 字节 ({block_size_bytes/1024:.2f} KB)")
    print(
        f"总传输字节数: {total_possible_bytes} 字节 ({total_possible_bytes/1024:.2f} KB)"
    )
    print(
        f"有效数据字节数: {total_effective_bytes} 字节 ({total_effective_bytes/1024:.2f} KB)"
    )
    print(f"总体带宽利用率: {overall_utilization:.2%}")

    # 打印每种数据类型的带宽利用率
    print("\n=== 各数据类型带宽利用率 ===")
    print(
        f"Values 带宽利用率: {values_utilization:.2%} ({total_values_bytes}/{max_values_bytes} 字节)"
    )
    print(
        f"Column Indices 带宽利用率: {col_indices_utilization:.2%} ({total_col_indices_bytes}/{max_col_indices_bytes} 字节)"
    )
    print(
        f"Row Pointers 带宽利用率: {row_pointers_utilization:.2%} ({total_row_pointers_bytes}/{max_row_pointers_bytes} 字节)"
    )

    print("\n利用率最低的三个块:")
    for info in lowest_util_blocks:
        print(f"  块 {info[0]}: {info[1]}/{info[2]} 字节, 利用率 {info[3]:.2%}")
        print(f"    Values: {info[4]} 字节, 利用率 {info[7]:.2%}")
        print(f"    Column Indices: {info[5]} 字节, 利用率 {info[8]:.2%}")
        print(f"    Row Pointers: {info[6]} 字节, 利用率 {info[9]:.2%}")

    print("\n利用率最高的三个块:")
    for info in highest_util_blocks:
        print(f"  块 {info[0]}: {info[1]}/{info[2]} 字节, 利用率 {info[3]:.2%}")
        print(f"    Values: {info[4]} 字节, 利用率 {info[7]:.2%}")
        print(f"    Column Indices: {info[5]} 字节, 利用率 {info[8]:.2%}")
        print(f"    Row Pointers: {info[6]} 字节, 利用率 {info[9]:.2%}")


    return {
        "total_blocks": total_blocks,
        "block_size_bytes": block_size_bytes,
        "total_possible_bytes": total_possible_bytes,
        "total_effective_bytes": total_effective_bytes,
        "overall_utilization": overall_utilization,
        "values_bytes": total_values_bytes,
        "col_indices_bytes": total_col_indices_bytes,
        "row_pointers_bytes": total_row_pointers_bytes,
        "values_utilization": values_utilization,
        "col_indices_utilization": col_indices_utilization,
        "row_pointers_utilization": row_pointers_utilization,
        "block_utilization": block_utilization,
        "lowest_util_blocks": lowest_util_blocks,
        "highest_util_blocks": highest_util_blocks,
    }


def test_csr_blocking(
    matrix, values_per_block=256, col_indices_per_block=256, row_pointers_per_block=256
):
    """测试CSR矩阵分块存储和重建"""
    print(f"\n测试矩阵形状: {matrix.shape}")

    # 转换为CSR格式
    csr = csr_matrix(matrix)

    print(f"非零元素: {len(csr.data)}")
    print(
        f"Values: {csr.data[:10]}..." if len(csr.data) > 10 else f"Values: {csr.data}"
    )
    print(
        f"Column indices: {csr.indices[:10]}..."
        if len(csr.indices) > 10
        else f"Column indices: {csr.indices}"
    )
    print(
        f"Row pointers: {csr.indptr[:10]}..."
        if len(csr.indptr) > 10
        else f"Row pointers: {csr.indptr}"
    )

    block_size = calculate_block_size_bytes(
        values_per_block, col_indices_per_block, row_pointers_per_block
    )
    print(f"块大小: {block_size} 字节 ({block_size/1024:.2f} KB)")

    # 分块存储
    start_time = time.time()
    blocks = store_csr_in_blocks(
        csr, values_per_block, col_indices_per_block, row_pointers_per_block
    )
    blocking_time = time.time() - start_time
    print(f"分块时间: {blocking_time:.6f} 秒")

    # 打印第一个块的内容
    if blocks:
        print("\n第一个块的内容:")
        print(
            f"Values: {blocks[0]['values'][:5]}..."
            if blocks[0]["values"]
            else "Values: []"
        )
        print(
            f"Column indices: {blocks[0]['col_indices'][:5]}..."
            if blocks[0]["col_indices"]
            else "Column indices: []"
        )
        print(
            f"Row pointers: {blocks[0]['row_pointers'][:5]}..."
            if blocks[0]["row_pointers"]
            else "Row pointers: []"
        )
    block_analysis = analyze_block_sizes(blocks)
    bandwidth_analysis = analyze_bandwidth_utilization(blocks, block_size)

    # 重建CSR
    start_time = time.time()
    reconstructed_csr = reconstruct_csr_from_blocks(
        blocks, csr.shape[0], csr.shape[1], values_per_block, col_indices_per_block
    )
    reconstruction_time = time.time() - start_time
    print(f"重建时间: {reconstruction_time:.6f} 秒")

    # 验证
    is_valid = verify_csr_blocks(csr, reconstructed_csr)
    print(f"\n验证结果: {'成功' if is_valid else '失败'}")

    return is_valid, blocks


# 测试用例
def run_tests():
    print("===== 测试CSR矩阵分块存储 =====")

    # 测试1: 小矩阵
    print("\n测试1: 小矩阵")
    small_matrix = np.array([[1, 1, 1, 0], [0, 1, 1, 0]])
    test_csr_blocking(small_matrix)

    # 测试2: 中等大小矩阵
    print("\n测试2: 中等大小矩阵")
    medium_matrix = np.random.choice([0, 1], size=(100, 100), p=[0.95, 0.05])
    test_csr_blocking(medium_matrix)

    # 测试3: 大矩阵(在计算资源允许的情况下)
    print("\n测试3: 较大矩阵")
    large_matrix = np.random.choice([0, 1], size=(4096, 4096), p=[0.90, 0.10])
    test_csr_blocking(large_matrix)

    # 测试4: 自定义块大小
    print("\n测试4: 自定义块大小 (128-128-64)")
    test_csr_blocking(medium_matrix, 128, 128, 64)


# 运行测试
run_tests()
