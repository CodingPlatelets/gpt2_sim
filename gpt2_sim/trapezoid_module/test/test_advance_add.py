from ..utils import bf16_to_float
from ..module import AdvanceAddUnit

def AdvanceAdd_output_to_float(output: dict):
    new_output = {}
    for key, values in output.items():
        temp = [bf16_to_float(v) for v in values]
        new_output[key] = temp
    return new_output


def test_AdvanceAdd():
    # 示例用法
    adder = AdvanceAddUnit(c_values=[0] * 4, M=2, N=2)

    # 准备输入数据
    input_pairs = [
        #    ({0: [5]}, {0: [3]}),           # 周期1: 合并得到 {1: [5, 3]}
        #    ({2: [1]}, {1: [1]}),           # 周期2: {2: [4]}, {3: [7]}
        ({2: [2]}, {1: [3]}),
        ({2: [1]}, {-1: [0]}),
        #    ({}, {})                        # 周期3: 空输入
    ]

    evict_indices = [
        #   [],                           # 周期1: 驱逐索引1
        [],  # 周期2: 驱逐索引2
        [],  # 周期3: 无驱逐
        #    [],
        #    [],
    ]

    # 运行流水线
    results = adder.run_pipeline_with_bf16(
        input_pairs, evict_indices, print_states=True
    )

    # 检查结果
    for i, res in enumerate(results):
        if res["valid"]:
            print(f"周期 {i+1} 输出: {AdvanceAdd_output_to_float(res['output'])}")
    c_values_float = [bf16_to_float(c) for c in adder.c_values]
    print(c_values_float)

test_AdvanceAdd()