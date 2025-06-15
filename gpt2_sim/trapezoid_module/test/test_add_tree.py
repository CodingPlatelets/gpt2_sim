import numpy as np
from ..module import AddTree
from ..utils import bf16_to_float, bf16_add

def test_AddTree():
    print("\n===== 测试 AddTree =====")

    c_values = [0] * 5
    M = 1
    N = 5

    add_tree = AddTree(PE_num=4, c_values=c_values, M=M, N=N)

    # input_maps_queue = [
    #    [
    #        {0: [1.0]},
    #        {2: [1.0]},
    #        {2: [1.0]},
    #        {3: [1.0]},
    #        {-1: [0.0]},
    #        {-1: [0.0]},
    #        {-1: [0.0]},
    #        {-1: [0.0]},
    #
    #    ]
    # ]

    input_maps_queue = [
        [
            {0: [1.0]},
            {1: [1.0]},
            {2: [1.0]},
            {3: [1.0]},
        ],
        [
            {3: [1.0]},
            {3: [1.0]},
            {4: [1.0]},
            {-1: [0.0]},
        ],
    ]

    print("\n使用BF16模式运行加法树...")
    results = add_tree.run_pipeline_with_bf16(
        input_maps_queue, max_cycles=40, print_states=True
    )

    # 打印输出结果
    print("\n加法树输出结果:")
    for i, res in enumerate(results):
        if res["valid"]:
            # 将BF16输出转换为浮点数
            float_outputs = {}
            if res["output"]:
                for out_map in res["output"]:
                    for idx, vals in out_map.items():
                        if idx not in float_outputs:
                            float_outputs[idx] = []
                        float_outputs[idx].extend([bf16_to_float(v) for v in vals])

            print(f"周期 {i+1}: {float_outputs}")

    # output加至C
    for res in results:
        if res["valid"]:
            if res["output"][0]:
                index = next(iter(res["output"][0]))
                value = res["output"][0][index][0]
                m = index % M
                n = int(index / M)
                if index != -1:
                    # c_values[m * N + n] += value
                    c_values[m * N + n] = bf16_add(c_values[m * N + n], value)

    # 打印驱逐到C矩阵的值
    print("\n结果矩阵C:")
    print(c_values)
    float_c_values = [bf16_to_float(v) for v in c_values]
    c_matrix = np.array(float_c_values).reshape(M, N)
    print(c_matrix)

test_AddTree()