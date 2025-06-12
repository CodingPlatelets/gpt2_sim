import matplotlib.pyplot as plt
import numpy as np

# gpt2数据
sparse = [0, 10, 20, 30, 40, 50, 60, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
          81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]

xwwx_ppl = [21.1049, 21.1124, 21.1396, 21.2209, 21.3887, 21.7158, 22.3004, 23.4604,
            23.6035, 23.7968, 24.003, 24.2277, 24.4489, 24.711, 25.0103, 25.3251,
            25.6764, 26.086, 26.5018, 27.0274, 27.6074, 28.3403, 29.1539, 30.0279,
            31.0182, 32.2939, 33.7805, 35.5199, 37.8724, 40.5717, 44.1183, 48.7415,
            54.845, 63.3303, 76.1484, 97.734]

ori_ppl = [21.1049, 21.1115, 21.1402, 21.2124, 21.3866, 21.7158, 22.2983, 23.4494,
           23.6184, 23.7997, 24.0031, 24.2093, 24.4309, 24.7013, 25.0173, 25.3195,
           25.675, 26.0573, 26.5121, 27.0414, 27.6197, 28.318, 29.1315, 30.0422,
           31.0796, 32.2801, 33.7333, 35.5406, 37.8273, 40.6079, 44.1082, 48.7387,
           54.7316, 63.2537, 76.1988, 97.8877]

xwwx_sxv_ppl = [21.0896, 21.1004, 21.1346, 21.2065, 21.3754, 21.6967, 22.2696, 23.4319,
                23.5757, 23.7577, 23.9674, 24.183, 24.3822, 24.6516, 24.9322, 25.2557,
                25.5764, 26.019, 26.4457, 26.9553, 27.5419, 28.1934, 29.0296, 29.9248,
                30.9113, 32.1361, 33.5666, 35.3009, 37.5547, 40.2113, 43.6479, 48.2496,
                54.2682, 62.5635, 75.0698, 96.4918]

ori_sxv_ppl = [21.0896, 21.1038, 21.1315, 21.2058, 21.3721, 21.696, 22.2772, 23.4118,
               23.5677, 23.7635, 23.953, 24.1753, 24.4063, 24.663, 24.9539, 25.2306,
               25.5931, 25.9891, 26.435, 26.9551, 27.5481, 28.204, 28.9848, 29.8861,
               30.9103, 32.113, 33.6005, 35.3375, 37.5591, 40.2378, 43.6575, 48.2136,
               54.1999, 62.6429, 75.1008, 96.5439]

# gpt2-medium数据
medium_sparse = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80, 81, 82, 83, 84, 85, 86, 87, 88,
                 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

medium_xwwx_ppl = [15.9429, 15.9485, 15.9673, 16.0166, 16.1017, 16.2648, 16.5972, 17.2612, 
                   17.8452, 18.8045, 19.0297, 19.3119, 19.7037, 20.0496, 20.5045, 20.996, 
                   21.6096, 22.3405, 23.1668, 24.242, 25.5651, 27.2532, 29.516, 32.706, 
                   37.2048, 44.3858, 56.0831, 76.6245, 112.873]

medium_ori_ppl = [15.9551, 15.964, 15.9827, 16.0311, 16.1154, 16.2905, 16.6353, 17.3258, 
                  17.9318, 18.94, 19.2301, 19.549, 19.9113, 20.3135, 20.7979, 21.3594, 
                  21.9969, 22.7935, 23.7559, 24.9474, 26.4516, 28.4233, 30.9585, 34.5343, 
                  39.9204, 48.4031, 61.5321, 83.4686, 121.8298]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(sparse, xwwx_ppl, 'b-o', label='xwwx-ppl', markersize=4)
plt.plot(sparse, ori_ppl, 'r-s', label='ori-ppl', markersize=4)

# 添加标题和标签
plt.title('gpt2', fontsize=14)
plt.xlabel('sparse(%)', fontsize=12)
plt.ylabel('PPL', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 聚焦显示变化较大的区间
plt.xticks(np.arange(0, 101, 10))

# 保存图片但不显示
plt.tight_layout()
plt.savefig('ppl_comparison.png', dpi=300)
# plt.show()

# 再创建一个图表，这次关注gpt2稀疏度较高区间（70-98%）
plt.figure(figsize=(12, 7))  # 略微增大图表尺寸以容纳数据标签
plt.plot(sparse[7:], xwwx_ppl[7:], 'b-o', label='xwwx-ppl', markersize=5)
plt.plot(sparse[7:], ori_ppl[7:], 'r-s', label='ori-ppl', markersize=5)

# 添加数据标签
for i in range(len(sparse[7:])):
    # xwwx_ppl的标签（微微向上偏移）
    plt.annotate(f'{xwwx_ppl[i+7]:.2f}', 
                 (sparse[i+7], xwwx_ppl[i+7]),
                 textcoords="offset points", 
                 xytext=(0,7), 
                 ha='center',
                 fontsize=8)
    
    # ori_ppl的标签（微微向下偏移）
    plt.annotate(f'{ori_ppl[i+7]:.2f}', 
                 (sparse[i+7], ori_ppl[i+7]),
                 textcoords="offset points", 
                 xytext=(0,-14), 
                 ha='center',
                 fontsize=8)

# 添加标题和标签
plt.title('gpt2 (70%-98%)', fontsize=14)
plt.xlabel('sparse(%)', fontsize=12)
plt.ylabel('PPL', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 保存图片但不显示
plt.tight_layout()
plt.savefig('ppl_comparison_high_sparsity.png', dpi=300)
# plt.show()

# ================ 处理gpt2-medium数据 ================

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(medium_sparse, medium_xwwx_ppl, 'b-o', label='xwwx-ppl', markersize=4)
plt.plot(medium_sparse, medium_ori_ppl, 'r-s', label='ori-ppl', markersize=4)

# 添加标题和标签
plt.title('gpt2-medium (1024)', fontsize=14)
plt.xlabel('sparse(%)', fontsize=12)
plt.ylabel('PPL', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 聚焦显示变化较大的区间
plt.xticks(np.arange(0, 101, 10))

# 保存图片但不显示
plt.tight_layout()
plt.savefig('medium_ppl_comparison.png', dpi=300)
# plt.show()

# 再创建一个图表，这次关注gpt2-medium稀疏度较高区间（70-99%）
plt.figure(figsize=(12, 7))  # 略微增大图表尺寸以容纳数据标签
high_start_index = 7  # 索引7开始是70%
plt.plot(medium_sparse[high_start_index:], medium_xwwx_ppl[high_start_index:], 'b-o', label='xwwx-ppl', markersize=5)
plt.plot(medium_sparse[high_start_index:], medium_ori_ppl[high_start_index:], 'r-s', label='ori-ppl', markersize=5)

# 添加数据标签
for i in range(len(medium_sparse[high_start_index:])):
    # xwwx_ppl的标签（微微向上偏移）
    plt.annotate(f'{medium_xwwx_ppl[i+high_start_index]:.2f}', 
                 (medium_sparse[i+high_start_index], medium_xwwx_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(0,7), 
                 ha='center',
                 fontsize=8)
    
    # ori_ppl的标签（微微向下偏移）
    plt.annotate(f'{medium_ori_ppl[i+high_start_index]:.2f}', 
                 (medium_sparse[i+high_start_index], medium_ori_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(0,-14), 
                 ha='center',
                 fontsize=8)

# 添加标题和标签
plt.title('gpt2-medium (70%-99%)', fontsize=14)
plt.xlabel('sparse(%)', fontsize=12)
plt.ylabel('PPL', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 保存图片但不显示
plt.tight_layout()
plt.savefig('medium_ppl_comparison_high_sparsity.png', dpi=300)
# plt.show()

# ================ 创建包含全部四组数据的图表 ================

# 完整稀疏度范围图表
plt.figure(figsize=(12, 8))

plt.plot(sparse, xwwx_ppl, 'b-o', label='xwwx-ppl', markersize=4)
plt.plot(sparse, ori_ppl, 'r-s', label='ori-ppl', markersize=4)
plt.plot(sparse, xwwx_sxv_ppl, 'g-^', label='xwwx-sxv-ppl', markersize=4)
plt.plot(sparse, ori_sxv_ppl, 'm-d', label='ori-sxv-ppl', markersize=4)

# 添加标题和标签
plt.title('GPT2 all', fontsize=16)
plt.xlabel('sparse(%)', fontsize=14)
plt.ylabel('PPL', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 设置x轴刻度
plt.xticks(np.arange(0, 101, 10))

# 保存图片
plt.tight_layout()
plt.savefig('all_methods_comparison.png', dpi=300)
# plt.show()

# 高稀疏度范围图表 (70%-98%)
plt.figure(figsize=(14, 10))

high_start_index = 7  # 索引7开始是70%
plt.plot(sparse[high_start_index:], xwwx_ppl[high_start_index:], 'b-o', label='xwwx-ppl', markersize=5)
plt.plot(sparse[high_start_index:], ori_ppl[high_start_index:], 'r-s', label='ori-ppl', markersize=5)
plt.plot(sparse[high_start_index:], xwwx_sxv_ppl[high_start_index:], 'g-^', label='xwwx-sxv-ppl', markersize=5)
plt.plot(sparse[high_start_index:], ori_sxv_ppl[high_start_index:], 'm-d', label='ori-sxv-ppl', markersize=5)

# 添加标题和标签
plt.title('GPT2 (70%-98%) all', fontsize=16)
plt.xlabel('sparse(%)', fontsize=14)
plt.ylabel('PPL', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 为了避免图表过于拥挤，只对一组数据添加标签
for i in range(len(sparse[high_start_index:])):
    y_offset = 0
    x_pos = sparse[i+high_start_index]
    
    # 找到该x位置四个值的最大值和最小值
    all_y_values = [
        xwwx_ppl[i+high_start_index], 
        ori_ppl[i+high_start_index], 
        xwwx_sxv_ppl[i+high_start_index], 
        ori_sxv_ppl[i+high_start_index]
    ]
    min_y = min(all_y_values)
    max_y = max(all_y_values)
    
    # 计算y轴的比例，用于放置标签
    y_range = plt.gca().get_ylim()[1] - plt.gca().get_ylim()[0]
    y_pos_factor = 0.02  # 标签位置因子
    
    # 为每个点添加数值标签，位置略有不同以避免重叠
    plt.annotate(f'{xwwx_ppl[i+high_start_index]:.2f}', 
                 (x_pos, xwwx_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(-20, 10), 
                 ha='center',
                 fontsize=7,
                 color='blue')
    
    plt.annotate(f'{ori_ppl[i+high_start_index]:.2f}', 
                 (x_pos, ori_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center',
                 fontsize=7,
                 color='red')
    
    plt.annotate(f'{xwwx_sxv_ppl[i+high_start_index]:.2f}', 
                 (x_pos, xwwx_sxv_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(-20, -15), 
                 ha='center',
                 fontsize=7,
                 color='green')
    
    plt.annotate(f'{ori_sxv_ppl[i+high_start_index]:.2f}', 
                 (x_pos, ori_sxv_ppl[i+high_start_index]),
                 textcoords="offset points", 
                 xytext=(0, -15), 
                 ha='center',
                 fontsize=7,
                 color='magenta')

# 保存图片
plt.tight_layout()
plt.savefig('all_methods_high_sparsity_comparison.png', dpi=300)
# plt.show()