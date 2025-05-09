from transformers import AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_model_from_checkpoint(checkpoint_path):
    # 获取最新的检查点路径
    last_checkpoint = get_last_checkpoint(checkpoint_path)
    print(f"找到最新检查点：{last_checkpoint}")

    # 加载检查点参数到模型
    if last_checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(last_checkpoint)
        return model
    else:
        raise ValueError(f"未在 {checkpoint_path} 找到检查点")

# 加载四个模型
print("加载原始GPT-2模型...")
model_raw = AutoModelForCausalLM.from_pretrained("../../gpt2")  # 加载未微调的原始权重
print("加载default模型...")
model_default = load_model_from_checkpoint("./tmp/test-clm")
print("加载ori模型...")
model_ori = load_model_from_checkpoint("./tmp/test-clm-ori")
print("加载xwwx模型...")
model_xwwx = load_model_from_checkpoint("./tmp/test-clm-xwwx")

# 比较模型权重
def compare_weights(model1, model2, model3, model4, model1_name="raw", model2_name="default", model3_name="ori", model4_name="xwwx"):
    print(f"\n===== {model1_name} vs {model2_name} vs {model3_name} vs {model4_name} weight =====")
    
    differences = []
    param_names = []
    
    # 遍历所有参数并比较
    for (name1, param1), (name2, param2), (name3, param3), (name4, param4) in zip(
        model1.named_parameters(), 
        model2.named_parameters(),
        model3.named_parameters(),
        model4.named_parameters()
    ):
        if name1 != name2 or name2 != name3 or name3 != name4:
            print(f"警告：参数名不匹配: {name1} vs {name2} vs {name3} vs {name4}")
            continue
            
        # 计算差异
        diff12 = torch.abs(param1 - param2)
        diff23 = torch.abs(param2 - param3)
        diff34 = torch.abs(param3 - param4)
        diff13 = torch.abs(param1 - param3)
        diff14 = torch.abs(param1 - param4)
        diff24 = torch.abs(param2 - param4)
        
        avg_diff = max(
            torch.mean(diff12).item(),
            torch.mean(diff23).item(),
            torch.mean(diff34).item(),
            torch.mean(diff13).item(),
            torch.mean(diff14).item(),
            torch.mean(diff24).item()
        )
        max_diff = max(
            torch.max(diff12).item(),
            torch.max(diff23).item(),
            torch.max(diff34).item(),
            torch.max(diff13).item(),
            torch.max(diff14).item(),
            torch.max(diff24).item()
        )
        
        # 记录有意义的差异
        if avg_diff > 1e-6:  # 忽略非常小的差异
            differences.append((name1, avg_diff, max_diff))
            param_names.append(name1)
            
    # 排序并打印差异最大的参数
    differences.sort(key=lambda x: x[1], reverse=True)
    
    if differences:
        print(f"\n差异最大的前10个参数:")
        for i, (name, avg_diff, max_diff) in enumerate(differences[:10]):
            print(f"{i+1}. {name}: 平均差异={avg_diff:.6f}, 最大差异={max_diff:.6f}")
    else:
        print("未发现显著差异")
    
    return differences, param_names

# 执行比较
differences, param_names = compare_weights(model_raw, model_default, model_ori, model_xwwx)

# # 可视化一些差异
# def visualize_weight_difference(model1, model2, model3, model4, param_name, model1_name="raw", model2_name="default", model3_name="ori", model4_name="xwwx"):
#     # 获取指定参数
#     param1 = dict(model1.named_parameters())[param_name].detach().cpu().numpy().flatten()
#     param2 = dict(model2.named_parameters())[param_name].detach().cpu().numpy().flatten()
#     param3 = dict(model3.named_parameters())[param_name].detach().cpu().numpy().flatten()
#     param4 = dict(model4.named_parameters())[param_name].detach().cpu().numpy().flatten()
    
#     # 计算差异
#     diff12 = np.abs(param1 - param2)
#     diff23 = np.abs(param2 - param3)
#     diff34 = np.abs(param3 - param4)
#     diff13 = np.abs(param1 - param3)
#     diff14 = np.abs(param1 - param4)
#     diff24 = np.abs(param2 - param4)
    
#     # 创建图表
#     plt.figure(figsize=(15, 5))
    
#     # 绘制权重分布直方图
#     plt.subplot(1, 3, 1)
#     plt.hist(param1, bins=50, alpha=0.25, label=model1_name)
#     plt.hist(param2, bins=50, alpha=0.25, label=model2_name)
#     plt.hist(param3, bins=50, alpha=0.25, label=model3_name)
#     plt.hist(param4, bins=50, alpha=0.25, label=model4_name)
#     plt.legend()
#     plt.title(f"{param_name} weight")
    
#     # 绘制差异直方图
#     plt.subplot(1, 3, 2)
#     plt.hist(diff12, bins=50, alpha=0.25, label=f"{model1_name}-{model2_name}")
#     plt.hist(diff23, bins=50, alpha=0.25, label=f"{model2_name}-{model3_name}")
#     plt.hist(diff34, bins=50, alpha=0.25, label=f"{model3_name}-{model4_name}")
#     plt.hist(diff13, bins=50, alpha=0.25, label=f"{model1_name}-{model3_name}")
#     plt.hist(diff14, bins=50, alpha=0.25, label=f"{model1_name}-{model4_name}")
#     plt.hist(diff24, bins=50, alpha=0.25, label=f"{model2_name}-{model4_name}")
#     plt.legend()
#     plt.title(f"{param_name} diff")
    
#     # 绘制前100个值的对比
#     plt.subplot(1, 3, 3)
#     sample_size = min(100, len(param1))
#     indices = np.arange(sample_size)
#     plt.plot(indices, param1[:sample_size], label=model1_name)
#     plt.plot(indices, param2[:sample_size], label=model2_name)
#     plt.plot(indices, param3[:sample_size], label=model3_name)
#     plt.plot(indices, param4[:sample_size], label=model4_name)
#     plt.legend()
#     plt.title(f"{param_name} {sample_size}values diff")
    
#     plt.tight_layout()
#     plt.savefig(f"{param_name.replace('.', '_')}_comparison.png")
#     print(f"已保存 {param_name} 的比较图表")

# # 如果找到差异，可视化前三个差异最大的参数
# if differences:
#     for name, _, _ in differences[:3]:
#         visualize_weight_difference(model_raw, model_default, model_ori, model_xwwx, name)

# 比较特定层的注意力权重
def compare_attention_weights(model1, model2, model3, model4, layer_idx=0, model1_name="raw", model2_name="default", model3_name="ori", model4_name="xwwx"):
    # 提取第layer_idx层的注意力权重
    attn1_qkv = model1.transformer.h[layer_idx].attn.c_attn.weight
    attn2_qkv = model2.transformer.h[layer_idx].attn.c_attn.weight
    attn3_qkv = model3.transformer.h[layer_idx].attn.c_attn.weight
    attn4_qkv = model4.transformer.h[layer_idx].attn.c_attn.weight
    
    attn1_proj = model1.transformer.h[layer_idx].attn.c_proj.weight
    attn2_proj = model2.transformer.h[layer_idx].attn.c_proj.weight
    attn3_proj = model3.transformer.h[layer_idx].attn.c_proj.weight
    attn4_proj = model4.transformer.h[layer_idx].attn.c_proj.weight
    
    # 计算差异
    qkv_diff12 = torch.abs(attn1_qkv - attn2_qkv)
    qkv_diff23 = torch.abs(attn2_qkv - attn3_qkv)
    qkv_diff34 = torch.abs(attn3_qkv - attn4_qkv)
    qkv_diff13 = torch.abs(attn1_qkv - attn3_qkv)
    qkv_diff14 = torch.abs(attn1_qkv - attn4_qkv)
    qkv_diff24 = torch.abs(attn2_qkv - attn4_qkv)
    
    proj_diff12 = torch.abs(attn1_proj - attn2_proj)
    proj_diff23 = torch.abs(attn2_proj - attn3_proj)
    proj_diff34 = torch.abs(attn3_proj - attn4_proj)
    proj_diff13 = torch.abs(attn1_proj - attn3_proj)
    proj_diff14 = torch.abs(attn1_proj - attn4_proj)
    proj_diff24 = torch.abs(attn2_proj - attn4_proj)
    
    print(f"\n===== 第{layer_idx}层注意力权重比较 =====")
    print(f"Q/K/V投影权重差异:")
    print(f"{model1_name}-{model2_name}: 平均={torch.mean(qkv_diff12).item():.6f}, 最大={torch.max(qkv_diff12).item():.6f}")
    print(f"{model2_name}-{model3_name}: 平均={torch.mean(qkv_diff23).item():.6f}, 最大={torch.max(qkv_diff23).item():.6f}")
    print(f"{model3_name}-{model4_name}: 平均={torch.mean(qkv_diff34).item():.6f}, 最大={torch.max(qkv_diff34).item():.6f}")
    print(f"{model1_name}-{model3_name}: 平均={torch.mean(qkv_diff13).item():.6f}, 最大={torch.max(qkv_diff13).item():.6f}")
    print(f"{model1_name}-{model4_name}: 平均={torch.mean(qkv_diff14).item():.6f}, 最大={torch.max(qkv_diff14).item():.6f}")
    print(f"{model2_name}-{model4_name}: 平均={torch.mean(qkv_diff24).item():.6f}, 最大={torch.max(qkv_diff24).item():.6f}")
    
    print(f"\n输出投影权重差异:")
    print(f"{model1_name}-{model2_name}: 平均={torch.mean(proj_diff12).item():.6f}, 最大={torch.max(proj_diff12).item():.6f}")
    print(f"{model2_name}-{model3_name}: 平均={torch.mean(proj_diff23).item():.6f}, 最大={torch.max(proj_diff23).item():.6f}")
    print(f"{model3_name}-{model4_name}: 平均={torch.mean(proj_diff34).item():.6f}, 最大={torch.max(proj_diff34).item():.6f}")
    print(f"{model1_name}-{model3_name}: 平均={torch.mean(proj_diff13).item():.6f}, 最大={torch.max(proj_diff13).item():.6f}")
    print(f"{model1_name}-{model4_name}: 平均={torch.mean(proj_diff14).item():.6f}, 最大={torch.max(proj_diff14).item():.6f}")
    print(f"{model2_name}-{model4_name}: 平均={torch.mean(proj_diff24).item():.6f}, 最大={torch.max(proj_diff24).item():.6f}")
    
    # 打印部分具体数据示例
    sample_size = 5
    hidden_size = attn1_qkv.shape[1] // 3  # 获取每个头的隐藏维度大小
    
    # 分别获取Q、K、V的权重
    q1 = attn1_qkv[:, :hidden_size]
    k1 = attn1_qkv[:, hidden_size:2*hidden_size]
    v1 = attn1_qkv[:, 2*hidden_size:]
    
    q2 = attn2_qkv[:, :hidden_size]
    k2 = attn2_qkv[:, hidden_size:2*hidden_size]
    v2 = attn2_qkv[:, 2*hidden_size:]
    
    q3 = attn3_qkv[:, :hidden_size]
    k3 = attn3_qkv[:, hidden_size:2*hidden_size]
    v3 = attn3_qkv[:, 2*hidden_size:]
    
    q4 = attn4_qkv[:, :hidden_size]
    k4 = attn4_qkv[:, hidden_size:2*hidden_size]
    v4 = attn4_qkv[:, 2*hidden_size:]
    
    print(f"\n{model1_name}模型 Q权重示例 (前{sample_size}行，前5列):")
    print(q1[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model2_name}模型 Q权重示例 (前{sample_size}行，前5列):")
    print(q2[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model3_name}模型 Q权重示例 (前{sample_size}行，前5列):")
    print(q3[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model4_name}模型 Q权重示例 (前{sample_size}行，前5列):")
    print(q4[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model1_name}模型 K权重示例 (前{sample_size}行，前5列):")
    print(k1[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model2_name}模型 K权重示例 (前{sample_size}行，前5列):")
    print(k2[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model3_name}模型 K权重示例 (前{sample_size}行，前5列):")
    print(k3[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model4_name}模型 K权重示例 (前{sample_size}行，前5列):")
    print(k4[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model1_name}模型 V权重示例 (前{sample_size}行，前5列):")
    print(v1[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model2_name}模型 V权重示例 (前{sample_size}行，前5列):")
    print(v2[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model3_name}模型 V权重示例 (前{sample_size}行，前5列):")
    print(v3[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\n{model4_name}模型 V权重示例 (前{sample_size}行，前5列):")
    print(v4[:sample_size, :5].detach().cpu().numpy())
    
    # 计算Q、K、V的差异
    q_diff12 = torch.abs(q1 - q2)
    q_diff23 = torch.abs(q2 - q3)
    q_diff34 = torch.abs(q3 - q4)
    q_diff13 = torch.abs(q1 - q3)
    q_diff14 = torch.abs(q1 - q4)
    q_diff24 = torch.abs(q2 - q4)
    
    k_diff12 = torch.abs(k1 - k2)
    k_diff23 = torch.abs(k2 - k3)
    k_diff34 = torch.abs(k3 - k4)
    k_diff13 = torch.abs(k1 - k3)
    k_diff14 = torch.abs(k1 - k4)
    k_diff24 = torch.abs(k2 - k4)
    
    v_diff12 = torch.abs(v1 - v2)
    v_diff23 = torch.abs(v2 - v3)
    v_diff34 = torch.abs(v3 - v4)
    v_diff13 = torch.abs(v1 - v3)
    v_diff14 = torch.abs(v1 - v4)
    v_diff24 = torch.abs(v2 - v4)
    
    print(f"\nQ权重差异 (前{sample_size}行，前5列):")
    print(f"{model1_name}-{model2_name}:")
    print(q_diff12[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model3_name}:")
    print(q_diff23[:sample_size, :5].detach().cpu().numpy())
    print(f"{model3_name}-{model4_name}:")
    print(q_diff34[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model3_name}:")
    print(q_diff13[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model4_name}:")
    print(q_diff14[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model4_name}:")
    print(q_diff24[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\nK权重差异 (前{sample_size}行，前5列):")
    print(f"{model1_name}-{model2_name}:")
    print(k_diff12[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model3_name}:")
    print(k_diff23[:sample_size, :5].detach().cpu().numpy())
    print(f"{model3_name}-{model4_name}:")
    print(k_diff34[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model3_name}:")
    print(k_diff13[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model4_name}:")
    print(k_diff14[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model4_name}:")
    print(k_diff24[:sample_size, :5].detach().cpu().numpy())
    
    print(f"\nV权重差异 (前{sample_size}行，前5列):")
    print(f"{model1_name}-{model2_name}:")
    print(v_diff12[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model3_name}:")
    print(v_diff23[:sample_size, :5].detach().cpu().numpy())
    print(f"{model3_name}-{model4_name}:")
    print(v_diff34[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model3_name}:")
    print(v_diff13[:sample_size, :5].detach().cpu().numpy())
    print(f"{model1_name}-{model4_name}:")
    print(v_diff14[:sample_size, :5].detach().cpu().numpy())
    print(f"{model2_name}-{model4_name}:")
    print(v_diff24[:sample_size, :5].detach().cpu().numpy())
    

# 比较第0层的注意力权重
compare_attention_weights(model_raw, model_default, model_ori, model_xwwx, layer_idx=0)

