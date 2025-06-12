from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from accelerate.test_utils.testing import get_backend
from datasets import load_dataset,load_from_disk
from transformers.trainer_utils import get_last_checkpoint
import torch
from tqdm import tqdm
# import evaluate
from evaluate import load
from transformers import TrainingArguments, Trainer
import math
import matplotlib.pyplot as plt
import numpy as np


def calculate_perplexity(model_type="raw"):
    device, _, _ = get_backend()
    
    # 根据模型类型选择checkpoint
    if model_type == "raw":
        model_path = "../../gpt2"  # 未微调的原始GPT-2模型
    elif model_type == "default":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm")  # 使用wikitext 微调，未改变gpt模型的chekpoint
        model_path = last_checkpoint
    elif model_type == "ori":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到具体的原注意力计算模式
        model_path = last_checkpoint
    elif model_type == "xwwx":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到xwwx注意力计算模式
        model_path = last_checkpoint
    else:
        raise ValueError("model_type must be one of: raw, default, ori, xwwx")

    print(f"\n正在加载 {model_type} 模型...")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    # 加载测试数据
    print("正在加载测试数据...")
    raw = load_from_disk("../datasets/wikitext-2-raw-v1")
    test = raw["test"]

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    print(f"开始计算 {model_type} 模型的困惑度...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl.item()

def calculate_perplexity_sparse(model_type="xwwx", sparsity=0):
    device, _, _ = get_backend()
    
    # 构建模型路径
    if sparsity == 0:
        model_path = f"./tmp/test-clm-{model_type}"
    else:
        model_path = f"./tmp/test-clm-{model_type}-{sparsity}"
    
    print(f"\n正在加载 {model_type} 模型 (sparsity={sparsity})...")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)

    # 加载测试数据
    print("正在加载测试数据...")
    raw = load_from_disk("../datasets/wikitext-2-raw-v1")
    test = raw["test"]

    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    
    print(f"开始计算 {model_type} 模型 (sparsity={sparsity}) 的困惑度...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        num_valid_tokens = (target_ids != -100).sum().item()
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
    return ppl.item()

if __name__ == "__main__":
    # 测试不同稀疏度的模型
    model_types = ["xwwx", "ori"]
    sparsities = [0,0.05, 0.1,0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    results = {}
    
    for model_type in model_types:
        results[model_type] = {}
        for sparsity in sparsities:
            try:
                ppl = calculate_perplexity_sparse(model_type, sparsity)
                results[model_type][sparsity] = ppl
                print(f"\n{model_type} 模型 (sparsity={sparsity}) 在测试集上的困惑度(Perplexity): {ppl:.2f}")
            except Exception as e:
                print(f"\n计算 {model_type} 模型 (sparsity={sparsity}) 困惑度时出错: {str(e)}")
    
    print("\n=== 所有模型的困惑度对比 ===")
    for model_type in model_types:
        print(f"\n{model_type} 模型:")
        for sparsity in sparsities:
            if sparsity in results[model_type]:
                print(f"  sparsity={sparsity}: {results[model_type][sparsity]:.2f}")

    # 绘制困惑度对比图
    # 准备绘图数据
    plot_sparsities = [s for s in sparsities]  
    xwwx_ppl = [results["xwwx"].get (s) for s in sparsities]
    ori_ppl = [results["ori"].get(s) for s in sparsities]

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制折线
    plt.plot(plot_sparsities, xwwx_ppl, 'b-o', label='XWWX Model', linewidth=2)
    plt.plot(plot_sparsities, ori_ppl, 'r-o', label='ORI Model', linewidth=2)

    # 在每个点旁边显示具体ppl数值
    for x, y in zip(sparsities, xwwx_ppl):
        plt.text(x, y, f'{y:.2f}', color='blue', fontsize=9, ha='left', va='bottom')
    for x, y in zip(sparsities, ori_ppl):
        plt.text(x, y, f'{y:.2f}', color='red', fontsize=9, ha='right', va='top')
    # 设置标题和标签
    plt.title('Perplexity vs Sparsity', fontsize=14)
    plt.xlabel('Sparsity', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    plt.legend(fontsize=10)

    # 设置x轴刻度
    plt.xticks(plot_sparsities)

    # 保存图片
    plt.savefig('perplexity_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


# perplexity = load("./metrics/perplexity", module_type="metric")
# input_texts = load_from_disk("../datasets/wikitext-2-raw-v1")["test"]["text"]
# input_texts = [s for s in input_texts if s!='']
# results = perplexity.compute(model_id='../../gpt2',
#                              predictions=input_texts)
# print(list(results.keys()))
# # >>>['perplexities', 'mean_perplexity']    
# print(round(results["mean_perplexity"], 2))
# # >>>576.76
# print(round(results["perplexities"][0], 2))
# # >>>889.28
# print(results)

# def calculate_perplexity_with_trainer(model_type="raw"):
#     device, _, _ = get_backend()
    
#     # 根据模型类型选择checkpoint
#     if model_type == "raw":
#         model_path = "../../gpt2"  # 未微调的原始GPT-2模型
#     elif model_type == "default":
#         last_checkpoint = get_last_checkpoint("./tmp/test-clm")  # 使用wikitext 微调，未改变gpt模型的chekpoint
#         model_path = last_checkpoint
#     elif model_type == "ori":
#         last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到具体的原注意力计算模式
#         model_path = last_checkpoint
#     elif model_type == "xwwx":
#         last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到xwwx注意力计算模式
#         model_path = last_checkpoint
#     else:
#         raise ValueError("model_type must be one of: raw, default, ori, xwwx")

#     print(f"\n正在加载 {model_type} 模型...")
#     model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
#     tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
#     # 设置padding token
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id

#     # 加载测试数据
#     print("正在加载测试数据...")
#     raw = load_from_disk("../datasets/wikitext-2-raw-v1")
#     test = raw["test"]

#     # 准备训练参数
#     training_args = TrainingArguments(
#         output_dir="./tmp/test-perplexity",
#         do_train=False,
#         do_eval=True,
#         per_device_eval_batch_size=8,
#         dataloader_num_workers=4,
#     )

#     # 准备数据集
#     def tokenize_function(examples):
#         return tokenizer(examples["text"], padding=True, truncation=True)

#     tokenized_datasets = test.map(
#         tokenize_function,
#         batched=True,
#         remove_columns=["text"],
#     )

#     # 定义计算指标的函数
#     def compute_metrics(eval_preds):
#         return {"perplexity": math.exp(eval_preds.loss)}

#     # 初始化Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         eval_dataset=tokenized_datasets,
#         tokenizer=tokenizer,
#         compute_metrics=compute_metrics,
#     )

#     # 评估
#     print(f"开始计算 {model_type} 模型的困惑度...")
#     metrics = trainer.evaluate()
    
#     try:
#         perplexity = metrics["perplexity"]
#     except KeyError:
#         perplexity = math.exp(metrics["eval_loss"])
    
#     return perplexity

# if __name__ == "__main__":
#     # 计算所有模型类型的困惑度
#     model_types = ["raw", "default", "ori", "xwwx"]
#     results = {}
    
#     for model_type in model_types:
#         try:
#             ppl = calculate_perplexity_with_trainer(model_type)
#             results[model_type] = ppl
#             print(f"\n{model_type} 模型在测试集上的困惑度(Perplexity): {ppl:.2f}")
#         except Exception as e:
#             print(f"\n计算 {model_type} 模型困惑度时出错: {str(e)}")
    
#     print("\n=== 所有模型的困惑度对比 ===")
#     for model_type, ppl in results.items():
#         print(f"{model_type}: {ppl:.2f}")

# 保留原有的计算方式作为参考
# perplexity = load("./metrics/perplexity", module_type="metric")
# input_texts = load_from_disk("../datasets/wikitext-2-raw-v1")["test"]["text"]
# input_texts = [s for s in input_texts if s!='']
# results = perplexity.compute(model_id='../../gpt2',
#                              predictions=input_texts)
# print(list(results.keys()))
# # >>>['perplexities', 'mean_perplexity']    
# print(round(results["mean_perplexity"], 2))
# # >>>576.76
# print(round(results["perplexities"][0], 2))
# # >>>889.28