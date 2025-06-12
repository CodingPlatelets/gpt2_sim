from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.trainer_utils import get_last_checkpoint
import torch
import numpy as np
from transformers.models.gpt2 import modeling_gpt2
import torch.cuda.nvtx as nvtx  # 添加nvtx支持

def generate_text(prompt, model_type="default", max_length=30):
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 根据模型类型选择checkpoint
    if model_type == "raw":
        model_path = "../../gpt2"  # 未微调的原始GPT-2模型
    elif model_type == "default":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm")  # 使用wikitext 微调，未改变gpt模型的chekpoint
        model_path = last_checkpoint
    elif model_type == "ori":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-ori-sxv")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到具体的原注意力计算模式
        model_path = last_checkpoint
    elif model_type == "xwwx":
        last_checkpoint = get_last_checkpoint("./tmp/test-clm-xwwx-sxv")  # 使用wikitext 微调，改变gpt模型Attn部分的指定到xwwx注意力计算模式
        model_path = last_checkpoint
    else:
        raise ValueError("model_type must be one of: raw, default, ori, xwwx")

    # 加载模型和分词器
    nvtx.range_push("模型加载")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    from model import GPT2AttentionXWWX
    modeling_gpt2.GPT2Attention = GPT2AttentionXWWX
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        use_cache=False  # 设置use_cache为False
    )
    # 将模型移动到GPU
    model = model.to(device)
    nvtx.range_pop()

    # 打印模型结构
    print(model)
    # 打印模型配置
    print("\n模型配置:")
    print(model.config)
    
    # 将模型设置为评估模式
    model.eval()
    
    # 对输入文本进行编码
    nvtx.range_push("文本编码")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)
    # 将输入数据移动到GPU
    inputs = inputs.to(device)
    attention_mask = attention_mask.to(device)
    nvtx.range_pop()
    
    # 生成文本
    nvtx.range_push("文本生成")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,  # 确保输出hidden_states
            output_attentions=True,     # 确保输出attention weights
            return_dict_in_generate=True  # 返回字典格式
        )
    nvtx.range_pop()
    
    # 解码生成的文本
    nvtx.range_push("文本解码")
    # 将输出移回CPU进行解码
    generated_text = tokenizer.decode(outputs.sequences[0].cpu().numpy(), skip_special_tokens=True)
    nvtx.range_pop()
    
    return generated_text

if __name__ == "__main__":
    # 测试提示
    prompt = "Hello, I'm a student,"
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")

    print("=== 测试改变模型定向到XWWX注意力计算模式 ===")
    generated_xwwx = generate_text(prompt, model_type="xwwx")
    print(f"输入: {prompt}")
    print(f"生成: {generated_xwwx}")
    
    if torch.cuda.is_available():
        print(f"GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
