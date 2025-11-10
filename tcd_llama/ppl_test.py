import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math

# 確保 TCDLlamaForCausalLM 已經被導入
from TCDAttention import TCDLlamaForCausalLM, TokenCache
from peft import PeftModel

# --- 1. 配置模型路徑 ---
base_model_path = "/home/tgx/data/models/Llama-3-8B-Instruct"
adapter_model_path = "/home/tgx/data/projects/gpt2_sim/tcd_llama/tcd-llama-squad-lora-v2/final_checkpoint" 

# --- 2. 加載模型和 Tokenizer ---
print("正在加載 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("正在加載基礎模型 (TCD)...")
base_model = TCDLlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

print("正在應用 LoRA 適配器...")
model = PeftModel.from_pretrained(base_model, adapter_model_path)
model.eval()

# --- 3. 加載並準備 SQuAD 驗證集 ---
print("正在加載 SQuAD 驗證集...")

def create_squad_prompt(example):
    context = example["context"]
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer."
    # 為了計算 PPL，我們需要完整的文本序列
    return {"text": f"Context: {context}\nQuestion: {question}\nAnswer: {answer}{tokenizer.eos_token}"}

# 使用 SQuAD 的 validation split
validation_data = load_dataset("squad", split="validation") 
# 將數據集格式化為與訓練時相同的提示格式
formatted_texts = validation_data.map(create_squad_prompt, remove_columns=validation_data.column_names)["text"]
encodings = tokenizer("\n\n".join(formatted_texts), return_tensors="pt")


# --- 4. 計算 Perplexity (PPL) ---
max_length = 1024 # 對於 SQuAD 樣本，1024 的窗口通常足夠了
stride = 512 
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0

print("正在計算 Perplexity (on SQuAD validation set)...")
# --- 修正循环范围，确保处理整个数据集 ---
for begin_loc in tqdm(range(0, 4096, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    
    # --- 关键修复：修正 target_ids 的 masking 逻辑 ---
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
    target_ids = input_ids.clone()
    
    # 我们只对新滑入窗口的 stride 部分计算 loss。
    # 因此，对于每个窗口，前面的 (max_length - stride) 部分都应该被 mask。
    # 注意：这假设 max_length > stride
    mask_len = max_length - stride
    target_ids[:, :mask_len] = -100

    # 丢弃最后一个不足一个完整窗口的批次，以避免偏差
    if input_ids.size(1) != max_length:
        continue

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        # outputs.loss 会自动计算未被 mask 的部分的平均 loss
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

# 計算最終的 PPL
try:
    ppl = torch.exp(torch.stack(nlls).mean())
    print("\n" + "="*30)
    print(f"模型在 SQuAD 驗證集上的 Perplexity: {ppl.item():.4f}")
    print("="*30)
except Exception as e:
    print(f"\n計算 PPL 時出錯: {e}")
