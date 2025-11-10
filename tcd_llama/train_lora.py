import os
import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    #DataCollatorForLanguageModeling
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel # 用於後續測試
)

# -----------------------------------------------------------------
# 步驟 0: 導入你的自訂 TCD 模型
# (假設你的 TCDLlamaForCausalLM 類在 TCDAttention.py 文件中)
# -----------------------------------------------------------------
try:
    from TCDAttention import TCDLlamaForCausalLM, TokenCache
except ImportError:
    print("錯誤：找不到 TCDAttention.py 文件。")
    print("請確保此腳本與 TCDAttention.py 儲存在同一個目錄中。")
    exit(1)

# --- 1. 配置模型和 Tokenizer ---
model_name = "/home/tgx/data/models/Llama-3-8B-Instruct"
output_dir = "./tcd-llama-squad-lora" # LoRA 適配器(adapter)的儲存位置

print(f"正在加載 Tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Llama 3 沒有 pad token，我們將它設置為 eos_token (結束符)
# 這對於 DataCollator (數據整理器) 進行填充 (padding) 至關重要
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"正在加載基礎模型 (TCDLlamaForCausalLM): {model_name}")
model = TCDLlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# 確保 post_init 已經運行 (這一步很重要，因為你的 post_init 會計算 q_j_proj)
#model.post_init()


# --- 2. PEFT / LoRA 配置 ---
print("正在配置 LoRA (PEFT)...")

lora_config = LoraConfig(
    r=16,                # LoRA 的秩 (rank)，8, 16, 32 是常用值
    lora_alpha=32,       # Alpha (通常是 r 的 2 倍)
    lora_dropout=0.1,    # Dropout 比例
    bias="none",         # "none", "all" 或 "lora_only"
    task_type=TaskType.CAUSAL_LM, # 必須設置為因果語言模型
    
    # 這是最關鍵的一步：
    # 指定要應用 LoRA 的模組 (nn.Linear 層)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
    ]
)

# 將基礎模型轉換為 PeftModel (LoRA 模型)
model = get_peft_model(model, lora_config)

print("可訓練參數 (Trainable parameters):")
model.print_trainable_parameters()


# --- 3. 數據集處理 (SQuAD) ---

def create_squad_prompt(example):
    """將 SQuAD 樣本轉換為提示格式"""
    context = example["context"]
    question = example["question"]
    # 確保答案存在
    answer = example["answers"]["text"][0] if example["answers"]["text"] else "No answer."
    
    # 這是模型將學習的格式
    prompt_template = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    return {"text": prompt_template}

def tokenize_and_mask_labels(example):
    """
    對樣本進行 Tokenize，並對「提示」部分進行 Mask，
    使模型只學習預測「答案」部分。
    """
    
    # 1. 創建提示 (不含答案) 和 完整文本 (含答案)
    prompt_template = example["text"].split("\nAnswer:")[0] + "\nAnswer: "
    full_text = example["text"] + tokenizer.eos_token # 添加結束符
    
    # 2. Tokenize 兩者
    # (max_length 應根據你的 GPU 顯存調整)
    tokenized_full = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
    
    # (只 Tokenize 提示，用於計算長度)
    tokenized_prompt = tokenizer(prompt_template, truncation=True, max_length=512, padding=False)
    
    # 3. 計算提示的 Token 長度
    prompt_len = len(tokenized_prompt["input_ids"])

    # 4. 創建標籤 (Labels)
    # 我們複製 input_ids 作為標籤
    labels = list(tokenized_full["input_ids"])
    
    # 5. 關鍵：將「提示」部分的標籤設置為 -100
    # PyTorch 的 CrossEntropyLoss 會自動忽略 -100 的標籤
    # 這確保了模型只在「答案」部分計算損失 (Loss)
    labels[:prompt_len] = [-100] * prompt_len
    
    tokenized_full["labels"] = labels
    return tokenized_full

print("正在加載和處理 SQuAD 數據集...")
# 為了快速演示，我們只使用 5000 個樣本。
# 在正式訓練時，請移除 "[:5000]"
data = load_dataset("squad", split="train[:5000]")

# 步驟 1: 格式化為提示
data = data.map(create_squad_prompt, remove_columns=data.column_names)
# 步驟 2: Tokenize 並 Mask 標籤
# === 修正 ===
tokenized_data = data.map(tokenize_and_mask_labels, remove_columns=data.column_names)
# ============

print(f"數據集樣本數: {len(tokenized_data)}")


# --- 4. 配置 Trainer ---

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,      # 根據你的顯存調整
    gradient_accumulation_steps=8,      # 模擬更大的批次大小 (2*8=16)
    learning_rate=2e-4,                 # LoRA 通常使用比全量微調更高的學習率
    num_train_epochs=1,                 # 1 個 epoch 通常足夠用於演示
    bf16=True,                          # 使用 bfloat16 (如果你的 GPU 支持)
    logging_steps=20,
    save_steps=100,
    report_to="none",                   # 禁用 wandb/tensorboard (可選)
    save_total_limit=2,
)

# 數據整理器 (Data Collator)
# DataCollatorForLanguageModeling 會自動處理 input_ids 和 labels 的填充
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model, 
    label_pad_token_id=-100 # 顯式指定 labels 的填充符
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator,
)

# --- 5. 開始訓練 ---
print("="*30)
print("     開始 LoRA 微調     ")
print("="*30)
trainer.train()

# --- 6. 儲存最終的 LoRA 適配器 ---
final_model_dir = os.path.join(output_dir, "final_checkpoint")
print(f"訓練完成。正在儲存 LoRA 適配器到: {final_model_dir}")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

# --- 7. (可選) 測試微調後的模型 ---
print("\n" + "="*30)
print("     測試微調後的模型     ")
print("="*30)

# 加載基礎模型 (TCD)
base_model = TCDLlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# 應用 LoRA 適配器
lora_model = PeftModel.from_pretrained(base_model, final_model_dir)
lora_model.eval()

# 準備一個 SQuAD 格式的測試提示
test_context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
test_question = "Where is the Eiffel Tower located?"
test_prompt = f"Context: {test_context}\nQuestion: {test_question}\nAnswer: "

config = lora_model.config
past_key_values = TokenCache(config=config)

inputs = tokenizer(test_prompt, return_tensors="pt").to(lora_model.device)
output = lora_model.generate(**inputs, max_new_tokens=20, past_key_values=past_key_values)
print("模型輸出:")
print(tokenizer.decode(output[0], skip_special_tokens=True))