import os
import torch
from transformers import (
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling # <-- 更改 DataCollator
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType
)

# -----------------------------------------------------------------
# 步驟 0: 導入你的自訂 TCD 模型
# -----------------------------------------------------------------
try:
    from TCDAttention import TCDLlamaForCausalLM
except ImportError:
    print("錯誤：找不到 TCDAttention.py 文件。")
    exit(1)

# --- 1. 配置模型和 Tokenizer ---
model_name = "/home/tgx/data/models/Llama-3-8B-Instruct"
output_dir = "./tcd-llama-wikitext-lora" # 新的輸出目錄

print(f"正在加載 Tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
print(f"正在加載基礎模型 (TCDLlamaForCausalLM): {model_name}")
model = TCDLlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# --- 2. PEFT / LoRA 配置 ---
print("正在配置 LoRA (PEFT)...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
print("可訓練參數 (Trainable parameters):")
model.print_trainable_parameters()

# --- 3. 數據集處理 (WikiText) ---

def tokenize_function(examples):
    """對 WikiText 數據集進行 Tokenize"""
    # 將每個樣本的文本與結束符拼接
    # Llama3 的訓練方式建議在每個文檔後添加結束符
    text_with_eos = [text + tokenizer.eos_token for text in examples["text"]]
    
    # 進行 Tokenize
    return tokenizer(
        text_with_eos,
        truncation=True, 
        max_length=512, 
        # 不再需要 padding，DataCollator 會處理
    )

print("正在加載和處理 WikiText 數據集...")
# 加載 WikiText 訓練集，並過濾掉空行
data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) > 0)

# Tokenize 數據集
tokenized_data = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

print(f"數據集樣本數: {len(tokenized_data)}")

# --- 4. 配置 Trainer ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10, # WikiText 數據集較大，可以增加 logging 步數
    save_steps=200,
    report_to="none",
    save_total_limit=2,
)

# --- 關鍵更改：使用 DataCollatorForLanguageModeling ---
# 它會自動處理 padding 並創建 labels (通過複製 input_ids)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False # 這不是 Masked Language Model，而是 Causal LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator,
)

# --- 5. 開始訓練 ---
print("="*30)
print("     開始 LoRA 微調 (on WikiText)     ")
print("="*30)
trainer.train()

# --- 6. 儲存最終的 LoRA 適配器 ---
final_model_dir = os.path.join(output_dir, "final_checkpoint")
print(f"訓練完成。正在儲存 LoRA 適配器到: {final_model_dir}")
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print("\n微調完成！")
print(f"模型已儲存至: {final_model_dir}")
print("您現在可以使用 ppl_test.py 來評估此模型在 WikiText 驗證集上的 Perplexity。")