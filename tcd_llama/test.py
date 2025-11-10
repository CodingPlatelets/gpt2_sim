from transformers import LlamaForCausalLM, AutoTokenizer
# 確保 TCDLlamaForCausalLM 已經被導入
from TCDAttention import TCDLlamaForCausalLM, TokenCache
import torch
# === 1. 導入 PeftModel ===
from peft import PeftModel

# --- 1. 指定路徑 ---
base_model_path = "/home/tgx/data/models/Llama-3-8B-Instruct"
# 這是你 LoraConfig 中的 output_dir + "final_checkpoint"
adapter_model_path = "/home/tgx/data/projects/gpt2_sim/tcd_llama/tcd-llama-wikitext-lora/final_checkpoint" 

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
print("\n" + "="*30)
print("     測試微調後的模型     ")
print("="*30)

# 加載基礎模型 (TCD)
base_model = TCDLlamaForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# 應用 LoRA 適配器
lora_model = PeftModel.from_pretrained(base_model, adapter_model_path)
lora_model.eval()

# 準備一個 SQuAD 格式的測試提示
#test_context = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower."
#test_question = "Where is the Eiffel Tower located?"
test_prompt = f"please write a python function to calculate the factorial of a number."
config = lora_model.config
past_key_values = TokenCache(config=config)
inputs = tokenizer(test_prompt, return_tensors="pt").to(lora_model.device)
output = lora_model.generate(**inputs, max_new_tokens=50, past_key_values=past_key_values)
print("模型輸出:")
print(tokenizer.decode(output[0], skip_special_tokens=True))