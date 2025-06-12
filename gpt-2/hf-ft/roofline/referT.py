from transformers import AutoTokenizer
from model_gpt2 import GPT2LMHeadModel
from model_llama import LlamaForCausalLM
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
from transformers.trainer_utils import get_last_checkpoint
import time
# from deepspeed.profiling.flops_profiler import get_model_profile

def get_model_profile(model,
                      args=[],
                      kwargs={},
                      print_profile=True,
                      detailed=True,
                      module_depth=-1,
                      top_modules=1,
                      warm_up=1,
                      as_string=True,
                      output_file=None,
                      ignore_modules=None,
                      mode='forward'):

    prof = FlopsProfiler(model)

    for _ in range(warm_up):
        if mode == 'forward':
            _ = model(*args, **kwargs)
        if mode == 'generate':
            _ = model.generate(*args, **kwargs)

    prof.start_profile(ignore_list=ignore_modules)

    if mode == 'forward':
        with torch.no_grad():   
            _ = model(*args, **kwargs)
    if mode == 'generate':
        with torch.no_grad():
            _ = model.generate(*args, **kwargs)


    flops = prof.get_total_flops(as_string)
    macs = prof.get_total_macs(as_string)
    params = prof.get_total_params(as_string)
    duration = prof.get_total_duration(as_string)
    if print_profile:
        prof.print_model_profile(profile_step=warm_up,
                                 module_depth=module_depth,
                                 top_modules=top_modules,
                                 detailed=detailed,
                                 output_file=output_file)

    prof.end_profile()
    return flops, macs, params, duration

def setup_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    last_checkpoint = get_last_checkpoint("./tmp/test-clm-medium")
    tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)
    
    model = GPT2LMHeadModel.from_pretrained(last_checkpoint, use_cache=True).to(device)
    # 转换为FP16
    # model = model.to(torch.float16)
    model = model.half()
    # 只保留第一层
    model.transformer.h = model.transformer.h[:1]
    model.n_layers = 1
    model.eval()
    # print(f"model.config: {model.config}")
    # 检查第一个参数的数据类型
    # first_param_dtype = next(model.parameters()).dtype
    # print(f"模型参数类型: {first_param_dtype}")
    # print(f"max_position_embeddings: {model.config.max_position_embeddings}")
    # print(model)
    # print(f"model.n_layers: {model.n_layers}")
    return model, tokenizer, device

def get_llama3_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llama3_model_path = "../../Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llama3_model_path)
    model = LlamaForCausalLM.from_pretrained(llama3_model_path, use_cache=True).to(device)
    model = model.half()
    # 检查第一个参数的数据类型
    first_param_dtype = next(model.parameters()).dtype
    print(f"模型参数类型: {first_param_dtype}")
    model.model.layers = model.model.layers[:1]
    model.eval()
    return model, tokenizer, device


def run_inference(model, batch_size, seq_len, use_cache=True,use_profiler=True):
    # 确保序列长度不超过模型的最大长度限制
    max_length = model.config.max_position_embeddings
    if seq_len - 1 >= max_length:
        print(f"Warning: sequence length {seq_len} exceeds model max length {max_length}, using {max_length-1}")
        seq_len = max_length - 1  # 留出一个位置给单步推理
    
    print(f"\n=== Run inference - batch size: {batch_size}, sequence length: {seq_len}  use_cache: {use_cache} ===")
    
    # 准备历史缓存
    if (use_cache):
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len - 1)).to(device)
        print(f"get past key values")
        with torch.no_grad():
            past = model(input_ids, use_cache=use_cache).past_key_values
    else:
        past = None

    # 准备数据
    if (use_cache):
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, 1)).to(device)
    else:
        input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)

    print(f"warmup")
    # warmup
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past, use_cache=use_cache)
    # warmup
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=past, use_cache=use_cache)
    

    print(f"test time of decode next token")
    if (use_profiler):
        profiler = FlopsProfiler(model)
        profiler.start_profile()
        start = time.time()
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past, use_cache=use_cache)
        end = time.time()
        profiler.stop_profile()
        print(f"FLOPs: {profiler.get_total_flops(as_string=True)}")
        print(f"MACs: {profiler.get_total_macs(as_string=True)}")
        print(f"Parameters: {profiler.get_total_params(as_string=True)}")
        print(f"Latency: {profiler.get_total_duration(as_string=True)}")
        print(f"Total Time: {end - start:.6f} 秒")
        profiler.print_model_profile(profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None)
        profiler.end_profile()

    else:
        for i in range(5):
            print(f"test {i}")
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past, use_cache=use_cache)
            torch.cuda.synchronize()
    return


if __name__ == "__main__":
    # 初始化模型
    # model, tokenizer, device = setup_model()
    model, tokenizer, device = get_llama3_model()
    # run_inference(model, batch_size=8, seq_len=4096, use_cache=True,use_profiler=False)
    run_inference(model, batch_size=8, seq_len=4096, use_cache=False,use_profiler=False)

