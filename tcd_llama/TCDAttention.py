from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Any
from typing_extensions import Unpack
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging, TransformersKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm
from transformers.models.llama.modeling_llama import create_causal_mask
from transformers.utils.generic import check_model_inputs


logger = logging.get_logger(__name__)


import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig

def build_tcd_j_matrix(
    head_dim: int, 
    device: torch.device = torch.device("cpu"), 
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
   
    j_2d_block = torch.tensor(
        [[0, 1], 
         [-1, 0]], 
        dtype=dtype, 
        device=device
    )

    J_matrix = torch.zeros((head_dim, head_dim), dtype=dtype, device=device)

    num_blocks = head_dim // 2
    for i in range(num_blocks):
        start = i * 2
        end = start + 2
        J_matrix[start:end, start:end] = j_2d_block
        
    return J_matrix



class TokenLayer(DynamicLayer):  
    def __init__(self):
        super().__init__()
        del self.keys, self.values
        self.tokens: Optional[torch.Tensor] = None
    
    def lazy_initialization(self, hidden_states: torch.Tensor):
        self.dtype, self.device = hidden_states.dtype, hidden_states.device
        self.tokens = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True
        
    def update(
        self,
        hidden_states: torch.Tensor,
        no_used_hidden_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(hidden_states)
        self.tokens = torch.cat([self.tokens, hidden_states], dim=-2)
        return self.tokens, None
    
    def get_seq_length(self) -> int:
        if not self.is_initialized or self.tokens.numel() == 0:
            return 0
        return self.tokens.shape[-2]
    
class TokenCache(DynamicCache):
    def __init__(
        self,
        ddp_cache_data = None,
        config = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(ddp_cache_data, config, offloading, offload_only_non_sliding)
        for i in range(len(self.layers)):
            self.layers[i] = TokenLayer()


class TCDRotaryEmbedding(LlamaRotaryEmbedding):
 
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__(config=config, device=device)
        r = getattr(config, "num_rope_bands", 4) 

        if r <= 1:
            theta_head = self.inv_freq[0]
            num_freqs = self.inv_freq.shape[0]
            inv_freq_fixed = theta_head.repeat(num_freqs)
        else:
            num_freqs = self.inv_freq.shape[0] # D_h / 2
            
            if num_freqs % r != 0:
                raise ValueError(f"D_head/2 ({num_freqs}) must be divisible by r ({r}) for r-band RoPE.")
            band_size = num_freqs // r

            #    indices: [0, band_size, 2*band_size, ...]
            indices = torch.arange(0, num_freqs, band_size, device=self.inv_freq.device)
            
            # theta_bands: [θ_A, θ_B, θ_C, ...] (shape: [r])
            theta_bands = self.inv_freq[indices]
    
            inv_freq_fixed = theta_bands.repeat_interleave(band_size, dim=0)

        self.register_buffer("inv_freq", inv_freq_fixed.to(self.inv_freq.device), persistent=False)
        self.original_inv_freq = inv_freq_fixed
    
    @torch.no_grad()
    #@dynamic_rope_update 
    def forward(self, 
                cache_position: torch.LongTensor, 
                kv_len: int, 
                r: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        為 TCD r-band *動態* 計算 cos(Δ_b) 和 sin(Δ_b)。
        (這必須在 TCDLlamaAttention.forward 內部被調用)
        """
        device = self.inv_freq.device
        dtype = self.inv_freq.dtype
        q_len = cache_position.shape[0]
        
        # 1. 獲取 i (查詢位置) 和 j (鍵位置)
        # i_pos shape: [Q_len, 1]
        i_pos = cache_position.view(-1, 1)    
        
        # j_pos shape: [1, K_len]
        j_pos = torch.arange(kv_len, device=device).view(1, -1) 

        # 2. 計算相對位置 (i - j)
        # delta_pos shape: [Q_len, K_len]
        delta_pos = (i_pos - j_pos).float()

        # 3. 獲取 r 個頻帶的頻率 (θ_b)
        # self.inv_freq (D_h/2) -> (r, band_size)
        num_freqs = self.inv_freq.shape[0]
        band_size = num_freqs // r
        inv_freq_bands = self.inv_freq.view(r, band_size)
        
        # theta_bands: [θ_A, θ_B, ...] (shape: [r])
        theta_bands = inv_freq_bands[:, 0] # 取每個頻帶的第一個頻率
        
        # 4. 為 r 個頻帶計算 Δ_b = (i - j) * θ_b
        # theta_r shape: [r, 1, 1]
        theta_r = theta_bands.view(r, 1, 1)
        
        # delta_r shape: [r, Q_len, K_len]
        delta_r = delta_pos.unsqueeze(0) * theta_r
        
        # [r, Q_len, K_len]
        cos = torch.cos(delta_r).to(dtype)
        sin = torch.sin(delta_r).to(dtype)
        
        # 5. 調整形狀以匹配 (B, H, Q, r, K)
        # 返回: [1, 1, Q, r, K] (B 和 H 會被廣播)
        return cos.permute(1, 0, 2).unsqueeze(0).unsqueeze(0), sin.permute(1, 0, 2).unsqueeze(0).unsqueeze(0)

class TCDLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, rotary_emb: TCDRotaryEmbedding):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.r = 4
        self.rotary_emb = rotary_emb

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        j_matrix = build_tcd_j_matrix(
            self.head_dim, 
            dtype=torch.get_default_dtype() 
        )
        self.register_buffer("j_matrix", j_matrix, persistent=True)

    #@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        #position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[TokenCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        _, _, hidden_dim = hidden_states.shape
        
        r = self.r
        assert self.head_dim % r == 0, "head_dim must be divisible by r"
        
        band_dim = self.head_dim // r
        
        q_A_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        q_B_states = q_A_states @ self.j_matrix
        #q_B_states = self.q_j_proj(hidden_states).view(hidden_shape).transpose(1, 2) # (B, H， N, D_h)
        
        # k_T per-head shape: [H_kv, D_h, D_in]
        k_T_per_head = self.k_proj.weight.view(
            self.num_key_value_heads, self.head_dim, hidden_dim
        )
        k_T_per_head_repeated = k_T_per_head.repeat_interleave(self.num_key_value_groups, dim=0) #(H, D_h, D_in)
        if past_key_value is not None and self.training is False:
            # 推理 (Generation) 模式
            # past_tokens 是 TokenCache 對象, hidden_states 是 (B, 1, D)
            X_cache, _ = past_key_value.update(hidden_states, None, self.layer_idx, {})
        else:
            # 訓練 (Training) 或 提示 (Prompt) 模式
            # hidden_states 是 (B, Q_len, D)
            X_cache = hidden_states
        #X_cache = hidden_states
        # 2. 在 *更新後* 的 X_cache 上計算 kv_len
        kv_len = X_cache.shape[-2]
            
        # X_cache.transpose: (B, N_cache, D_in) -> (B, D_in, N_cache)
        X_j_T = X_cache.transpose(-1, -2)
        
        # r-band gate
        
        # D_h -> (r, band_dim)
        
        # q_A_bands: (B, H, N, r, band_dim)
        q_A_bands = q_A_states.view(*q_A_states.shape[:-1], r, band_dim)
        
        # q_B_bands: (B, H, N, r, band_dim)
        q_B_bands = q_B_states.view(*q_B_states.shape[:-1], r, band_dim)
        
        # k_T_bands: (H, r, band_dim, D_in)
        k_T_bands = k_T_per_head_repeated.view(
            self.num_heads, r, band_dim, hidden_dim
        )
        
        # (B,H,N,r,D_h/r) @ (H,r,D_h/r,D_in) -> (B,H,N,r,D_in)
        q_A_Wk_T_bands = torch.einsum('bhqrd, hrdi -> bhqri', q_A_bands, k_T_bands)
        q_B_Wk_T_bands = torch.einsum('bhqrd, hrdi -> bhqri', q_B_bands, k_T_bands)
        
        #(B,H,N,r,D_in) @ (B, D_in, N_cache) -> (B,H,N,r,N_cache)
        
        attn_scores_A_bands = torch.einsum('bhnrd, bdk -> bhnrk', q_A_Wk_T_bands, X_j_T)
        attn_scores_B_bands = torch.einsum('bhnrd, bdk -> bhnrk', q_B_Wk_T_bands, X_j_T)

        
        cos, sin = self.rotary_emb(
            cache_position=cache_position, 
            kv_len=kv_len, 
            r=self.r
        )
        #print(cos.shape)
        #print(attn_scores_A_bands.shape)
        
        # (B,H,Q,r,K) * (B,1,Q,r,K) -> (B,H,Q,r,K)
        attn_scores_bands = attn_scores_A_bands * cos + attn_scores_B_bands * sin
        
        # (B,H,Q,r,K) -> (B,H,Q,K)
        attn_scores = attn_scores_bands.sum(dim=3)
        
        attn_scores = attn_scores * self.scaling
        # 5. 应用掩码和 Softmax (p)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_A_states.dtype)
        attn_probs = nn.functional.dropout(attn_probs, p=0.0 if not self.training else self.attention_dropout,training=self.training)

        # 6. TCD Value 聚合: (p * X) * W_v
        pX = torch.matmul(attn_probs, X_cache.unsqueeze(1)) 

        # (pX) * W_v

        # w_v shape: [H_kv * D_h, D_in]
        
        v_T_per_head = self.v_proj.weight.view(
            self.num_key_value_heads, self.head_dim, hidden_dim
        ).permute(2, 0, 1)  # shape: [D_in, H_kv, D_h]
        
        # shape: [D_in, H, D_h]
        v_T_per_head_repeated = v_T_per_head.repeat_interleave(self.num_key_value_groups, dim=1)
        
        v_T_per_head_repeated = v_T_per_head_repeated.transpose(0, 1)  # shape: [H, D_in, D_h]
        
        
        attn_output = pX @ v_T_per_head_repeated
        
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    

class TCDLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, rotary_emb):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = TCDLlamaAttention(config=config, layer_idx=layer_idx, rotary_emb=rotary_emb)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    #@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            #position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config: LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": TCDLlamaDecoderLayer,
        "attentions": TCDLlamaAttention,
    }


@auto_docstring
class TCDLlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.rotary_emb = TCDRotaryEmbedding(config=config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TCDLlamaDecoderLayer(config, layer_idx, self.rotary_emb) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        #position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                #position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class TCDLlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = TCDLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
    
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    