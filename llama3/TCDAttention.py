from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
from typing import Optional, Callable
from typing_extensions import Unpack
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging, TransformersKwargs
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, LlamaRotaryEmbedding
from transformers.modeling_rope_utils import dynamic_rope_update

logger = logging.get_logger(__name__)


import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig


class TCDRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: LlamaConfig, device=None):
        # 1. 让父类计算 [θ_0, θ_1, θ_2, ...]
        super().__init__(config=config, device=device)
        
        # 2. "近似" 操作：用 [θ_0, θ_0, ...] 覆盖它
        #    (这个操作只在模型加载时执行一次)
        theta_head = self.inv_freq[0]
        num_freqs = self.inv_freq.shape[0]
        inv_freq_fixed = theta_head.repeat(num_freqs)
        
        # 3. 用 "近似版" 覆盖掉 "标准版"
        #    (确保 device 一致)
        self.register_buffer("inv_freq", inv_freq_fixed.to(self.inv_freq.device), persistent=False)
        self.original_inv_freq = inv_freq_fixed
    
    @torch.no_grad()
    # @dynamic_rope_update # 假设这个装饰器存在
    def forward(self, x, position_ids):
        
        # ==================== MODIFICATION REMOVED ====================
        # 这里不再需要任何修改了！
        # self.inv_freq 已经被 __init__ 永久地改成了 [θ_0, θ_0, ...]
        # ============================================================

        # 4. 后续所有计算直接使用 "被修改过的" self.inv_freq
        
        # inv_freq_expanded shape: (B, D_head/2, 1)
        # self.inv_freq 现在是 [θ_0, θ_0, ...]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        
        # position_ids_expanded shape: (B, 1, S)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autast(device_type=device_type, enabled=False):  # Force float32
            
            # (B, D_head/2, 1) @ (B, 1, S) -> (B, D_head/2, S)
            # 这里的 @ 乘法现在使用的是 [θ_0, θ_0, ...] 
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # emb shape: (B, S, D_head)
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # cos/sin shape: (B, S, D_head)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class TCDAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

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

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
