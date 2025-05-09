from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.pytorch_utils import (
    Conv1D,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer,
)
from transformers.utils import logging

from transformers.models.gpt2.modeling_gpt2 import (
    eager_attention_forward,
)

from transformers.models.gpt2 import modeling_gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention


logger = logging.get_logger(__name__)


# 与 from transformers.models.gpt2.modeling_gpt2 import GPT2Attention 等价，只是多了注释
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        # 保存配置并获取最大位置嵌入数。
        self.config = config
        max_positions = config.max_position_embeddings
        # 注册一个下三角矩阵作为buffer，用于实现因果自注意力掩码
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册一个负无穷大作为buffer，用于实现掩码
        self.register_buffer(
            "masked_bias", torch.tensor(-1e4), persistent=False)
        # 获取隐藏维度、头数和头维度。
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 设置分割大小为隐藏维度。
        self.split_size = self.embed_dim
        # 检查头数和隐藏维度是否匹配。
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 获取是否需要缩放注意力权重。
        self.scale_attn_weights = config.scale_attn_weights
        # 获取是否为交叉注意力。
        self.is_cross_attention = is_cross_attention

        # 层级注意力缩放、重新排序和上转换
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # 获取层级索引。
        self.layer_idx = layer_idx
        # 获取是否需要重新排序和上转换。
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果为交叉注意力，则使用两块KV，否则使用三块QKV。
        # Conv1D基本上就像一个线性层，但权重是转置的。
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 获取注意力dropout和残差dropout。
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 设置是否为因果注意力。
        self.is_causal = True
        # 初始化一个空集合，用于存储已修剪的头。
        self.pruned_heads = set()

    # 修剪头部。
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的头和索引。
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads)
        # 将索引连接起来。
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)])
        # 修剪卷积层。
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 更新超参数。
        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * \
            (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # 上转换和重新排序注意力。
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用 `torch.baddbmm`（一个更高效的alpha参数用于缩放--来自Megatron-LM）
        # 获取输入形状。bsz: batch_size, num_heads: 头数, q_seq_len: 查询序列长度, dk: 每个头维度
        bsz, num_heads, q_seq_len, dk = query.size()
        # 获取键序列长度。
        _, _, k_seq_len, _ = key.size()
        # 预分配attn_weights用于`baddbmm`
        attn_weights = torch.empty(
            bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        # 计算缩放因子。
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        # 如果需要，按层级缩放。
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        # 关闭自动计算并重新排序（缩放K为1/根号(dk)）
        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with torch.amp.autocast(query.device.type, enabled=False):
            # q k^T
            q, k = query.reshape(-1, q_seq_len,
                                 dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            # 使用baddbmm进行批量矩阵乘法，计算Q·K^T
            attn_weights = torch.baddbmm(
                attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            # 重塑回原始形状
            attn_weights = attn_weights.reshape(
                bsz, num_heads, q_seq_len, k_seq_len)
        # 对于自注意力，应用因果掩码
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 序列长度
            query_length, key_length = query.size(-2), key.size(-2)
            # 计算因果掩码
            causal_mask = self.bias[:, :, key_length -
                                    query_length: key_length, :key_length]
            # 获取掩码值（负无穷大）
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(
                mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            # 应用掩码
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        # 应用softmax得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32")

        # 转换数据类型
        attn_weights = attn_weights.type(value.dtype)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        # 应用头部掩码（如果提供）
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        # attn_weights: [bsz,num_heads,q_seq_len,k_seq_len]
        # value: [bsz,num_heads,k_seq_len,head_dim]
        # attn_output: [bsz,num_heads,q_seq_len,head_dim]
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        # attn_output: [bsz,q_seq_len,num_heads,head_dim]

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            # 如果是交叉注意力，计算Q和KV
            # Q来自当前序列
            query_states = self.q_attn(hidden_states)
            # K、V来自编码器
            key_states, value_states = self.c_attn(
                encoder_hidden_states).split(self.split_size, dim=2)
            # 使用编码器的注意力掩码
            attention_mask = encoder_attention_mask
        else:
            # 自注意力机制：Q、K、V都来自同一个输入
            query_states, key_states, value_states = self.c_attn(
                hidden_states).split(self.split_size, dim=2)

        # query_states:[bsz,q_seq_len,hid_dim]
        # shape_q:[bsz,q_seq_len,num_heads,head_dim]
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        # query_states:[[bsz,num_heads,q_seq_len,head_dim]
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)
        # 处理缓存（用于加速自回归生成）
        if layer_past is not None:
            # 如果有过去的缓存，将其与当前的K、V拼接
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        # 如果需要缓存，准备当前的K、V用于下一次迭代
        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None
        # 解码器输入不为空，则为交叉注意力机制
        is_cross_attention = encoder_hidden_states is not None
        # attention_mask为空，且不是交叉注意力，且序列长度大于1，则为因果注意力
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        # 选择注意力实现方式
        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            # 如果使用SDPA但需要输出注意力权重或头掩码，则回退到eager实现
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                # 使用配置指定的注意力实现
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            # 使用优化的上转换和重排序方法
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            # 使用标准注意力接口
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        # 重塑输出并应用投影
        attn_output = attn_output.reshape(
            *attn_output.shape[:-2], -1).contiguous()
        # attn_output:[bsz,q_seq_len,hid_dim]
        attn_output = self.c_proj(attn_output)
        # attn_output:[bsz,q_seq_len,hid_dim]
        # 应用残差连接和dropout
        attn_output = self.resid_dropout(attn_output)
        # present 缓存的K、V
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2AttentionXWWX(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        # 保存配置并获取最大位置嵌入数。
        self.config = config
        max_positions = config.max_position_embeddings
        # 注册一个下三角矩阵作为buffer，用于实现因果自注意力掩码
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        # 注册一个负无穷大作为buffer，用于实现掩码
        self.register_buffer(
            "masked_bias", torch.tensor(-1e4), persistent=False)
        # 获取隐藏维度、头数和头维度。
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 设置分割大小为隐藏维度。
        self.split_size = self.embed_dim
        # 检查头数和隐藏维度是否匹配。
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 获取是否需要缩放注意力权重。
        self.scale_attn_weights = config.scale_attn_weights
        # 获取是否为交叉注意力。
        self.is_cross_attention = is_cross_attention

        # 层级注意力缩放、重新排序和上转换
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # 获取层级索引。
        self.layer_idx = layer_idx
        # 获取是否需要重新排序和上转换。
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果为交叉注意力，则使用两块KV，否则使用三块QKV。
        # Conv1D基本上就像一个线性层，但权重是转置的。
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 获取注意力dropout和残差dropout。
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 设置是否为因果注意力。
        self.is_causal = True
        # 初始化一个空集合，用于存储已修剪的头。
        self.pruned_heads = set()

    # 修剪头部。（（未使用
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的头和索引。
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        # 将索引连接起来。
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)]
        )
        # 修剪卷积层。
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 更新超参数。
        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (
            self.num_heads - len(heads)
        )
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # 上转换和重新排序注意力。（（未使用
    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None
    ):
        # 使用 `torch.baddbmm`（一个更高效的alpha参数用于缩放--来自Megatron-LM）
        # 获取输入形状。bsz: batch_size, num_heads: 头数, q_seq_len: 查询序列长度, dk: 每个头维度
        bsz, num_heads, q_seq_len, dk = query.size()
        # 获取键序列长度。
        _, _, k_seq_len, _ = key.size()
        # 预分配attn_weights用于`baddbmm`
        # attn_weights = torch.empty(
        #     bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        # 计算缩放因子。
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        # 如果需要，按层级缩放。
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        # 关闭自动计算并重新排序（缩放K为1/根号(dk)）
        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        # with torch.amp.autocast(query.device.type, enabled=False):
        # q k^T
        q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
            -1, dk, k_seq_len
        )
        # 使用baddbmm进行批量矩阵乘法，计算Q·K^T
        # attn_weights = torch.baddbmm(
        #     attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)

        attn_weights = torch.matmul(q, k)

        attn_weights = attn_weights * scale_factor

        # # 重塑回原始形状
        attn_weights = attn_weights.reshape(
            bsz, num_heads, q_seq_len, k_seq_len)
        # 对于自注意力，应用因果掩码
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 序列长度
            query_length, key_length = query.size(-2), key.size(-2)
            # 计算因果掩码
            causal_mask = self.bias[
                :, :, key_length - query_length: key_length, :key_length
            ]
            # 获取掩码值（负无穷大）
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(
                mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            # 应用掩码
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        # 应用softmax得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )

        # 转换数据类型
        attn_weights = attn_weights.type(value.dtype)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        # 应用头部掩码（如果提供）
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        # attn_weights: [bsz,num_heads,q_seq_len,k_seq_len]
        # value: [bsz,num_heads,k_seq_len,head_dim]
        # attn_output: [bsz,num_heads,q_seq_len,head_dim]
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        # attn_output: [bsz,q_seq_len,num_heads,head_dim]

        return attn_output, attn_weights

    # 新的注意力计算方法
    def _xwwx_attn_bias(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_mask=None,
        head_mask=None,
        use_bf16=False,
    ):

        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            # 提取 w_q, w_k, w_v , q_bias , k_bias, v_bias
            w_q = self.q_attn.weight
            q_bias = self.q_attn.bias
            # c_attn.weight的形状是[embed_dim, 2*embed_dim]
            w_k, w_v = self.c_attn.weight.split(self.split_size, dim=1)
            # c_attn.bias的形状是[2*embed_dim,]
            k_bias, v_bias = self.c_attn.bias.split(self.split_size, dim=0)
            attention_mask = encoder_attention_mask
        else:

            # c_attn.weight的形状是[embed_dim, 3*embed_dim]
            w_q, w_k, w_v = self.c_attn.weight.split(self.split_size, dim=1)
            # c_attn.bias的形状是[3*embed_dim,]
            q_bias, k_bias, v_bias = self.c_attn.bias.split(
                self.split_size, dim=0)

            encoder_hidden_states = hidden_states
            
        # 如果启用bf16，转换精度
        if use_bf16 and torch.cuda.is_available():
            # 转换权重和偏置
            w_q = w_q.to(torch.bfloat16)
            w_k = w_k.to(torch.bfloat16)
            w_v = w_v.to(torch.bfloat16)
            q_bias = q_bias.to(torch.bfloat16)
            k_bias = k_bias.to(torch.bfloat16)
            v_bias = v_bias.to(torch.bfloat16)
            # 转换输入
            hidden_states = hidden_states.to(torch.bfloat16)
            encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)

        """使用x·W_q·W_k^T·x^T直接计算注意力分数"""
        """(x*W_q + q_bias)(x*W_k + k_bias)^T === query*(W_k^T*x^T + k_bias^T) ===  query*W_k^T*x^T + query*k_bias^T"""

        # step1 计算query = x·w_q + q_bias 以及 value = x·w_v + v_bias
        # x:[bsz, q/k_seq_len, embed_dim] ,w_q/k:[embed_dim, embed_dim], q/k_bias:[embed_dim,]
        query_states = torch.matmul(hidden_states, w_q) + q_bias
        value_states = torch.matmul(encoder_hidden_states, w_v) + v_bias

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query = query_states.view(shape_q).transpose(
            1, 2)  # [bsz, num_heads, q_seq_len,  head_dim]

        shape_v = (*value_states.shape[:-1], -1, self.head_dim)
        value = value_states.view(shape_v).transpose(
            1, 2)  # [bsz, num_heads, k_seq_len,  head_dim]

        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = value.size()

        # 计算缩放因子。
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        # 如果需要，按层级缩放。
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # step2 计算query*w_k^T 和query*k_bias^T
        # query [bsz, num_heads, q_seq_len,  head_dim] ,w_k:[embed_dim, embed_dim], k_bias:[embed_dim,]

        # query * w_k^T
        w_k = w_k.view(self.embed_dim, -1, self.head_dim)#[embed_dim, num_heads, head_dim]
        q_k_t = torch.einsum("bnsh,enh->bnse", query, w_k)#[bsz,num_heads,q_seq_len,embed_dim]

        # query * k_b^T
        k_bias = k_bias.view(-1, self.head_dim)  # [num_heads,head_dim]
        k_b_t_expanded = k_bias.unsqueeze(0).unsqueeze(2)  # [1,num_heads,1,head_dim]
        k_bias_T = torch.matmul(query, k_b_t_expanded.transpose(-1, -2))#[bsz,num_heads,q_seq_len,1]

        # query * w_k^T * x^T
        x_t = encoder_hidden_states.transpose(-1, -2)#[batch_size, embed_dim, k_seq_len]
        x_t_expand = x_t.unsqueeze(1).expand(-1, num_heads, -1, -1)#[batch_size,num_heads,embed_dim, k_seq_len]

        attn_weights = torch.matmul(q_k_t, x_t_expand) + k_bias_T#[bsz,num_heads,q_seq_len,s_seq_len]

        # scale
        attn_weights = attn_weights * scale_factor
        
        # 如果使用了bf16，将结果转回原始精度用于后续处理
        if use_bf16 and torch.cuda.is_available():
            attn_weights = attn_weights.to(torch.float32)

        # 对于自注意力，应用因果掩码
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = q_seq_len, k_seq_len
            # 计算因果掩码
            causal_mask = self.bias[
                :, :, key_length - query_length: key_length, :key_length
            ]
            # 获取掩码值（负无穷大）
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(
                mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            # 应用掩码
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        # 应用softmax得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )

        # 转换数据类型
        attn_weights = attn_weights.type(value.dtype)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 应用头部掩码（如果提供）
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # attn_weights: [bsz,num_heads,q_seq_len,k_seq_len]
        # value: [bsz,num_heads,k_seq_len,head_dim]
        # attn_output: [bsz,num_heads,q_seq_len,head_dim]
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        # attn_output: [bsz,q_seq_len,num_heads,head_dim]
        
        # 如果使用了bf16，将输出转回原始精度
        if use_bf16 and torch.cuda.is_available():
            attn_output = attn_output.to(torch.float32)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:

        attn_output, attn_weights = self._xwwx_attn_bias(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            attention_mask,
            head_mask=head_mask,
            # use_bf16=True
        )

        # 重塑输出并应用投影
        attn_output = attn_output.reshape(
            *attn_output.shape[:-2], -1).contiguous()
        # attn_output:[bsz,q_seq_len,hid_dim]
        attn_output = self.c_proj(attn_output)
        # attn_output:[bsz,q_seq_len,hid_dim]
        # 应用残差连接和dropout
        attn_output = self.resid_dropout(attn_output)
        # present 缓存的K、V
        present = None
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class GPT2AttentionOri(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        # 保存配置并获取最大位置嵌入数。
        self.config = config
        max_positions = config.max_position_embeddings
        # 注册一个下三角矩阵作为buffer，用于实现因果自注意力掩码
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
            persistent=False,
        )
        # 注册一个负无穷大作为buffer，用于实现掩码
        self.register_buffer(
            "masked_bias", torch.tensor(-1e4), persistent=False)
        # 获取隐藏维度、头数和头维度。
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 设置分割大小为隐藏维度。
        self.split_size = self.embed_dim
        # 检查头数和隐藏维度是否匹配。
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 获取是否需要缩放注意力权重。
        self.scale_attn_weights = config.scale_attn_weights
        # 获取是否为交叉注意力。
        self.is_cross_attention = is_cross_attention

        # 层级注意力缩放、重新排序和上转换
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # 获取层级索引。
        self.layer_idx = layer_idx
        # 获取是否需要重新排序和上转换。
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果为交叉注意力，则使用两块KV，否则使用三块QKV。
        # Conv1D基本上就像一个线性层，但权重是转置的。
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        # 获取注意力dropout和残差dropout。
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 设置是否为因果注意力。
        self.is_causal = True
        # 初始化一个空集合，用于存储已修剪的头。
        self.pruned_heads = set()

    # 修剪头部。（（未使用
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可修剪的头和索引。
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_heads, self.head_dim, self.pruned_heads
        )
        # 将索引连接起来。
        index_attn = torch.cat(
            [index, index + self.split_size, index + (2 * self.split_size)]
        )
        # 修剪卷积层。
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 更新超参数。
        # Update hyper params
        self.split_size = (self.split_size // self.num_heads) * (
            self.num_heads - len(heads)
        )
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    # 上转换和重新排序注意力。（（未使用
    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None
    ):
        # 使用 `torch.baddbmm`（一个更高效的alpha参数用于缩放--来自Megatron-LM）
        # 获取输入形状。bsz: batch_size, num_heads: 头数, q_seq_len: 查询序列长度, dk: 每个头维度
        bsz, num_heads, q_seq_len, dk = query.size()
        # 获取键序列长度。
        _, _, k_seq_len, _ = key.size()
        # 预分配attn_weights用于`baddbmm`
        # attn_weights = torch.empty(
        #     bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        # 计算缩放因子。
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5
        # 如果需要，按层级缩放。
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)
        # 关闭自动计算并重新排序（缩放K为1/根号(dk)）
        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        # with torch.amp.autocast(query.device.type, enabled=False):
        # q k^T
        q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(
            -1, dk, k_seq_len
        )
        # 使用baddbmm进行批量矩阵乘法，计算Q·K^T
        # attn_weights = torch.baddbmm(
        #     attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)

        attn_weights = torch.matmul(q, k)

        attn_weights = attn_weights * scale_factor

        # # 重塑回原始形状
        attn_weights = attn_weights.reshape(
            bsz, num_heads, q_seq_len, k_seq_len)
        # 对于自注意力，应用因果掩码
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 序列长度
            query_length, key_length = query.size(-2), key.size(-2)
            # 计算因果掩码
            causal_mask = self.bias[
                :, :, key_length - query_length: key_length, :key_length
            ]
            # 获取掩码值（负无穷大）
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(
                mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            # 应用掩码
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        # 应用softmax得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )

        # 转换数据类型
        attn_weights = attn_weights.type(value.dtype)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        # 应用头部掩码（如果提供）
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        # attn_weights: [bsz,num_heads,q_seq_len,k_seq_len]
        # value: [bsz,num_heads,k_seq_len,head_dim]
        # attn_output: [bsz,num_heads,q_seq_len,head_dim]
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)
        # attn_output: [bsz,q_seq_len,num_heads,head_dim]

        return attn_output, attn_weights


    # 原来的注意力计算方式，但是指定到确定的计算方式中
    def _ori_attn(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask,
        attention_mask=None,
        head_mask=None,
        use_bf16=False,
    ):
        # 如果启用bf16，转换输入
        if use_bf16 and torch.cuda.is_available():
            hidden_states = hidden_states.to(torch.bfloat16)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)
                # 使用with torch.no_grad()来避免修改requires_grad
                with torch.no_grad():
                    self.q_attn.weight.data = self.q_attn.weight.data.to(torch.bfloat16)
                    self.q_attn.bias.data = self.q_attn.bias.data.to(torch.bfloat16)
            with torch.no_grad():
                self.c_attn.weight.data = self.c_attn.weight.data.to(torch.bfloat16)
                self.c_attn.bias.data = self.c_attn.bias.data.to(torch.bfloat16)
                
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(
                self.split_size, dim=2
            )
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )

        # query_states:[bsz,q_seq_len,hid_dim]
        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        # query_states:[[bsz,num_heads,q_seq_len,head_dim]
        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        bsz, num_heads, q_seq_len, dk = query_states.size()
        _, _, k_seq_len, _ = key_states.size()
        
        # 计算缩放因子。
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value_states.size(-1)) ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # q k^T
        q, k = query_states.reshape(-1, q_seq_len, dk), key_states.transpose(
            -1, -2
        ).reshape(-1, dk, k_seq_len)
        
        # 使用bf16精度进行矩阵乘法
        attn_weights = torch.matmul(q, k)
        attn_weights = attn_weights * scale_factor
        
        # 如果使用了bf16，将权重转回float32用于后续处理
        if use_bf16 and torch.cuda.is_available():
            attn_weights = attn_weights.to(torch.float32)

        # 重塑回原始形状
        attn_weights = attn_weights.reshape(
            bsz, num_heads, q_seq_len, k_seq_len)
            
        # 对于自注意力，应用因果掩码
        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            # 序列长度
            query_length, key_length = query_states.size(
                -2), key_states.size(-2)
            # 计算因果掩码
            causal_mask = self.bias[
                :, :, key_length - query_length: key_length, :key_length
            ]
            # 获取掩码值（负无穷大）
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(
                mask_value, dtype=attn_weights.dtype, device=attn_weights.device
            )
            # 应用掩码
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        # 应用额外的注意力掩码（如果提供）
        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask
        # 应用softmax得到注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )

        # 转换数据类型
        attn_weights = attn_weights.type(value_states.dtype)
        # 应用dropout
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算注意力输出
        # attn_weights: [bsz,num_heads,q_seq_len,k_seq_len]
        # value: [bsz,num_heads,k_seq_len,head_dim]
        # attn_output: [bsz,num_heads,q_seq_len,head_dim]
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2)
        # attn_output: [bsz,q_seq_len,num_heads,head_dim]
        
        # 如果使用了bf16，将输出转回原始精度
        if use_bf16 and torch.cuda.is_available():
            attn_output = attn_output.to(torch.float32)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:


        attn_output, attn_weights = self._ori_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            attention_mask,
            head_mask=head_mask,
            # use_bf16=True
        )

        # 重塑输出并应用投影
        attn_output = attn_output.reshape(
            *attn_output.shape[:-2], -1).contiguous()
        # attn_output:[bsz,q_seq_len,hid_dim]
        attn_output = self.c_proj(attn_output)
        # attn_output:[bsz,q_seq_len,hid_dim]
        # 应用残差连接和dropout
        attn_output = self.resid_dropout(attn_output)
        # present 缓存的K、V
        present = None
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

