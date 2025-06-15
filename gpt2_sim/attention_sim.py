import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
import torch
from collections import defaultdict, OrderedDict
from memory_sim import SRAMOverflowError, DDROverflowError, _HardwareSimulator

# you should first init it
global_hardware_simulator = _HardwareSimulator()
global_hardware_simulator.initialize()

'''
attention 包含三个子函数，自注意力乘法，合并头，分解头
init 包含一个读权重
'''


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        # when used, the weight should be loaded to the SRAM
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Conv1D_Sim(nn.Module):
    def __init__(self, nf, nx, hw_sim):
        super().__init__()
        self.hw = hw_sim
        self.nf = nf

        # 尝试加载权重到SRAM（可能抛出异常）
        try:
            w = torch.randn(nx, nf) * 0.02
            b = torch.zeros(nf)
            self.hw.sram_load(f"conv1d_w_{nx}x{nf}", w)
            self.hw.sram_load(f"conv1d_b_{nf}", b)
        except SRAMOverflowError as e:
            print(f"Failed to initialize Conv1D: {e}")
            raise

    def forward(self, x):
        try:
            w = self.hw.sram_read(f"conv1d_w_{x.size(-1)}x{self.nf}")
            b = self.hw.sram_read(f"conv1d_b_{self.nf}")
            return x @ w + b
        except KeyError as e:
            print(f"SRAM cache miss: {e}")
            raise


class Attention_Sim(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        '''nx: input token, n_ctx: token length'''
        super(Attention_Sim, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        # should store in global SRAM
        self.register_buffer("bias", torch.tril(
            torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
