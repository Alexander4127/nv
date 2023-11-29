import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from typing import Tuple, Iterable, Dict, Union

from nv.base.base_model import BaseModel
from .utils import conv_2d_layer


class SubMPD(nn.Module):
    def __init__(self,
                 period: int,
                 num_channels: int = 32,
                 kernel_size: Tuple[int, int] = (5, 1),
                 stride: Tuple[int, int] = (3, 1),
                 head_channels: int = 1024,
                 head_kernel: Tuple[int, int] = (3, 1)):
        super().__init__()
        assert kernel_size[-1] == 1, f'{kernel_size} expected to have second dimension = 1'
        assert head_kernel[-1] == 1, f'{head_kernel} expected to have second dimension = 1'

        self.p = period

        self.conv_2d_layers = nn.ModuleList([
            conv_2d_layer(1, num_channels, kernel_size, stride),
            conv_2d_layer(num_channels, num_channels * 2, kernel_size, stride),
            conv_2d_layer(num_channels * 2, num_channels * 4, kernel_size, stride),
            conv_2d_layer(num_channels * 4, num_channels * 8, kernel_size, stride),
            conv_2d_layer(num_channels * 8, num_channels * 16, kernel_size, stride),
            conv_2d_layer(num_channels * 16, num_channels * 32, kernel_size, stride),
            conv_2d_layer(num_channels * 32, num_channels * 32, kernel_size, stride)
        ])

        self.head = conv_2d_layer(head_channels, 1, head_kernel)

    def forward(self, wave):
        assert len(wave.shape) == 3, f'Expected len({wave.shape}) == 3'
        B, C, T = wave.shape
        rem = self.p - T % self.p if T % self.p > 0 else 0
        T_padded = T + rem
        assert T_padded % self.p == 0
        wave = F.pad(wave, [0, rem], "reflect").reshape([B, C, T_padded // self.p, self.p])

        out = wave
        out_layers = []
        for layer in self.conv_2d_layers:
            out = F.leaky_relu(layer(out))
            out_layers.append(out)

        out = self.head(out)
        out_layers.append(out)

        return out, out_layers


class MultiPeriodDiscriminant(BaseModel):
    def __init__(self, sub_params, periods: Iterable[int] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminants = nn.ModuleList([SubMPD(period=p, **sub_params) for p in periods])

    def forward(self, wave, wave_gen):
        res_real, res_gen, outputs_real, outputs_gen = [], [], [], []
        for sub_disc in self.discriminants:
            real, out_real = sub_disc(wave)
            res_real.append(real)
            outputs_real.append(out_real)

            gen, out_gen = sub_disc(wave_gen)
            res_gen.append(gen)
            outputs_gen.append(out_gen)

        assert len(res_real) == len(res_gen) == len(outputs_real) == len(outputs_gen)

        return res_real, res_gen, outputs_real, outputs_gen
