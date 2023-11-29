import torch
import torch.nn as nn

from nv.logger import logger


class MelLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Compute mel-spec loss for generator
        L = ||Phi(x) - Phi(G(s))||_1
        Phi - makes mel-spectrogram from waveform
        """
        super().__init__(**kwargs)
        self.l1 = nn.L1Loss()

    def __call__(self, real_spec, gen_spec):
        return self.l1(gen_spec, real_spec)
