import torch
import torch.nn as nn

from nv.logger import logger


class GeneratorLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Compute generator loss
        L = (D(G(s)) - 1)^2
        """
        super().__init__(**kwargs)

    def __call__(self, d_gen):
        gen_loss = 0
        for dg in d_gen:
            gen_loss += torch.mean((dg - 1)**2)
        return gen_loss
