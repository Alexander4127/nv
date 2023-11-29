import torch
import torch.nn as nn

from .generator import Generator
from .period_discriminator import MultiPeriodDiscriminant
from .scale_discriminator import MultiScaleDiscriminator


class HiFiGANModel(nn.Module):
    def __init__(self, mpd_params, msd_params, generator_params):
        super().__init__()
        self.mpd_discriminator = MultiPeriodDiscriminant(**mpd_params)
        self.msd_discriminator = MultiScaleDiscriminator(**msd_params)
        self.generator = Generator(**generator_params)

    def __str__(self):
        return self.mpd_discriminator.__str__() + "\n\n\n" + self.msd_discriminator.__str__() + "\n\n\n" + \
            self.generator.__str__()

    def forward(self):
        raise NotImplementedError("Cannot forward, please, use MSD/MPD/Generator components directly instead")
