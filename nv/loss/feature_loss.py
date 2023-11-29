import torch
import torch.nn as nn

from nv.logger import logger


class FeatureLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        Compute feature discriminator loss (for MSD and MPD)
        """
        super().__init__(**kwargs)
        self.l1 = nn.L1Loss()

    def __call__(self, outputs_real, outputs_gen):
        feature_loss = 0
        for output_real, output_gen in zip(outputs_real, outputs_gen):
            for out_real, out_gen in zip(output_real, output_gen):
                feature_loss += self.l1(out_gen, out_real)
        return feature_loss
