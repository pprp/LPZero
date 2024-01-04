import torch
import torch.nn as nn
import torch.nn.functional as F

class PKD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, s):
        t = self.norm(t)
        s = self.norm(s)
        loss = F.mse_loss(t, s)
        return loss

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        """Normalize the feature maps to have zero mean and unit variances.

        Args:
          feat (torch.Tensor): The original feature map with shape
            (B, N, C).
        """
        assert len(feat.shape) == 3
        B, N, C = feat.shape

        feat = feat.transpose(-2, -1) # [B, C, N]
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)

        return feat.transpose(-2, -1) # [B, N, C]