"""Full eMamba model for MARS pose estimation (FP32 training version).

Pipeline:  (B,8,8,5) --patchify P=2--> (B,16,20) --M Mamba blocks--> (B,16,20)
           --RangeNorm--> Output Head --> (B,57)  [19 joints x 3]
"""
import torch
import torch.nn as nn

from .mamba_block import MambaBlock
from .range_norm import RangeNorm
from .heads import KinematicHead, BaselineHead, ResidualMLPHead


class PatchEmbed(nn.Module):
    """(B,H,W,C) -> (B, (H/p)*(W/p), C*p*p) tokens, then a linear embedding.
    For MARS: (B,8,8,5), p=2 -> 16 tokens of dim 2*2*5 = 20 = D."""
    def __init__(self, patch=2, in_ch=5, d_model=20):
        super().__init__()
        self.p = patch
        self.embed = nn.Linear(patch * patch * in_ch, d_model)

    def forward(self, x):                          # (B,H,W,C)
        B, H, W, C = x.shape
        p = self.p
        x = x.permute(0, 3, 1, 2)                  # (B,C,H,W)
        x = x.unfold(2, p, p).unfold(3, p, p)      # (B,C,H/p,W/p,p,p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, (H // p) * (W // p), C * p * p)
        return self.embed(x)                       # (B, L, d_model)


class EMambaMARS(nn.Module):
    def __init__(self, d_model=20, d_state=8, expand=2, n_blocks=2,
                 patch=2, in_ch=5, head="kinematic", hidden=32, seq_len=16,
                 norm_before_head=False, dt_act="relu", conv_silu=True):
        super().__init__()
        self.patch_embed = PatchEmbed(patch, in_ch, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state, expand, dt_act=dt_act, conv_silu=conv_silu)
             for _ in range(n_blocks)]
        )
        # a norm right before a regression head removes scale info -> off by default
        self.final_norm = RangeNorm(d_model) if norm_before_head else None
        if head == "kinematic":
            self.head = KinematicHead(d_model, hidden)
        elif head == "mlp":
            self.head = ResidualMLPHead(d_model, hidden)
        elif head == "baseline":
            self.head = BaselineHead(d_model, seq_len)
        else:
            raise ValueError(f"unknown head: {head}")

    def forward(self, x):                          # x: (B,8,8,5)
        h = self.patch_embed(x)                    # (B,L,D)
        for blk in self.blocks:
            h = blk(h)
        if self.final_norm is not None:
            h = self.final_norm(h)
        out57, pos = self.head(h)                  # (B,57), (B,N,3)
        return out57, pos

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_pwl(self, flag=True):
        """Bật/tắt PWL (SiLU/exp) cho mọi MambaBlock — chỉ tác dụng khi eval()."""
        for blk in self.blocks:
            blk.use_pwl = flag
