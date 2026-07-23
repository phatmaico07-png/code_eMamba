"""
mamba_model.py — 1-1 từ emamba_top.sv, integer arithmetic.
Đã đồng bộ PatchEmbedding Conv2D và Residual Head phần cứng.
"""
import torch
import torch.nn as nn
from .mamba_block import MambaBlock, RangeNorm
from .ssm import RTLLinear, RTLConv2D, fq, rtl_relu, FRAC, SCALE

# =====================================================================
# PATCH EMBEDDING (SỬ DỤNG RTLConv2D)
# =====================================================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=2, in_channels=5, d_model=20):
        super().__init__()
        self.patch_size = patch_size
        
        self.proj = RTLConv2D(in_channels=in_channels, 
                              out_channels=d_model, 
                              kernel_size=patch_size, 
                              stride=patch_size, 
                              padding=0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        
        B, C, H_out, W_out = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H_out * W_out, C)
        return x

# =====================================================================
# POOLING
# =====================================================================
class RTLMeanPool(nn.Module):
    def __init__(self, seq_len=16):
        super().__init__()
        self.seq_len = seq_len
        self.inv_seq = 32768 // seq_len

    def forward(self, x):
        # 1. HARD PATH
        x_int = (x * SCALE).round().clamp(-128, 127)
        acc   = x_int.sum(dim=1)
        mult  = acc * self.inv_seq
        pool  = ((mult + (1<<14)).div(1<<15, rounding_mode='floor').clamp(-128, 127))
        y_hard = pool.float() / SCALE
        
        # 2. SOFT PATH
        y_soft = fq(x.mean(dim=1), SCALE)
        
        # 3. KẾT HỢP STE
        return y_hard.detach() + (y_soft - y_soft.detach())

# =====================================================================
# RESIDUAL HEAD
# =====================================================================
class RTLResidualHead(nn.Module):
    def __init__(self, d_model=20, hidden_dim=64, output_dim=57):
        super().__init__()
        self.ffn_in  = RTLLinear(d_model, hidden_dim, has_bias=True)
        self.ffn_out = RTLLinear(hidden_dim, d_model, has_bias=True)
        self.proj    = RTLLinear(d_model, output_dim, has_bias=True)

    def forward(self, x):
        x_res = (x * SCALE).round().clamp(-128, 127)
        
        h = fq(rtl_relu(self.ffn_in(x)))
        h_out = self.ffn_out(h)
        
        h_out_i = (h_out * SCALE).round().clamp(-128, 127)
        x_add_i = (x_res + h_out_i).clamp(-128, 127)
        x_add_hard = x_add_i.float() / SCALE
        
        x_add_soft = x + h_out
        x_residual_exact = x_add_hard.detach() + (x_add_soft - x_add_soft.detach())
        
        out = self.proj(x_residual_exact)
        return out

# =====================================================================
# MAIN MODEL
# =====================================================================
class EMambaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding = PatchEmbedding(cfg.patch_size, cfg.in_channels, cfg.d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=cfg.d_model, d_inner=cfg.d_inner,
                       d_state=cfg.d_state, d_conv=cfg.d_conv,
                       norm_type=cfg.norm_type, softplus_type=cfg.softplus_type,
                       silu_type=cfg.silu_type, dt_rank=cfg.dt_rank)
            for _ in range(cfg.n_mamba_blocks)
        ])
        self.pool  = RTLMeanPool(seq_len=cfg.seq_len)
        self.norm  = RangeNorm(cfg.d_model)
        self.head  = RTLResidualHead(d_model=cfg.d_model, hidden_dim=64, output_dim=cfg.output_dim)

    def forward(self, x):
        x = self.embedding(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        x = self.norm(x)
        x = self.head(x)
        return x

    def summary(self):
        n = sum(p.numel() for p in self.parameters())
        print("="*58)
        print("eMamba — Integer-exact, 1-1 với RTL (Đã Update Conv2D + Residual)")
        print("="*58)
        print(f"  d_model={self.cfg.d_model}  d_inner={self.cfg.d_inner}")
        print(f"  d_state={self.cfg.d_state}  d_conv={self.cfg.d_conv}")
        print(f"  dt_rank={self.cfg.dt_rank}  n_blocks={self.cfg.n_mamba_blocks}")
        print(f"  seq_len={self.cfg.seq_len}  patch_dim={self.cfg.patch_dim}")
        print(f"  output_dim={self.cfg.output_dim}")
        print(f"  params: {n:,}")
        print("="*58)

def build_model(cfg):
    return EMambaModel(cfg)