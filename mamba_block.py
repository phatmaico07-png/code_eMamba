import torch
import torch.nn as nn
from .ssm import (RTLLinear, RTLConv1D, RTLRangeNorm, RTLSiLU,
                  SSM, fq, rtl_relu, FRAC, SCALE)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, d_state,
                 d_conv=4, norm_type="range_norm",
                 softplus_type="relu", silu_type="silu", dt_rank=3):
        super().__init__()
        self.d_model  = d_model
        self.d_inner  = d_inner
        self.norm     = RTLRangeNorm(d_model)
        self.in_proj  = RTLLinear(d_model, 2*d_inner, has_bias=False)
        self.silu     = RTLSiLU()
        self.conv1d   = RTLConv1D(d_inner, kernel=d_conv)
        self.ssm      = SSM(d_inner=d_inner, n_state=d_state, dt_rank=dt_rank)
        self.out_proj = RTLLinear(d_inner, d_model, has_bias=False)

    def forward(self, x):
        # ==========================================
        # 1. HARD PATH Latch residual (integer)
        # ==========================================
        x_res  = (x * SCALE).round().clamp(-128, 127)

        x_norm = self.norm(x)
        x_exp  = self.in_proj(x_norm)
        x1     = x_exp[..., :self.d_inner]
        z      = x_exp[..., self.d_inner:]

        z_act   = self.silu(z)
        x1_conv = self.conv1d(x1)
        y_ssm   = self.ssm(x1_conv)

        # ==========================================
        # 2. GATING (Đã gắn STE wrapper)
        # ==========================================
        # Hard path Gating
        y_i = (y_ssm  * SCALE).round().clamp(-128, 127)
        z_i = (z_act  * SCALE).round().clamp(-128, 127)
        mult = y_i * z_i                               
        half = 1 << (FRAC - 1)                        
        gated = ((mult + half).div(SCALE, rounding_mode='floor').clamp(-128, 127))
        y_gated_hard = gated.float() / SCALE
        
        # Soft path Gating
        y_gated_soft = fq(y_ssm * z_act, SCALE)
        
        # Gating STE
        y_gated = y_gated_hard.detach() + (y_gated_soft - y_gated_soft.detach())

        y_proj = self.out_proj(y_gated)

        # ==========================================
        # 3. RESIDUAL ADD (Đã gắn STE wrapper)
        # ==========================================
        # Hard path Residual
        yp_i = (y_proj * SCALE).round().clamp(-128, 127)
        y_hard = (yp_i + x_res).clamp(-128, 127).float() / SCALE
        
        # Soft path Residual
        y_soft = fq(y_proj + x, SCALE)

        # Residual STE
        y = y_hard.detach() + (y_soft - y_soft.detach())

        return y

RangeNorm = RTLRangeNorm