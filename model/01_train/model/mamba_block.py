"""eMamba Mamba block (Mamba1 / eMamba-style selective state space, FP32).

Bias policy (theo yêu cầu): **bias BẬT ở mọi lớp TRỪ B_proj và C_proj**.
  - in_proj, conv1d, dt_in, dt_proj, out_proj : bias = True
  - B_proj, C_proj (SSM input-dependent B, C)  : bias = False
  - RangeNorm β cung cấp shift cho normalization.

Khác: RangeNorm thay LayerNorm; Δ = dt_act(dt_proj(dt_in(x))) (dt_act="relu" mặc
định = eMamba, thay Softplus); A đầy đủ (d_inner, d_state) với |A| ≤ a_clamp.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .range_norm import RangeNorm


class MambaBlock(nn.Module):
    def __init__(self, d_model=20, d_state=8, expand=2, d_conv=4, dt_rank=None,
                 dt_act="relu", a_clamp=4.0, conv_silu=True):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model            # E*D = 40 for MARS
        self.d_state = d_state                     # N = 8
        self.dt_rank = dt_rank or max(1, math.ceil(self.d_inner / 16))
        self.dt_act = dt_act
        self.a_clamp = a_clamp
        self.conv_silu = conv_silu                 # False -> BỎ SiLU sau conv1d
        self.use_pwl = False                       # True -> dùng PWL SiLU/exp khi eval

        self.norm = RangeNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=True)      # bias ON
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)  # bias ON
        # SSM input-dependent projections
        self.dt_in = nn.Linear(self.d_inner, self.dt_rank, bias=True)       # dt branch, bias ON
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)     # B: bias OFF
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)     # C: bias OFF
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)     # bias ON

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)         # bias ON

    def _delta(self, raw):
        if self.dt_act == "relu":
            return F.relu(raw)
        if self.dt_act == "softplus":
            return F.softplus(raw)
        raise ValueError("unknown dt_act: " + str(self.dt_act))

    def _silu(self, x):
        if self.use_pwl and not self.training:
            from .approximations import pwl_silu
            return pwl_silu(x)
        return F.silu(x)

    def _exp(self, x):
        if self.use_pwl and not self.training:
            from .approximations import pwl_exp
            return pwl_exp(x)
        return torch.exp(x)

    def forward(self, x):                          # x: (B, L, d_model)
        res = x
        x = self.norm(x)
        B, L, _ = x.shape

        xz = self.in_proj(x)
        xx, z = xz.chunk(2, dim=-1)                # each (B, L, d_inner)

        xx = self.conv1d(xx.transpose(1, 2))[..., :L].transpose(1, 2)
        if self.conv_silu:
            xx = self._silu(xx)                    # SiLU after Conv1d (BỎ khi conv_silu=False)

        delta = self._delta(self.dt_proj(self.dt_in(xx)))   # (B, L, d_inner)
        Bm = self.B_proj(xx)                                # (B, L, N)
        Cm = self.C_proj(xx)                                # (B, L, N)

        A = -torch.exp(self.A_log.clamp(max=math.log(self.a_clamp)))    # (d_inner, d_state), |A|<=a_clamp
        deltaA = self._exp(delta.unsqueeze(-1) * A)                     # (B,L,d_inner,N) PWL khi eval
        deltaBx = delta.unsqueeze(-1) * Bm.unsqueeze(2) * xx.unsqueeze(-1)  # (B,L,d_inner,N)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaBx[:, t]                        # (B, d_inner, N)
            ys.append(torch.einsum('bdn,bn->bd', h, Cm[:, t]))
        y = torch.stack(ys, dim=1) + xx * self.D

        y = y * self._silu(z)                      # gate (PWL khi eval)
        return self.out_proj(y) + res
