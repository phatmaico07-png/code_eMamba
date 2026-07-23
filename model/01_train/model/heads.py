"""Output heads for eMamba-MARS.

KinematicHead (proposed, balanced accuracy + HW):
  temporal aggregation -> MLP bottleneck (+ReLU) -> decode (root + bone vectors)
  -> kinematic integrate (cumulative add along the skeleton tree).
BaselineHead (paper-style): a single linear projection (for ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .skeleton import N_JOINTS, ROOT, PARENT, TOPO


class KinematicHead(nn.Module):
    def __init__(self, d_model=20, hidden=32, n_joints=N_JOINTS,
                 parent=PARENT, topo=TOPO):
        super().__init__()
        self.n_joints = n_joints
        self.parent = list(parent)
        self.topo = list(topo)
        # (1) temporal aggregation: depthwise conv over the token axis, then mean-pool
        self.agg = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        # (2) residual MLP trunk (FFN): Linear(D->H) -> ReLU -> Linear(H->D) -> + residual
        self.ffn_in = nn.Linear(d_model, hidden)
        self.ffn_out = nn.Linear(hidden, d_model)
        # (3) decode: root (3) + 18 bone vectors (18*3) = 57
        self.decode = nn.Linear(d_model, n_joints * 3)

    def forward(self, feats):                      # feats: (B, L, D)
        x = self.agg(feats.transpose(1, 2)).mean(dim=-1)   # (B, D)
        x = x + self.ffn_out(F.relu(self.ffn_in(x)))       # residual MLP trunk (+residual)
        raw = self.decode(x).view(-1, self.n_joints, 3)    # (B, N, 3): root + bones

        # (4) kinematic integrate: pos[j] = pos[parent[j]] + bone[j], in topo order
        pos = [None] * self.n_joints
        pos[ROOT] = raw[:, ROOT]
        for j in self.topo:
            if j == ROOT:
                continue
            pos[j] = pos[self.parent[j]] + raw[:, j]
        pos = torch.stack(pos, dim=1)              # (B, N, 3)
        return pos.reshape(pos.size(0), -1), pos   # (B,57), (B,N,3)


class ResidualMLPHead(nn.Module):
    """Residual MLP head (per the architecture diagram).

      mean-pool over L  ->  x + FFN_out(ReLU(FFN_in(x)))  ->  Linear(D->57)
    FFN_in: Linear(D->h), FFN_out: Linear(h->D), Proj: Linear(D->57). h=2*D=40.
    (Deviation from the paper's single Linear head.)
    """
    def __init__(self, d_model=20, hidden=40, n_joints=N_JOINTS):
        super().__init__()
        self.ffn_in = nn.Linear(d_model, hidden)
        self.ffn_out = nn.Linear(hidden, d_model)
        self.proj = nn.Linear(d_model, n_joints * 3)

    def forward(self, feats):                      # (B, L, D)
        x = feats.mean(dim=1)                      # mean pool over L -> (B, D)
        x = x + self.ffn_out(F.relu(self.ffn_in(x)))   # residual MLP
        out = self.proj(x)                         # (B, 57)
        return out, out.view(out.size(0), -1, 3)


class BaselineHead(nn.Module):
    """Paper-style single linear projection. pool=True: mean over tokens then Linear(D->57).
    pool=False: flatten (L*D->57)."""
    def __init__(self, d_model=20, seq_len=16, n_joints=N_JOINTS, pool=True):
        super().__init__()
        self.pool = pool
        in_dim = d_model if pool else d_model * seq_len
        self.proj = nn.Linear(in_dim, n_joints * 3)

    def forward(self, feats):                      # (B, L, D)
        x = feats.mean(dim=1) if self.pool else feats.flatten(1)
        out = self.proj(x)
        return out, out.view(out.size(0), -1, 3)
