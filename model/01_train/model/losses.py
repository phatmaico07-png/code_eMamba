"""Losses & metrics for eMamba-MARS pose regression (joint-major layout).

  total = pose_loss + lambda_sym·bone_symmetry + lambda_bonelen·bone_length
pose_loss: l1 (default) / mse / smooth_l1, optional per-coordinate weight.
bone_length: |‖pred_bone‖ − ‖gt_bone‖|  — ép độ dài xương về GT (chỉ ý nghĩa
             với head kinematic; chặn lỗi tích luỹ ở khớp xa → giảm RMSE).
bone_symmetry: |‖bone_L‖ − ‖bone_R‖| — đối xứng trái/phải.
"""
import torch
import torch.nn.functional as F

from .skeleton import PARENT, SYM_PAIRS, N_JOINTS


def _elt(pred, tgt, kind):
    if kind == "l1":
        return (pred - tgt).abs()
    if kind == "mse":
        return (pred - tgt) ** 2
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, tgt, reduction="none")
    raise ValueError("unknown loss kind: " + str(kind))


def pose_loss(pred57, tgt57, kind="l1", weight=None):
    e = _elt(pred57, tgt57, kind)          # (B, 57)
    if weight is not None:
        e = e * weight                      # (57,) per-coordinate weight
    return e.mean()


def bone_symmetry_loss(pos):                # pos: (B, N, 3)
    total = pos.new_zeros(())
    for (l, r) in SYM_PAIRS:
        bl = (pos[:, l] - pos[:, PARENT[l]]).norm(dim=-1)
        br = (pos[:, r] - pos[:, PARENT[r]]).norm(dim=-1)
        total = total + (bl - br).abs().mean()
    return total / len(SYM_PAIRS)


def bone_length_loss(pred_pos, gt_pos):     # (B, N, 3) both
    total = pred_pos.new_zeros(())
    cnt = 0
    for j in range(N_JOINTS):
        p = PARENT[j]
        if p < 0:
            continue
        bl = (pred_pos[:, j] - pred_pos[:, p]).norm(dim=-1)
        gl = (gt_pos[:, j] - gt_pos[:, p]).norm(dim=-1)
        total = total + (bl - gl).abs().mean()
        cnt += 1
    return total / cnt


def total_loss(pred57, pos, tgt57, lambda_sym=0.05, kind="l1",
               weight=None, lambda_bonelen=0.0):
    p = pose_loss(pred57, tgt57, kind, weight)
    sym = bone_symmetry_loss(pos)
    total = p + lambda_sym * sym
    logs = {"pose": p.item(), "sym": sym.item()}
    if lambda_bonelen > 0:
        gt_pos = tgt57.view(tgt57.size(0), N_JOINTS, 3)
        bl = bone_length_loss(pos, gt_pos)
        total = total + lambda_bonelen * bl
        logs["bonelen"] = bl.item()
    return total, logs


@torch.no_grad()
def mae_rmse(pred57, tgt57, n_joints=19):
    """Overall MAE and RMSE over all coordinates (same units as labels, e.g. cm)."""
    d = pred57 - tgt57
    return d.abs().mean().item(), d.pow(2).mean().sqrt().item()


@torch.no_grad()
def mpjpe(pred57, tgt57, n_joints=19):
    """Mean Per Joint Position Error: trung bình khoảng cách Euclid 3D mỗi khớp."""
    d = (pred57 - tgt57).view(-1, n_joints, 3)
    return d.pow(2).sum(-1).sqrt().mean().item()


@torch.no_grad()
def per_axis_mae(pred57, tgt57, n_joints=19):
    d = (pred57 - tgt57).view(-1, n_joints, 3).abs().mean(dim=(0, 1))
    return {"x": d[0].item(), "y": d[1].item(), "z": d[2].item()}
