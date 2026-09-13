"""Piecewise-linear (PWL) approximations cho INFERENCE (eMamba §4.2 & §4.3).

  pwl_silu : PWL trong [-7, 7];  x < -7 -> 0 ;  x > 7 -> x
  pwl_exp  : PWL trong [-4, 1];  x < -4 -> 0 ;  x > 1 -> e (hằng)

Knot được đặt **non-uniform, error-bounded** (greedy: thêm knot tại chỗ sai số lớn
nhất tới khi mọi đoạn ≤ tol) — dày ở vùng cong, thưa ở vùng phẳng → ít đoạn mà sai
số nhỏ, ánh xạ thẳng sang LUT(breakpoint, slope, intercept) trên FPGA.

PWL CHỈ dùng khi eval (không train — đổi gradient, eMamba §4.5).
"""
import torch
import torch.nn.functional as F


def _fit_knots(f, lo, hi, tol, rel=False, max_knots=200):
    """Greedy: chèn knot tại đoạn có sai số nội suy lớn nhất tới khi ≤ tol."""
    knots = [float(lo), float(hi)]
    for _ in range(max_knots):
        worst_e, worst_i, worst_x = 0.0, -1, None
        for i in range(len(knots) - 1):
            a, b = knots[i], knots[i + 1]
            xs = torch.linspace(a, b, 33)
            ya = f(torch.tensor(a)); yb = f(torch.tensor(b))
            interp = ya + (xs - a) / (b - a) * (yb - ya)
            err = (f(xs) - interp).abs()
            if rel:
                err = err / f(xs).abs().clamp_min(1e-6)
            e = err.max().item(); j = int(err.argmax())
            if e > worst_e:
                worst_e, worst_i, worst_x = e, i, float(xs[j])
        if worst_e <= tol:
            break
        knots.insert(worst_i + 1, worst_x)
    return torch.tensor(sorted(set(knots)))


_SILU_KNOTS = _fit_knots(lambda x: F.silu(x), -7.0, 7.0, tol=0.01)
_SILU_VALS = F.silu(_SILU_KNOTS)
_EXP_KNOTS = _fit_knots(lambda x: torch.exp(x), -4.0, 1.0, tol=0.01, rel=True)
_EXP_VALS = torch.exp(_EXP_KNOTS)


def _pwl(x, knots, vals):
    knots = knots.to(device=x.device, dtype=x.dtype)
    vals = vals.to(device=x.device, dtype=x.dtype)
    lo, hi = float(knots[0]), float(knots[-1])
    xc = x.clamp(lo, hi).contiguous()
    idx = (torch.searchsorted(knots, xc, right=True) - 1).clamp(0, knots.numel() - 2)
    x0 = knots[idx]; x1 = knots[idx + 1]
    y0 = vals[idx]; y1 = vals[idx + 1]
    return y0 + (xc - x0) / (x1 - x0) * (y1 - y0)


def pwl_silu(x):
    y = _pwl(x, _SILU_KNOTS, _SILU_VALS)
    y = torch.where(x < -7.0, torch.zeros_like(x), y)
    y = torch.where(x > 7.0, x, y)
    return y


def pwl_exp(x):
    y = _pwl(x, _EXP_KNOTS, _EXP_VALS)
    y = torch.where(x < -4.0, torch.zeros_like(x), y)
    return y


def pwl_info():
    return {"silu_segments": _SILU_KNOTS.numel() - 1, "exp_segments": _EXP_KNOTS.numel() - 1}
