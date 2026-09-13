"""Train eMamba-MARS.

Smoke-train on synthetic data (no real data needed):
    python train.py --synthetic --epochs 30

Train on real MARS data:
    python train.py --feature path/to/train_feature.npy --label path/to/train_label.npy \
                    --val-feature path/to/val_feature.npy --val-label path/to/val_label.npy \
                    --epochs 100 --batch 64

Targets are standardized (scalar mean/std from the train set) for stable
optimization; MAE/RMSE are reported in the ORIGINAL units (e.g. cm).
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split

from model.emamba_mars import EMambaMARS
from model.losses import total_loss, mae_rmse, per_axis_mae
from model.dataset import SyntheticMARS, MARSDataset

OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop", "nadam", "radam"]

def make_opt(params, name, lr, weight_decay=0.0, momentum=0.9):
    """Tạo optimizer theo tên (chung cho train.py và nas.py)."""
    n = name.lower()
    if n == "adam":    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "adamw":   return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay or 1e-2)
    if n == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if n == "rmsprop": return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if n == "nadam":   return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    if n == "radam":   return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("unknown optimizer: " + str(name))


def build_loaders(args):
    if args.synthetic or not args.feature:
        full = SyntheticMARS(n=args.n_synth, seed=args.seed)
        n_val = int(len(full) * args.val_frac)
        tr, va = random_split(full, [len(full) - n_val, n_val],
                              generator=torch.Generator().manual_seed(args.seed))
        tag = "SYNTHETIC"
    else:
        tr = MARSDataset(args.feature, args.label)
        if args.val_feature:
            va = MARSDataset(args.val_feature, args.val_label)
        else:
            n_val = int(len(tr) * args.val_frac)
            tr, va = random_split(tr, [len(tr) - n_val, n_val],
                                  generator=torch.Generator().manual_seed(args.seed))
        tag = "MARS"
    return (DataLoader(tr, batch_size=args.batch, shuffle=True),
            DataLoader(va, batch_size=args.batch), tag)


def target_stats(loader, device):
    """Scalar mean/std of targets (preserves skeleton geometry: uniform scale+shift)."""
    s = ss = 0.0; cnt = 0
    for _, Y in loader:
        Y = Y.to(device)
        s += Y.sum().item(); ss += (Y * Y).sum().item(); cnt += Y.numel()
    mean = s / cnt
    std = max(ss / cnt - mean * mean, 1e-8) ** 0.5
    return mean, std


def feature_stats(loader, device):
    """Per-channel (5) mean/std of the input feature map (channels differ in scale)."""
    s = torch.zeros(5); ss = torch.zeros(5); cnt = 0
    for X, _ in loader:
        Xf = X.reshape(-1, 5)
        s += Xf.sum(0); ss += (Xf * Xf).sum(0); cnt += Xf.shape[0]
    mean = s / cnt
    std = (ss / cnt - mean * mean).clamp_min(1e-8).sqrt()
    return mean.to(device), std.to(device)


def feature_p99(loader, device, pct=99.0, cap=2_000_000):
    """Per-channel (5) scale = phân vị `pct` của |X| → input chuẩn hóa = clip(X/scale, -1, 1)."""
    buf = [[] for _ in range(5)]; n = 0
    for X, _ in loader:
        Xf = X.reshape(-1, 5).abs()
        for c in range(5):
            buf[c].append(Xf[:, c])
        n += Xf.shape[0]
        if n >= cap:
            break
    scale = torch.stack([torch.cat(b).quantile(pct / 100.0) for b in buf]).clamp_min(1e-8)
    return scale.to(device)


def run_epoch(model, loader, opt, device, lambda_sym, xnorm, y_mean, y_std,
              train=True, loss_kind="l1", grad_clip=0.0, weight=None, lambda_bonelen=0.0):
    model.train(train)
    tot, n, mae_s, rmse_s = 0.0, 0, 0.0, 0.0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        X = xnorm(X)                                   # chuẩn hóa input (zscore | p99→[-1,1])
        Yn = (Y - y_mean) / y_std                      # standardized target
        if train:
            opt.zero_grad()
        pred, pos = model(X)                            # model works in standardized space
        loss, _ = total_loss(pred, pos, Yn, lambda_sym, loss_kind, weight, lambda_bonelen)
        if train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        bs = X.size(0); tot += loss.item() * bs; n += bs
        pred_cm = pred.detach() * y_std + y_mean        # de-normalize for metrics
        m, r = mae_rmse(pred_cm, Y); mae_s += m * bs; rmse_s += r * bs
    return tot / n, mae_s / n, rmse_s / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--feature"); ap.add_argument("--label")
    ap.add_argument("--val-feature"); ap.add_argument("--val-label")
    ap.add_argument("--head", default="kinematic", choices=["kinematic", "mlp", "baseline"])
    ap.add_argument("--dt-act", default="relu", choices=["softplus", "relu"])
    ap.add_argument("--norm", default="zscore", choices=["zscore", "p99"],
                    help="chuẩn hóa input: zscore (cũ) | p99 (clip phân vị -> [-1,1])")
    ap.add_argument("--pct-in", type=float, default=99.0, help="phân vị cho --norm p99")
    ap.add_argument("--no-conv-silu", action="store_true", help="BỎ SiLU sau conv1d")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=20)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--patch", type=int, default=2, help="patch size (chia hết 8); D độc lập với patch")
    ap.add_argument("--scheduler", default="none", choices=["none", "cosine", "exp", "plateau"])
    ap.add_argument("--warmup", type=int, default=0, help="linear LR warmup epochs before cosine/exp")
    ap.add_argument("--lr-gamma", type=float, default=0.95, help="ExponentialLR decay/epoch (nhỏ hơn = giảm nhanh hơn)")
    ap.add_argument("--grad-clip", type=float, default=0.0, help="max grad norm (0 = off)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--loss", default="l1", choices=["l1", "mse", "smooth_l1"])
    ap.add_argument("--select", default="mae", choices=["mae", "rmse"])
    ap.add_argument("--wrist-weight", type=float, default=1.0, help=">1: upweight elbows+wrists")
    ap.add_argument("--lambda-bonelen", type=float, default=0.0, help="bone-length-vs-GT prior weight")
    ap.add_argument("--lambda-sym", type=float, default=0.05)
    ap.add_argument("--n-synth", type=int, default=2048)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results")
    ap.add_argument("--eval-pwl", action="store_true", help="cuối train: đo val FP32 vs PWL")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    tr_loader, va_loader, tag = build_loaders(args)

    if args.norm == "p99":
        x_scale = feature_p99(tr_loader, args.device, args.pct_in)
        def xnorm(X): return (X / x_scale).clamp(-1.0, 1.0)
        print(f"[norm] p99 (pct={args.pct_in}) scale/kênh = {[round(v,4) for v in x_scale.cpu().tolist()]}")
    else:
        x_mean, x_std = feature_stats(tr_loader, args.device)
        def xnorm(X): return (X - x_mean) / x_std
    y_mean, y_std = target_stats(tr_loader, args.device)

    model = EMambaMARS(d_model=args.d_model, d_state=args.d_state, expand=args.expand,
                       n_blocks=args.blocks, patch=args.patch, head=args.head, hidden=args.hidden,
                       dt_act=args.dt_act, conv_silu=not args.no_conv_silu).to(args.device)
    opt = make_opt(model.parameters(), args.optimizer, args.lr, args.weight_decay)
    jw = None
    if args.wrist_weight != 1.0:
        w19 = torch.ones(19)
        for j in (5, 6, 8, 9):           # ElbowLeft/Right, WristLeft/Right (joint-major idx)
            w19[j] = args.wrist_weight
        jw = w19.repeat_interleave(3).to(args.device)   # (57,) joint-major per-coordinate
    sched = None
    if args.scheduler == "cosine":
        if args.warmup > 0:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warm = LinearLR(opt, start_factor=0.01, total_iters=args.warmup)
            cos = CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup), eta_min=args.lr * 0.01)
            sched = SequentialLR(opt, schedulers=[warm, cos], milestones=[args.warmup])
        else:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.scheduler == "exp":
        if args.warmup > 0:
            from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
            warm = LinearLR(opt, start_factor=0.01, total_iters=args.warmup)
            ex = ExponentialLR(opt, gamma=args.lr_gamma)
            sched = SequentialLR(opt, schedulers=[warm, ex], milestones=[args.warmup])
        else:
            sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_gamma)
    elif args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8)
    print(f"[{tag}] head={args.head} params={model.num_params():,} device={args.device} "
          f"| target mean={y_mean:.2f} std={y_std:.2f}")
    print(f"(baseline khi đoán hằng-trung-bình: MAE≈{0.8*y_std:.2f}, RMSE≈{y_std:.2f})")

    log = []; best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_mae, _ = run_epoch(model, tr_loader, opt, args.device, args.lambda_sym, xnorm, y_mean, y_std, True, args.loss, args.grad_clip, jw, args.lambda_bonelen)
        with torch.no_grad():
            va_loss, va_mae, va_rmse = run_epoch(model, va_loader, opt, args.device, args.lambda_sym, xnorm, y_mean, y_std, False, args.loss)
        if sched is not None:
            sched.step(va_mae) if args.scheduler == "plateau" else sched.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {ep:3d} | lr {lr_now:.2e} | train_loss {tr_loss:7.4f} | val_loss {va_loss:7.4f} "
              f"| val_MAE {va_mae:7.4f} | val_RMSE {va_rmse:7.4f}")
        log.append((ep, tr_loss, va_loss, va_mae, va_rmse))
        sel = va_rmse if args.select == "rmse" else va_mae
        if sel < best:
            best = sel
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))

    with open(os.path.join(args.out, "train_log.csv"), "w") as f:
        f.write("epoch,train_loss,val_loss,val_mae,val_rmse\n")
        for row in log:
            f.write(",".join(str(x) for x in row) + "\n")
    if args.eval_pwl:
        model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location=args.device))
        with torch.no_grad():
            model.set_pwl(False)
            _, m0, r0 = run_epoch(model, va_loader, opt, args.device, args.lambda_sym, xnorm, y_mean, y_std, False, args.loss)
            model.set_pwl(True)
            _, m1, r1 = run_epoch(model, va_loader, opt, args.device, args.lambda_sym, xnorm, y_mean, y_std, False, args.loss)
            model.set_pwl(False)
        print(f"[eval] FP32  val_MAE={m0:.4f}  val_RMSE={r0:.4f}")
        print(f"[eval] +PWL  val_MAE={m1:.4f}  val_RMSE={r1:.4f}  (Δ MAE {m1-m0:+.3f}, Δ RMSE {r1-r0:+.3f})")
    print(f"done. best val_{args.select.upper()}={best:.4f} (cm). checkpoint+log in '{args.out}/'.")


if __name__ == "__main__":
    main()
