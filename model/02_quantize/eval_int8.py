"""Đo INT8 vs FP32 (và PWL) trên tập test — không train lại.

Bảng: FP32 | PWL | INT8 | INT8+pow2(requant=shift), kèm MAE/RMSE/MPJPE/per-axis (cm).
Nếu RMSE(INT8) − RMSE(FP32) > ngưỡng (--qat-th, mặc định 0.5 cm) → khuyến nghị QAT.

    python calibrate.py --ckpt results/best.pt --hidden 64 --out results/calib_int8.pt
    python eval_int8.py  --ckpt results/best.pt --hidden 64 --calib results/calib_int8.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader

from model.emamba_mars import EMambaMARS
from model.dataset import MARSDataset
from model.losses import mae_rmse, mpjpe, per_axis_mae
from quant import QEMambaMARS, QUANT


@torch.no_grad()
def run(predict, loader, x_mean, x_std, y_mean, y_std, dev):
    P, T = [], []
    for X, Y in loader:
        out = predict((X.to(dev) - x_mean) / x_std)
        P.append(out * y_std + y_mean)
        T.append(Y.to(dev))
    pred, tgt = torch.cat(P), torch.cat(T)
    mae, rmse = mae_rmse(pred, tgt)
    d = (pred - tgt).reshape(len(pred), 19, 3)            # (N,19,3) tách trục
    amae  = {k: d[..., i].abs().mean().item()           for i, k in enumerate("xyz")}
    armse = {k: (d[..., i] ** 2).mean().sqrt().item()   for i, k in enumerate("xyz")}
    return mae, rmse, mpjpe(pred, tgt), amae, armse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature", default="data/raw/MARS/feature/featuremap_test.npy")
    ap.add_argument("--label", default="data/raw/MARS/feature/labels_test.npy")
    ap.add_argument("--train-feature", default="data/raw/MARS/feature/featuremap_train.npy")
    ap.add_argument("--train-label", default="data/raw/MARS/feature/labels_train.npy")
    ap.add_argument("--ckpt", default="results/best.pt")
    ap.add_argument("--calib", default="results/calib_int8.pt")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--qat-th", type=float, default=0.5, help="ngưỡng RMSE drop (cm) → gợi ý QAT")
    ap.add_argument("--w-per-tensor", action="store_true",
                    help="weight 1 scale CHUNG (per-tensor) thay per-channel — HW gọn hơn")
    ap.add_argument("--w-pow2", action="store_true",
                    help="ép scale weight về pow2 → kết hợp pow2_act thì requant CHỈ DỊCH (không nhân)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.calib, map_location=dev)
    cfg = ck["cfg"]
    tr = MARSDataset(a.train_feature, a.train_label)
    te = MARSDataset(a.feature, a.label)
    x_mean = tr.X.reshape(-1, 5).mean(0).to(dev)
    x_std = (tr.X.reshape(-1, 5).std(0) + 1e-8).to(dev)
    y_mean = tr.Y.mean().item(); y_std = (tr.Y.std() + 1e-8).item()
    loader = DataLoader(te, batch_size=a.batch)

    fp = EMambaMARS(d_model=cfg["d_model"], d_state=cfg["d_state"], expand=cfg["expand"],
                    n_blocks=cfg["blocks"], head=cfg["head"], hidden=a.hidden,
                    dt_act=cfg["dt_act"]).to(dev)
    fp.load_state_dict(torch.load(a.ckpt, map_location=dev)); fp.eval()

    rows = {}
    fp.set_pwl(False)
    rows["FP32"] = run(lambda x: fp(x)[0], loader, x_mean, x_std, y_mean, y_std, dev)
    fp.set_pwl(True)
    rows["PWL"] = run(lambda x: fp(x)[0], loader, x_mean, x_std, y_mean, y_std, dev)
    fp.set_pwl(False)

    for tag, p2 in [("INT8", False), ("INT8+pow2", True)]:
        q = QEMambaMARS(fp, w_bits=cfg["w_bits"], a_bits=cfg["a_bits"],
                        state_bits=cfg["state_bits"], pow2_act=p2).to(dev)
        q.load_calib(ck["calib"]); q.eval(); q.set_mode(QUANT)
        if a.w_per_tensor:
            q.set_w_per_tensor(True)
        if a.w_pow2:
            q.set_w_pow2(True)
        if cfg.get("da_scale", 0) > 0:
            q.set_da_scale(cfg["da_scale"])
        rows[tag] = run(lambda x: q(x)[0], loader, x_mean, x_std, y_mean, y_std, dev)

    cm = cfg.get("method", "minmax")
    cm += f" p{cfg.get('pct', 99.9)}" if cm == "percentile" else ""
    wmode = "per-tensor" if a.w_per_tensor else "per-channel"
    das = cfg.get("da_scale", 0)
    dastr = f"  dA=2^{round(__import__('math').log2(das))}" if das > 0 else ""
    print(f"\n  config: INT{cfg['w_bits']}/{cfg['a_bits']}  state INT{cfg['state_bits']}{dastr}  "
          f"calib={cm}  (weight {wmode}, act per-tensor, bias INT32)\n")
    print(f"  {'variant':11s} {'MAE':>6s} {'RMSE':>6s} {'MPJPE':>6s}    "
          f"{'MAEx':>5s} {'MAEy':>5s} {'MAEz':>5s}    {'RMSx':>5s} {'RMSy':>5s} {'RMSz':>5s}   {'ΔRMSE':>6s}")
    print("  " + "-" * 94)
    base = rows["FP32"][1]
    for k, (mae, rmse, mp, am, ar) in rows.items():
        d = "" if k == "FP32" else f"{rmse - base:+.3f}"
        print(f"  {k:11s} {mae:6.3f} {rmse:6.3f} {mp:6.3f}    "
              f"{am['x']:5.2f} {am['y']:5.2f} {am['z']:5.2f}    "
              f"{ar['x']:5.2f} {ar['y']:5.2f} {ar['z']:5.2f}   {d:>6s}")
    print("  " + "-" * 94 + "  (cm)")

    drop = rows["INT8"][1] - base
    if drop > a.qat_th:
        print(f"\n  → INT8 RMSE drop {drop:+.3f} cm > {a.qat_th} → NÊN chạy QAT fine-tune.")
    else:
        print(f"\n  → INT8 RMSE drop {drop:+.3f} cm ≤ {a.qat_th} → PTQ ĐỦ, không cần QAT.")


if __name__ == "__main__":
    main()
