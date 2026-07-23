"""PTQ calibration — thu absmax cho mọi activation rồi lưu (results/calib_int8.pt).

Chạy model (đã train) ở mode CALIB trên một ít batch TRAIN, observer ghi absmax →
scale. Không train lại, không sửa weight.

    python calibrate.py --ckpt results/best.pt --head kinematic --hidden 64 \
        --n-batches 16 --out results/calib_int8.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader

from model.emamba_mars import EMambaMARS
from model.dataset import MARSDataset
from quant import QEMambaMARS, CALIB


def add_model_args(ap):
    ap.add_argument("--ckpt", default="results/best.pt")
    ap.add_argument("--head", default="kinematic", choices=["kinematic", "mlp"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=20)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--d-state", type=int, default=8)
    ap.add_argument("--dt-act", default="relu", choices=["relu", "softplus"])
    ap.add_argument("--w-bits", type=int, default=8)
    ap.add_argument("--a-bits", type=int, default=8)
    ap.add_argument("--state-bits", type=int, default=16, help="bit cho state SSM (paper: 24)")
    ap.add_argument("--da-scale", type=float, default=0.0,
                    help="cố định scale dA = giá trị này (paper: 0.0078125 = 2^-7); 0 = calibrate")


def build(a, dev):
    tr = MARSDataset(a.train_feature, a.train_label)
    x_mean = tr.X.reshape(-1, 5).mean(0).to(dev)
    x_std = (tr.X.reshape(-1, 5).std(0) + 1e-8).to(dev)
    fp = EMambaMARS(d_model=a.d_model, d_state=a.d_state, expand=a.expand,
                    n_blocks=a.blocks, head=a.head, hidden=a.hidden, dt_act=a.dt_act).to(dev)
    fp.load_state_dict(torch.load(a.ckpt, map_location=dev)); fp.eval()
    q = QEMambaMARS(fp, w_bits=a.w_bits, a_bits=a.a_bits,
                    state_bits=a.state_bits, pow2_act=False).to(dev); q.eval()
    return tr, x_mean, x_std, q


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    ap.add_argument("--train-feature", default="data/raw/MARS/feature/featuremap_train.npy")
    ap.add_argument("--train-label", default="data/raw/MARS/feature/labels_train.npy")
    ap.add_argument("--n-batches", type=int, default=16, help="số batch train dùng calibrate")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--method", default="mse", choices=["minmax", "percentile", "mse"],
                    help="minmax | percentile | mse (dò ngưỡng tối ưu RMSE, mặc định)")
    ap.add_argument("--pct", type=float, default=99.9, help="phân vị clip (chỉ percentile)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/calib_int8.pt")
    a = ap.parse_args()
    dev = a.device

    tr, x_mean, x_std, q = build(a, dev)
    q.set_calib_method(a.method, a.pct)
    q.set_mode(CALIB)
    loader = DataLoader(tr, batch_size=a.batch, shuffle=True)
    used = 0
    with torch.no_grad():
        for i, (X, _) in enumerate(loader):
            if i >= a.n_batches:
                break
            q((X.to(dev) - x_mean) / x_std)
            used += X.size(0)
    q.finalize_calib()                     # chốt ngưỡng (mse)

    cfg = {k: getattr(a, k) for k in
           ["head", "hidden", "blocks", "d_model", "expand", "d_state",
            "dt_act", "w_bits", "a_bits", "state_bits", "method", "pct", "da_scale"]}
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"calib": q.calib_state(), "cfg": cfg}, a.out)
    m = a.method + (f" p{a.pct}" if a.method == "percentile" else "")
    print(f"calib -> {a.out} | {used} frame, {a.n_batches} batch | "
          f"INT{a.w_bits}/{a.a_bits} state{a.state_bits} | calib={m}")


if __name__ == "__main__":
    main()
