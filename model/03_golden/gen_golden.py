"""Sinh golden I/O vectors cho RTL verify bit-exact.

Chạy IntModel (golden integer) trên vài frame test → :
  hw_export/golden/input.mem      mã INT8 đầu vào (320/frame = 8×8×5), hex 2 chữ số
  hw_export/golden/golden_out.mem mã INT8 đầu ra  (57/frame = 19 khớp×3) — KỲ VỌNG của RTL
  hw_export/golden/meta.txt       layout + cách dùng

RTL nạp input.mem → chạy datapath INT → so từng số với golden_out.mem.

    python gen_golden.py --ckpt results_qat/best.pt --calib results_qat/calib_paper.pt --frames 8
"""
import argparse, os
import numpy as np
from int_reference import IntModel, clampq


def hexs(arr, nib=2):
    mask = (1 << (nib * 4)) - 1
    return "\n".join(format(int(v) & mask, f"0{nib}x") for v in np.asarray(arr).ravel().astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_qat/best.pt")
    ap.add_argument("--calib", default="results_qat/calib_paper.pt")
    ap.add_argument("--feature", default="data/raw/MARS/feature/featuremap_test.npy")
    ap.add_argument("--train-feature", default="data/raw/MARS/feature/featuremap_train.npy")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default="rtl_export/golden")   # khớp run_sim.tcl ($SIM/../golden)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = IntModel(a.ckpt, a.calib)
    fm_tr = np.load(a.train_feature).astype(np.float64)
    fm_te = np.load(a.feature).astype(np.float64)[:a.frames]
    mean = fm_tr.reshape(-1, 5).mean(0); std = fm_tr.reshape(-1, 5).std(0) + 1e-8
    Xn = (fm_te - mean) / std

    s_in = m.asc("a_in")
    q_in = clampq(Xn / s_in).astype(np.int64)              # (F,8,8,5) mã INT8 vào
    out_real = m.forward(Xn)                               # (F,57) thực (chuẩn hoá)
    s_o = m.asc("head.a_out")
    out_code = np.clip(np.round(out_real / s_o), -128, 127).astype(np.int64)  # mã INT8, clamp[-128,127] KHỚP RTL

    open(os.path.join(a.out, "input.mem"), "w", encoding="utf-8").write(hexs(q_in, 2))
    open(os.path.join(a.out, "golden_out.mem"), "w", encoding="utf-8").write(hexs(out_code, 2))
    with open(os.path.join(a.out, "meta.txt"), "w", encoding="utf-8") as f:
        f.write("# golden I/O — eMamba-MARS\n")
        f.write(f"frames        : {a.frames}\n")
        f.write(f"input  layout : {a.frames} × (8×8×5=320) mã INT8, scale 2^{round(np.log2(s_in))}\n")
        f.write(f"output layout : {a.frames} × (19 khớp × 3 = 57) mã INT8, scale 2^{round(np.log2(s_o))}\n")
        f.write("dùng          : RTL nạp input.mem → datapath INT → so golden_out.mem từng số\n")
        f.write(f"output→cm     : code × 2^{round(np.log2(s_o))} × y_std + y_mean  (y_std/y_mean từ train)\n")

    print(f"golden -> {a.out}/ | {a.frames} frame")
    print(f"  input.mem  {q_in.size} mã (= {a.frames}×320)   scale 2^{round(np.log2(s_in))}")
    print(f"  golden_out.mem {out_code.size} mã (= {a.frames}×57)   scale 2^{round(np.log2(s_o))}")
    print(f"  ví dụ out_code[0][:6] = {out_code[0][:6].tolist()}")


if __name__ == "__main__":
    main()
