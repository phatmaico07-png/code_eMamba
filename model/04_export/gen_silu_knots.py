"""gen_silu_knots.py — sinh knot PWL SiLU (siluG/siluC _bp/_sl/_ic.hex) cho MODEL HIỆN TẠI.

KHỚP CHÍNH XÁC int_reference.silu_lut256 (17 đoạn [-7,7], round-half-up). Gate & conv
SiLU phụ thuộc scale model (in_proj.out_q, a_gatez, a_silu1) → PHẢI regenerate mỗi khi
đổi model, nếu không silu_rom đọc knot cũ (sai scale) → RTL lệch golden.
(exp KHÔNG phụ thuộc model: S_EXPIN=2^-5 cố định → không regen.)

    python gen_silu_knots.py --ckpt results_all/best.pt --calib results_all/calib.pt \
        --wdir rtl_export/weights
"""
import argparse, os, math
import numpy as np
from int_reference import IntModel, fit_knots_n, _silu


def s8(v):  return format(int(v) & 0xFF, "02x")
def s16(v): return format(int(np.clip(v, -32768, 32767)) & 0xFFFF, "04x")
def wr(d, name, txt): open(os.path.join(d, name), "w", encoding="utf-8").write(txt + "\n")


def knots(s_in, s_out):
    """bp(18 INT8) + sl/ic(17 INT16) — Y HỆT silu_lut256 trong int_reference."""
    kn = fit_knots_n(_silu, -7.0, 7.0, 17)              # 18 knot, 17 đoạn
    bp = np.round(kn / s_in).astype(np.int64)
    sl, ic = [], []
    for i in range(17):
        a, b = kn[i], kn[i + 1]
        ya, yb = _silu(a) / s_out, _silu(b) / s_out
        m = (yb - ya) / (b - a)
        sl.append(round(m * s_in * 128))
        ic.append(round((ya - m * a) * 128))
    return bp, sl, ic


def dump(wd, base, s_in, s_out):
    bp, sl, ic = knots(s_in, s_out)
    # bp INT16: gate s_in thô (2^-5) → bp = ±224 vượt INT8. silu_rom/silu_pwl_comp
    # nay so sanh bp INT16 (input INT8 sign-extend) → bp>127 chi khong bao gio duoc
    # chon (input INT8 ≤127) → KHOP CHINH XAC golden khong-kep. (truoc: s8 wrap → sai)
    wr(wd, base + "_bp.hex", "\n".join(s16(v) for v in bp))
    wr(wd, base + "_sl.hex", "\n".join(s16(v) for v in sl))
    wr(wd, base + "_ic.hex", "\n".join(s16(v) for v in ic))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_all/best.pt")
    ap.add_argument("--calib", default="results_all/calib.pt")
    ap.add_argument("--wdir", default="rtl_export/weights")
    a = ap.parse_args()
    m = IntModel(a.ckpt, a.calib)
    for b in range(m.cfg["blocks"]):
        s_gin = m.asc(f"blocks.{b}.in_proj.out_q")     # gate: silu_lut256(qz, s_xz, s_xz, s_gz)
        s_gout = m.asc(f"blocks.{b}.a_gatez")
        s_cin = 2.0 ** -4                              # conv: silu_lut256(cacc, ., 2^-4, s_sl)
        s_cout = m.asc(f"blocks.{b}.a_silu1")
        dump(a.wdir, f"siluG_b{b}", s_gin, s_gout)
        dump(a.wdir, f"siluC_b{b}", s_cin, s_cout)
        print(f"b{b}: GATE in=2^{round(math.log2(s_gin))} out=2^{round(math.log2(s_gout))}"
              f" | CONV in=2^-4 out=2^{round(math.log2(s_cout))}")
    print(f"-> {a.wdir}/siluG_b*, siluC_b*  (_bp/_sl/_ic, 17 đoạn, khớp int_reference)")


if __name__ == "__main__":
    main()
