"""export_rtl.py — Sinh weight .hex + shift + synth wrapper ĐÚNG FORMAT RTL của bạn.

Đọc results_qat/best.pt + calib (numpy), xuất cho emamba_top.v:
  Linear weight : column-major wide-word (IN dòng × OUT·2 hex, output 0 ở LSB)
  Linear bias   : 1 dòng × OUT·4 hex (INT16)
  conv weight   : K dòng × ED·2 hex (tap-major) ; conv bias 1 dòng INT16
  in_proj       : TÁCH upper(xx, hàng 0..39) / lower(z, hàng 40..79)
  delta         : GỘP dt_proj@dt_in → 40×40 (+ bias gộp)
  gamma INT8 (DIM dòng) · beta INT16 (DIM dòng) · A_log (N dòng×ED, INT8) · D INT16 (ED dòng)
  + Linear/head SHIFT (= log2(s_out/(s_x·s_w))) ; HEAD_H=80

PHẦN (a): weight format + Linear shift + synth wrapper. (SSM DA/DB/DX/Y, exp-PWL, SiLU-LUT
= phần (b)). Chạy: python export_rtl.py --ckpt results_qat/best.pt --calib results_qat/calib_paper.pt
"""
import argparse, math, os
import numpy as np
from int_reference import IntModel, pow2, QM


def s8(v):  return format(int(v) & 0xFF, "02x")
def s16(v): return format(int(np.clip(v, -32768, 32767)) & 0xFFFF, "04x")
def wcol(qw):                                   # (OUT,IN) → IN dòng, output OUT-1..0
    OUT, IN = qw.shape
    return "\n".join("".join(s8(qw[o, i]) for o in range(OUT - 1, -1, -1)) for i in range(IN))
def bias_line(bi):                              # (OUT,) INT16 → 1 dòng
    return "".join(s16(bi[o]) for o in range(len(bi) - 1, -1, -1))
def per8(a):  return "\n".join(s8(v) for v in a)
def per16(a): return "\n".join(s16(v) for v in a)
def wr(d, name, txt): open(os.path.join(d, name), "w", encoding="utf-8").write(txt)


def qtensor(W, bits=8):                          # per-tensor pow2 → (codes, scale)
    q = (1 << (bits - 1)) - 1                     # 127 — SYMMETRIC clamp [-127,127] (KHỚP golden clampq)
    s = pow2(np.abs(W).max() / q)
    return np.clip(np.round(W / s), -q, q), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_qat/best.pt")
    ap.add_argument("--calib", default="results_qat/calib_paper.pt")
    ap.add_argument("--out", default="rtl_export")
    a = ap.parse_args()
    wd = os.path.join(a.out, "weights"); os.makedirs(wd, exist_ok=True)
    m = IntModel(a.ckpt, a.calib); W = m.W
    sh = lambda sx, sw, so: round(math.log2(so / (sx * sw)))
    shifts = {}

    def lin(prefix, Wm, bm, six, sox, split=None):
        """Xuất 1 linear (column-major + bias INT16) + trả shift. split: None|'upper'|'lower'."""
        qw, sw = qtensor(Wm)
        sx, so = m.asc(six), m.asc(sox)
        wr(wd, prefix + "_weight.hex", wcol(qw))
        if bm is not None:
            bi = np.round(bm / (sx * sw))
            wr(wd, prefix + "_bias.hex", bias_line(bi))
        return sh(sx, sw, so)

    for b in range(m.cfg["blocks"]):
        p = f"blocks.{b}."; tag = f"b{b}"
        # in_proj — RTL: upper→SiLU→gate dùng z (in_proj[40:]); lower→conv→SSM dùng xx (in_proj[:40]).
        # chunk model: xx=[:40], z=[40:]; RTL nối ngược nên upper=z, lower=xx.
        Win = W[p + "in_proj.weight"]; bin_ = W[p + "in_proj.bias"]
        shifts[f"{tag}_upper"] = lin(f"{tag}_upper", Win[40:], bin_[40:], p + "a_norm", p + "in_proj.out_q")
        shifts[f"{tag}_lower"] = lin(f"{tag}_lower", Win[:40], bin_[:40], p + "a_norm", p + "in_proj.out_q")
        # conv1d depthwise: K dòng × ED byte (tap-major)
        cw = W[p + "conv1d.weight"]; csw_s = pow2(np.abs(cw).max() / 127)
        cq = np.clip(np.round(cw / csw_s), -128, 127)        # (ED,1,K)
        ED, _, Kk = cq.shape
        conv_lines = "\n".join("".join(s8(cq[c, 0, k]) for c in range(ED - 1, -1, -1)) for k in range(Kk))
        wr(wd, f"{tag}_conv_weight.hex", conv_lines)
        cb = np.round(W[p + "conv1d.bias"] / (m.asc(p + "in_proj.out_q") * csw_s))
        wr(wd, f"{tag}_conv_bias.hex", bias_line(cb))
        shifts[f"{tag}_conv"] = sh(m.asc(p + "in_proj.out_q"), csw_s, 2.0 ** -4)  # conv→scale vào SiLU 2^-4
        # delta 2 BƯỚC (đúng model): dt_in (ED→rank → dt_in.out_q) rồi dt_proj (rank→ED → a_delta)
        shifts[f"{tag}_dtin"]  = lin(f"{tag}_dtin",  W[p + "dt_in.weight"],   W[p + "dt_in.bias"],   p + "a_silu1",      p + "dt_in.out_q")
        shifts[f"{tag}_delta"] = lin(f"{tag}_delta", W[p + "dt_proj.weight"], W[p + "dt_proj.bias"], p + "dt_in.out_q",  p + "a_delta")
        # B, C (no bias)
        shifts[f"{tag}_B"] = lin(f"{tag}_Bproj", W[p + "B_proj.weight"], None, p + "a_silu1", p + "B_proj.out_q")
        shifts[f"{tag}_C"] = lin(f"{tag}_Cproj", W[p + "C_proj.weight"], None, p + "a_silu1", p + "C_proj.out_q")
        # out_proj
        shifts[f"{tag}_out"] = lin(f"{tag}_outproj", W[p + "out_proj.weight"], W[p + "out_proj.bias"], p + "a_gate", p + "a_block")
        # gamma INT8 / beta INT16 — β PHẢI cùng scale γ (s_gam=gsw): RTL cộng β trực tiếp
        # vào (γ·quot)>>15, không có shift bù. (Trước đây β dùng scale riêng → tràn.)
        g = W[p + "norm.gamma"]; gsw = pow2(np.abs(g).max() / 127)
        wr(wd, f"{tag}_gamma.hex", per8(np.clip(np.round(g / gsw), -127, 127)))
        wr(wd, f"{tag}_beta.hex", per16(np.round(W[p + "norm.beta"] / gsw)))
        # A_log: N dòng × ED byte (column per state) ; D INT16
        A = -np.exp(np.minimum(W[p + "A_log"], math.log(4.0)))    # (ED,N)
        Aq, sA = qtensor(A)
        N = A.shape[1]
        wr(wd, f"{tag}_A_log.hex", "\n".join("".join(s8(Aq[e, n]) for e in range(ED - 1, -1, -1)) for n in range(N)))
        Dp = W[p + "D"]; Dsw = pow2(np.abs(Dp).max() / 32767)
        wr(wd, f"{tag}_D_param.hex", per16(np.round(Dp / Dsw)))

    # PatchEmbed — hoán vị cột weight: model patchify (c,pi,pj) → RTL patch_data (pi,pj,c)
    #   RTL cột k=(pi=k//10, pj=(k%10)//5, c=k%5) phải lấy cột model (c*4+pi*2+pj)
    pe_perm = np.array([(k % 5) * 4 + (k // 10) * 2 + ((k % 10) // 5) for k in range(20)])
    Wpe = W["patch_embed.embed.weight"][:, pe_perm]
    shifts["pe"] = lin("pe", Wpe, W["patch_embed.embed.bias"], "a_in", "embed.out_q")
    # Head
    shifts["hd_ffn_in"]  = lin("head_ffn_in",  W["head.ffn_in.weight"],  W["head.ffn_in.bias"],  "head.a_pool",        "head.ffn_in.out_q")
    shifts["hd_ffn_out"] = lin("head_ffn_out", W["head.ffn_out.weight"], W["head.ffn_out.bias"], "head.ffn_in.out_q",  "head.a_trunk")
    shifts["hd_proj"]    = lin("head_proj",    W["head.proj.weight"],    W["head.proj.bias"],    "head.a_trunk",       "head.a_out")

    print("Shift (Linear) suy ra cho model hiện tại:")
    for k in ["pe", "b0_upper", "b0_lower", "b0_conv", "b0_delta", "b0_B", "b0_C", "b0_out",
              "b1_upper", "b1_lower", "b1_conv", "b1_delta", "b1_B", "b1_C", "b1_out",
              "hd_ffn_in", "hd_ffn_out", "hd_proj"]:
        print(f"  {k:12s} = {shifts[k]}")
    nf = len([f for f in os.listdir(wd) if f.endswith('.hex')])
    print("\n-> %s/ : %d hex files (column-major, bias INT16, split upper/lower)" % (wd, nf))
    print("   HEAD hidden = 80 ; synth set HEAD_H=80")
    print("   weights regenerated with SYMMETRIC clamp [-127,127]")


if __name__ == "__main__":
    main()
