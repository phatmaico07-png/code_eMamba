"""gen_synth.py — Sinh emamba_top_synth.v cấu hình SẴN cho model hiện tại.

Derive MỌI shift (Linear + SSM DA/DB/DX/Y/H + gamma + gating + res) từ calib, set
HEAD_H=80, trỏ file .hex (do export_rtl.py sinh). Khớp datapath int_reference đã verify.
"""
import argparse, math
import numpy as np
from int_reference import IntModel, pow2

S_EXPIN = 2.0 ** -5
S_H = 2.0 ** -3                      # scale state INT17 (đã quét: RMSE tốt nhất)
def e(s): return round(math.log2(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results_qat/best.pt")
    ap.add_argument("--calib", default="results_qat/calib_paper.pt")
    ap.add_argument("--wdir", default="rtl_export/weights")
    ap.add_argument("--out", default="rtl_export/emamba_top_synth.v")
    a = ap.parse_args()
    m = IntModel(a.ckpt, a.calib); W = m.W

    def wsc(key): return pow2(np.abs(W[key]).max() / 127)
    def lin(six, Wm, sox):                       # shift = log2(s_out/(s_x·s_w))
        return e(m.asc(sox)) - e(m.asc(six)) - e(pow2(np.abs(Wm).max() / 127))

    S = {}
    S["pe"] = lin("a_in", W["patch_embed.embed.weight"], "embed.out_q")
    blk = {}
    for b in range(m.cfg["blocks"]):
        p = f"blocks.{b}."; d = {}
        Win = W[p + "in_proj.weight"]
        d["upper"] = lin(p + "a_norm", Win[40:], p + "in_proj.out_q")   # KHỚP export_rtl: upper=Win[40:]
        d["lower"] = lin(p + "a_norm", Win[:40], p + "in_proj.out_q")   # KHỚP export_rtl: lower=Win[:40]
        d["conv"] = e(2.0 ** -4) - e(m.asc(p + "in_proj.out_q")) - e(pow2(np.abs(W[p + "conv1d.weight"]).max() / 127))  # conv→scale SiLU 2^-4
        # delta 2 BƯỚC: dt_in (a_silu1→dt_in.out_q) rồi dt_proj (dt_in.out_q→a_delta)
        d["dtin"]  = e(m.asc(p + "dt_in.out_q")) - e(m.asc(p + "a_silu1"))      - e(pow2(np.abs(W[p + "dt_in.weight"]).max() / 127))
        d["delta"] = e(m.asc(p + "a_delta"))     - e(m.asc(p + "dt_in.out_q")) - e(pow2(np.abs(W[p + "dt_proj.weight"]).max() / 127))
        d["B"] = lin(p + "a_silu1", W[p + "B_proj.weight"], p + "B_proj.out_q")
        d["C"] = lin(p + "a_silu1", W[p + "C_proj.weight"], p + "C_proj.out_q")
        d["out"] = lin(p + "a_gate", W[p + "out_proj.weight"], p + "a_block")
        # SSM
        A = -np.exp(np.minimum(W[p + "A_log"], math.log(4.0))); s_A = pow2(np.abs(A).max() / 127)
        s_xx = m.asc(p + "a_silu1"); s_del = m.asc(p + "a_delta")
        s_B = m.asc(p + "B_proj.out_q"); s_C = m.asc(p + "C_proj.out_q")
        s_y = m.asc(p + "a_y"); s_D = pow2(np.abs(W[p + "D"]).max() / 32767); s_h24 = (2.0 ** -7) * S_H
        d["DA"] = e(S_EXPIN) - e(s_del) - e(s_A)
        d["DB"] = e(s_h24 / s_xx) - e(s_del) - e(s_B)
        d["DX"] = e(s_C * s_h24) - e(s_D) - e(s_xx)
        d["Y"] = e(s_y) - e(s_C * s_h24)
        # gamma: GAMMA_SHIFT = log2(s_n / s_gamma)
        s_gamma = pow2(np.abs(W[p + "norm.gamma"]).max() / 127)
        d["gamma"] = e(m.asc(p + "a_norm")) - e(s_gamma)
        # gating: y·SiLU(z) → a_gate ; shift = log2(s_g/(s_y·s_gz))
        d["gating"] = e(m.asc(p + "a_gate")) - e(s_y) - e(m.asc(p + "a_gatez"))
        # residual: out_proj(s_bo=a_block) + res(sres=block input). Align res→s_bo.
        s_bo = m.asc(p + "a_block"); sres = m.asc("embed.out_q" if b == 0 else f"blocks.{b-1}.a_block")
        rsh = e(s_bo) - e(sres)        # >0: res mịn hơn → dịch phải res
        d["res_b_rshift"] = max(0, rsh); d["res_b_lshift"] = max(0, -rsh); d["res_shift"] = 0; d["res_a_lshift"] = 0
        blk[b] = d
    S["hd_ffn_in"] = lin("head.a_pool", W["head.ffn_in.weight"], "head.ffn_in.out_q")
    S["hd_ffn_out"] = lin("head.ffn_in.out_q", W["head.ffn_out.weight"], "head.a_trunk")
    S["hd_proj"] = lin("head.a_trunk", W["head.proj.weight"], "head.a_out")
    # residual head: qt = fo_out + rsh(qp, e(s_tr)-e(s_p)). Pool (b) dich ve s_tr.
    _hdres = e(m.asc("head.a_trunk")) - e(m.asc("head.a_pool"))
    S["hd_res_b_lshift"] = max(0, -_hdres)   # s_tr tho hon s_p -> pool dich trai
    S["hd_res_b_rshift"] = max(0,  _hdres)

    def bp(b, k): return blk[b][k]
    Wp = a.wdir.replace("\\", "/")
    def f(n): return '{W, "/' + n + '"}'
    v = []
    v.append("`timescale 1ns / 1ps")
    v.append("// emamba_top_synth.v — CẤU HÌNH SẴN cho model hiện tại (QAT, head=80)")
    v.append("// Sinh tự động bởi gen_synth.py. Shift Linear+SSM+gamma+gating+res derive từ calib.")
    v.append("module emamba_top_synth (")
    v.append("    input wire clk, input wire rst_n,")
    v.append("    input wire in_valid, output wire in_ready, input wire [2559:0] in_data,")
    v.append("    output wire out_valid, input wire out_ready, output wire [455:0] out_data,")
    v.append("    output wire busy);")
    v.append(f'    localparam W = "{Wp}";')
    v.append("    emamba_top #(")
    v.append("        .HEAD_H(80),")
    v.append(f"        .PE_SHIFT({S['pe']}), .PE_W_FILE({f('pe_weight.hex')}), .PE_B_FILE({f('pe_bias.hex')}),")
    for b in range(m.cfg["blocks"]):
        B = f"B{b}"; pre = f"b{b}"
        v.append(f"        .{B}_GAMMA_SHIFT({bp(b,'gamma')}), .{B}_UPPER_SHIFT({bp(b,'upper')}), .{B}_LOWER_SHIFT({bp(b,'lower')}),")
        v.append(f"        .{B}_CONV_SHIFT({bp(b,'conv')}), .{B}_DELTA_SHIFT({bp(b,'delta')}), .{B}_DT_IN_SHIFT({bp(b,'dtin')}), .{B}_B_SHIFT({bp(b,'B')}),")
        v.append(f"        .{B}_C_SHIFT({bp(b,'C')}), .{B}_OUT_SHIFT({bp(b,'out')}), .{B}_SSM_Y_SHIFT({bp(b,'Y')}),")
        v.append(f"        .{B}_SSM_DA_SHIFT({bp(b,'DA')}), .{B}_SSM_DB_SHIFT({bp(b,'DB')}), .{B}_SSM_DX_ALIGN({bp(b,'DX')}),")
        v.append(f"        .{B}_GATING_SHIFT({bp(b,'gating')}), .{B}_RES_SHIFT({bp(b,'res_shift')}),")
        v.append(f"        .{B}_RES_A_LSHIFT({bp(b,'res_a_lshift')}), .{B}_RES_B_LSHIFT({bp(b,'res_b_lshift')}), .{B}_RES_B_RSHIFT({bp(b,'res_b_rshift')}),")
        for nm, fn in [("GAMMA_FILE", "gamma"), ("BETA_FILE", "beta"), ("UPPER_W", "upper_weight"), ("UPPER_B", "upper_bias"),
                       ("LOWER_W", "lower_weight"), ("LOWER_B", "lower_bias"), ("CONV_W", "conv_weight"), ("CONV_B", "conv_bias"),
                       ("DELTA_W", "delta_weight"), ("DELTA_B", "delta_bias"),
                       ("DT_IN_W", "dtin_weight"), ("DT_IN_B", "dtin_bias"), ("B_W", "Bproj_weight"), ("C_W", "Cproj_weight"),
                       ("OUT_W", "outproj_weight"), ("OUT_B", "outproj_bias"), ("A_FILE", "A_log"), ("D_FILE", "D_param")]:
            v.append(f"        .{B}_{nm}({f(pre + '_' + fn + '.hex')}),")
        v.append(f"        .{B}_SILU_LUT({f('siluG_' + pre)}),")
        v.append(f"        .{B}_CONV_SILU_LUT({f('siluC_' + pre)}),")
    v.append(f"        .EXP_BP_FILE({f('exp_bp.hex')}), .EXP_SL_FILE({f('exp_sl.hex')}), .EXP_IC_FILE({f('exp_ic.hex')}),")
    v.append(f"        .HD_FFN_IN_SHIFT({S['hd_ffn_in']}), .HD_FFN_OUT_SHIFT({S['hd_ffn_out']}), .HD_PROJ_SHIFT({S['hd_proj']}),")
    v.append(f"        .HD_RES_B_LSHIFT({S['hd_res_b_lshift']}), .HD_RES_B_RSHIFT({S['hd_res_b_rshift']}),")
    v.append(f"        .HD_FFN_IN_W({f('head_ffn_in_weight.hex')}), .HD_FFN_IN_B({f('head_ffn_in_bias.hex')}),")
    v.append(f"        .HD_FFN_OUT_W({f('head_ffn_out_weight.hex')}), .HD_FFN_OUT_B({f('head_ffn_out_bias.hex')}),")
    v.append(f"        .HD_PROJ_W({f('head_proj_weight.hex')}), .HD_PROJ_B({f('head_proj_bias.hex')})")
    v.append("    ) u_core (")
    v.append("        .clk(clk), .rst_n(rst_n),")
    v.append("        .in_valid(in_valid), .in_ready(in_ready), .in_data(in_data),")
    v.append("        .out_valid(out_valid), .out_ready(out_ready), .out_data(out_data), .busy(busy),")
    v.append("        .wbus(16'd0), .use_shift_ovr(1'b0), .shift_ovr_b0(90'd0), .shift_ovr_b1(90'd0));")
    v.append("endmodule")
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(v) + "\n")

    print("shift derive (model hiện tại):")
    print(f"  PE={S['pe']}  HD: ffn_in={S['hd_ffn_in']} ffn_out={S['hd_ffn_out']} proj={S['hd_proj']}")
    for b in range(m.cfg["blocks"]):
        d = blk[b]
        print(f"  B{b}: up/lo={d['upper']}/{d['lower']} conv={d['conv']} delta={d['delta']} B={d['B']} C={d['C']} out={d['out']}")
        print(f"      SSM DA={d['DA']} DB={d['DB']} DX={d['DX']} Y={d['Y']} H=7 | gamma={d['gamma']} gating={d['gating']} res_b_rsh={d['res_b_rshift']}")
    print(f"\n-> {a.out}  (HEAD_H=80, mọi shift cấu hình sẵn)")


if __name__ == "__main__":
    main()
