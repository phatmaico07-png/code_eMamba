"""gen_weight_blob.py — Gộp 51 file .hex tham số → 1 BLOB nhị phân để PS DMA nạp runtime,
kèm LOAD-MAP (offset/size/đích BRAM/nửa b0|b1) cho weight_loader FSM bám theo.

    python gen_weight_blob.py
      → rtl_export/weights.bin          (blob, thứ tự cố định, MSB-first mỗi word)
        rtl_export/weight_loadmap.txt   (bảng người đọc)
        rtl_export/weight_loadmap.vh    (ROM init cho loader: {sel,half,depth,width})

Quy ước byte: mỗi dòng .hex là 1 WORD của BRAM (rộng = OUT_DIM byte). bytes.fromhex giữ
MSB trước → loader dồn trái (word = {word, byte}) là khớp $readmemh.
b0_*/b1_* (và *_b0/*_b1) cùng 1 BRAM vật lý double-depth: half=0 nạp nửa thấp, half=1 nửa cao.
"""
import os

WDIR    = "rtl_export/weights"
OUT_BIN = "rtl_export/weights.bin"
OUT_TXT = "rtl_export/weight_loadmap.txt"
OUT_VH  = "rtl_export/weight_loadmap.vh"

# Thứ tự nạp — KHỚP thứ tự $readmemh trong emamba_top_synth.v
ORDER = [
    "pe_weight", "pe_bias",
    "b0_gamma", "b0_beta", "b0_upper_weight", "b0_upper_bias", "b0_lower_weight", "b0_lower_bias",
    "b0_conv_weight", "b0_conv_bias", "b0_delta_weight", "b0_delta_bias", "b0_dtin_weight", "b0_dtin_bias",
    "b0_Bproj_weight", "b0_Cproj_weight", "b0_outproj_weight", "b0_outproj_bias", "b0_A_log", "b0_D_param",
    "silu_lut_b0", "silu_conv_b0",
    "b1_gamma", "b1_beta", "b1_upper_weight", "b1_upper_bias", "b1_lower_weight", "b1_lower_bias",
    "b1_conv_weight", "b1_conv_bias", "b1_delta_weight", "b1_delta_bias", "b1_dtin_weight", "b1_dtin_bias",
    "b1_Bproj_weight", "b1_Cproj_weight", "b1_outproj_weight", "b1_outproj_bias", "b1_A_log", "b1_D_param",
    "silu_lut_b1", "silu_conv_b1",
    "exp_bp", "exp_sl", "exp_ic",
    "head_ffn_in_weight", "head_ffn_in_bias", "head_ffn_out_weight", "head_ffn_out_bias",
    "head_proj_weight", "head_proj_bias",
]


def group_half(name):
    """Trả (group, half). b0/b1 cùng group (BRAM vật lý), half phân nửa thấp/cao."""
    if name.startswith("b0_"): return name[3:], 0
    if name.startswith("b1_"): return name[3:], 1
    if name.endswith("_b0"):   return name[:-3], 0
    if name.endswith("_b1"):   return name[:-3], 1
    return name, 0


def main():
    blob = bytearray()
    rows = []          # (name, group, half, depth, width, offset, nbytes)
    off = 0
    for nm in ORDER:
        path = os.path.join(WDIR, nm + ".hex")
        lines = [l.strip() for l in open(path) if l.strip()]
        depth = len(lines)
        width = len(lines[0]) // 2                      # byte / word
        seg = bytearray()
        for ln in lines:
            b = bytes.fromhex(ln)
            assert len(b) == width, f"{nm}: word {len(b)}B != {width}B (file không đều)"
            seg += b
        g, h = group_half(nm)
        rows.append((nm, g, h, depth, width, off, len(seg)))
        blob += seg
        off += len(seg)

    # gán sel cho từng GROUP (BRAM vật lý) theo thứ tự xuất hiện
    sel = {}
    for nm, g, h, *_ in rows:
        if g not in sel:
            sel[g] = len(sel)

    os.makedirs(os.path.dirname(OUT_BIN), exist_ok=True)
    open(OUT_BIN, "wb").write(blob)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# idx  seg                   group              sel half depth widthB  byteOff  nbytes\n")
        for i, (nm, g, h, d, w, o, nb) in enumerate(rows):
            f.write("%3d  %-20s %-18s %3d  %d  %5d %5d %8d %6d\n" % (i, nm, g, sel[g], h, d, w, o, nb))
        f.write("# %d segment, %d BRAM vật lý, TOTAL %d byte\n" % (len(rows), len(sel), len(blob)))

    with open(OUT_VH, "w", encoding="utf-8") as f:
        f.write("// weight_loadmap.vh — auto gen_weight_blob.py. %d segment.\n" % len(rows))
        f.write("// mỗi entry: {sel, half, depth, width_byte, byte_off}\n")
        f.write("localparam integer LM_N = %d;\n" % len(rows))
        f.write("localparam integer LM_TOTAL = %d;\n" % len(blob))
        for i, (nm, g, h, d, w, o, nb) in enumerate(rows):
            f.write("// [%2d] %-20s sel=%2d half=%d depth=%-4d width=%-2d off=%d\n" % (i, nm, sel[g], h, d, w, o))

    print("weights.bin : %d byte, %d segment, %d BRAM vật lý" % (len(blob), len(rows), len(sel)))
    print("load-map    : %s , %s" % (OUT_TXT, OUT_VH))
    print("nhóm BRAM (sel):")
    for g, s in sel.items():
        print("  sel %2d  %s" % (s, g))


if __name__ == "__main__":
    main()
