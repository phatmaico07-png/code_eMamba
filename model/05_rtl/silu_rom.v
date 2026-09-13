`timescale 1ns / 1ps
//============================================================================
// silu_rom.v — 1 ROM dùng CHUNG chứa 4 bảng SiLU PWL (gate_b0/b1, conv_b0/b1).
//   2 cổng đọc TỔ HỢP (port A = gate, port B = conv); mỗi cổng chọn block bằng
//   sel_*_blk (0=b0,1=b1). Trả full bp(N_SEG+1 INT8)+sl/ic(N_SEG INT16) cho bộ
//   tính silu_pwl_comp. Bảng nhỏ -> distributed (LUT). Nhiều bộ tính đọc chung.
//   LOAD_MODE=0: $readmemh từ {BASE}_bp/_sl/_ic.hex. (LOAD_MODE=1: TODO board.)
//============================================================================
module silu_rom #(
    parameter N_SEG   = 17,
    parameter GATE_B0 = "none",   // base path (tự thêm _bp/_sl/_ic.hex)
    parameter GATE_B1 = "none",
    parameter CV_B0   = "none",
    parameter CV_B1   = "none",
    parameter LOAD_MODE = 0
)(
    input  wire                       clk,
    // ── port A: gate ──
    input  wire                       sel_a_blk,
    output wire [(N_SEG+1)*16-1:0]    bpA,
    output wire [N_SEG*16-1:0]        slA,
    output wire [N_SEG*16-1:0]        icA,
    // ── port B: conv ──
    input  wire                       sel_b_blk,
    output wire [(N_SEG+1)*16-1:0]    bpB,
    output wire [N_SEG*16-1:0]        slB,
    output wire [N_SEG*16-1:0]        icB
);
    localparam NB = N_SEG + 1;

    (* ram_style="distributed" *) reg signed [15:0] g_bp [0:2*NB-1];   // bp INT16 (gate s_in thô → bp vượt ±127)
    (* ram_style="distributed" *) reg signed [15:0] g_sl [0:2*N_SEG-1];
    (* ram_style="distributed" *) reg signed [15:0] g_ic [0:2*N_SEG-1];
    (* ram_style="distributed" *) reg signed [15:0] c_bp [0:2*NB-1];   // bp INT16
    (* ram_style="distributed" *) reg signed [15:0] c_sl [0:2*N_SEG-1];
    (* ram_style="distributed" *) reg signed [15:0] c_ic [0:2*N_SEG-1];

    initial if (LOAD_MODE == 0) begin
        if (GATE_B0!="none") begin
            $readmemh({GATE_B0,"_bp.hex"}, g_bp, 0,     NB-1);
            $readmemh({GATE_B0,"_sl.hex"}, g_sl, 0,     N_SEG-1);
            $readmemh({GATE_B0,"_ic.hex"}, g_ic, 0,     N_SEG-1);
        end
        if (GATE_B1!="none") begin
            $readmemh({GATE_B1,"_bp.hex"}, g_bp, NB,    2*NB-1);
            $readmemh({GATE_B1,"_sl.hex"}, g_sl, N_SEG, 2*N_SEG-1);
            $readmemh({GATE_B1,"_ic.hex"}, g_ic, N_SEG, 2*N_SEG-1);
        end
        if (CV_B0!="none") begin
            $readmemh({CV_B0,"_bp.hex"}, c_bp, 0,     NB-1);
            $readmemh({CV_B0,"_sl.hex"}, c_sl, 0,     N_SEG-1);
            $readmemh({CV_B0,"_ic.hex"}, c_ic, 0,     N_SEG-1);
        end
        if (CV_B1!="none") begin
            $readmemh({CV_B1,"_bp.hex"}, c_bp, NB,    2*NB-1);
            $readmemh({CV_B1,"_sl.hex"}, c_sl, N_SEG, 2*N_SEG-1);
            $readmemh({CV_B1,"_ic.hex"}, c_ic, N_SEG, 2*N_SEG-1);
        end
    end

    // đọc tổ hợp: offset block * (NB hoặc N_SEG)
    wire [$clog2(2*NB)-1:0]    aoff_bp = sel_a_blk ? NB[$clog2(2*NB)-1:0]       : {$clog2(2*NB){1'b0}};
    wire [$clog2(2*N_SEG)-1:0] aoff_s  = sel_a_blk ? N_SEG[$clog2(2*N_SEG)-1:0] : {$clog2(2*N_SEG){1'b0}};
    wire [$clog2(2*NB)-1:0]    boff_bp = sel_b_blk ? NB[$clog2(2*NB)-1:0]       : {$clog2(2*NB){1'b0}};
    wire [$clog2(2*N_SEG)-1:0] boff_s  = sel_b_blk ? N_SEG[$clog2(2*N_SEG)-1:0] : {$clog2(2*N_SEG){1'b0}};

    genvar k;
    generate
        for (k=0; k<NB; k=k+1) begin : g_bpk
            assign bpA[k*16 +: 16] = g_bp[aoff_bp + k];
            assign bpB[k*16 +: 16] = c_bp[boff_bp + k];
        end
        for (k=0; k<N_SEG; k=k+1) begin : g_slk
            assign slA[k*16 +: 16] = g_sl[aoff_s + k];
            assign icA[k*16 +: 16] = g_ic[aoff_s + k];
            assign slB[k*16 +: 16] = c_sl[boff_s + k];
            assign icB[k*16 +: 16] = c_ic[boff_s + k];
        end
    endgenerate
endmodule
