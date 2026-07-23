`timescale 1ns / 1ps
//============================================================================
// emamba_top.v — eMamba Accelerator Top Level
//
// Full inference pipeline:
//   Input (8×8×5 INT8) → PatchEmbed → 16×20 tokens
//   → Shared MambaBlock pass-0 (block_sel=0, block0 weights & shifts)
//   → Shared MambaBlock pass-1 (block_sel=1, block1 weights & shifts)
//   → MeanPool → OutputHead → Output (57 INT8)
//
// Single shared MambaBlock (SHARED=1) with double-depth BRAMs.
// FSM drives block_sel=0 for pass-0 then block_sel=1 for pass-1.
// A frame_start pulse resets SSM state and conv1d shift register between
// passes.
//
// FSM: IDLE → PE → BLK0 → BLK1 → POOL → HEAD → DONE
//
// Latency estimate:
//   PE: ~368 cyc
//   Block0 pass: ~1045 cyc
//   Block1 pass: ~1045 cyc
//   Pool: ~17 cyc
//   Head: ~113 cyc
//   Total: ~2588 cyc @ 100 MHz = 25.9 µs/frame
//============================================================================

module emamba_top #(
    // ── Architecture ─────────────────────────────────────────────
    parameter H          = 8,
    parameter W          = 8,
    parameter C          = 5,
    parameter P          = 2,
    parameter D          = 20,
    parameter ED         = 40,
    parameter L          = 16,
    parameter N_STATE    = 8,
    parameter K          = 4,
    parameter N_CU       = 20,
    parameter HEAD_H     = 64,
    parameter OUT_DIM    = 57,
    parameter LOAD_MODE  = 0,

    // ── PatchEmbed weights ───────────────────────────────────────
    parameter PE_SHIFT    = 8,
    parameter PE_W_FILE   = "none",
    parameter PE_B_FILE   = "none",

    // ── Block 0 weights ──────────────────────────────────────────
    parameter B0_GAMMA_FILE   = "none", parameter B0_BETA_FILE    = "none",
    parameter B0_UPPER_W      = "none", parameter B0_UPPER_B      = "none",
    parameter B0_LOWER_W      = "none", parameter B0_LOWER_B      = "none",
    parameter B0_CONV_W       = "none", parameter B0_CONV_B       = "none",
    parameter B0_DELTA_W      = "none", parameter B0_DELTA_B      = "none",
    parameter B0_DT_IN_W      = "none", parameter B0_DT_IN_B      = "none",
    parameter B0_B_W          = "none", parameter B0_C_W          = "none",
    parameter B0_OUT_W        = "none", parameter B0_OUT_B        = "none",
    parameter B0_A_FILE       = "none", parameter B0_D_FILE       = "none",
    parameter B0_SILU_LUT     = "none",
    parameter B0_CONV_SILU_LUT = "none",
    parameter EXP_BP_FILE     = "none", parameter EXP_SL_FILE     = "none", parameter EXP_IC_FILE = "none",

    // ── Block 0 shifts ───────────────────────────────────────────
    parameter B0_GAMMA_SHIFT  = 0,
    parameter B0_UPPER_SHIFT  = 7,
    parameter B0_LOWER_SHIFT  = 7,
    parameter B0_CONV_SHIFT   = 6,
    parameter B0_DELTA_SHIFT  = 7,
    parameter B0_DT_IN_SHIFT  = 6,
    parameter B0_B_SHIFT      = 7,
    parameter B0_C_SHIFT      = 6,
    parameter B0_OUT_SHIFT    = 7,
    parameter B0_SSM_Y_SHIFT  = 10,
    parameter B0_SSM_DA_SHIFT = 3,
    parameter B0_SSM_DB_SHIFT = 3,
    parameter B0_SSM_DX_ALIGN = 8,
    parameter B0_SSM_PER_CH_YSHIFT = 0,
    parameter B0_SSM_YSHIFT_FILE   = "none",
    parameter B0_GATING_SHIFT = 7,
    parameter B0_RES_SHIFT    = 3,
    parameter B0_RES_A_LSHIFT = 0,
    parameter B0_RES_B_LSHIFT = 0,
    parameter B0_RES_B_RSHIFT = 0,

    // ── Block 1 weights ──────────────────────────────────────────
    parameter B1_GAMMA_FILE   = "none", parameter B1_BETA_FILE    = "none",
    parameter B1_UPPER_W      = "none", parameter B1_UPPER_B      = "none",
    parameter B1_LOWER_W      = "none", parameter B1_LOWER_B      = "none",
    parameter B1_CONV_W       = "none", parameter B1_CONV_B       = "none",
    parameter B1_DELTA_W      = "none", parameter B1_DELTA_B      = "none",
    parameter B1_DT_IN_W      = "none", parameter B1_DT_IN_B      = "none",
    parameter B1_B_W          = "none", parameter B1_C_W          = "none",
    parameter B1_OUT_W        = "none", parameter B1_OUT_B        = "none",
    parameter B1_A_FILE       = "none", parameter B1_D_FILE       = "none",
    parameter B1_SILU_LUT     = "none",
    parameter B1_CONV_SILU_LUT = "none",

    // ── Block 1 shifts ───────────────────────────────────────────
    parameter B1_GAMMA_SHIFT  = 0,
    parameter B1_UPPER_SHIFT  = 8,
    parameter B1_LOWER_SHIFT  = 8,
    parameter B1_CONV_SHIFT   = 6,
    parameter B1_DELTA_SHIFT  = 6,
    parameter B1_DT_IN_SHIFT  = 7,
    parameter B1_B_SHIFT      = 7,
    parameter B1_C_SHIFT      = 6,
    parameter B1_OUT_SHIFT    = 8,
    parameter B1_SSM_Y_SHIFT  = 7,
    parameter B1_SSM_DA_SHIFT = 3,
    parameter B1_SSM_DB_SHIFT = 3,
    parameter B1_SSM_DX_ALIGN = 8,
    parameter B1_SSM_PER_CH_YSHIFT = 0,
    parameter B1_SSM_YSHIFT_FILE   = "none",
    parameter B1_GATING_SHIFT = 7,
    parameter B1_RES_SHIFT    = 4,
    parameter B1_RES_A_LSHIFT = 0,
    parameter B1_RES_B_LSHIFT = 2,
    parameter B1_RES_B_RSHIFT = 0,

    // ── Output Head weights ──────────────────────────────────────
    parameter HD_FFN_IN_SHIFT  = 6,
    parameter HD_FFN_OUT_SHIFT = 6,
    parameter HD_PROJ_SHIFT    = 8,
    parameter HD_RES_B_LSHIFT  = 0,   // dich residual pool trong head (align s_p -> s_tr)
    parameter HD_RES_B_RSHIFT  = 0,
    parameter HD_FFN_IN_W      = "none", parameter HD_FFN_IN_B  = "none",
    parameter HD_FFN_OUT_W     = "none", parameter HD_FFN_OUT_B = "none",
    parameter HD_PROJ_W        = "none", parameter HD_PROJ_B    = "none"
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // ── Input: one frame ─────────────────────────────────────────
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [H*W*C*8-1:0]      in_data,     // 8×8×5 = 320 INT8

    // ── Output: joint predictions ────────────────────────────────
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [OUT_DIM*8-1:0]    out_data,     // 57 INT8

    // ── Status ───────────────────────────────────────────────────
    output wire                     busy,
    input  wire [15:0]              wbus,

    // P1.4: shift override (0 = param verified ; 1 = vector PS ghi qua AXI-Lite)
    input  wire                     use_shift_ovr,
    input  wire [89:0]              shift_ovr_b0,   // 18 shift × 5b (block0)
    input  wire [89:0]              shift_ovr_b1    // 18 shift × 5b (block1)
);

    // ══════════════════════════════════════════════════════════════
    // TOP FSM
    // ══════════════════════════════════════════════════════════════
    localparam S_IDLE   = 3'd0;
    localparam S_PE     = 3'd1;   // Patch Embedding
    localparam S_BLK0   = 3'd2;   // Mamba Block 0
    localparam S_BLK1   = 3'd3;   // Mamba Block 1
    localparam S_POOL   = 3'd4;   // Mean Pool
    localparam S_HEAD   = 3'd5;   // Output Head
    localparam S_DONE   = 3'd6;

    reg [2:0] state;

    // ══════════════════════════════════════════════════════════════
    // INTER-STAGE WIRES
    // ══════════════════════════════════════════════════════════════
    // PE output
    wire                pe_ov;               // PE out_valid (stream 1 token/lan)
    wire [D*8-1:0]      pe_sdata;            // PE out_data (1 token = 160-bit)
    wire                pe_or;               // PE out_ready (thu khi con cho)
    reg  [L*D*8-1:0]   pe_tokens;            // buffer THU stream PE (16×20) — GIU logic feed cu bit-exact
    reg  [4:0]          pe_tdone;            // so token DA thu tu PE (0..16)

    // Token feed registers (feed tokens 1-by-1 to shared mamba_block)
    reg [L*D*8-1:0]    tok_buf;      // token buffer for current pass
    reg [$clog2(L):0]  tok_feed_cnt;
    wire [D*8-1:0]     tok_feed_data;

    // Shared mamba_block I/O
    reg                mb_iv;
    wire               mb_ir;
    reg                mb_fs;
    reg                mb_block_sel;  // 0=pass0, 1=pass1
    wire               mb_ov;
    reg                mb_or;
    wire [D*8-1:0]     mb_out;

    // Pass-0 output collection buffer
    reg [L*D*8-1:0]    b0_tok_buf;
    reg [$clog2(L):0]  b0_out_cnt;

    // Pool
    reg                pool_fs;
    wire               pool_ov;
    reg                pool_or;
    wire [D*8-1:0]     pool_out;

    // Head
    wire               head_ov;
    wire [OUT_DIM*8-1:0] head_out;

    // ══════════════════════════════════════════════════════════════
    // SUB-MODULE INSTANCES
    // ══════════════════════════════════════════════════════════════

    // ── Patch Embedding ──────────────────────────────────────────
    wire pe_ir;
    patch_embed #(
        .H(H), .W(W), .C(C), .P(P), .D(D),
        .SHIFT_RIGHT(PE_SHIFT),
        .WEIGHT_FILE(PE_W_FILE), .BIAS_FILE(PE_B_FILE),
        .LOAD_MODE(LOAD_MODE)
    ) u_pe (
        .clk(clk), .rst_n(rst_n),
        .in_valid(state == S_PE),          // FIX: giữ trigger suốt pha S_PE (tránh lỡ cửa sổ 1-cycle)
        .in_ready(pe_ir),
        .in_data(in_data),
        .out_valid(pe_ov), .out_ready(pe_or),
        .out_data(pe_sdata), .wbus(wbus)
    );

    // ── THU stream PE (160-bit/token) vao buffer pe_tokens + dem pe_tdone ──
    //   Giu nguyen mux tok_src + logic feed S_BLK0 (da bit-exact). Token dung thu tu.
    assign pe_or = (state != S_IDLE) && (pe_tdone < L);   // thu khi con cho (0..15)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pe_tokens <= 0; pe_tdone <= 0;
        end else if (state == S_IDLE) begin
            pe_tdone <= 0;                                 // reset moi frame
        end else if (pe_ov && pe_or) begin
            case (pe_tdone[$clog2(L)-1:0])
                4'd0:  pe_tokens[  0*D*8 +: D*8] <= pe_sdata;
                4'd1:  pe_tokens[  1*D*8 +: D*8] <= pe_sdata;
                4'd2:  pe_tokens[  2*D*8 +: D*8] <= pe_sdata;
                4'd3:  pe_tokens[  3*D*8 +: D*8] <= pe_sdata;
                4'd4:  pe_tokens[  4*D*8 +: D*8] <= pe_sdata;
                4'd5:  pe_tokens[  5*D*8 +: D*8] <= pe_sdata;
                4'd6:  pe_tokens[  6*D*8 +: D*8] <= pe_sdata;
                4'd7:  pe_tokens[  7*D*8 +: D*8] <= pe_sdata;
                4'd8:  pe_tokens[  8*D*8 +: D*8] <= pe_sdata;
                4'd9:  pe_tokens[  9*D*8 +: D*8] <= pe_sdata;
                4'd10: pe_tokens[ 10*D*8 +: D*8] <= pe_sdata;
                4'd11: pe_tokens[ 11*D*8 +: D*8] <= pe_sdata;
                4'd12: pe_tokens[ 12*D*8 +: D*8] <= pe_sdata;
                4'd13: pe_tokens[ 13*D*8 +: D*8] <= pe_sdata;
                4'd14: pe_tokens[ 14*D*8 +: D*8] <= pe_sdata;
                4'd15: pe_tokens[ 15*D*8 +: D*8] <= pe_sdata;
            endcase
            pe_tdone <= pe_tdone + 1'b1;
        end
    end

    // ── Shared Mamba Block (pass 0: block_sel=0, pass 1: block_sel=1) ──
    // Runtime shift MUX: pass-0 uses B0 shifts, pass-1 uses B1 shifts
    // P1.4: mỗi shift = use_shift_ovr ? vector(PS ghi) : param B0/B1 (verified).
    // shift_ovr_b{0,1}[89:0] = 18 shift × 5b: up0 lo5 conv10 out15 res20 ral25 rbl30 rbr35
    //   ssmY40 sda45 sdb50 sdx55 gam60 del65 dti70 bp75 cp80 gat85.
    wire [4:0] mb_shift_upper = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[ 0+:5] : shift_ovr_b0[ 0+:5]) : (mb_block_sel ? B1_UPPER_SHIFT[4:0] : B0_UPPER_SHIFT[4:0]);
    wire [4:0] mb_shift_lower = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[ 5+:5] : shift_ovr_b0[ 5+:5]) : (mb_block_sel ? B1_LOWER_SHIFT[4:0] : B0_LOWER_SHIFT[4:0]);
    wire [4:0] mb_shift_conv  = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[10+:5] : shift_ovr_b0[10+:5]) : (mb_block_sel ? B1_CONV_SHIFT[4:0]  : B0_CONV_SHIFT[4:0]);
    wire [4:0] mb_shift_out   = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[15+:5] : shift_ovr_b0[15+:5]) : (mb_block_sel ? B1_OUT_SHIFT[4:0]   : B0_OUT_SHIFT[4:0]);
    wire [4:0] mb_shift_res   = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[20+:5] : shift_ovr_b0[20+:5]) : (mb_block_sel ? B1_RES_SHIFT[4:0]   : B0_RES_SHIFT[4:0]);
    wire [3:0] mb_res_a_lshift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[25+:4] : shift_ovr_b0[25+:4]) : (mb_block_sel ? B1_RES_A_LSHIFT[3:0] : B0_RES_A_LSHIFT[3:0]);
    wire [3:0] mb_res_b_lshift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[30+:4] : shift_ovr_b0[30+:4]) : (mb_block_sel ? B1_RES_B_LSHIFT[3:0] : B0_RES_B_LSHIFT[3:0]);
    wire [3:0] mb_res_b_rshift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[35+:4] : shift_ovr_b0[35+:4]) : (mb_block_sel ? B1_RES_B_RSHIFT[3:0] : B0_RES_B_RSHIFT[3:0]);
    wire [4:0] mb_ssm_y_shift  = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[40+:5] : shift_ovr_b0[40+:5]) : (mb_block_sel ? B1_SSM_Y_SHIFT[4:0]  : B0_SSM_Y_SHIFT[4:0]);
    wire [3:0] mb_ssm_da_shift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[45+:4] : shift_ovr_b0[45+:4]) : (mb_block_sel ? B1_SSM_DA_SHIFT[3:0] : B0_SSM_DA_SHIFT[3:0]);
    wire [3:0] mb_ssm_db_shift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[50+:4] : shift_ovr_b0[50+:4]) : (mb_block_sel ? B1_SSM_DB_SHIFT[3:0] : B0_SSM_DB_SHIFT[3:0]);
    wire [3:0] mb_ssm_dx_align = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[55+:4] : shift_ovr_b0[55+:4]) : (mb_block_sel ? B1_SSM_DX_ALIGN[3:0] : B0_SSM_DX_ALIGN[3:0]);
    wire signed [3:0] mb_gamma_shift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[60+:4] : shift_ovr_b0[60+:4]) : (mb_block_sel ? B1_GAMMA_SHIFT[3:0] : B0_GAMMA_SHIFT[3:0]);
    wire [4:0] mb_shift_delta = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[65+:5] : shift_ovr_b0[65+:5]) : (mb_block_sel ? B1_DELTA_SHIFT[4:0] : B0_DELTA_SHIFT[4:0]);
    wire [4:0] mb_shift_dtin  = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[70+:5] : shift_ovr_b0[70+:5]) : (mb_block_sel ? B1_DT_IN_SHIFT[4:0] : B0_DT_IN_SHIFT[4:0]);
    wire [4:0] mb_shift_bproj = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[75+:5] : shift_ovr_b0[75+:5]) : (mb_block_sel ? B1_B_SHIFT[4:0]     : B0_B_SHIFT[4:0]);
    wire [4:0] mb_shift_cproj = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[80+:5] : shift_ovr_b0[80+:5]) : (mb_block_sel ? B1_C_SHIFT[4:0]     : B0_C_SHIFT[4:0]);
    wire [4:0] mb_gating_shift = use_shift_ovr ? (mb_block_sel ? shift_ovr_b1[85+:5] : shift_ovr_b0[85+:5]) : (mb_block_sel ? B1_GATING_SHIFT[4:0] : B0_GATING_SHIFT[4:0]);

    mamba_block #(
        .D(D), .ED(ED), .L(L), .N_STATE(N_STATE), .K(K), .N_CU(N_CU),
        // Use block0 parameter shifts as structural defaults (overridden at runtime)
        .GAMMA_SHIFT(B0_GAMMA_SHIFT),
        .UPPER_SHIFT(B0_UPPER_SHIFT), .LOWER_SHIFT(B0_LOWER_SHIFT),
        .CONV_SHIFT(B0_CONV_SHIFT),   .DELTA_SHIFT(B0_DELTA_SHIFT), .DT_IN_SHIFT(B0_DT_IN_SHIFT),
        .B_SHIFT(B0_B_SHIFT),         .C_SHIFT(B0_C_SHIFT),
        .OUT_SHIFT(B0_OUT_SHIFT),     .SSM_Y_SHIFT(B0_SSM_Y_SHIFT),
        .SSM_DA_SHIFT(B0_SSM_DA_SHIFT), .SSM_DB_SHIFT(B0_SSM_DB_SHIFT),
        .SSM_DX_ALIGN(B0_SSM_DX_ALIGN),
        .SSM_PER_CH_YSHIFT(B0_SSM_PER_CH_YSHIFT),
        .SSM_YSHIFT_FILE(B0_SSM_YSHIFT_FILE),
        .SSM_YSHIFT_FILE_B1(B1_SSM_YSHIFT_FILE),
        .GATING_SHIFT(B0_GATING_SHIFT),
        // Shared mode enabled
        .SHARED(1),
        // Block-0 weight files
        .GAMMA_FILE(B0_GAMMA_FILE),   .BETA_FILE(B0_BETA_FILE),
        .UPPER_W_FILE(B0_UPPER_W),    .UPPER_B_FILE(B0_UPPER_B),
        .LOWER_W_FILE(B0_LOWER_W),    .LOWER_B_FILE(B0_LOWER_B),
        .CONV_W_FILE(B0_CONV_W),      .CONV_B_FILE(B0_CONV_B),
        .DELTA_W_FILE(B0_DELTA_W),    .DELTA_B_FILE(B0_DELTA_B),
        .DT_IN_W_FILE(B0_DT_IN_W),    .DT_IN_B_FILE(B0_DT_IN_B),
        .B_W_FILE(B0_B_W),            .C_W_FILE(B0_C_W),
        .OUT_W_FILE(B0_OUT_W),        .OUT_B_FILE(B0_OUT_B),
        .A_FILE(B0_A_FILE),           .D_FILE(B0_D_FILE),
        .SILU_LUT_FILE(B0_SILU_LUT),
        .CONV_SILU_LUT(B0_CONV_SILU_LUT),
        .EXP_BP_FILE(EXP_BP_FILE), .EXP_SL_FILE(EXP_SL_FILE), .EXP_IC_FILE(EXP_IC_FILE),
        // Block-1 weight files
        .GAMMA_FILE_B1(B1_GAMMA_FILE),   .BETA_FILE_B1(B1_BETA_FILE),
        .UPPER_W_FILE_B1(B1_UPPER_W),    .UPPER_B_FILE_B1(B1_UPPER_B),
        .LOWER_W_FILE_B1(B1_LOWER_W),    .LOWER_B_FILE_B1(B1_LOWER_B),
        .CONV_W_FILE_B1(B1_CONV_W),      .CONV_B_FILE_B1(B1_CONV_B),
        .DELTA_W_FILE_B1(B1_DELTA_W),    .DELTA_B_FILE_B1(B1_DELTA_B),
        .DT_IN_W_FILE_B1(B1_DT_IN_W),    .DT_IN_B_FILE_B1(B1_DT_IN_B),
        .B_W_FILE_B1(B1_B_W),            .C_W_FILE_B1(B1_C_W),
        .OUT_W_FILE_B1(B1_OUT_W),        .OUT_B_FILE_B1(B1_OUT_B),
        .A_FILE_B1(B1_A_FILE),           .D_FILE_B1(B1_D_FILE),
        .SILU_LUT_FILE_B1(B1_SILU_LUT),
        .CONV_SILU_LUT_B1(B1_CONV_SILU_LUT),
        .LOAD_MODE(LOAD_MODE)
    ) u_shared_block (
        .clk(clk), .rst_n(rst_n),
        .block_sel(mb_block_sel),
        .frame_start(mb_fs),
        .shift_upper(mb_shift_upper), .shift_lower(mb_shift_lower),
        .shift_conv(mb_shift_conv),   .shift_out(mb_shift_out),
        .shift_res(mb_shift_res),
        .res_a_lshift(mb_res_a_lshift),
        .res_b_lshift(mb_res_b_lshift),
        .res_b_rshift(mb_res_b_rshift),
        .ssm_y_shift_in(mb_ssm_y_shift),
        .ssm_da_shift_in(mb_ssm_da_shift),
        .ssm_db_shift_in(mb_ssm_db_shift),
        .ssm_dx_align_in(mb_ssm_dx_align),
        .gamma_shift_in(mb_gamma_shift),
        .shift_delta(mb_shift_delta),
        .shift_dtin(mb_shift_dtin),
        .shift_bproj(mb_shift_bproj),
        .shift_cproj(mb_shift_cproj),
        .gating_shift_in(mb_gating_shift),
        .in_valid(mb_iv), .in_ready(mb_ir),
        .in_data(tok_feed_data),
        .out_valid(mb_ov), .out_ready(mb_or),
        .out_data(mb_out), .wbus(wbus)
    );

    // ── Mean Pool ────────────────────────────────────────────────
    // Pool receives tokens from shared block pass-1 (block_sel=1) output
    wire pool_iv_w = mb_ov && (state == S_BLK1);
    wire pool_ir;

    // SHIFT = log2(L) - (e(s_block1)-e(a_pool)) = 4 - 2 = 2  (mean /16 RỒI đổi scale 2^-4->2^-6).
    // (Trước SHIFT(4) giữ scale block1 2^-4; model cần a_pool 2^-6 mịn hơn 2 bit.)
    mean_pool #(.D(D), .L(L), .SHIFT(2))
    u_pool (
        .clk(clk), .rst_n(rst_n),
        .frame_start(pool_fs),
        .in_valid(pool_iv_w), .in_ready(pool_ir),
        .in_data(mb_out),
        .out_valid(pool_ov), .out_ready(pool_or),
        .out_data(pool_out)
    );

    // ── Output Head ──────────────────────────────────────────────
    wire hd_iv_w = pool_ov && (state == S_POOL || state == S_HEAD);
    wire hd_ir;

    output_head #(
        .D(D), .H(HEAD_H), .OUT_DIM(OUT_DIM),
        .FFN_IN_SHIFT(HD_FFN_IN_SHIFT), .FFN_OUT_SHIFT(HD_FFN_OUT_SHIFT),
        .PROJ_SHIFT(HD_PROJ_SHIFT),
        .RES_B_LSHIFT(HD_RES_B_LSHIFT), .RES_B_RSHIFT(HD_RES_B_RSHIFT),
        .FFN_IN_W_FILE(HD_FFN_IN_W), .FFN_IN_B_FILE(HD_FFN_IN_B),
        .FFN_OUT_W_FILE(HD_FFN_OUT_W), .FFN_OUT_B_FILE(HD_FFN_OUT_B),
        .PROJ_W_FILE(HD_PROJ_W), .PROJ_B_FILE(HD_PROJ_B),
        .LOAD_MODE(LOAD_MODE)
    ) u_head (
        .clk(clk), .rst_n(rst_n),
        .in_valid(hd_iv_w), .in_ready(hd_ir),
        .in_data(pool_out),
        .out_valid(head_ov), .out_ready(out_ready),
        .out_data(head_out), .wbus(wbus)
    );

    // ══════════════════════════════════════════════════════════════
    // TOKEN FEED MUX — extract 1 token from tok_buf
    // Variable part-select via case MUX (Verilog-2001 safe)
    // ══════════════════════════════════════════════════════════════
    // C: pass0 doc THANG tu PE (overlap); pass1 doc tu tok_buf (=b0 output)
    wire [L*D*8-1:0] tok_src = mb_block_sel ? tok_buf : pe_tokens;
    reg [D*8-1:0] tok_feed_data_r;
    always @(*) begin
        case (tok_feed_cnt[$clog2(L)-1:0])
            4'd0:  tok_feed_data_r = tok_src[  0*D*8 +: D*8];
            4'd1:  tok_feed_data_r = tok_src[  1*D*8 +: D*8];
            4'd2:  tok_feed_data_r = tok_src[  2*D*8 +: D*8];
            4'd3:  tok_feed_data_r = tok_src[  3*D*8 +: D*8];
            4'd4:  tok_feed_data_r = tok_src[  4*D*8 +: D*8];
            4'd5:  tok_feed_data_r = tok_src[  5*D*8 +: D*8];
            4'd6:  tok_feed_data_r = tok_src[  6*D*8 +: D*8];
            4'd7:  tok_feed_data_r = tok_src[  7*D*8 +: D*8];
            4'd8:  tok_feed_data_r = tok_src[  8*D*8 +: D*8];
            4'd9:  tok_feed_data_r = tok_src[  9*D*8 +: D*8];
            4'd10: tok_feed_data_r = tok_src[ 10*D*8 +: D*8];
            4'd11: tok_feed_data_r = tok_src[ 11*D*8 +: D*8];
            4'd12: tok_feed_data_r = tok_src[ 12*D*8 +: D*8];
            4'd13: tok_feed_data_r = tok_src[ 13*D*8 +: D*8];
            4'd14: tok_feed_data_r = tok_src[ 14*D*8 +: D*8];
            4'd15: tok_feed_data_r = tok_src[ 15*D*8 +: D*8];
            default: tok_feed_data_r = 0;
        endcase
    end
    assign tok_feed_data = tok_feed_data_r;

    // ══════════════════════════════════════════════════════════════
    // TOP-LEVEL I/O
    // ══════════════════════════════════════════════════════════════
    assign in_ready  = (state == S_IDLE) && pe_ir;
    // mb_or: driven by FSM below
    assign out_valid = head_ov;
    assign out_data  = head_out;
    assign busy      = (state != S_IDLE);

    // ══════════════════════════════════════════════════════════════
    // MAIN FSM
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= S_IDLE;
            mb_iv         <= 0; mb_fs <= 0; mb_or <= 0;
            mb_block_sel  <= 0;
            pool_fs       <= 0; pool_or <= 0;
            tok_feed_cnt  <= 0;
            tok_buf       <= 0;
            b0_tok_buf    <= 0;
            b0_out_cnt    <= 0;
        end else begin
            // Pulse signals default to 0
            mb_fs   <= 0;
            mb_iv   <= 0;
            pool_fs <= 0;

            case (state)
                // ──────────────────────────────────────────────────
                // S_IDLE: Wait for input, PE starts automatically
                // ──────────────────────────────────────────────────
                S_IDLE: begin
                    if (in_valid && pe_ir) begin
                        state <= S_PE;
                    end
                end

                // ──────────────────────────────────────────────────
                // S_PE: Wait for PatchEmbed to produce all L tokens
                // ──────────────────────────────────────────────────
                S_PE: begin
                    if (pe_tdone > 0) begin          // C: token 0 san -> vao BLK0, PE chay tiep SONG SONG
                        tok_feed_cnt <= 0;
                        b0_out_cnt   <= 0;
                        mb_block_sel <= 0;           // pass 0 → block0 weights
                        mb_fs        <= 1;           // frame_start: reset SSM + conv1d
                        mb_or        <= 1;           // ready to accept output
                        state        <= S_BLK0;
                    end
                end

                // ──────────────────────────────────────────────────
                // S_BLK0: Shared block pass-0 (block_sel=0)
                //   Feed PE tokens, collect all L output tokens
                // ──────────────────────────────────────────────────
                S_BLK0: begin
                    // Feed tokens one-by-one — C: CHI feed token PE da chot (overlap PE//block)
                    if (tok_feed_cnt < L) begin
                        if (mb_ir && !mb_iv && (tok_feed_cnt < pe_tdone))
                            mb_iv <= 1;
                        if (mb_iv && mb_ir)
                            tok_feed_cnt <= tok_feed_cnt + 1;
                    end

                    // Collect output tokens
                    if (mb_ov) begin
                        case (b0_out_cnt[$clog2(L)-1:0])
                            4'd0:  b0_tok_buf[  0*D*8 +: D*8] <= mb_out;
                            4'd1:  b0_tok_buf[  1*D*8 +: D*8] <= mb_out;
                            4'd2:  b0_tok_buf[  2*D*8 +: D*8] <= mb_out;
                            4'd3:  b0_tok_buf[  3*D*8 +: D*8] <= mb_out;
                            4'd4:  b0_tok_buf[  4*D*8 +: D*8] <= mb_out;
                            4'd5:  b0_tok_buf[  5*D*8 +: D*8] <= mb_out;
                            4'd6:  b0_tok_buf[  6*D*8 +: D*8] <= mb_out;
                            4'd7:  b0_tok_buf[  7*D*8 +: D*8] <= mb_out;
                            4'd8:  b0_tok_buf[  8*D*8 +: D*8] <= mb_out;
                            4'd9:  b0_tok_buf[  9*D*8 +: D*8] <= mb_out;
                            4'd10: b0_tok_buf[ 10*D*8 +: D*8] <= mb_out;
                            4'd11: b0_tok_buf[ 11*D*8 +: D*8] <= mb_out;
                            4'd12: b0_tok_buf[ 12*D*8 +: D*8] <= mb_out;
                            4'd13: b0_tok_buf[ 13*D*8 +: D*8] <= mb_out;
                            4'd14: b0_tok_buf[ 14*D*8 +: D*8] <= mb_out;
                            4'd15: b0_tok_buf[ 15*D*8 +: D*8] <= mb_out;
                        endcase
                        b0_out_cnt <= b0_out_cnt + 1;

                        if (b0_out_cnt == L - 1) begin
                            // All pass-0 tokens collected → switch to pass-1
                            mb_or        <= 0;
                            tok_buf      <= b0_tok_buf;  // will be fixed up next cycle
                            tok_feed_cnt <= 0;
                            mb_block_sel <= 1;           // pass 1 → block1 weights
                            mb_fs        <= 1;           // frame_start: reset SSM + conv1d
                            mb_or        <= 1;
                            pool_fs      <= 1;           // reset pool accumulator
                            state        <= S_BLK1;
                        end
                    end
                end

                // ──────────────────────────────────────────────────
                // S_BLK1: Shared block pass-1 (block_sel=1)
                //   Feed pass-0 output tokens, pass-1 output → Pool
                // ──────────────────────────────────────────────────
                S_BLK1: begin
                    // Feed tokens one-by-one
                    if (tok_feed_cnt < L) begin
                        if (mb_ir && !mb_iv)
                            mb_iv <= 1;
                        if (mb_iv && mb_ir)
                            tok_feed_cnt <= tok_feed_cnt + 1;
                    end

                    // mb_out → pool via pool_iv_w wire; pool accumulates automatically

                    // When pool is done → go to HEAD
                    if (pool_ov) begin
                        mb_or   <= 0;
                        pool_or <= 1;
                        state   <= S_HEAD;
                    end
                end

                // ──────────────────────────────────────────────────
                // S_HEAD: Wait for OutputHead to finish
                // ──────────────────────────────────────────────────
                S_HEAD: begin
                    if (head_ov && out_ready) begin
                        pool_or <= 0;
                        state   <= S_DONE;
                    end
                end

                // ──────────────────────────────────────────────────
                // S_DONE: Output consumed, return to IDLE
                // ──────────────────────────────────────────────────
                S_DONE: begin
                    state <= S_IDLE;
                end
            endcase

            // Fix-up: last b0 token written same cycle as tok_buf <= b0_tok_buf,
            // so the last entry misses. Patch it in the first cycle of S_BLK1.
            if (state == S_BLK1 && tok_feed_cnt == 0) begin
                case (L - 1)
                    4'd15: tok_buf[15*D*8 +: D*8] <= b0_tok_buf[15*D*8 +: D*8];
                    default: ;
                endcase
            end
        end
    end

    // ══════════════════════════════════════════════════════════════
    // DEBUG (synthesis translate_off)
    // Debug prints disabled for clean output
    // synthesis translate_off
    // (all TOP debug $display removed)
    // synthesis translate_on

endmodule
