`timescale 1ns / 1ps
//============================================================================
// mamba_block.v — Mamba Block with Pipelined Token Processing (paper §5)
//
// Multi-stage pipeline: each stage operates independently with local
// valid/ready handshake. When token t leaves a stage, that stage
// immediately accepts token t+1 from upstream.
//
// Pipeline stages (5 macro-stages):
//   S0: RangeNorm + Upper/Lower Linear + SiLU    (~53 cyc)
//   S1: Conv1D + ReLU                             (~7 cyc)
//   S2: SSM projections (delta+B+C parallel)      (~43 cyc)
//   S3: SSM recurrence                            (~43 cyc) ← BOTTLENECK
//   S4: Gating + OutProj + Residual               (~24 cyc)
//
// Throughput: 1 token / 43 cycles (SSM recurrence bottleneck)
// First token latency: ~170 cycles
// L=16 total: 170 + 15×43 = ~815 cycles/block
//
// Each stage has:
//   - Input register (latched data from previous stage)
//   - Valid bit (data available)
//   - Busy flag (processing in progress)
//   - Done flag (result ready for next stage)
//============================================================================

module mamba_block #(
    parameter D              = 20,
    parameter ED             = 40,
    parameter L              = 16,
    parameter N_STATE        = 8,
    parameter K              = 4,
    parameter N_CU           = 20,
    parameter GAMMA_SHIFT    = 0,
    parameter UPPER_SHIFT    = 7,
    parameter LOWER_SHIFT    = 7,
    parameter CONV_SHIFT     = 6,
    parameter DELTA_SHIFT    = 7,
    parameter DT_IN_SHIFT    = 6,
    parameter B_SHIFT        = 7,
    parameter C_SHIFT        = 6,
    parameter OUT_SHIFT      = 7,
    parameter SSM_Y_SHIFT    = 10,
    parameter SSM_DA_SHIFT   = 3,            // dA shift (calibrated)
    parameter SSM_DB_SHIFT   = 3,            // B_bar shift (calibrated)
    parameter SSM_DX_ALIGN   = 8,            // D×x alignment shift (calibrated)
    parameter SSM_PER_CH_YSHIFT = 0,         // 1 = per-channel Y_SHIFT
    parameter SSM_YSHIFT_FILE    = "none",   // per-channel Y_SHIFT hex (block0)
    parameter SSM_YSHIFT_FILE_B1 = "none",   // per-channel Y_SHIFT hex (block1, SHARED only)
    parameter GATING_SHIFT   = 7,
    // ── Shared block mode ─────────────────────────────────────────
    parameter SHARED             = 0,        // 1 = shared double-weight mode
    // ── Block-0 weight files (also used as sole files when SHARED=0) ─
    parameter GAMMA_FILE         = "none",
    parameter BETA_FILE          = "none",
    parameter UPPER_W_FILE       = "none",
    parameter UPPER_B_FILE       = "none",
    parameter LOWER_W_FILE       = "none",
    parameter LOWER_B_FILE       = "none",
    parameter CONV_W_FILE        = "none",
    parameter CONV_B_FILE        = "none",
    parameter DELTA_W_FILE       = "none",
    parameter DELTA_B_FILE       = "none",
    parameter DT_IN_W_FILE       = "none",
    parameter DT_IN_B_FILE       = "none",
    parameter B_W_FILE           = "none",
    parameter C_W_FILE           = "none",
    parameter OUT_W_FILE         = "none",
    parameter OUT_B_FILE         = "none",
    parameter A_FILE             = "none",
    parameter D_FILE             = "none",
    parameter SILU_LUT_FILE      = "none",   // 256-entry SiLU direct LUT (GATE: SiLU(z))
    parameter CONV_SILU_LUT      = "none",   // SiLU SAU conv: in 2^-4 -> out a_silu1
    parameter EXP_BP_FILE        = "none",   // PWL Exp breakpoints (11-seg)
    parameter EXP_SL_FILE        = "none",   // PWL Exp slopes (INT16)
    parameter EXP_IC_FILE        = "none",   // PWL Exp intercepts (INT16)
    // ── Block-1 weight files (SHARED=1 only) ─────────────────────
    parameter GAMMA_FILE_B1      = "none",
    parameter BETA_FILE_B1       = "none",
    parameter UPPER_W_FILE_B1    = "none",
    parameter UPPER_B_FILE_B1    = "none",
    parameter LOWER_W_FILE_B1    = "none",
    parameter LOWER_B_FILE_B1    = "none",
    parameter CONV_W_FILE_B1     = "none",
    parameter CONV_B_FILE_B1     = "none",
    parameter DELTA_W_FILE_B1    = "none",
    parameter DELTA_B_FILE_B1    = "none",
    parameter DT_IN_W_FILE_B1    = "none",
    parameter DT_IN_B_FILE_B1    = "none",
    parameter B_W_FILE_B1        = "none",
    parameter C_W_FILE_B1        = "none",
    parameter OUT_W_FILE_B1      = "none",
    parameter OUT_B_FILE_B1      = "none",
    parameter A_FILE_B1          = "none",
    parameter D_FILE_B1          = "none",
    parameter SILU_LUT_FILE_B1   = "none",
    parameter CONV_SILU_LUT_B1   = "none",
    parameter LOAD_MODE          = 0         // 0=$readmemh (verified) ; 1=runtime load
)(
    input  wire                clk,
    input  wire                rst_n,
    // Shared block select (0=block0, 1=block1); ignored when SHARED=0
    input  wire                block_sel,
    input  wire                frame_start,
    input  wire [4:0]          shift_upper,
    input  wire [4:0]          shift_lower,
    input  wire [4:0]          shift_conv,
    input  wire [4:0]          shift_out,
    input  wire [4:0]          shift_res,     // legacy residual b_shift
    input  wire [3:0]          res_a_lshift,  // residual a left-shift
    input  wire [3:0]          res_b_lshift,  // residual b left-shift (b_scale > a_scale)
    input  wire [3:0]          res_b_rshift,  // residual b right-shift (b_scale < a_scale)
    input  wire [4:0]          ssm_y_shift_in,// SSM Y_SHIFT runtime override per-block
    input  wire [3:0]          ssm_da_shift_in,// SSM DA_SHIFT per-block
    input  wire [3:0]          ssm_db_shift_in,// SSM DB_SHIFT per-block
    input  wire [3:0]          ssm_dx_align_in,// SSM DX_ALIGN per-block
    input  wire signed [3:0]   gamma_shift_in,
    input  wire [4:0]          shift_delta,   // runtime delta proj shift (0=use param)
    input  wire [4:0]          shift_dtin,    // runtime dt_in shift (0=use param)
    input  wire [4:0]          shift_bproj,   // runtime B proj shift (0=use param)
    input  wire [4:0]          shift_cproj,   // runtime C proj shift (0=use param)
    input  wire [4:0]          gating_shift_in,// runtime gating shift (block_sel switch)
    input  wire                in_valid,
    output wire                in_ready,
    input  wire [D*8-1:0]      in_data,
    output wire                out_valid,
    input  wire                out_ready,
    output wire [D*8-1:0]      out_data,

    // write-bus runtime-load (bundled): [15:8]=byte [7]=half [6:2]=sel [1]=seg_rst [0]=en
    input  wire [15:0]         wbus
);

    // ══════════════════════════════════════════════════════════════
    // PIPELINE STAGE REGISTERS
    //
    // Each stage: data reg + valid + busy + sub-FSM
    // Handshake: stage accepts when !busy, produces when done
    // ══════════════════════════════════════════════════════════════

    // ── Stage 0 input latch ─────────────────────────────────────────
    reg [D*8-1:0]   s0_input_latch;   // latched in_data (stable during RN processing)

    // ── Stage 0 output: norm done, upper+lower done ──────────────
    reg [D*8-1:0]   s0_residual;      // saved input for skip connection
    reg [ED*8-1:0]  s0_upper_silu;    // SiLU(upper_lin) result
    reg [ED*8-1:0]  s0_lower;         // lower_lin result
    reg             s0_valid;          // data ready for stage 1
    reg             s0_busy;

    // ── Stage 1 input latch + output ─────────────────────────────
    reg [ED*8-1:0]  s1_lower_latch;   // latched s0_lower for conv input
    reg [D*8-1:0]   s1_residual;
    reg [ED*8-1:0]  s1_upper_silu;
    reg [ED*8-1:0]  s1_conv_relu;     // conv1d + relu result
    reg             s1_valid;
    reg             s1_busy;

    // ── Stage 2 output: SSM projections done ─────────────────────
    reg [D*8-1:0]   s2_residual;
    reg [ED*8-1:0]  s2_upper_silu;
    reg [ED*8-1:0]  s2_conv_relu;     // x for SSM
    reg [ED*8-1:0]  s2_delta_relu;
    reg [N_STATE*8-1:0] s2_B, s2_C;
    reg             s2_valid;
    reg             s2_busy;

    // ── Stage 2b (B): dt_proj rieng (tach khoi S2 de bottleneck 49->43) ──
    reg [D*8-1:0]   s2b_residual;
    reg [ED*8-1:0]  s2b_upper_silu;
    reg [ED*8-1:0]  s2b_conv_relu;
    reg [ED*8-1:0]  s2b_delta_relu;
    reg [N_STATE*8-1:0] s2b_B, s2b_C;
    reg             s2b_valid;
    reg             s2b_busy;

    // ── Stage 3 output: SSM recurrence done ──────────────────────
    reg [D*8-1:0]   s3_residual;
    reg [ED*8-1:0]  s3_upper_silu;
    reg [ED*8-1:0]  s3_ssm_out;
    reg             s3_valid;
    reg             s3_busy;

    // ── Stage 4 output: gating + outproj + residual ──────────────
    reg [D*8-1:0]   s4_result;
    reg             s4_valid;
    reg             s4_busy;

    // ══════════════════════════════════════════════════════════════
    // SUB-MODULE WIRES
    // ══════════════════════════════════════════════════════════════
    // RangeNorm
    reg  rn_iv; wire rn_ir, rn_ov; reg rn_or;
    wire [D*8-1:0] rn_out;

    // Upper/Lower Linear
    reg  up_iv, lo_iv; wire up_ir, lo_ir, up_ov, lo_ov; reg up_or, lo_or;
    wire [ED*8-1:0] up_out, lo_out;

    // SiLU (tuần tự qua BRAM — handshake start/done)
    wire [ED*8-1:0] silu_out;
    reg  silu_start;       wire silu_done;
    reg  silu_conv_start;  wire silu_conv_done;

    // Conv1D
    reg  cv_iv; wire cv_ir, cv_ov; reg cv_or;
    wire [ED*8-1:0] cv_out;
    wire [ED*8-1:0] relu_out;

    // SSM projections
    reg  dp_iv, bp_iv, cp_iv;
    wire dp_ir, bp_ir, cp_ir, dp_ov, bp_ov, cp_ov;
    reg  dp_or, bp_or, cp_or;
    wire [ED*8-1:0] dp_out, dp_relu;
    wire [N_STATE*8-1:0] bp_out, cp_out;

    // SSM recurrence
    reg  ssm_start; wire ssm_done;
    wire [ED*8-1:0] ssm_y;

    // Gating (combinational)
    wire [ED*8-1:0] gate_out;

    // OutProj
    reg  op_iv; wire op_ir, op_ov; reg op_or;
    wire [D*8-1:0] op_out;

    // Residual (combinational)
    wire [D*8-1:0] res_out;

    // Frame control
    reg frame_start_r;
    wire do_clear = frame_start_r;
    reg [$clog2(L):0] tok_out_cnt;

    // ══════════════════════════════════════════════════════════════
    // SUB-MODULE INSTANCES
    // ══════════════════════════════════════════════════════════════
    // RangeNorm: recip-multiply (1/range, K=23, FSMD 4 cycle/token thay divider 31).
    // Cau hinh DIM=20/SHARED/N_CU=20/IDs/K=23 da HARDCODE trong range_norm_recip.
    // (Ban divider goc: range_norm — giu lai trong rtl/range_norm.v de tham chieu.)
    range_norm_recip #(
        .GAMMA_FILE(GAMMA_FILE),    .BETA_FILE(BETA_FILE),
        .GAMMA_FILE_B1(GAMMA_FILE_B1), .BETA_FILE_B1(BETA_FILE_B1),
        .LOAD_MODE(LOAD_MODE))
    u_norm (.clk(clk), .rst_n(rst_n),
        .block_sel(block_sel), .gamma_shift_in(gamma_shift_in),
        .in_valid(rn_iv), .in_ready(rn_ir), .in_data(s0_input_latch),
        .out_valid(rn_ov), .out_ready(rn_or), .out_data(rn_out), .wbus(wbus));

    linear_layer #(.IN_DIM(D), .OUT_DIM(ED), .SHIFT_RIGHT(UPPER_SHIFT), .USE_BIAS(1),
        .SHARED(SHARED),
        .WEIGHT_FILE(UPPER_W_FILE),    .BIAS_FILE(UPPER_B_FILE),
        .WEIGHT_FILE_B1(UPPER_W_FILE_B1), .BIAS_FILE_B1(UPPER_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd4), .BID(5'd5))
    u_upper (.clk(clk), .rst_n(rst_n), .shift_right(shift_upper),
        .block_sel(block_sel),
        .in_valid(up_iv), .in_ready(up_ir), .in_data(rn_out),
        .out_valid(up_ov), .out_ready(up_or), .out_data(up_out), .wbus(wbus));

    linear_layer #(.IN_DIM(D), .OUT_DIM(ED), .SHIFT_RIGHT(LOWER_SHIFT), .USE_BIAS(1),
        .SHARED(SHARED),
        .WEIGHT_FILE(LOWER_W_FILE),    .BIAS_FILE(LOWER_B_FILE),
        .WEIGHT_FILE_B1(LOWER_W_FILE_B1), .BIAS_FILE_B1(LOWER_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd6), .BID(5'd7))
    u_lower (.clk(clk), .rst_n(rst_n), .shift_right(shift_lower),
        .block_sel(block_sel),
        .in_valid(lo_iv), .in_ready(lo_ir), .in_data(rn_out),
        .out_valid(lo_ov), .out_ready(lo_or), .out_data(lo_out), .wbus(wbus));

    // SiLU PWL: thay direct_lut (42 cyc tuan tu) bang silu_pwl_comp (2 cyc, 40 lane song song).
    //   1 silu_rom dung chung (port A=gate, port B=conv) ; 2 datapath rieng (gate, conv).
    //   Tai dung 4 o tham so cu: SILU_LUT_FILE/_B1 = base path gate ; CONV_SILU_LUT/_B1 = base conv
    //   (silu_rom tu them _bp/_sl/_ic.hex). Interface start/done GIU NGUYEN -> FSM khong doi.
    localparam SILU_NSEG = 17;
    wire [(SILU_NSEG+1)*16-1:0] silu_bpA, silu_bpB;   // bp INT16 (gate bp vượt ±127)
    wire [SILU_NSEG*16-1:0]    silu_slA, silu_icA, silu_slB, silu_icB;
    silu_rom #(.N_SEG(SILU_NSEG),
        .GATE_B0(SILU_LUT_FILE), .GATE_B1(SILU_LUT_FILE_B1),
        .CV_B0(CONV_SILU_LUT),   .CV_B1(CONV_SILU_LUT_B1),
        .LOAD_MODE(LOAD_MODE))
    u_silu_rom (.clk(clk),
        .sel_a_blk(block_sel), .bpA(silu_bpA), .slA(silu_slA), .icA(silu_icA),
        .sel_b_blk(block_sel), .bpB(silu_bpB), .slB(silu_slB), .icB(silu_icB));

    // datapath GATE (vi tri 1): SiLU(up_out)
    silu_pwl_comp #(.N_ELEM(ED), .N_SEG(SILU_NSEG), .FRAC(7))
    u_silu (.clk(clk), .rst_n(rst_n), .start(silu_start), .x_in(up_out),
        .bp_bus(silu_bpA), .sl_bus(silu_slA), .ic_bus(silu_icA),
        .y_out(silu_out), .done(silu_done));

    conv1d_dw #(.ED(ED), .K(K), .SHIFT_RIGHT(CONV_SHIFT),
        .SHARED(SHARED),
        .WEIGHT_FILE(CONV_W_FILE),    .BIAS_FILE(CONV_B_FILE),
        .WEIGHT_FILE_B1(CONV_W_FILE_B1), .BIAS_FILE_B1(CONV_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd8), .BID(5'd9))
    u_conv (.clk(clk), .rst_n(rst_n), .seq_clear(do_clear),
        .block_sel(block_sel), .shift_right(shift_conv),
        .in_valid(cv_iv), .in_ready(cv_ir), .in_data(s1_lower_latch),
        .out_valid(cv_ov), .out_ready(cv_or), .out_data(cv_out), .wbus(wbus));

    // SiLU SAU Conv1D (khớp model: xx -> conv -> SiLU -> SSM): datapath CONV (vi tri 2).
    //   silu_pwl_comp 2 cyc; bang tu silu_rom port B (CV_B0/B1).
    wire [ED*8-1:0] silu_conv_out;
    silu_pwl_comp #(.N_ELEM(ED), .N_SEG(SILU_NSEG), .FRAC(7))
    u_silu_conv (.clk(clk), .rst_n(rst_n), .start(silu_conv_start), .x_in(cv_out),
        .bp_bus(silu_bpB), .sl_bus(silu_slB), .ic_bus(silu_icB),
        .y_out(silu_conv_out), .done(silu_conv_done));
    assign relu_out = silu_conv_out;

    // delta 2 BƯỚC (đúng model): dt_in (ED→DT_RANK → dt_in.out_q) chuỗi-> dt_proj (DT_RANK→ED → a_delta)
    localparam DT_RANK = (ED + 15) / 16;     // ceil(ED/16) = 3 cho ED=40
    wire [DT_RANK*8-1:0] dtin_out;
    wire dtin_ov;
    reg  dtin_or;                  // B: S2 giu/nha output dt_in
    reg  [DT_RANK*8-1:0] s2_dtin;  // B: chot dt_in.out o S2, dua sang S2b
    reg  dtp_iv;  wire dtp_ir;     // B: S2b khoi dong dt_proj
    linear_layer #(.IN_DIM(ED), .OUT_DIM(DT_RANK), .SHIFT_RIGHT(DT_IN_SHIFT), .USE_BIAS(1),
        .SHARED(SHARED),
        .WEIGHT_FILE(DT_IN_W_FILE),    .BIAS_FILE(DT_IN_B_FILE),
        .WEIGHT_FILE_B1(DT_IN_W_FILE_B1), .BIAS_FILE_B1(DT_IN_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd12), .BID(5'd13))
    u_dt_in (.clk(clk), .rst_n(rst_n), .shift_right(shift_dtin),
        .block_sel(block_sel),
        .in_valid(dp_iv), .in_ready(dp_ir), .in_data(s2_conv_relu),
        .out_valid(dtin_ov), .out_ready(dtin_or), .out_data(dtin_out), .wbus(wbus));

    linear_layer #(.IN_DIM(DT_RANK), .OUT_DIM(ED), .SHIFT_RIGHT(DELTA_SHIFT), .USE_BIAS(1),
        .SHARED(SHARED),
        .WEIGHT_FILE(DELTA_W_FILE),    .BIAS_FILE(DELTA_B_FILE),
        .WEIGHT_FILE_B1(DELTA_W_FILE_B1), .BIAS_FILE_B1(DELTA_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd10), .BID(5'd11))
    u_dt_proj (.clk(clk), .rst_n(rst_n), .shift_right(shift_delta),
        .block_sel(block_sel),
        .in_valid(dtp_iv), .in_ready(dtp_ir), .in_data(s2_dtin),
        .out_valid(dp_ov), .out_ready(dp_or), .out_data(dp_out), .wbus(wbus));

    relu_unit #(.N_ELEM(ED)) u_drelu (.x_in(dp_out), .y_out(dp_relu));

    linear_layer #(.IN_DIM(ED), .OUT_DIM(N_STATE), .SHIFT_RIGHT(B_SHIFT), .USE_BIAS(0),
        .SHARED(SHARED),
        .WEIGHT_FILE(B_W_FILE),    .BIAS_FILE("none"),
        .WEIGHT_FILE_B1(B_W_FILE_B1), .BIAS_FILE_B1("none"),
        .LOAD_MODE(LOAD_MODE), .WID(5'd14))
    u_bproj (.clk(clk), .rst_n(rst_n), .shift_right(shift_bproj),
        .block_sel(block_sel),
        .in_valid(bp_iv), .in_ready(bp_ir), .in_data(s2_conv_relu),
        .out_valid(bp_ov), .out_ready(bp_or), .out_data(bp_out), .wbus(wbus));

    linear_layer #(.IN_DIM(ED), .OUT_DIM(N_STATE), .SHIFT_RIGHT(C_SHIFT), .USE_BIAS(0),
        .SHARED(SHARED),
        .WEIGHT_FILE(C_W_FILE),    .BIAS_FILE("none"),
        .WEIGHT_FILE_B1(C_W_FILE_B1), .BIAS_FILE_B1("none"),
        .LOAD_MODE(LOAD_MODE), .WID(5'd15))
    u_cproj (.clk(clk), .rst_n(rst_n), .shift_right(shift_cproj),
        .block_sel(block_sel),
        .in_valid(cp_iv), .in_ready(cp_ir), .in_data(s2_conv_relu),
        .out_valid(cp_ov), .out_ready(cp_or), .out_data(cp_out), .wbus(wbus));

    ssm_core #(.ED(ED), .N(N_STATE), .Y_SHIFT(SSM_Y_SHIFT),
        .DA_SHIFT(SSM_DA_SHIFT), .DB_SHIFT(SSM_DB_SHIFT), .DX_ALIGN(SSM_DX_ALIGN),
        .SHARED(SHARED),
        .PER_CH_YSHIFT(SSM_PER_CH_YSHIFT),
        .YSHIFT_FILE(SSM_YSHIFT_FILE),
        .YSHIFT_FILE_B1(SSM_YSHIFT_FILE_B1),
        .A_FILE(A_FILE),          .D_FILE(D_FILE),
        .EXP_BP_FILE(EXP_BP_FILE), .EXP_SL_FILE(EXP_SL_FILE), .EXP_IC_FILE(EXP_IC_FILE),
        .A_FILE_B1(A_FILE_B1),    .D_FILE_B1(D_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .A_ID(5'd18), .D_ID(5'd19), .BP_ID(5'd22), .SL_ID(5'd23), .IC_ID(5'd24))
    u_ssm (.clk(clk), .rst_n(rst_n),
        .block_sel(block_sel),
        .y_shift_rt(ssm_y_shift_in),
        .da_shift_rt(ssm_da_shift_in), .db_shift_rt(ssm_db_shift_in), .dx_align_rt(ssm_dx_align_in),
        .token_start(ssm_start), .seq_clear(do_clear),
        .x_in(s2b_conv_relu), .delta_in(s2b_delta_relu),
        .B_in(s2b_B), .C_in(s2b_C),
        .y_out(ssm_y), .done(ssm_done), .wbus(wbus));

    elem_multiply #(.N_ELEM(ED), .SHIFT(GATING_SHIFT), .PIPELINE(0))
    u_gate (.clk(clk), .shift_right(gating_shift_in),   // runtime: B0=5 / B1=4 (trước kẹt param B0)
        .a_in(s3_upper_silu), .b_in(s3_ssm_out), .y_out(gate_out));

    linear_layer #(.IN_DIM(ED), .OUT_DIM(D), .SHIFT_RIGHT(OUT_SHIFT), .USE_BIAS(1),
        .SHARED(SHARED),
        .WEIGHT_FILE(OUT_W_FILE),    .BIAS_FILE(OUT_B_FILE),
        .WEIGHT_FILE_B1(OUT_W_FILE_B1), .BIAS_FILE_B1(OUT_B_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .WID(5'd16), .BID(5'd17))
    u_outproj (.clk(clk), .rst_n(rst_n), .shift_right(shift_out),
        .block_sel(block_sel),
        .in_valid(op_iv), .in_ready(op_ir), .in_data(gate_out),
        .out_valid(op_ov), .out_ready(op_or), .out_data(op_out), .wbus(wbus));

    // FIX: a=out_proj (giữ scale s_bo), b=residual → b_ls/b_rs dịch ĐÚNG residual để align.
    // (Trước: a=residual,b=out_proj → b_rshift dịch nhầm out_proj, sai block1.)
    reg [D*8-1:0] s4_residual;   // khai báo TRƯỚC chỗ dùng (residual_add) — tránh implicit wire 1-bit
    residual_add #(.N_ELEM(D)) u_res (
        .a_in(op_out), .b_in(s4_residual),
        .a_lshift_rt(res_a_lshift),
        .b_lshift_rt(res_b_lshift),
        .b_rshift_rt(res_b_rshift),
        .b_shift(shift_res),
        .y_out(res_out));

    // ══════════════════════════════════════════════════════════════
    // TOP-LEVEL I/O
    // ══════════════════════════════════════════════════════════════
    // Input accepted when stage 0 not busy
    assign in_ready = !s0_busy && !s0_valid && (tok_out_cnt < L);  // +!s0_valid: backpressure (khong nhan khi output chua tieu thu)
    assign out_valid = s4_valid;
    assign out_data  = s4_result;

    // ══════════════════════════════════════════════════════════════
    // STAGE 0: RangeNorm + Upper/Lower Linear + SiLU
    // Sub-FSM: IDLE → NORM → LIN → LATCH
    // ══════════════════════════════════════════════════════════════
    localparam S0_IDLE = 2'd0, S0_NORM = 2'd1, S0_LIN = 2'd2, S0_SILU = 2'd3;
    reg [1:0] s0_fsm;

    // Debug prints disabled
    // synthesis translate_off
    // (S0 debug $display removed)
    // synthesis translate_on

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s0_fsm <= S0_IDLE; s0_busy <= 0; s0_valid <= 0;
            s0_input_latch <= 0;
            s0_residual <= 0; s0_upper_silu <= 0; s0_lower <= 0;
            rn_iv <= 0; rn_or <= 0; up_iv <= 0; up_or <= 0; lo_iv <= 0; lo_or <= 0;
            frame_start_r <= 0; tok_out_cnt <= 0; silu_start <= 0;
        end else begin
            rn_iv <= 0; up_iv <= 0; lo_iv <= 0; silu_start <= 0;
            frame_start_r <= 0;

            // tok_out_cnt: single driver (avoids multi-driven net)
            if (frame_start && s0_fsm == S0_IDLE)
                tok_out_cnt <= 0;
            else if (s4_valid && out_ready)
                tok_out_cnt <= tok_out_cnt + 1;

            // Frame start flag
            if (frame_start && s0_fsm == S0_IDLE) begin
                frame_start_r <= 1;
            end

            // S0 output consumed by S1
            if (s0_valid && !s1_busy && !s1_valid) s0_valid <= 0;  // xoa KHOP voi accept S1

            case (s0_fsm)
                S0_IDLE: begin
                    if (in_valid && !s0_busy && !s0_valid && (tok_out_cnt < L || frame_start)) begin
                        s0_busy <= 1;
                        s0_input_latch <= in_data;
                        s0_residual <= in_data;
                        rn_iv <= 1;
                        s0_fsm <= S0_NORM;
                    end
                end

                S0_NORM: begin
                    rn_or <= 1;
                    if (rn_ov) begin
                        rn_or <= 0;
                        up_iv <= 1;
                        lo_iv <= 1;
                        s0_fsm <= S0_LIN;
                    end
                end

                S0_LIN: begin
                    up_or <= 1; lo_or <= 1;
                    if (up_ov && lo_ov) begin
                        up_or <= 0; lo_or <= 0;
                        s0_lower   <= lo_out;
                        silu_start <= 1;            // chốt up_out -> tra SiLU tuần tự (BRAM)
                        s0_fsm     <= S0_SILU;
                    end
                end

                S0_SILU: begin                      // chờ SiLU đọc xong
                    if (silu_done) begin
                        s0_upper_silu <= silu_out;
                        s0_valid <= 1;
                        s0_busy  <= 0;
                        s0_fsm   <= S0_IDLE;
                    end
                end
            endcase
        end
    end

    // ══════════════════════════════════════════════════════════════
    // STAGE 1: Conv1D + ReLU
    // ══════════════════════════════════════════════════════════════
    localparam S1_IDLE = 2'd0, S1_CONV = 2'd1, S1_SILU = 2'd2;
    reg [1:0] s1_fsm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s1_fsm <= S1_IDLE; s1_busy <= 0; s1_valid <= 0;
            s1_lower_latch <= 0;
            s1_residual <= 0; s1_upper_silu <= 0; s1_conv_relu <= 0;
            cv_iv <= 0; cv_or <= 0; silu_conv_start <= 0;
        end else begin
            cv_iv <= 0; silu_conv_start <= 0;
            if (s1_valid && !s2_busy && !s2_valid) s1_valid <= 0;  // xoa KHOP voi accept S2

            case (s1_fsm)
                S1_IDLE: begin
                    if (s0_valid && !s1_busy && !s1_valid) begin
                        s1_busy <= 1;
                        s1_residual <= s0_residual;
                        s1_upper_silu <= s0_upper_silu;
                        s1_lower_latch <= s0_lower;  // latch before S0 overwrites
                        cv_iv <= 1;
                        s1_fsm <= S1_CONV;
                    end
                end

                S1_CONV: begin
                    cv_or <= 1;
                    if (cv_ov) begin
                        cv_or <= 0;
                        silu_conv_start <= 1;       // chốt cv_out -> tra SiLU-conv tuần tự
                        s1_fsm <= S1_SILU;
                    end
                end

                S1_SILU: begin                      // chờ SiLU-conv đọc xong
                    if (silu_conv_done) begin
                        s1_conv_relu <= relu_out;
                        s1_valid <= 1;
                        s1_busy  <= 0;
                        s1_fsm   <= S1_IDLE;
                    end
                end
            endcase
        end
    end

    // ══════════════════════════════════════════════════════════════
    // STAGE 2: SSM Projections (delta+B+C parallel)
    // ══════════════════════════════════════════════════════════════
    localparam S2_IDLE = 1'd0, S2_PROJ = 1'd1;
    reg s2_fsm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2_fsm <= S2_IDLE; s2_busy <= 0; s2_valid <= 0;
            s2_residual <= 0; s2_upper_silu <= 0; s2_conv_relu <= 0;
            s2_dtin <= 0; s2_B <= 0; s2_C <= 0;
            dp_iv <= 0; dtin_or <= 0;
            bp_iv <= 0; bp_or <= 0;
            cp_iv <= 0; cp_or <= 0;
        end else begin
            dp_iv <= 0; bp_iv <= 0; cp_iv <= 0;
            dtin_or <= 0; bp_or <= 0; cp_or <= 0;   // out_ready mac dinh 0
            if (s2_valid && !s2b_busy && !s2b_valid) s2_valid <= 0;  // xoa KHOP voi accept S2b

            case (s2_fsm)
                S2_IDLE: begin
                    if (s1_valid && !s2_busy && !s2_valid) begin
                        s2_busy <= 1;
                        s2_residual <= s1_residual;
                        s2_upper_silu <= s1_upper_silu;
                        s2_conv_relu <= s1_conv_relu;
                        dp_iv <= 1; bp_iv <= 1; cp_iv <= 1;
                        s2_fsm <= S2_PROJ;
                    end
                end

                S2_PROJ: begin
                    // FIX: chỉ consume (out_ready) khi CẢ 3 xong — delta 2-bước chậm hơn B/C nên
                    // trước đây bp_or/cp_or bật sớm → B/C bị nhả trước khi delta xong → kẹt.
                    if (dtin_ov && bp_ov && cp_ov) begin   // B: cho dt_IN (43) + B/C, KHONG cho dt_proj
                        dtin_or <= 1; bp_or <= 1; cp_or <= 1;
                        s2_dtin <= dtin_out;
                        s2_B <= bp_out;
                        s2_C <= cp_out;
                        s2_valid <= 1;
                        s2_busy <= 0;
                        s2_fsm <= S2_IDLE;
                    end
                end
            endcase
        end
    end

    // ══════════════════════════════════════════════════════════════
    // STAGE 2b (B): dt_proj rieng -> tach chuoi dt (49) -> dt_in(43)|dt_proj(6)
    // ══════════════════════════════════════════════════════════════
    localparam S2B_IDLE = 1'd0, S2B_PROJ = 1'd1;
    reg s2b_fsm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s2b_fsm <= S2B_IDLE; s2b_busy <= 0; s2b_valid <= 0;
            s2b_residual <= 0; s2b_upper_silu <= 0; s2b_conv_relu <= 0;
            s2b_delta_relu <= 0; s2b_B <= 0; s2b_C <= 0;
            dtp_iv <= 0; dp_or <= 0;
        end else begin
            dtp_iv <= 0; dp_or <= 0;
            if (s2b_valid && !s3_busy && !s3_valid) s2b_valid <= 0;  // xoa KHOP voi accept S3

            case (s2b_fsm)
                S2B_IDLE: begin
                    if (s2_valid && !s2b_busy && !s2b_valid) begin
                        s2b_busy <= 1;
                        s2b_residual   <= s2_residual;
                        s2b_upper_silu <= s2_upper_silu;
                        s2b_conv_relu  <= s2_conv_relu;
                        s2b_B <= s2_B;
                        s2b_C <= s2_C;
                        dtp_iv <= 1;               // khoi dong dt_proj voi s2_dtin
                        s2b_fsm <= S2B_PROJ;
                    end
                end
                S2B_PROJ: begin
                    if (dp_ov) begin
                        dp_or <= 1;
                        s2b_delta_relu <= dp_relu;
                        s2b_valid <= 1;
                        s2b_busy <= 0;
                        s2b_fsm <= S2B_IDLE;
                    end
                end
            endcase
        end
    end

    // ══════════════════════════════════════════════════════════════
    // STAGE 3: SSM Recurrence (BOTTLENECK — 43 cycles)
    // ══════════════════════════════════════════════════════════════
    localparam S3_IDLE = 1'd0, S3_REC = 1'd1;
    reg s3_fsm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s3_fsm <= S3_IDLE; s3_busy <= 0; s3_valid <= 0;
            s3_residual <= 0; s3_upper_silu <= 0; s3_ssm_out <= 0;
            ssm_start <= 0;
        end else begin
            ssm_start <= 0;
            if (s3_valid && !s4_busy) s3_valid <= 0;

            case (s3_fsm)
                S3_IDLE: begin
                    // accept KHOP voi s2b_valid-clear (s2b_valid && !s3_busy && !s3_valid)
                    if (s2b_valid && !s3_busy && !s3_valid) begin
                        s3_busy <= 1;
                        s3_residual   <= s2b_residual;
                        s3_upper_silu <= s2b_upper_silu;
                        ssm_start <= 1;
                        s3_fsm <= S3_REC;
                    end
                end

                S3_REC: begin
                    if (ssm_done) begin
                        s3_ssm_out <= ssm_y;
                        s3_valid <= 1;
                        s3_busy <= 0;
                        s3_fsm <= S3_IDLE;
                    end
                end
            endcase
        end
    end

    // ══════════════════════════════════════════════════════════════
    // STAGE 4: Gating + OutProj + Residual
    // ══════════════════════════════════════════════════════════════
    localparam S4_IDLE = 2'd0, S4_GATE = 2'd1, S4_OUTPROJ = 2'd2, S4_DONE = 2'd3;
    reg [1:0] s4_fsm;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s4_fsm <= S4_IDLE; s4_busy <= 0; s4_valid <= 0;
            s4_result <= 0; s4_residual <= 0;
            op_iv <= 0; op_or <= 0;
        end else begin
            op_iv <= 0;
            if (s4_valid && out_ready) begin
                s4_valid <= 0;
            end

            case (s4_fsm)
                S4_IDLE: begin
                    if (s3_valid && !s4_busy) begin
                        s4_busy <= 1;
                        s4_residual <= s3_residual;
                        op_iv <= 1;
                        s4_fsm <= S4_OUTPROJ;
                    end
                end

                S4_OUTPROJ: begin
                    op_or <= 1;
                    if (op_ov) begin
                        op_or <= 0;
                        s4_result <= res_out;
                        s4_valid <= 1;
                        s4_busy <= 0;
                        s4_fsm <= S4_IDLE;
                    end
                end
            endcase
        end
    end

endmodule
