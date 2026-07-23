`timescale 1ns / 1ps
//============================================================================
// piecewise_lut.v — Piecewise Linear LUT (eMamba paper §4.2-4.3)
//
// Combinational INT8 → INT8 function approximation.
// Reusable for SiLU (17 segments) and Exp (11 segments).
// [FIX] Exp MODE=1: HI_CONST=128, output UNSIGNED [0,128].
//       exp(0)=1.0 → 128/128 = 1.000 exact (was 127 = 0.992).
//
// Algorithm per element:
//   1. Parallel comparators: find segment (x >= bp[i]?)
//   2. Select slope[seg], intercept[seg]
//   3. y = (slope_int16 × x_int8 + intercept_int16) >> FRAC
//      Giống linear layer: w × x + b >> shift
//   4. Boundary: x < lo → 0, x > hi → x (SiLU) or HI_CONST (Exp)
//   5. Clamp to INT8
//
// Processes N_ELEM elements in parallel (1 instance per element).
// Purely combinational — 0 cycle latency.
// Optional 1-cycle pipeline register (PIPELINE=1).
//
// Resource per element (N_SEG=17):
//   ~17 comparators + priority MUX + 1 MUL(INT16×INT8) + shift/clamp
//   ≈ 1 DSP48 + ~80 LUT
//
// SiLU config: N_SEG=17, MODE=0, x∈[-7,7], outside: 0 / identity
// Exp config:  N_SEG=11, MODE=1, x∈[-4,1], outside: 0 / HI_CONST
//============================================================================

module piecewise_lut #(
    parameter N_SEG    = 17,       // number of linear segments
    parameter FRAC     = 7,        // fractional bits for slope precision
    parameter MODE     = 0,        // 0=SiLU, 1=Exp
    parameter HI_CONST = 128,      // output when x > hi (MODE=1 only). 128=exp(0) exact.
    parameter N_ELEM   = 40,        // elements processed in parallel
    parameter PIPELINE = 0,        // 1 = add output register (1-cycle latency)
    parameter BP_FILE  = "none",   // breakpoints hex (N_SEG+1 INT8)
    parameter SL_FILE  = "none",   // slopes hex (N_SEG INT16)
    parameter IC_FILE  = "none",   // intercepts hex (N_SEG INT16, at acc scale)
    parameter LOAD_MODE   = 0,         // 0=$readmemh (verified), 1=runtime
    parameter [4:0] MY_ID_BP = 5'd0,   // sel cho bảng bp / sl / ic
    parameter [4:0] MY_ID_SL = 5'd0,
    parameter [4:0] MY_ID_IC = 5'd0
)(
    input  wire                     clk,        // only used if PIPELINE=1
    input  wire [N_ELEM*8-1:0]     x_in,       // N_ELEM × INT8 packed
    output wire [N_ELEM*8-1:0]     y_out,      // N_ELEM × INT8 packed
    // ── write-bus dùng chung (chỉ tác dụng khi LOAD_MODE=1) ──
    input  wire                     w_en,
    input  wire                     w_seg_rst,
    input  wire [4:0]               w_sel,
    input  wire                     w_half,     // không dùng (exp single) — giữ cho đồng bộ bus
    input  wire [7:0]               w_byte
);

    // ══════════════════════════════════════════════════════════════
    // LUT TABLES — loaded from hex, treated as ROM by Vivado
    // ══════════════════════════════════════════════════════════════
    reg signed [7:0]  bp [0:N_SEG];       // breakpoints (N_SEG+1 entries) INT8
    reg signed [15:0] sl [0:N_SEG-1];     // slopes      (N_SEG entries)  INT16
    reg signed [15:0] ic [0:N_SEG-1];     // intercepts  (N_SEG entries)  INT16 at acc scale

    initial begin
        if (LOAD_MODE == 0 && BP_FILE != "none") $readmemh(BP_FILE, bp);
        if (LOAD_MODE == 0 && SL_FILE != "none") $readmemh(SL_FILE, sl);
        if (LOAD_MODE == 0 && IC_FILE != "none") $readmemh(IC_FILE, ic);
    end

    // ══════════════════════════════════════════════════════════════
    // WRITE-BUS byte-serial — 3 bảng, 3 sel riêng (bp 1B; sl/ic 2B MSB-first)
    // ══════════════════════════════════════════════════════════════
    generate if (LOAD_MODE) begin : gen_wr
        // bp: 1 byte/entry
        reg [$clog2(N_SEG+2)-1:0] bp_addr;
        always @(posedge clk) if (w_sel == MY_ID_BP) begin
            if (w_seg_rst)      bp_addr <= 0;
            else if (w_en) begin bp[bp_addr] <= w_byte; bp_addr <= bp_addr + 1'b1; end
        end
        // sl: 2 byte/entry
        reg [$clog2(N_SEG > 1 ? N_SEG : 2)-1:0] sl_addr; reg sl_bc; reg [7:0] sl_hi;
        always @(posedge clk) if (w_sel == MY_ID_SL) begin
            if (w_seg_rst) begin sl_addr <= 0; sl_bc <= 0; end
            else if (w_en) begin
                if (sl_bc == 0) begin sl_hi <= w_byte; sl_bc <= 1; end
                else begin sl[sl_addr] <= {sl_hi, w_byte}; sl_addr <= sl_addr + 1'b1; sl_bc <= 0; end
            end
        end
        // ic: 2 byte/entry
        reg [$clog2(N_SEG > 1 ? N_SEG : 2)-1:0] ic_addr; reg ic_bc; reg [7:0] ic_hi;
        always @(posedge clk) if (w_sel == MY_ID_IC) begin
            if (w_seg_rst) begin ic_addr <= 0; ic_bc <= 0; end
            else if (w_en) begin
                if (ic_bc == 0) begin ic_hi <= w_byte; ic_bc <= 1; end
                else begin ic[ic_addr] <= {ic_hi, w_byte}; ic_addr <= ic_addr + 1'b1; ic_bc <= 0; end
            end
        end
    end endgenerate

    // ══════════════════════════════════════════════════════════════
    // PARALLEL PWL COMPUTE — N_ELEM instances
    // ══════════════════════════════════════════════════════════════
    wire [N_ELEM*8-1:0] y_comb;  // combinational result

    genvar e;
    generate
        for (e = 0; e < N_ELEM; e = e + 1) begin : gen_elem

            wire signed [7:0] x_val = $signed(x_in[e*8 +: 8]);

            // ── Step 1: Segment search (parallel comparators) ────
            // Find last i where x >= bp[i+1], giving seg = i+1
            // Breakpoints sorted: bp[0] < bp[1] < ... < bp[N_SEG]
            reg [$clog2(N_SEG > 1 ? N_SEG : 2)-1:0] seg;
            integer si;
            always @(*) begin
                seg = 0;
                for (si = 0; si < N_SEG - 1; si = si + 1)
                    if (x_val >= bp[si + 1])
                        seg = si + 1;
            end

            // ── Step 2: Lookup slope & intercept ─────────────────
            wire signed [15:0] s_val = sl[seg];    // INT16 slope
            wire signed [15:0] i_val = ic[seg];    // INT16 intercept at acc scale

            // ── Step 3: Compute y = (slope_int16 * x_int8 + ic_int16) >> FRAC
            //    Giống linear layer: w × x + b >> shift
            (* use_dsp = "yes" *)
            wire signed [23:0] product    = s_val * x_val;              // INT16 × INT8 = INT24 (DSP48)
            wire signed [23:0] ic_ext     = {{8{i_val[15]}}, i_val};    // sign-extend INT16 → INT24
            wire signed [23:0] sum_val    = product + ic_ext;           // INT24
            wire signed [23:0] shifted    = (sum_val + (1 <<< (FRAC-1))) >>> FRAC;  // round-half-up

            // ── Step 4: Clamp ────────────────────────────────────
            // MODE 0 (SiLU): signed INT8 [-128, 127]
            // MODE 1 (Exp):  unsigned [0, HI_CONST] (exp always ≥ 0)
            wire signed [7:0] y_pwl;
            if (MODE == 0) begin : clamp_silu
                assign y_pwl = (shifted > 127)  ? 8'sd127 :
                               (shifted < -128) ? -8'sd128 :
                               shifted[7:0];
            end else begin : clamp_exp
                // Exp output ≥ 0 always. Clamp [0, HI_CONST].
                assign y_pwl = (shifted > HI_CONST) ? HI_CONST[7:0] :
                               (shifted < 0)        ? 8'd0 :
                               shifted[7:0];
            end

            // ── Step 5: Boundary handling ─────────────────────────
            wire below = (x_val < bp[0]);        // x < lo_bound
            wire above = (x_val >= bp[N_SEG]);   // x >= hi_bound (match Python semantic)

            // MODE 0 (SiLU): below→0, above→x (identity)
            // MODE 1 (Exp):  below→0, above→HI_CONST (128 = exp(0) = 1.0 exact)
            wire [7:0] y_final;  // unsigned for MODE=1 compatibility
            if (MODE == 0) begin : mode_silu
                assign y_final = below ? 8'd0 :
                                 above ? x_val :
                                 y_pwl;
            end else begin : mode_exp
                assign y_final = below ? 8'd0 :
                                 above ? HI_CONST[7:0] :
                                 y_pwl;
            end

            assign y_comb[e*8 +: 8] = y_final;
        end
    endgenerate

    // ══════════════════════════════════════════════════════════════
    // OUTPUT — combinational or pipelined
    // ══════════════════════════════════════════════════════════════
    generate
        if (PIPELINE) begin : gen_pipe
            reg [N_ELEM*8-1:0] y_reg;
            always @(posedge clk)
                y_reg <= y_comb;
            assign y_out = y_reg;
        end else begin : gen_comb
            assign y_out = y_comb;
        end
    endgenerate

endmodule
