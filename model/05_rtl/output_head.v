`timescale 1ns / 1ps
//============================================================================
// output_head.v — Residual MLP Output Head (eMamba)
//
// Structure:
//   Linear(D→H) → ReLU → Linear(H→D) → +residual → Linear(D→OUT)
//
// D=20 (token dim), H=64 (hidden), OUT=57 (joints×3)
//
// FSM: IDLE → FFN_IN → FFN_OUT → PROJ → DONE
// Latency: (D+3) + (H+3) + (D+3) = 23 + 67 + 23 = 113 cycles
//============================================================================

module output_head #(
    parameter D              = 20,
    parameter H              = 64,
    parameter OUT_DIM        = 57,
    parameter FFN_IN_SHIFT   = 6,
    parameter FFN_OUT_SHIFT  = 6,
    parameter PROJ_SHIFT     = 8,
    parameter RES_B_LSHIFT   = 0,   // dich residual pool (b) trai: = max(0, -(e(s_tr)-e(s_p)))
    parameter RES_B_RSHIFT   = 0,   // dich residual pool (b) phai: = max(0,  (e(s_tr)-e(s_p)))
    parameter FFN_IN_W_FILE  = "none",
    parameter FFN_IN_B_FILE  = "none",
    parameter FFN_OUT_W_FILE = "none",
    parameter FFN_OUT_B_FILE = "none",
    parameter PROJ_W_FILE    = "none",
    parameter PROJ_B_FILE    = "none",
    parameter LOAD_MODE      = 0
)(
    input  wire                  clk,
    input  wire                  rst_n,

    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire [D*8-1:0]        in_data,      // pooled vector D INT8

    output reg                   out_valid,
    input  wire                  out_ready,
    output reg  [OUT_DIM*8-1:0]  out_data,       // 57 INT8
    input  wire [15:0]           wbus            // write-bus runtime-load
);

    // ── FSM ──────────────────────────────────────────────────────
    localparam S_IDLE    = 3'd0;
    localparam S_FFN_IN  = 3'd1;
    localparam S_FFN_OUT = 3'd2;
    localparam S_PROJ    = 3'd3;
    localparam S_DONE    = 3'd4;

    reg [2:0] state, next_state;

    // ── Data registers ───────────────────────────────────────────
    reg [D*8-1:0] input_latch;        // latched input (stable during processing)
    reg [D*8-1:0] residual_reg;       // skip connection (pooled input)

    // ── Sub-module signals ───────────────────────────────────────
    // FFN_IN: Linear(D→H)
    reg  fi_iv; wire fi_ir, fi_ov; reg fi_or;
    wire [H*8-1:0] fi_out;
    wire [H*8-1:0] fi_relu;

    // FFN_OUT: Linear(H→D)
    reg  fo_iv; wire fo_ir, fo_ov; reg fo_or;
    wire [D*8-1:0] fo_out;

    // Residual: pooled + ffn_out
    wire [D*8-1:0] res_out;

    // PROJ: Linear(D→OUT)
    reg  pj_iv; wire pj_ir, pj_ov; reg pj_or;
    wire [OUT_DIM*8-1:0] pj_out;

    // ── Sub-modules ──────────────────────────────────────────────
    linear_layer #(.IN_DIM(D), .OUT_DIM(H), .SHIFT_RIGHT(FFN_IN_SHIFT), .USE_BIAS(1),
        .WEIGHT_FILE(FFN_IN_W_FILE), .BIAS_FILE(FFN_IN_B_FILE),
        .LOAD_MODE(LOAD_MODE), .WID(5'd25), .BID(5'd26))
    u_ffn_in (.clk(clk), .rst_n(rst_n), .shift_right(5'd0), .block_sel(1'b0),
        .in_valid(fi_iv), .in_ready(fi_ir), .in_data(input_latch),
        .out_valid(fi_ov), .out_ready(fi_or), .out_data(fi_out), .wbus(wbus));

    relu_unit #(.N_ELEM(H)) u_relu (.x_in(fi_out), .y_out(fi_relu));

    linear_layer #(.IN_DIM(H), .OUT_DIM(D), .SHIFT_RIGHT(FFN_OUT_SHIFT), .USE_BIAS(1),
        .WEIGHT_FILE(FFN_OUT_W_FILE), .BIAS_FILE(FFN_OUT_B_FILE),
        .LOAD_MODE(LOAD_MODE), .WID(5'd27), .BID(5'd28))
    u_ffn_out (.clk(clk), .rst_n(rst_n), .shift_right(5'd0), .block_sel(1'b0),
        .in_valid(fo_iv), .in_ready(fo_ir), .in_data(fi_relu),
        .out_valid(fo_ov), .out_ready(fo_or), .out_data(fo_out), .wbus(wbus));

    // SWAP: a=ffn_out (giu scale s_tr), b=residual pool (can dich ve s_tr).
    //   b co ca B_LSHIFT/B_RSHIFT -> xu ly duoc ca 2 huong (s_tr>s_p hoac <).
    //   golden: qt = fo_out + rsh(qp, e(s_tr)-e(s_p)).
    residual_add #(.N_ELEM(D), .B_LSHIFT(RES_B_LSHIFT), .B_RSHIFT(RES_B_RSHIFT)) u_res (
        .a_in(fo_out), .b_in(residual_reg),
        .a_lshift_rt(4'd0), .b_lshift_rt(4'd0), .b_rshift_rt(4'd0),
        .b_shift(5'd0), .y_out(res_out));

    linear_layer #(.IN_DIM(D), .OUT_DIM(OUT_DIM), .SHIFT_RIGHT(PROJ_SHIFT), .USE_BIAS(1),
        .WEIGHT_FILE(PROJ_W_FILE), .BIAS_FILE(PROJ_B_FILE),
        .LOAD_MODE(LOAD_MODE), .WID(5'd29), .BID(5'd30))
    u_proj (.clk(clk), .rst_n(rst_n), .shift_right(5'd0), .block_sel(1'b0),
        .in_valid(pj_iv), .in_ready(pj_ir), .in_data(res_out),
        .out_valid(pj_ov), .out_ready(pj_or), .out_data(pj_out), .wbus(wbus));

    // ── I/O ──────────────────────────────────────────────────────
    assign in_ready = (state == S_IDLE);

    // Debug prints disabled
    // synthesis translate_off
    // (HEAD debug $display removed)
    // synthesis translate_on

    // ── State register ───────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;
    end

    // ── Next state ───────────────────────────────────────────────
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:    if (in_valid)  next_state = S_FFN_IN;
            S_FFN_IN:  if (fi_ov)    next_state = S_FFN_OUT;
            S_FFN_OUT: if (fo_ov)    next_state = S_PROJ;
            S_PROJ:    if (pj_ov)    next_state = S_DONE;
            S_DONE:    if (out_ready) next_state = S_IDLE;
            default:                  next_state = S_IDLE;
        endcase
    end

    // ── Datapath ─────────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_latch  <= 0;
            residual_reg <= 0;
            out_valid    <= 0;
            out_data     <= 0;
            fi_iv <= 0; fi_or <= 0;
            fo_iv <= 0; fo_or <= 0;
            pj_iv <= 0; pj_or <= 0;
        end else begin
            fi_iv <= 0; fo_iv <= 0; pj_iv <= 0;

            if (out_valid && out_ready)
                out_valid <= 0;

            case (state)
                S_IDLE: begin
                    if (in_valid) begin
                        input_latch  <= in_data;  // latch before source changes
                        residual_reg <= in_data;
                        fi_iv <= 1;     // start ffn_in
                    end
                end

                S_FFN_IN: begin
                    fi_or <= 1;
                    if (fi_ov) begin
                        fi_or <= 0;
                        // fi_relu ready combinationally
                        fo_iv <= 1;     // start ffn_out with relu'd input
                    end
                end

                S_FFN_OUT: begin
                    fo_or <= 1;
                    if (fo_ov) begin
                        fo_or <= 0;
                        // res_out = residual + fo_out (combinational)
                        pj_iv <= 1;     // start projection with residual sum
                    end
                end

                S_PROJ: begin
                    pj_or <= 1;
                    if (pj_ov) begin
                        pj_or <= 0;
                        out_data  <= pj_out;
                        out_valid <= 1;
                    end
                end

                S_DONE: begin
                    // hold out_data
                end
            endcase
        end
    end

endmodule
