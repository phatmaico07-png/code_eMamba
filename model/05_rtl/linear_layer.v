`timescale 1ns / 1ps
//============================================================================
// linear_layer.v — Parametric INT8 Linear Layer (y = Wx + b)
//
// Output Stationary: OUT_DIM accumulators song song.
// BRAM column-major: 1 addr → OUT_DIM weights cùng lúc.
// Pipeline delay tracking (_d1) che BRAM 1-cycle latency.
// Bias pre-load vào accumulator cycle đầu (gộp IDLE + LOAD_BIAS).
//
// Forward: y[o] = saturate( (Σ w[o][i]×x[i] + bias[o]) >>> shift )
//
// Requant: Case MUX cho runtime shift_right (shared Mamba block).
//   SHIFT_RIGHT parameter = default. shift_right port = runtime override.
//   Fixed >>> = bit selection (0 LUT). MUX ~12 LUT per output.
//
// FSM: IDLE → ACC (IN_DIM cycles) → REQUANT → DONE  (4 states)
// Latency: IN_DIM + 2 cycles
// Resource: OUT_DIM multipliers (40 DSP cho in_proj_upper)
//============================================================================

module linear_layer #(
    parameter IN_DIM      = 20,
    parameter OUT_DIM     = 40,
    parameter ACC_WIDTH   = 24,
    parameter SHIFT_RIGHT = 8,        // default shift (used when shift_right port = 0)
    parameter USE_BIAS    = 1,        // 1 = có bias, 0 = không bias (B_proj, C_proj)
    parameter SHARED      = 0,        // 1 = shared mode (2 blocks in 1 BRAM)
    parameter WEIGHT_FILE = "none",
    parameter BIAS_FILE   = "none",
    parameter WEIGHT_FILE_B1 = "none", // block1 weights (SHARED=1 only)
    parameter BIAS_FILE_B1   = "none", // block1 bias (SHARED=1 only)
    parameter LOAD_MODE      = 0,      // 0=$readmemh (verified) ; 1=runtime byte-load
    parameter [4:0] WID      = 5'd0,   // weight bram MY_ID (load-map sel)
    parameter [4:0] BID      = 5'd0,   // bias bram MY_ID
    parameter REQ_PARALLEL   = 1       // 0=requant noi tiep (it LUT, OUT_DIM cyc); 1=song song (1 cyc)
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // Runtime shift override (0 = use SHIFT_RIGHT parameter)
    input  wire [4:0]                  shift_right,

    // Shared block select (0=block0, 1=block1)
    input  wire                        block_sel,

    // Input: parallel bus
    input  wire                        in_valid,
    output wire                        in_ready,
    input  wire [IN_DIM*8-1:0]         in_data,

    // Output: parallel bus
    output wire                        out_valid,
    input  wire                        out_ready,
    output wire [OUT_DIM*8-1:0]        out_data,

    // write-bus runtime-load (bundled): [15:8]=byte [7]=half [6:2]=sel [1]=seg_rst [0]=en
    input  wire [15:0]                 wbus
);

    // ══════════════════════════════════════════════════════════════
    // FSM — 4 states one-hot
    // ══════════════════════════════════════════════════════════════
    localparam S_IDLE    = 4'b0001;   // chờ input + BRAM addr
    localparam S_ACC     = 4'b0010;   // MAC song song, cnt runs 0..IN_DIM
    localparam S_REQUANT = 4'b0100;   // shift + clamp → INT8
    localparam S_DONE    = 4'b1000;   // output ready

    (* fsm_encoding = "one_hot" *)
    reg [3:0] state, next_state;

    // ══════════════════════════════════════════════════════════════
    // COUNTER — runs to IN_DIM (1 extra cycle cho last MAC settle)
    // ══════════════════════════════════════════════════════════════
    localparam CNT_W = $clog2((IN_DIM+1) > 1 ? (IN_DIM+1) : 2);
    reg [CNT_W-1:0] cnt;
    wire cnt_done = (cnt == IN_DIM);  // 1 cycle sau MAC cuối

    // ══════════════════════════════════════════════════════════════
    // INPUT REGISTER
    // ══════════════════════════════════════════════════════════════
    reg [IN_DIM*8-1:0] x_reg;

    // ══════════════════════════════════════════════════════════════
    // WEIGHT BRAM — column-major: 1 addr → OUT_DIM weights
    // ══════════════════════════════════════════════════════════════
    localparam W_WIDTH = OUT_DIM * 8;

    reg  [CNT_W-1:0]   w_addr;
    wire [W_WIDTH-1:0]  w_data;

    weight_bram #(
        .DEPTH(IN_DIM), .WIDTH(W_WIDTH), .SHARED(SHARED),
        .INIT_FILE(WEIGHT_FILE), .INIT_FILE_B1(WEIGHT_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .MY_ID(WID)
    ) u_weight_bram (
        .clk(clk), .addr(w_addr), .block_sel(block_sel), .dout(w_data),
        .w_en(wbus[0]), .w_seg_rst(wbus[1]), .w_sel(wbus[6:2]), .w_half(wbus[7]), .w_byte(wbus[15:8])
    );

    // ══════════════════════════════════════════════════════════════
    // BIAS BRAM — chỉ tạo nếu USE_BIAS=1, Vivado optimize away nếu =0
    // ══════════════════════════════════════════════════════════════
    localparam B_WIDTH = OUT_DIM * 16;

    wire [B_WIDTH-1:0] b_data;

    generate
        if (USE_BIAS) begin : gen_bias
            weight_bram #(
                .DEPTH(1), .WIDTH(B_WIDTH), .SHARED(SHARED),
                .INIT_FILE(BIAS_FILE), .INIT_FILE_B1(BIAS_FILE_B1),
                .LOAD_MODE(LOAD_MODE), .MY_ID(BID)
            ) u_bias_bram (
                .clk(clk), .addr(1'b0), .block_sel(block_sel), .dout(b_data),
                .w_en(wbus[0]), .w_seg_rst(wbus[1]), .w_sel(wbus[6:2]), .w_half(wbus[7]), .w_byte(wbus[15:8])
            );
        end else begin : gen_no_bias
            assign b_data = {B_WIDTH{1'b0}};
        end
    endgenerate

    // ══════════════════════════════════════════════════════════════
    // PIPELINE DELAY — BRAM data chậm 1 cycle
    // ══════════════════════════════════════════════════════════════
    reg [CNT_W-1:0] cnt_d1;
    reg             acc_en;
    reg             bias_en;

    // ══════════════════════════════════════════════════════════════
    // PARALLEL ACCUMULATORS
    // ══════════════════════════════════════════════════════════════
    (* use_dsp = "yes" *)                       // ép MAC x*w + acc vào DSP48 (bit-exact)
    reg signed [ACC_WIDTH-1:0] acc [0:OUT_DIM-1];

    wire signed [7:0] x_cur;
    assign x_cur = $signed(x_reg[cnt_d1*8 +: 8]);   // mux dùng F7/F8 (rẻ hơn shift-reg vào LUT)

    // ══════════════════════════════════════════════════════════════
    // RE-QUANTIZE — Case MUX for runtime shift
    //   shift_right port = 0 → use SHIFT_RIGHT parameter (backward compat)
    //   shift_right port != 0 → use port value (shared block mode)
    //   Each case: fixed >>> = bit-select (0 LUT), MUX selects result
    // ══════════════════════════════════════════════════════════════
    wire [4:0] sh = (shift_right != 0) ? shift_right : SHIFT_RIGHT[4:0];

    // Rounding value via case (no barrel shifter)
    reg signed [ACC_WIDTH-1:0] rnd_val;
    always @(*) begin
        case (sh)
            5'd4:    rnd_val =       8;
            5'd5:    rnd_val =      16;
            5'd6:    rnd_val =      32;
            5'd7:    rnd_val =      64;
            5'd8:    rnd_val =     128;
            5'd9:    rnd_val =     256;
            5'd10:   rnd_val =     512;
            5'd11:   rnd_val =    1024;
            5'd12:   rnd_val =    2048;
            default: rnd_val =     128;  // fallback shift=8
        endcase
    end

    // A: requant NỐI TIẾP — 1 lane/chu kỳ, chia sẻ 1 bộ requant thay vì OUT_DIM.
    localparam RQ_W = $clog2((OUT_DIM > 1) ? OUT_DIM : 2);
    reg  [RQ_W-1:0] rq_cnt;
    wire            rq_done = (rq_cnt == OUT_DIM-1);

    reg [OUT_DIM*8-1:0] y_reg;

    // ══════════════════════════════════════════════════════════════
    // OUTPUT ASSIGNS
    // ══════════════════════════════════════════════════════════════
    assign in_ready  = (state == S_IDLE);
    assign out_valid = (state == S_DONE);
    assign out_data = y_reg;

    // ══════════════════════════════════════════════════════════════
    // BLOCK 1: STATE REGISTER (sequential)
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 2: NEXT STATE LOGIC (combinational)
    // ══════════════════════════════════════════════════════════════
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:    if (in_valid)    next_state = S_ACC;
            S_ACC:     if (cnt_done)    next_state = S_REQUANT;
            S_REQUANT: if (REQ_PARALLEL || rq_done) next_state = S_DONE;
            S_DONE:    if (out_ready)   next_state = S_IDLE;
            default:                    next_state = S_IDLE;
        endcase
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 3: DATAPATH (sequential)
    // ══════════════════════════════════════════════════════════════
    integer o;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg   <= {(IN_DIM*8){1'b0}};
            cnt     <= {CNT_W{1'b0}};
            w_addr  <= {CNT_W{1'b0}};
            cnt_d1  <= {CNT_W{1'b0}};
            acc_en  <= 1'b0;
            bias_en <= 1'b0;
            y_reg   <= {(OUT_DIM*8){1'b0}};
            rq_cnt  <= {RQ_W{1'b0}};
            for (o = 0; o < OUT_DIM; o = o + 1)
                acc[o] <= {ACC_WIDTH{1'b0}};
        end else begin

            // ── Pipeline delay tracking (mọi cycle) ──────────
            cnt_d1  <= cnt;
            acc_en  <= (state == S_ACC) && (cnt < IN_DIM);
            bias_en <= (state == S_IDLE && in_valid);

            // ── FSM datapath ─────────────────────────────────
            case (state)

                S_IDLE: begin
                    cnt    <= {CNT_W{1'b0}};
                    w_addr <= {CNT_W{1'b0}};
                    rq_cnt <= {RQ_W{1'b0}};
                    if (in_valid)
                        x_reg <= in_data;
                end

                S_ACC: begin
                    if (cnt < IN_DIM) begin
                        cnt <= cnt + 1'b1;
                    end
                    if (cnt < IN_DIM - 1) begin
                        w_addr <= w_addr + 1'b1;
                    end
                end

                S_REQUANT: begin
                  if (REQ_PARALLEL) begin : rqp   // #3: OUT_DIM lane trong 1 chu ky (it cycle, +LUT)
                        reg signed [ACC_WIDTH-1:0] arp;
                        reg signed [ACC_WIDTH-1:0] svp;
                        for (o = 0; o < OUT_DIM; o = o + 1) begin
                            arp = acc[o] + rnd_val;
                            case (sh)
                                5'd4:    svp = arp >>> 4;
                                5'd5:    svp = arp >>> 5;
                                5'd6:    svp = arp >>> 6;
                                5'd7:    svp = arp >>> 7;
                                5'd8:    svp = arp >>> 8;
                                5'd9:    svp = arp >>> 9;
                                5'd10:   svp = arp >>> 10;
                                5'd11:   svp = arp >>> 11;
                                5'd12:   svp = arp >>> 12;
                                default: svp = arp >>> 8;
                            endcase
                            if (svp > 127)        y_reg[o*8 +: 8] <= 8'sd127;
                            else if (svp < -128)  y_reg[o*8 +: 8] <= -8'sd128;
                            else                  y_reg[o*8 +: 8] <= svp[7:0];
                        end
                  end else begin : requant1       // serial 1 lane/chu ky (it LUT) — MAC DINH, da verify
                        reg signed [ACC_WIDTH-1:0] ar;
                        reg signed [ACC_WIDTH-1:0] sv;
                        ar = acc[rq_cnt] + rnd_val;
                        case (sh)
                            5'd4:    sv = ar >>> 4;
                            5'd5:    sv = ar >>> 5;
                            5'd6:    sv = ar >>> 6;
                            5'd7:    sv = ar >>> 7;
                            5'd8:    sv = ar >>> 8;
                            5'd9:    sv = ar >>> 9;
                            5'd10:   sv = ar >>> 10;
                            5'd11:   sv = ar >>> 11;
                            5'd12:   sv = ar >>> 12;
                            default: sv = ar >>> 8;
                        endcase
                        if (sv > 127)        y_reg[rq_cnt*8 +: 8] <= 8'sd127;
                        else if (sv < -128)  y_reg[rq_cnt*8 +: 8] <= -8'sd128;
                        else                 y_reg[rq_cnt*8 +: 8] <= sv[7:0];
                        rq_cnt <= rq_cnt + 1'b1;
                  end
                end

                S_DONE: begin
                    // hold y_reg
                end

            endcase

            // ── Accumulator update (delayed signals) ─────────
            if (bias_en) begin
                for (o = 0; o < OUT_DIM; o = o + 1)
                    acc[o] <= USE_BIAS ? {{(ACC_WIDTH-16){b_data[o*16+15]}}, b_data[o*16 +: 16]}
                                      : {ACC_WIDTH{1'b0}};
            end
            else if (acc_en) begin
                for (o = 0; o < OUT_DIM; o = o + 1)
                    acc[o] <= acc[o] + (x_cur * $signed(w_data[o*8 +: 8]));
            end

        end
    end

endmodule
