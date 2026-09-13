`timescale 1ns / 1ps
//============================================================================
// conv1d_dw.v — INT8 Depthwise 1D Convolution (causal, shift register)
//
// ED channels, kernel size K, causal (left-pad K-1 zeros).
// Each channel independent (depthwise = groups=ED in PyTorch).
//
// Architecture:
//   - Shift register: K entries × ED channels, persists across tokens
//   - Weight BRAM: K entries × (ED×8) bits, column-major
//   - Bias BRAM: 1 entry × (ED×16) bits
//   - ED parallel accumulators (Output Stationary, same as linear_layer)
//
// Data flow per token:
//   Token arrives → shift register updates → K MAC cycles → requant → output
//   Shift register starts all-zero (causal padding), cleared by rst_n/seq_clear
//
// Convolution formula per channel c:
//   y[c] = Σ_{k=0..K-1} w[c,k] × shift_reg[K-1-k][c] + b[c]
//   shift_reg[0] = newest (current token), shift_reg[K-1] = oldest
//
// Weight BRAM layout (column-major, same convention as linear_layer):
//   addr 0: w[0,0,0], w[1,0,0], ..., w[ED-1,0,0]  ← all channels, tap 0
//   addr 1: w[0,0,1], w[1,0,1], ..., w[ED-1,0,1]  ← all channels, tap 1
//   ...
//   addr K-1: all channels, tap K-1
//
// FSM: IDLE → ACC (K+1 cycles) → REQUANT → DONE
// Latency: K + 3 cycles per token (from in_valid to out_valid)
// Resource: ED multipliers (40 DSP for ED=40)
//============================================================================

module conv1d_dw #(
    parameter ED          = 40,
    parameter K           = 4,
    parameter ACC_WIDTH   = 24,
    parameter SHIFT_RIGHT = 8,
    parameter SHARED      = 0,
    parameter WEIGHT_FILE = "none",
    parameter BIAS_FILE   = "none",
    parameter WEIGHT_FILE_B1 = "none",
    parameter BIAS_FILE_B1   = "none",
    parameter LOAD_MODE      = 0,      // 0=$readmemh (verified) ; 1=runtime byte-load
    parameter [4:0] WID      = 5'd0,   // weight bram MY_ID
    parameter [4:0] BID      = 5'd0    // bias bram MY_ID
)(
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     seq_clear,
    input  wire [4:0]              shift_right,
    input  wire                     block_sel,     // shared block select

    // Input: one token (ED INT8 values)
    input  wire                     in_valid,
    output wire                     in_ready,
    input  wire [ED*8-1:0]         in_data,

    // Output: one token (ED INT8 values)
    output wire                     out_valid,
    input  wire                     out_ready,
    output wire [ED*8-1:0]         out_data,

    // write-bus runtime-load (bundled): [15:8]=byte [7]=half [6:2]=sel [1]=seg_rst [0]=en
    input  wire [15:0]              wbus
);

    // ══════════════════════════════════════════════════════════════
    // FSM — 4 states, one-hot
    // ══════════════════════════════════════════════════════════════
    localparam S_IDLE    = 4'b0001;
    localparam S_ACC     = 4'b0010;   // MAC K taps
    localparam S_REQUANT = 4'b0100;   // shift + clamp → INT8
    localparam S_DONE    = 4'b1000;   // output ready

    (* fsm_encoding = "one_hot" *)
    reg [3:0] state, next_state;

    // ══════════════════════════════════════════════════════════════
    // COUNTER — runs to K (1 extra cycle cho last MAC settle)
    // ══════════════════════════════════════════════════════════════
    localparam CNT_W = $clog2((K+1) > 1 ? (K+1) : 2);
    reg [CNT_W-1:0] cnt;
    wire cnt_done = (cnt == K);

    // ══════════════════════════════════════════════════════════════
    // SHIFT REGISTER — K entries × ED channels
    //   shift_reg[0] = newest (current token)
    //   shift_reg[K-1] = oldest
    //   Persists across tokens (maintains conv history)
    //   All-zero at reset/seq_clear = causal left-padding
    // ══════════════════════════════════════════════════════════════
    reg [ED*8-1:0] shift_reg [0:K-1];

    // ══════════════════════════════════════════════════════════════
    // WEIGHT BRAM — column-major: addr k → ED weights for tap k
    // ══════════════════════════════════════════════════════════════
    localparam W_WIDTH  = ED * 8;
    localparam W_ADDR_W = $clog2(K > 1 ? K : 2);

    reg  [W_ADDR_W-1:0] w_addr;
    wire [W_WIDTH-1:0]   w_data;

    weight_bram #(
        .DEPTH(K), .WIDTH(W_WIDTH), .SHARED(SHARED),
        .INIT_FILE(WEIGHT_FILE), .INIT_FILE_B1(WEIGHT_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .MY_ID(WID)
    ) u_wbram (
        .clk(clk), .addr(w_addr), .block_sel(block_sel), .dout(w_data),
        .w_en(wbus[0]), .w_seg_rst(wbus[1]), .w_sel(wbus[6:2]), .w_half(wbus[7]), .w_byte(wbus[15:8])
    );

    // ══════════════════════════════════════════════════════════════
    // BIAS BRAM — 1 entry, ED INT16 packed
    // ══════════════════════════════════════════════════════════════
    localparam B_WIDTH = ED * 16;

    wire [B_WIDTH-1:0] b_data;

    weight_bram #(
        .DEPTH(1), .WIDTH(B_WIDTH), .SHARED(SHARED),
        .INIT_FILE(BIAS_FILE), .INIT_FILE_B1(BIAS_FILE_B1),
        .LOAD_MODE(LOAD_MODE), .MY_ID(BID)
    ) u_bbram (
        .clk(clk), .addr(1'b0), .block_sel(block_sel), .dout(b_data),
        .w_en(wbus[0]), .w_seg_rst(wbus[1]), .w_sel(wbus[6:2]), .w_half(wbus[7]), .w_byte(wbus[15:8])
    );

    // ══════════════════════════════════════════════════════════════
    // PIPELINE DELAY — BRAM data chậm 1 cycle so với addr
    // ══════════════════════════════════════════════════════════════
    reg [CNT_W-1:0] cnt_d1;       // delayed counter (which BRAM entry arrived)
    reg             acc_en;        // accumulate enable (delayed)
    reg             bias_en;       // bias pre-load flag (delayed)

    // ══════════════════════════════════════════════════════════════
    // PARALLEL ACCUMULATORS — ED channels
    // ══════════════════════════════════════════════════════════════
    (* use_dsp = "yes" *)                       // ép MAC w*x + acc vào DSP48 (bit-exact)
    reg signed [ACC_WIDTH-1:0] acc [0:ED-1];

    // ── Shift register MUX ───────────────────────────────────────
    // At cnt_d1, BRAM returned w[all_c, cnt_d1]
    // Need to multiply with shift_reg[K-1-cnt_d1]
    //
    // Tap selection: combinational MUX on shift_reg
    // Safe because acc_en guards cnt_d1 < K
    reg [ED*8-1:0] shift_mux;

    integer mi;
    always @(*) begin
        shift_mux = shift_reg[0];   // default
        for (mi = 0; mi < K; mi = mi + 1) begin
            if (cnt_d1 == mi[CNT_W-1:0])
                shift_mux = shift_reg[K-1-mi];
        end
    end

    // ══════════════════════════════════════════════════════════════
    // REQUANT — Case MUX for runtime shift (same as linear_layer)
    // ══════════════════════════════════════════════════════════════
    wire [4:0] sh = (shift_right != 0) ? shift_right : SHIFT_RIGHT[4:0];

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
            default: rnd_val =     128;
        endcase
    end

    reg [ED*8-1:0] y_reg;

    // ══════════════════════════════════════════════════════════════
    // OUTPUT ASSIGNS
    // ══════════════════════════════════════════════════════════════
    assign in_ready  = (state == S_IDLE);
    assign out_valid = (state == S_DONE);
    assign out_data  = y_reg;

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
            S_REQUANT:                  next_state = S_DONE;
            S_DONE:    if (out_ready)   next_state = S_IDLE;
            default:                    next_state = S_IDLE;
        endcase
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 3: DATAPATH (sequential)
    // ══════════════════════════════════════════════════════════════
    integer c, t;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cnt     <= {CNT_W{1'b0}};
            cnt_d1  <= {CNT_W{1'b0}};
            w_addr  <= {W_ADDR_W{1'b0}};
            acc_en  <= 1'b0;
            bias_en <= 1'b0;
            y_reg   <= {(ED*8){1'b0}};
            for (t = 0; t < K; t = t + 1)
                shift_reg[t] <= {(ED*8){1'b0}};
            for (c = 0; c < ED; c = c + 1)
                acc[c] <= {ACC_WIDTH{1'b0}};
        end else begin

            // ── Seq clear: reset shift register for new sequence ──
            if (seq_clear) begin
                for (t = 0; t < K; t = t + 1)
                    shift_reg[t] <= {(ED*8){1'b0}};
            end

            // ── Pipeline delay tracking (every cycle) ────────────
            cnt_d1  <= cnt;
            acc_en  <= (state == S_ACC) && (cnt < K);
            bias_en <= (state == S_IDLE && in_valid);

            // ── FSM datapath ─────────────────────────────────────
            case (state)

                S_IDLE: begin
                    cnt    <= {CNT_W{1'b0}};
                    w_addr <= {W_ADDR_W{1'b0}};
                    if (in_valid) begin
                        // Shift register: push newest, oldest drops off
                        for (t = K-1; t > 0; t = t - 1)
                            shift_reg[t] <= shift_reg[t-1];
                        shift_reg[0] <= in_data;
                    end
                end

                S_ACC: begin
                    if (cnt < K)
                        cnt <= cnt + 1'b1;
                    if (cnt < K - 1)
                        w_addr <= w_addr + 1'b1;
                end

                S_REQUANT: begin
                    for (c = 0; c < ED; c = c + 1) begin : requant_loop
                        reg signed [ACC_WIDTH-1:0] ar, sv;
                        ar = acc[c] + rnd_val;
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
                        if (sv > 127)
                            y_reg[c*8 +: 8] <= 8'sd127;
                        else if (sv < -128)
                            y_reg[c*8 +: 8] <= -8'sd128;
                        else
                            y_reg[c*8 +: 8] <= sv[7:0];
                    end
                end

                S_DONE: begin
                    // hold y_reg until out_ready
                end

            endcase

            // ── Accumulator update (1-cycle delayed signals) ─────
            if (bias_en) begin
                // Pre-load bias into accumulators (sign-extend INT16 → ACC_WIDTH)
                for (c = 0; c < ED; c = c + 1)
                    acc[c] <= {{(ACC_WIDTH-16){b_data[c*16+15]}}, b_data[c*16 +: 16]};
            end
            else if (acc_en) begin
                // MAC: acc[c] += w[c, cnt_d1] × shift_reg[K-1-cnt_d1][c]
                for (c = 0; c < ED; c = c + 1)
                    acc[c] <= acc[c]
                            + ($signed(shift_mux[c*8 +: 8])
                             * $signed(w_data[c*8 +: 8]));
            end

        end
    end

endmodule
