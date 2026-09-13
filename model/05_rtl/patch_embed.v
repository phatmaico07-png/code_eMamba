`timescale 1ns / 1ps
//============================================================================
// patch_embed.v — Patch Embedding = Partition + Linear(P²×C → D)
//
// Tương đương Conv2d(C→D, k=P, s=P) nhưng reuse linear_layer module.
// Đã chứng minh toán học equivalence trong test_patch_embed_equivalence.py.
//
// Flow:
//   Input (H, W, C) = (8, 8, 5) = 320 INT8 values
//   → Partition thành L = (H/P)×(W/P) = 16 patches
//   → Mỗi patch flatten: P×P×C = 2×2×5 = 20 values
//   → Linear(20→D) cho mỗi patch (reuse cùng weights)
//   → Output: L tokens × D values = 16 × 20
//
// Architecture:
//   - Patch MUX: counter chọn patch → extract 20 values từ input buffer
//   - linear_layer(20→20): tái sử dụng 16 lần, 1 lần per token
//   - Token buffer: lưu 16 × 20 = 320 INT8 outputs
//
// Latency: 16 × (IN_DIM + 3) = 16 × 23 = 368 cycles
// Resource: reuse 1 linear_layer instance
//============================================================================

module patch_embed #(
    parameter H           = 8,       // input height
    parameter W           = 8,       // input width
    parameter C           = 5,       // input channels
    parameter P           = 2,       // patch size
    parameter D           = 20,      // embedding dim = P*P*C (for MARS)
    parameter ACC_WIDTH   = 24,
    parameter SHIFT_RIGHT = 8,
    parameter WEIGHT_FILE = "none",
    parameter BIAS_FILE   = "none",
    parameter LOAD_MODE   = 0
)(
    input  wire                           clk,
    input  wire                           rst_n,

    // Input: full frame flat bus
    input  wire                           in_valid,
    output wire                           in_ready,
    input  wire [H*W*C*8-1:0]            in_data,    // (H×W×C) INT8 packed

    // Output: STREAMING 1 token/lan (D INT8 = 160-bit) — bat tay valid/ready.
    //   Buffer token_buf giu ben trong; con tro out_idx day tung token cho lop ke.
    output wire                           out_valid,
    input  wire                           out_ready,
    output wire [D*8-1:0]                 out_data,     // 1 token (D INT8)
    input  wire [15:0]                   wbus          // write-bus runtime-load
);

    // ── Derived parameters ────────────────────────────────────────
    localparam L        = (H/P) * (W/P);    // 16 tokens
    localparam PATCH_SZ = P * P * C;         // 20 values per patch
    localparam NH       = H / P;             // 4 patches vertically
    localparam NW       = W / P;             // 4 patches horizontally

    // ── FSM ───────────────────────────────────────────────────────
    localparam S_IDLE   = 3'd0;
    localparam S_PATCH  = 3'd1;   // extract patch + feed linear
    localparam S_WAIT   = 3'd2;   // wait linear output
    localparam S_STORE  = 3'd3;   // store token output
    localparam S_DRAIN  = 3'd4;   // da tinh xong het; cho stream day het token ra

    reg [2:0] state, next_state;

    // ── Counters ──────────────────────────────────────────────────
    reg [$clog2(L):0] token_idx;  // 0..15  (con tro TINH token)
    wire token_done = (token_idx == L - 1);
    reg [4:0] n_done;             // so token DA chot vao token_buf (0..16)
    reg [$clog2(L):0] out_idx;    // 0..16  (con tro DAY token ra ngoai — streaming)

    // ── Input buffer ──────────────────────────────────────────────
    reg [H*W*C*8-1:0] frame_buf;

    // ── Output buffer ─────────────────────────────────────────────
    reg [L*D*8-1:0] token_buf;

    // ── Patch extraction: token_idx → 20 INT8 values ──────────────
    // token_idx = ph * NW + pw, where ph=0..NH-1, pw=0..NW-1
    // Patch position: rows [ph*P .. ph*P+P-1], cols [pw*P .. pw*P+P-1]
    // Input layout: channels-last (H, W, C) → flat index = (h*W + w)*C + c
    wire [$clog2(NH):0] ph = token_idx / NW;
    wire [$clog2(NW):0] pw = token_idx % NW;

    reg [PATCH_SZ*8-1:0] patch_data;

    // Extract patch — combinational
    integer pi, pj, pc, flat_idx;
    always @(*) begin
        patch_data = {(PATCH_SZ*8){1'b0}};
        for (pi = 0; pi < P; pi = pi + 1) begin
            for (pj = 0; pj < P; pj = pj + 1) begin
                for (pc = 0; pc < C; pc = pc + 1) begin
                    // Source: frame_buf[(h*W + w)*C + c]
                    // h = ph*P + pi, w = pw*P + pj
                    // Dest: patch_data[(pi*P*C + pj*C + pc)*8 +: 8]
                    flat_idx = ((ph*P + pi)*W + (pw*P + pj))*C + pc;
                    patch_data[(pi*P*C + pj*C + pc)*8 +: 8] = frame_buf[flat_idx*8 +: 8];
                end
            end
        end
    end

    // ── Linear layer instance (reuse for all 16 tokens) ───────────
    reg                    lin_valid;
    wire                   lin_ready;
    wire                   lin_ovalid;
    reg                    lin_oready;
    wire [D*8-1:0]        lin_out;

    linear_layer #(
        .IN_DIM(PATCH_SZ), .OUT_DIM(D),
        .ACC_WIDTH(ACC_WIDTH), .SHIFT_RIGHT(SHIFT_RIGHT),
        .USE_BIAS(1),
        .WEIGHT_FILE(WEIGHT_FILE), .BIAS_FILE(BIAS_FILE),
        .LOAD_MODE(LOAD_MODE), .WID(5'd0), .BID(5'd1)
    ) u_linear (
        .clk(clk), .rst_n(rst_n),
        .shift_right(5'd0),
        .block_sel(1'b0),      // PE is not shared
        .in_valid(lin_valid), .in_ready(lin_ready), .in_data(patch_data),
        .out_valid(lin_ovalid), .out_ready(lin_oready), .out_data(lin_out),
        .wbus(wbus)
    );

    // ── Output assigns (STREAMING) ────────────────────────────────
    //   out_valid khi con token da chot chua day het (out_idx < n_done).
    //   out_data = token_buf[out_idx] (MUX). Bat tay: (out_valid & out_ready) -> out_idx++.
    assign in_ready  = (state == S_IDLE);
    assign out_valid = (out_idx < n_done);
    reg [D*8-1:0] out_data_r;
    always @(*) begin
        case (out_idx[$clog2(L)-1:0])
            4'd0:  out_data_r = token_buf[  0*D*8 +: D*8];
            4'd1:  out_data_r = token_buf[  1*D*8 +: D*8];
            4'd2:  out_data_r = token_buf[  2*D*8 +: D*8];
            4'd3:  out_data_r = token_buf[  3*D*8 +: D*8];
            4'd4:  out_data_r = token_buf[  4*D*8 +: D*8];
            4'd5:  out_data_r = token_buf[  5*D*8 +: D*8];
            4'd6:  out_data_r = token_buf[  6*D*8 +: D*8];
            4'd7:  out_data_r = token_buf[  7*D*8 +: D*8];
            4'd8:  out_data_r = token_buf[  8*D*8 +: D*8];
            4'd9:  out_data_r = token_buf[  9*D*8 +: D*8];
            4'd10: out_data_r = token_buf[ 10*D*8 +: D*8];
            4'd11: out_data_r = token_buf[ 11*D*8 +: D*8];
            4'd12: out_data_r = token_buf[ 12*D*8 +: D*8];
            4'd13: out_data_r = token_buf[ 13*D*8 +: D*8];
            4'd14: out_data_r = token_buf[ 14*D*8 +: D*8];
            4'd15: out_data_r = token_buf[ 15*D*8 +: D*8];
            default: out_data_r = {(D*8){1'b0}};
        endcase
    end
    assign out_data  = out_data_r;

    // ══════════════════════════════════════════════════════════════
    // BLOCK 1: STATE REGISTER
    // ══════════════════════════════════════════════════════════════
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            state <= S_IDLE;
        else
            state <= next_state;
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 2: NEXT STATE LOGIC
    // ══════════════════════════════════════════════════════════════
    always @(*) begin
        next_state = state;
        case (state)
            S_IDLE:  if (in_valid)     next_state = S_PATCH;
            S_PATCH: if (lin_ready)    next_state = S_WAIT;
            S_WAIT:  if (lin_ovalid)   next_state = S_STORE;
            S_STORE: if (token_done)   next_state = S_DRAIN;
                     else              next_state = S_PATCH;
            S_DRAIN: if (out_idx == L) next_state = S_IDLE;  // da day het 16 token ra ngoai
            default:                   next_state = S_IDLE;
        endcase
    end

    // ══════════════════════════════════════════════════════════════
    // BLOCK 3: DATAPATH
    // ══════════════════════════════════════════════════════════════
    integer tk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            frame_buf  <= {(H*W*C*8){1'b0}};
            token_buf  <= {(L*D*8){1'b0}};
            token_idx  <= 0;
            n_done     <= 0;
            out_idx    <= 0;
            lin_valid  <= 1'b0;
            lin_oready <= 1'b0;
        end else begin
            // ── STREAM OUT: day token_buf[out_idx] khi bat tay ──
            if (out_valid && out_ready)
                out_idx <= out_idx + 1'b1;

            case (state)
                S_IDLE: begin
                    token_idx  <= 0;
                    n_done     <= 0;
                    out_idx    <= 0;
                    lin_valid  <= 1'b0;
                    lin_oready <= 1'b0;
                    if (in_valid)
                        frame_buf <= in_data;
                end

                S_PATCH: begin
                    // patch_data computed combinationally from token_idx
                    lin_valid <= 1'b1;   // send to linear
                end

                S_WAIT: begin
                    lin_valid  <= 1'b0;
                    lin_oready <= 1'b1;  // ready to accept linear output
                end

                S_STORE: begin
                    // Store linear output into token buffer
                    for (tk = 0; tk < D; tk = tk + 1)
                        token_buf[(token_idx*D + tk)*8 +: 8] <= lin_out[tk*8 +: 8];

                    lin_oready <= 1'b0;
                    n_done <= n_done + 1'b1;   // C: token nay da chot -> bao co them 1 token san

                    if (!token_done)
                        token_idx <= token_idx + 1;
                end

                S_DRAIN: begin
                    // Da tinh xong het token; cho stream day het (out_idx==L) roi ve IDLE.
                    // token_buf giu nguyen; out_idx tang o khoi STREAM OUT tren.
                end
            endcase
        end
    end

endmodule
