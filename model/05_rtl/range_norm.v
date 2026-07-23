`timescale 1ns / 1ps
//============================================================================
// range_norm.v — Range Normalization (eMamba paper §4.1 + §5.3)
//
//   ŷ[i] = γ[i] × (x[i] - μ) / range + β[i]
//
// Architecture (Phase 3.5 — explicit divider):
//   - N_CU parallel non-restoring dividers (divider.v, 25-cycle each)
//   - γ INT8, β INT16 (per-element, loaded from hex)
//   - Q1.15 normalized value from divider
//   - GAMMA_SHIFT for output re-scaling
//
// FSM:
//   S_IDLE  → latch input, compute sum (combinational)
//   S_STATS → compute mean (×6554>>>17), max, min, range (1 cycle)
//   S_DIV   → start N_CU dividers, wait 25 cycles for done (25 cycles)
//   S_POST  → γ × quot >>> 15 + β + GAMMA_SHIFT + clamp (1 cycle)
//             if more batches → S_DIV, else → S_DONE
//   S_DONE  → output valid, wait out_ready
//
// Latency per token:
//   2 (IDLE+STATS) + ceil(DIM/N_CU) × (25+1) + 1
//   DIM=20, N_CU=10: 2 + 2×26 + 1 = 55 cycles
//   DIM=20, N_CU=20: 2 + 1×26 + 1 = 29 cycles
//
// Resource: N_CU dividers + N_CU DSP(×γ) + comparators
//============================================================================

module range_norm #(
    parameter DIM           = 20,
    parameter N_CU          = 20,       // parallel compute units (paper §5.3)
    parameter GAMMA_SHIFT   = 0,        // default gamma shift
    parameter SHARED        = 0,        // 1 = shared mode (double-depth arrays)
    parameter GAMMA_FILE    = "none",   // DIM × INT8 hex (block0 or single)
    parameter BETA_FILE     = "none",   // DIM × INT16 hex (block0 or single)
    parameter GAMMA_FILE_B1 = "none",   // block1 gamma (SHARED=1 only)
    parameter BETA_FILE_B1  = "none",   // block1 beta  (SHARED=1 only)
    parameter LOAD_MODE     = 0,        // 0=$readmemh (verified) ; 1=runtime byte-load
    parameter [4:0] GAMMA_ID = 5'd0,    // γ MY_ID (load-map sel)
    parameter [4:0] BETA_ID  = 5'd0     // β MY_ID
)(
    input  wire                    clk,
    input  wire                    rst_n,

    // Shared block select (0=block0, 1=block1); ignored when SHARED=0
    input  wire                    block_sel,

    // Runtime gamma shift override (signed: -3..+3, 0x1F = use parameter)
    input  wire signed [3:0]       gamma_shift_in,

    input  wire                    in_valid,
    output wire                    in_ready,
    input  wire [DIM*8-1:0]       in_data,

    output wire                    out_valid,
    input  wire                    out_ready,
    output wire [DIM*8-1:0]       out_data,

    // write-bus runtime-load (bundled): [15:8]=byte [7]=half [6:2]=sel [1]=seg_rst [0]=en
    input  wire [15:0]             wbus
);

    // ══════════════════════════════════════════════════════════════
    // DERIVED PARAMETERS
    // ══════════════════════════════════════════════════════════════
    localparam N_ITER = (DIM + N_CU - 1) / N_CU;  // ceil(DIM/N_CU) batches

    // Runtime gamma shift: use port if != 4'sb0111 (sentinel), else parameter
    wire signed [3:0] gs = (gamma_shift_in != 4'sb0111) ? gamma_shift_in
                                                         : GAMMA_SHIFT[3:0];

    localparam FRAC_BITS = 15;  // Q1.15 from divider
    localparam NUM_BITS  = 24;  // divider numerator width = 9(centered) + 15(frac)
    localparam DEN_BITS  = 9;   // divider denominator width
    localparam QUOT_BITS = 16;  // Q1.15 signed output

    // ══════════════════════════════════════════════════════════════
    // FSM — 5 states, one-hot
    // ══════════════════════════════════════════════════════════════
    localparam S_IDLE  = 5'b00001;
    localparam S_STATS = 5'b00010;
    localparam S_DIV   = 5'b00100;
    localparam S_POST  = 5'b01000;
    localparam S_DONE  = 5'b10000;

    (* fsm_encoding = "one_hot" *)
    reg [4:0] state, next_state;

    // ══════════════════════════════════════════════════════════════
    // COUNTERS
    // ══════════════════════════════════════════════════════════════
    localparam ITER_W = $clog2(N_ITER > 1 ? N_ITER : 2);
    reg [ITER_W:0] iter_idx;             // which batch (0..N_ITER-1)
    wire iter_last = (iter_idx == N_ITER - 1);

    // ══════════════════════════════════════════════════════════════
    // BUFFERS
    // ══════════════════════════════════════════════════════════════
    reg signed [7:0] x_buf [0:DIM-1];    // input token
    reg signed [7:0] y_buf [0:DIM-1];    // output token

    // ══════════════════════════════════════════════════════════════
    // STATISTICS
    // ══════════════════════════════════════════════════════════════
    reg signed [15:0] sum_val;
    reg signed [7:0]  mean_val;
    reg signed [8:0]  max_val, min_val;
    reg signed [8:0]  range_val;         // always > 0 (clamped to 1)
    reg        [1:0]  stats_phase;       // pipeline S_STATS: 0=mean, 1=range, 2=prep-divider

    // ══════════════════════════════════════════════════════════════
    // GAMMA / BETA storage — loaded from hex files
    // SHARED=1: double-depth [0..DIM-1]=block0, [DIM..2*DIM-1]=block1
    // SHARED=0: single-depth [0..DIM-1] (unused upper half optimized away)
    // ══════════════════════════════════════════════════════════════
    localparam COEFF_DEPTH = SHARED ? 2*DIM : DIM;
    reg signed [7:0]  gamma_buf [0:COEFF_DEPTH-1];
    reg signed [15:0] beta_buf  [0:COEFF_DEPTH-1];

    initial begin
        if (LOAD_MODE == 0 && GAMMA_FILE != "none") begin
            if (SHARED)
                $readmemh(GAMMA_FILE, gamma_buf, 0, DIM-1);
            else
                $readmemh(GAMMA_FILE, gamma_buf);
        end
        if (LOAD_MODE == 0 && SHARED && GAMMA_FILE_B1 != "none")
            $readmemh(GAMMA_FILE_B1, gamma_buf, DIM, 2*DIM-1);
        if (LOAD_MODE == 0 && BETA_FILE != "none") begin
            if (SHARED)
                $readmemh(BETA_FILE, beta_buf, 0, DIM-1);
            else
                $readmemh(BETA_FILE, beta_buf);
        end
        if (LOAD_MODE == 0 && SHARED && BETA_FILE_B1 != "none")
            $readmemh(BETA_FILE_B1, beta_buf, DIM, 2*DIM-1);
    end

    // ══════════════════════════════════════════════════════════════
    // WRITE-BUS byte-serial — γ (1B, sel GAMMA_ID) + β (2B, sel BETA_ID)
    //   LOAD_MODE=1: nạp runtime; w_half=1 → nửa block1 (offset DIM)
    // ══════════════════════════════════════════════════════════════
    localparam CW = $clog2(COEFF_DEPTH > 1 ? COEFF_DEPTH : 2);
    generate if (LOAD_MODE) begin : gen_wr
        wire        w_en      = wbus[0];
        wire        w_seg_rst = wbus[1];
        wire [4:0]  w_sel     = wbus[6:2];
        wire        w_half    = wbus[7];
        wire [7:0]  w_byte    = wbus[15:8];
        // γ : 1 byte/entry
        reg [CW-1:0] g_addr;
        always @(posedge clk) if (w_sel == GAMMA_ID) begin
            if (w_seg_rst)      g_addr <= w_half ? DIM[CW-1:0] : {CW{1'b0}};
            else if (w_en) begin gamma_buf[g_addr] <= w_byte; g_addr <= g_addr + 1'b1; end
        end
        // β : 2 byte/entry (MSB-first)
        reg [CW-1:0] bt_addr; reg bt_bc; reg [7:0] bt_hi;
        always @(posedge clk) if (w_sel == BETA_ID) begin
            if (w_seg_rst) begin bt_addr <= w_half ? DIM[CW-1:0] : {CW{1'b0}}; bt_bc <= 0; end
            else if (w_en) begin
                if (bt_bc == 0) begin bt_hi <= w_byte; bt_bc <= 1; end
                else begin beta_buf[bt_addr] <= {bt_hi, w_byte}; bt_addr <= bt_addr + 1'b1; bt_bc <= 0; end
            end
        end
    end endgenerate

    // Helper wires to access gamma/beta with optional block_sel offset
    // (works for both SHARED and non-SHARED, Verilog-2001 safe via function)
    // Use a localparam-based macro style: read via always-combinational wires
    // captured per element in the POST stage (see datapath below).

    // ══════════════════════════════════════════════════════════════
    // DIVIDER INSTANCES — N_CU parallel
    //   Packed vectors for Vivado compatibility (unpacked wire arrays
    //   in generate blocks can cause simulation/synthesis issues)
    // ══════════════════════════════════════════════════════════════
    reg                          div_start;
    reg  signed [NUM_BITS-1:0]   div_num  [0:N_CU-1];
    reg         [DEN_BITS-1:0]   div_den  [0:N_CU-1];

    // Packed vectors — 1 bit / QUOT_BITS per divider
    wire [N_CU-1:0]                     div_done_vec;
    wire [N_CU*QUOT_BITS-1:0]           div_quot_vec;

    genvar g;
    generate
        for (g = 0; g < N_CU; g = g + 1) begin : gen_div
            divider #(
                .NUM_BITS(NUM_BITS),
                .DEN_BITS(DEN_BITS),
                .QUOT_BITS(QUOT_BITS)
            ) u_div (
                .clk(clk),
                .rst_n(rst_n),
                .start(div_start),
                .numerator(div_num[g]),
                .denominator(div_den[g]),
                .done(div_done_vec[g]),
                .quotient(div_quot_vec[g*QUOT_BITS +: QUOT_BITS])
            );
        end
    endgenerate

    // All dividers finish simultaneously (same start, same latency)
    wire all_div_done = div_done_vec[0];

    // ══════════════════════════════════════════════════════════════
    // OUTPUT ASSIGNS
    // ══════════════════════════════════════════════════════════════
    assign in_ready  = (state == S_IDLE);
    assign out_valid = (state == S_DONE);

    genvar gi;
    generate
        for (gi = 0; gi < DIM; gi = gi + 1) begin : gen_out
            assign out_data[gi*8 +: 8] = y_buf[gi];
        end
    endgenerate

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
            S_IDLE:  if (in_valid)      next_state = S_STATS;
            S_STATS: if (stats_phase==2'd2) next_state = S_DIV;
            S_DIV:   if (all_div_done)  next_state = S_POST;
            S_POST:  if (iter_last)     next_state = S_DONE;
                     else               next_state = S_DIV;
            S_DONE:  if (out_ready)     next_state = S_IDLE;
            default:                    next_state = S_IDLE;
        endcase
    end

    // ══════════════════════════════════════════════════════════════
    // DEBUG: trace FSM transitions and divider timing
    // ══════════════════════════════════════════════════════════════
    // Debug prints disabled
    // synthesis translate_off
    // (RN debug $display removed)
    // synthesis translate_on

    // ══════════════════════════════════════════════════════════════
    // BLOCK 3: DATAPATH
    // ══════════════════════════════════════════════════════════════
    integer k, cu;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            iter_idx  <= 0;
            sum_val   <= 0;
            mean_val  <= 0;
            max_val   <= -9'sd256;
            min_val   <= 9'sd255;
            range_val <= 9'sd1;
            div_start <= 1'b0;
            stats_phase <= 2'd0;
            for (k = 0; k < DIM; k = k + 1) begin
                x_buf[k] <= 8'sd0;
                y_buf[k] <= 8'sd0;
            end
            for (cu = 0; cu < N_CU; cu = cu + 1) begin
                div_num[cu] <= {NUM_BITS{1'b0}};
                div_den[cu] <= {{(DEN_BITS-1){1'b0}}, 1'b1};
            end
        end else begin

            // Default: clear div_start after 1 cycle
            div_start <= 1'b0;

            case (state)

                // ── IDLE: latch input + compute sum ──────────────
                S_IDLE: begin
                    iter_idx <= 0;
                    if (in_valid) begin
                        begin : compute_sum
                            reg signed [15:0] s;
                            s = 0;
                            for (k = 0; k < DIM; k = k + 1) begin
                                x_buf[k] <= $signed(in_data[k*8 +: 8]);
                                s = s + $signed(in_data[k*8 +: 8]);
                            end
                            sum_val <= s;
                        end
                    end
                end

                // ── STATS: mean (×6554>>>17), max, min, range ────
                // S_STATS pipeline 3 PHA — cắt đường tổ hợp 54-level (mean+max/min+prep):
                //   pha0: mean (×6554) -> reg ; pha1: max/min/range ; pha2: setup divider
                S_STATS: case (stats_phase)
                    2'd0: begin
                        mean_val    <= (sum_val * 17'sd6554 + 32'sd65536) >>> 17;
                        stats_phase <= 2'd1;
                    end

                    2'd1: begin : find_range
                        reg signed [8:0] mx, mn, cv;
                        mx = -9'sd256; mn = 9'sd255;
                        for (k = 0; k < DIM; k = k + 1) begin
                            cv = {x_buf[k][7], x_buf[k]} - {mean_val[7], mean_val};
                            if (cv > mx) mx = cv;
                            if (cv < mn) mn = cv;
                        end
                        max_val     <= mx;
                        min_val     <= mn;
                        range_val   <= (mx - mn > 0) ? (mx - mn) : 9'sd1;
                        stats_phase <= 2'd2;
                    end

                    2'd2: begin : prep_div_batch0
                        reg signed [8:0] cent0;
                        reg signed [NUM_BITS-1:0] num0;
                        for (cu = 0; cu < N_CU; cu = cu + 1) begin
                            if (cu < DIM) begin
                                cent0 = {x_buf[cu][7], x_buf[cu]} - {mean_val[7], mean_val};
                                num0  = cent0 <<< FRAC_BITS;
                                div_num[cu] <= num0;
                                div_den[cu] <= range_val[DEN_BITS-1:0];
                            end else begin
                                div_num[cu] <= {NUM_BITS{1'b0}};
                                div_den[cu] <= {{(DEN_BITS-1){1'b0}}, 1'b1};
                            end
                        end
                        div_start   <= 1'b1;
                        iter_idx    <= 0;
                        stats_phase <= 2'd0;
                    end
                endcase

                // ── DIV: wait for dividers (25 cycles) ───────────
                S_DIV: begin
                    // Dividers running, just wait for all_div_done
                    // div_start already cleared by default
                end

                // ── POST: apply γ×quot + β + shift + clamp ───────
                S_POST: begin
                    for (cu = 0; cu < N_CU; cu = cu + 1) begin : post_cu
                        reg [7:0] eidx;
                        reg [7:0] cidx;   // coeff index (adds block_sel offset)
                        reg signed [QUOT_BITS-1:0] q;
                        reg signed [QUOT_BITS+7:0] prod;  // INT8 × Q1.15 = 24 bits
                        reg signed [15:0] raw;

                        eidx = iter_idx * N_CU + cu;
                        cidx = (block_sel ? DIM : 0) + eidx;

                        if (eidx < DIM) begin
                            q = $signed(div_quot_vec[cu*QUOT_BITS +: QUOT_BITS]);

                            // γ × normalized: INT8 × Q1.15 → Q8.15
                            prod = $signed(gamma_buf[cidx]) * q;

                            // Shift right by FRAC_BITS (15) with rounding + add β INT16
                            raw = ((prod + (1 <<< (FRAC_BITS-1))) >>> FRAC_BITS)
                                  + beta_buf[cidx];

                            // GAMMA_SHIFT: case MUX (signed, -3..+3)
                            case (gs)
                                -4'sd3: raw = raw <<< 3;
                                -4'sd2: raw = raw <<< 2;
                                -4'sd1: raw = raw <<< 1;
                                 4'sd0: raw = raw;            // no shift
                                 4'sd1: raw = (raw + 1) >>> 1;
                                 4'sd2: raw = (raw + 2) >>> 2;
                                 4'sd3: raw = (raw + 4) >>> 3;
                                default: raw = raw;
                            endcase

                            // Clamp to INT8
                            if (raw > 127)
                                y_buf[eidx] <= 8'sd127;
                            else if (raw < -128)
                                y_buf[eidx] <= -8'sd128;
                            else
                                y_buf[eidx] <= raw[7:0];
                        end
                    end

                    // Advance to next batch or finish
                    if (!iter_last) begin
                        iter_idx <= iter_idx + 1;

                        // Prepare next batch of dividers
                        for (cu = 0; cu < N_CU; cu = cu + 1) begin : prep_next
                            reg [7:0] nidx;
                            reg signed [8:0] ncent;
                            reg signed [NUM_BITS-1:0] nnum;

                            nidx = (iter_idx + 1) * N_CU + cu;

                            if (nidx < DIM) begin
                                ncent = {x_buf[nidx][7], x_buf[nidx]}
                                      - {mean_val[7], mean_val};
                                nnum = ncent <<< FRAC_BITS;
                                div_num[cu] <= nnum;
                                div_den[cu] <= range_val[DEN_BITS-1:0];
                            end else begin
                                div_num[cu] <= {NUM_BITS{1'b0}};
                                div_den[cu] <= {{(DEN_BITS-1){1'b0}}, 1'b1};
                            end
                        end
                        div_start <= 1'b1;
                    end
                end

                S_DONE: begin
                    // hold y_buf
                end

            endcase
        end
    end

endmodule
